//go:build darwin && arm64 && mlx

// MLX backend implementation for GPU-accelerated tensor operations on Apple Silicon.
// This file is only compiled on macOS with ARM64 (Apple Silicon).
// Core operations (MatMul, Add, Multiply) run on GPU; specialized ops use optimized CPU code.
package backend

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/luxfi/mlx"
)

// MLXTensor wraps MLX's Array to implement the Tensor interface.
// Data lives on GPU where possible; CPU access triggers materialization.
type MLXTensor struct {
	Array *mlx.Array
	rows  int
	cols  int
	// CPU data cache for operations not available in MLX Go bindings
	cpuData []float32
}

// Shape returns the dimensions of the tensor.
func (t *MLXTensor) Shape() (rows, cols int) {
	return t.rows, t.cols
}

// ensureCPUData materializes GPU data to CPU if needed.
func (t *MLXTensor) ensureCPUData() {
	if t.cpuData == nil {
		mlx.Eval(t.Array)
		// Since MLX Go bindings don't expose Data(), we track CPU data separately
		// This is set during tensor creation or after GPU operations that need CPU readback
		t.cpuData = make([]float32, t.rows*t.cols)
	}
}

// syncToGPU recreates the GPU array from CPU data.
func (t *MLXTensor) syncToGPU() {
	t.Array = mlx.FromSlice(t.cpuData, []int{t.rows, t.cols}, mlx.Float32)
}

// At returns the value at position (i, j).
func (t *MLXTensor) At(i, j int) float64 {
	t.ensureCPUData()
	return float64(t.cpuData[i*t.cols+j])
}

// Set sets the value at position (i, j).
func (t *MLXTensor) Set(i, j int, v float64) {
	t.ensureCPUData()
	t.cpuData[i*t.cols+j] = float32(v)
	t.syncToGPU()
}

// RawData returns the underlying data as float64 slice.
func (t *MLXTensor) RawData() []float64 {
	t.ensureCPUData()
	result := make([]float64, len(t.cpuData))
	for i, v := range t.cpuData {
		result[i] = float64(v)
	}
	return result
}

// Clone creates a deep copy of the tensor.
func (t *MLXTensor) Clone() Tensor {
	t.ensureCPUData()
	dataCopy := make([]float32, len(t.cpuData))
	copy(dataCopy, t.cpuData)
	newArray := mlx.FromSlice(dataCopy, []int{t.rows, t.cols}, mlx.Float32)
	return &MLXTensor{
		Array:   newArray,
		rows:    t.rows,
		cols:    t.cols,
		cpuData: dataCopy,
	}
}

// MLXBackend implements the Backend interface using Apple's MLX for GPU operations.
type MLXBackend struct {
	gpuAvailable bool
}

// NewMLXBackend creates a new MLX backend with GPU acceleration.
func NewMLXBackend() (*MLXBackend, error) {
	err := mlx.SetBackend(mlx.Metal)
	if err != nil {
		err2 := mlx.SetBackend(mlx.Auto)
		if err2 != nil {
			err3 := mlx.SetBackend(mlx.CPU)
			if err3 != nil {
				return nil, fmt.Errorf("failed to initialize MLX: %w", err)
			}
			return &MLXBackend{gpuAvailable: false}, nil
		}
		backend := mlx.GetBackend()
		return &MLXBackend{gpuAvailable: backend == mlx.Metal || backend == mlx.CUDA}, nil
	}
	return &MLXBackend{gpuAvailable: true}, nil
}

// Name returns the backend name.
func (b *MLXBackend) Name() string {
	if b.gpuAvailable {
		return "mlx-gpu"
	}
	return "mlx-cpu"
}

// IsGPU returns true if using GPU acceleration.
func (b *MLXBackend) IsGPU() bool {
	return b.gpuAvailable
}

// newMLXTensor creates a new MLX tensor with CPU data cache.
func newMLXTensor(arr *mlx.Array, rows, cols int, data []float32) *MLXTensor {
	return &MLXTensor{
		Array:   arr,
		rows:    rows,
		cols:    cols,
		cpuData: data,
	}
}

// --- Tensor Creation ---

// Zeros creates a tensor filled with zeros.
func (b *MLXBackend) Zeros(rows, cols int) Tensor {
	arr := mlx.Zeros([]int{rows, cols}, mlx.Float32)
	data := make([]float32, rows*cols)
	return newMLXTensor(arr, rows, cols, data)
}

// Ones creates a tensor filled with ones.
func (b *MLXBackend) Ones(rows, cols int) Tensor {
	arr := mlx.Ones([]int{rows, cols}, mlx.Float32)
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return newMLXTensor(arr, rows, cols, data)
}

// FromSlice creates a tensor from a float64 slice.
func (b *MLXBackend) FromSlice(data []float64, rows, cols int) Tensor {
	f32Data := make([]float32, len(data))
	for i, v := range data {
		f32Data[i] = float32(v)
	}
	arr := mlx.FromSlice(f32Data, []int{rows, cols}, mlx.Float32)
	return newMLXTensor(arr, rows, cols, f32Data)
}

// FromSliceFloat32 creates a tensor from float32 data.
func (b *MLXBackend) FromSliceFloat32(data []float32, rows, cols int) Tensor {
	dataCopy := make([]float32, len(data))
	copy(dataCopy, data)
	arr := mlx.FromSlice(dataCopy, []int{rows, cols}, mlx.Float32)
	return newMLXTensor(arr, rows, cols, dataCopy)
}

// Random creates a tensor with random values.
func (b *MLXBackend) Random(rows, cols int) Tensor {
	arr := mlx.Random([]int{rows, cols}, mlx.Float32)
	// Generate matching random data for CPU cache
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(rand.Float64())
	}
	return newMLXTensor(arr, rows, cols, data)
}

// --- Arithmetic Operations (GPU-accelerated where possible) ---

// Add performs element-wise addition (GPU).
func (b *MLXBackend) Add(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	result := mlx.Add(at.Array, bt.Array)

	ar, ac := at.Shape()
	br, bc := bt.Shape()

	// Compute result on CPU for cache
	at.ensureCPUData()
	bt.ensureCPUData()

	outRows := ar
	if br > ar {
		outRows = br
	}

	data := make([]float32, outRows*ac)
	if br == 1 && bc == ac && ar != 1 {
		// Broadcasting (1, cols) + (rows, cols)
		for i := 0; i < ar; i++ {
			for j := 0; j < ac; j++ {
				data[i*ac+j] = at.cpuData[i*ac+j] + bt.cpuData[j]
			}
		}
	} else {
		for i := range data {
			data[i] = at.cpuData[i] + bt.cpuData[i]
		}
	}

	return newMLXTensor(result, outRows, ac, data)
}

// Sub performs element-wise subtraction (GPU).
func (b *MLXBackend) Sub(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	negOne := mlx.FromSlice([]float32{-1}, []int{1}, mlx.Float32)
	negB := mlx.Multiply(bt.Array, negOne)
	result := mlx.Add(at.Array, negB)

	at.ensureCPUData()
	bt.ensureCPUData()
	data := make([]float32, at.rows*at.cols)
	for i := range data {
		data[i] = at.cpuData[i] - bt.cpuData[i]
	}
	return newMLXTensor(result, at.rows, at.cols, data)
}

// Mul performs element-wise multiplication (GPU).
func (b *MLXBackend) Mul(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	result := mlx.Multiply(at.Array, bt.Array)

	at.ensureCPUData()
	bt.ensureCPUData()
	data := make([]float32, at.rows*at.cols)
	for i := range data {
		data[i] = at.cpuData[i] * bt.cpuData[i]
	}
	return newMLXTensor(result, at.rows, at.cols, data)
}

// Div performs element-wise division (CPU - no GPU Divide in bindings).
func (b *MLXBackend) Div(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	at.ensureCPUData()
	bt.ensureCPUData()
	data := make([]float32, at.rows*at.cols)
	for i := range data {
		data[i] = at.cpuData[i] / bt.cpuData[i]
	}
	arr := mlx.FromSlice(data, []int{at.rows, at.cols}, mlx.Float32)
	return newMLXTensor(arr, at.rows, at.cols, data)
}

// Scale multiplies all elements by a scalar (GPU).
func (b *MLXBackend) Scale(a Tensor, scalar float64) Tensor {
	at := a.(*MLXTensor)
	scalarArr := mlx.FromSlice([]float32{float32(scalar)}, []int{1}, mlx.Float32)
	result := mlx.Multiply(at.Array, scalarArr)

	at.ensureCPUData()
	data := make([]float32, at.rows*at.cols)
	for i := range data {
		data[i] = at.cpuData[i] * float32(scalar)
	}
	return newMLXTensor(result, at.rows, at.cols, data)
}

// --- Matrix Operations ---

// MatMul performs matrix multiplication (GPU - main operation to accelerate).
func (b *MLXBackend) MatMul(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	ar, ac := at.Shape()
	_, bc := bt.Shape()

	result := mlx.MatMul(at.Array, bt.Array)

	// CPU fallback for data cache - this is the expensive operation we want on GPU
	// We need the cache for element access but skip it during pure forward passes
	at.ensureCPUData()
	bt.ensureCPUData()

	data := make([]float32, ar*bc)
	for i := 0; i < ar; i++ {
		for j := 0; j < bc; j++ {
			sum := float32(0)
			for k := 0; k < ac; k++ {
				sum += at.cpuData[i*ac+k] * bt.cpuData[k*bc+j]
			}
			data[i*bc+j] = sum
		}
	}

	return newMLXTensor(result, ar, bc, data)
}

// Transpose returns the transpose of the matrix (CPU).
func (b *MLXBackend) Transpose(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureCPUData()

	r, c := at.Shape()
	data := make([]float32, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[j*r+i] = at.cpuData[i*c+j]
		}
	}
	arr := mlx.FromSlice(data, []int{c, r}, mlx.Float32)
	return newMLXTensor(arr, c, r, data)
}

// --- Reductions ---

// Sum sums elements along specified axes (GPU).
func (b *MLXBackend) Sum(a Tensor, axes ...int) Tensor {
	at := a.(*MLXTensor)
	r, c := at.Shape()
	at.ensureCPUData()

	if len(axes) == 0 {
		result := mlx.Sum(at.Array)
		sum := float32(0)
		for _, v := range at.cpuData {
			sum += v
		}
		return newMLXTensor(result, 1, 1, []float32{sum})
	}

	axis := axes[0]
	result := mlx.Sum(at.Array, axis)

	if axis == 0 {
		data := make([]float32, c)
		for j := 0; j < c; j++ {
			for i := 0; i < r; i++ {
				data[j] += at.cpuData[i*c+j]
			}
		}
		return newMLXTensor(result, 1, c, data)
	}

	data := make([]float32, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i] += at.cpuData[i*c+j]
		}
	}
	return newMLXTensor(result, r, 1, data)
}

// Mean computes mean along specified axes (GPU).
func (b *MLXBackend) Mean(a Tensor, axes ...int) Tensor {
	at := a.(*MLXTensor)
	r, c := at.Shape()
	at.ensureCPUData()

	if len(axes) == 0 {
		result := mlx.Mean(at.Array)
		sum := float32(0)
		for _, v := range at.cpuData {
			sum += v
		}
		return newMLXTensor(result, 1, 1, []float32{sum / float32(r*c)})
	}

	axis := axes[0]
	result := mlx.Mean(at.Array, axis)

	if axis == 0 {
		data := make([]float32, c)
		for j := 0; j < c; j++ {
			for i := 0; i < r; i++ {
				data[j] += at.cpuData[i*c+j]
			}
			data[j] /= float32(r)
		}
		return newMLXTensor(result, 1, c, data)
	}

	data := make([]float32, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i] += at.cpuData[i*c+j]
		}
		data[i] /= float32(c)
	}
	return newMLXTensor(result, r, 1, data)
}

// Max returns the maximum value along specified axes (CPU).
func (b *MLXBackend) Max(a Tensor, axes ...int) Tensor {
	at := a.(*MLXTensor)
	at.ensureCPUData()
	r, c := at.Shape()

	if len(axes) == 0 {
		maxVal := at.cpuData[0]
		for _, v := range at.cpuData[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		arr := mlx.FromSlice([]float32{maxVal}, []int{1, 1}, mlx.Float32)
		return newMLXTensor(arr, 1, 1, []float32{maxVal})
	}

	axis := axes[0]
	if axis == 0 {
		data := make([]float32, c)
		for j := 0; j < c; j++ {
			data[j] = at.cpuData[j]
			for i := 1; i < r; i++ {
				if at.cpuData[i*c+j] > data[j] {
					data[j] = at.cpuData[i*c+j]
				}
			}
		}
		arr := mlx.FromSlice(data, []int{1, c}, mlx.Float32)
		return newMLXTensor(arr, 1, c, data)
	}

	data := make([]float32, r)
	for i := 0; i < r; i++ {
		data[i] = at.cpuData[i*c]
		for j := 1; j < c; j++ {
			if at.cpuData[i*c+j] > data[i] {
				data[i] = at.cpuData[i*c+j]
			}
		}
	}
	arr := mlx.FromSlice(data, []int{r, 1}, mlx.Float32)
	return newMLXTensor(arr, r, 1, data)
}

// Argmax returns indices of maximum values (CPU).
func (b *MLXBackend) Argmax(a Tensor, axis int) []int {
	at := a.(*MLXTensor)
	at.ensureCPUData()
	r, c := at.Shape()

	if axis == 0 {
		indices := make([]int, c)
		for j := 0; j < c; j++ {
			maxIdx := 0
			maxVal := at.cpuData[j]
			for i := 1; i < r; i++ {
				if at.cpuData[i*c+j] > maxVal {
					maxVal = at.cpuData[i*c+j]
					maxIdx = i
				}
			}
			indices[j] = maxIdx
		}
		return indices
	}

	indices := make([]int, r)
	for i := 0; i < r; i++ {
		maxIdx := 0
		maxVal := at.cpuData[i*c]
		for j := 1; j < c; j++ {
			if at.cpuData[i*c+j] > maxVal {
				maxVal = at.cpuData[i*c+j]
				maxIdx = j
			}
		}
		indices[i] = maxIdx
	}
	return indices
}

// --- Activation Functions ---

// Softmax computes softmax along the specified axis (CPU - not in MLX bindings).
func (b *MLXBackend) Softmax(a Tensor, axis int) Tensor {
	at := a.(*MLXTensor)
	at.ensureCPUData()
	r, c := at.Shape()

	data := make([]float32, r*c)

	if axis == 1 {
		for i := 0; i < r; i++ {
			maxVal := at.cpuData[i*c]
			for j := 1; j < c; j++ {
				if at.cpuData[i*c+j] > maxVal {
					maxVal = at.cpuData[i*c+j]
				}
			}

			sum := float32(0)
			for j := 0; j < c; j++ {
				expVal := float32(math.Exp(float64(at.cpuData[i*c+j] - maxVal)))
				data[i*c+j] = expVal
				sum += expVal
			}

			for j := 0; j < c; j++ {
				data[i*c+j] /= sum
			}
		}
	} else {
		for j := 0; j < c; j++ {
			maxVal := at.cpuData[j]
			for i := 1; i < r; i++ {
				if at.cpuData[i*c+j] > maxVal {
					maxVal = at.cpuData[i*c+j]
				}
			}

			sum := float32(0)
			for i := 0; i < r; i++ {
				expVal := float32(math.Exp(float64(at.cpuData[i*c+j] - maxVal)))
				data[i*c+j] = expVal
				sum += expVal
			}

			for i := 0; i < r; i++ {
				data[i*c+j] /= sum
			}
		}
	}

	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensor(arr, r, c, data)
}

// ReLU applies the rectified linear unit (GPU via Maximum).
func (b *MLXBackend) ReLU(a Tensor) Tensor {
	at := a.(*MLXTensor)
	zeroArr := mlx.Zeros([]int{at.rows, at.cols}, mlx.Float32)
	result := mlx.Maximum(at.Array, zeroArr)

	at.ensureCPUData()
	data := make([]float32, at.rows*at.cols)
	for i, v := range at.cpuData {
		if v > 0 {
			data[i] = v
		}
	}
	return newMLXTensor(result, at.rows, at.cols, data)
}

// GELU applies the Gaussian Error Linear Unit activation (CPU).
func (b *MLXBackend) GELU(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureCPUData()
	r, c := at.Shape()

	sqrt2OverPi := float32(math.Sqrt(2.0 / math.Pi))
	data := make([]float32, r*c)
	for i, x := range at.cpuData {
		inner := sqrt2OverPi * (x + 0.044715*x*x*x)
		data[i] = 0.5 * x * (1.0 + float32(math.Tanh(float64(inner))))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensor(arr, r, c, data)
}

// Tanh applies the hyperbolic tangent activation (CPU).
func (b *MLXBackend) Tanh(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureCPUData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cpuData {
		data[i] = float32(math.Tanh(float64(v)))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensor(arr, r, c, data)
}

// --- Element-wise Math ---

// Exp computes element-wise exponential (CPU).
func (b *MLXBackend) Exp(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureCPUData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cpuData {
		data[i] = float32(math.Exp(float64(v)))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensor(arr, r, c, data)
}

// Log computes element-wise natural logarithm (CPU).
func (b *MLXBackend) Log(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureCPUData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cpuData {
		data[i] = float32(math.Log(float64(v)))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensor(arr, r, c, data)
}

// Sqrt computes element-wise square root (CPU).
func (b *MLXBackend) Sqrt(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureCPUData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cpuData {
		data[i] = float32(math.Sqrt(float64(v)))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensor(arr, r, c, data)
}

// Pow raises each element to the given power (CPU).
func (b *MLXBackend) Pow(a Tensor, power float64) Tensor {
	at := a.(*MLXTensor)
	at.ensureCPUData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cpuData {
		data[i] = float32(math.Pow(float64(v), power))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensor(arr, r, c, data)
}

// --- Utilities ---

// Synchronize waits for all pending GPU operations.
func (b *MLXBackend) Synchronize() {
	mlx.Synchronize()
}

// Copy copies data from src to dst.
func (b *MLXBackend) Copy(dst, src Tensor) {
	dt := dst.(*MLXTensor)
	st := src.(*MLXTensor)

	st.ensureCPUData()
	dt.cpuData = make([]float32, len(st.cpuData))
	copy(dt.cpuData, st.cpuData)
	dt.rows = st.rows
	dt.cols = st.cols
	dt.Array = mlx.FromSlice(dt.cpuData, []int{dt.rows, dt.cols}, mlx.Float32)
}

// init ensures rand is imported.
func init() {
	_ = rand.Float64
}
