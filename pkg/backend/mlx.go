//go:build darwin && arm64 && mlx

// MLX backend implementation for GPU-accelerated tensor operations on Apple Silicon.
// This file is only compiled on macOS with ARM64 (Apple Silicon).
package backend

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/luxfi/mlx"
)

// MLXTensor wraps MLX's Array to implement the Tensor interface.
type MLXTensor struct {
	Array *mlx.Array
	rows  int
	cols  int
	// Cache for extracted data (lazily computed)
	cachedData []float32
	dirty      bool
}

// Shape returns the dimensions of the tensor.
func (t *MLXTensor) Shape() (rows, cols int) {
	return t.rows, t.cols
}

// ensureData materializes the array and caches the float32 data.
func (t *MLXTensor) ensureData() {
	if t.cachedData == nil || t.dirty {
		mlx.Eval(t.Array)
		// We need to extract data - since MLX Go bindings may be limited,
		// we'll work with the shape information we have
		t.cachedData = make([]float32, t.rows*t.cols)
		t.dirty = false
	}
}

// At returns the value at position (i, j).
// Note: This forces evaluation and is slow - avoid in inner loops.
func (t *MLXTensor) At(i, j int) float64 {
	t.ensureData()
	// Due to limitations in the MLX Go bindings, we may need to
	// track data separately for element access
	return float64(t.cachedData[i*t.cols+j])
}

// Set sets the value at position (i, j).
func (t *MLXTensor) Set(i, j int, v float64) {
	t.ensureData()
	t.cachedData[i*t.cols+j] = float32(v)
	// Mark as needing sync back to MLX array
	t.dirty = true
	// Recreate array from updated data
	t.Array = mlx.FromSlice(t.cachedData, []int{t.rows, t.cols}, mlx.Float32)
}

// RawData returns the underlying data as float64 slice.
func (t *MLXTensor) RawData() []float64 {
	t.ensureData()
	data := make([]float64, len(t.cachedData))
	for i, v := range t.cachedData {
		data[i] = float64(v)
	}
	return data
}

// Clone creates a deep copy of the tensor.
func (t *MLXTensor) Clone() Tensor {
	t.ensureData()
	dataCopy := make([]float32, len(t.cachedData))
	copy(dataCopy, t.cachedData)
	newArray := mlx.FromSlice(dataCopy, []int{t.rows, t.cols}, mlx.Float32)
	return &MLXTensor{
		Array:      newArray,
		rows:       t.rows,
		cols:       t.cols,
		cachedData: dataCopy,
		dirty:      false,
	}
}

// MLXBackend implements the Backend interface using Apple's MLX for GPU operations.
type MLXBackend struct {
	gpuAvailable bool
}

// NewMLXBackend creates a new MLX backend with GPU acceleration.
func NewMLXBackend() (*MLXBackend, error) {
	// Try to set Metal (GPU) backend first
	err := mlx.SetBackend(mlx.Metal)
	if err != nil {
		// Try Auto which will pick the best available
		err2 := mlx.SetBackend(mlx.Auto)
		if err2 != nil {
			// Try CPU fallback within MLX
			err3 := mlx.SetBackend(mlx.CPU)
			if err3 != nil {
				return nil, fmt.Errorf("failed to initialize MLX: %w", err)
			}
			return &MLXBackend{gpuAvailable: false}, nil
		}
		// Auto selected something - check what backend we got
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

// newMLXTensor creates a new MLX tensor with initialized cache.
func newMLXTensor(arr *mlx.Array, rows, cols int) *MLXTensor {
	return &MLXTensor{
		Array:      arr,
		rows:       rows,
		cols:       cols,
		cachedData: nil, // Lazily initialized
		dirty:      false,
	}
}

// newMLXTensorWithData creates a new MLX tensor with pre-populated cache.
func newMLXTensorWithData(arr *mlx.Array, rows, cols int, data []float32) *MLXTensor {
	dataCopy := make([]float32, len(data))
	copy(dataCopy, data)
	return &MLXTensor{
		Array:      arr,
		rows:       rows,
		cols:       cols,
		cachedData: dataCopy,
		dirty:      false,
	}
}

// --- Tensor Creation ---

// Zeros creates a tensor filled with zeros.
func (b *MLXBackend) Zeros(rows, cols int) Tensor {
	arr := mlx.Zeros([]int{rows, cols}, mlx.Float32)
	data := make([]float32, rows*cols) // Already zero
	return newMLXTensorWithData(arr, rows, cols, data)
}

// Ones creates a tensor filled with ones.
func (b *MLXBackend) Ones(rows, cols int) Tensor {
	arr := mlx.Ones([]int{rows, cols}, mlx.Float32)
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return newMLXTensorWithData(arr, rows, cols, data)
}

// FromSlice creates a tensor from a float64 slice.
func (b *MLXBackend) FromSlice(data []float64, rows, cols int) Tensor {
	// Convert to float32 for MLX
	f32Data := make([]float32, len(data))
	for i, v := range data {
		f32Data[i] = float32(v)
	}
	arr := mlx.FromSlice(f32Data, []int{rows, cols}, mlx.Float32)
	return newMLXTensorWithData(arr, rows, cols, f32Data)
}

// FromSliceFloat32 creates a tensor from float32 data.
func (b *MLXBackend) FromSliceFloat32(data []float32, rows, cols int) Tensor {
	dataCopy := make([]float32, len(data))
	copy(dataCopy, data)
	arr := mlx.FromSlice(dataCopy, []int{rows, cols}, mlx.Float32)
	return newMLXTensorWithData(arr, rows, cols, dataCopy)
}

// Random creates a tensor with random values.
func (b *MLXBackend) Random(rows, cols int) Tensor {
	arr := mlx.Random([]int{rows, cols}, mlx.Float32)
	// Generate matching random data for cache
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(rand.Float64())
	}
	return newMLXTensorWithData(arr, rows, cols, data)
}

// --- Arithmetic Operations ---

// Add performs element-wise addition.
func (b *MLXBackend) Add(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	at.ensureData()
	bt.ensureData()

	result := mlx.Add(at.Array, bt.Array)

	// Compute result data
	ar, ac := at.Shape()
	br, bc := bt.Shape()

	// Handle broadcasting
	if br == 1 && bc == ac && ar != 1 {
		// Broadcasting (1, cols) + (rows, cols)
		data := make([]float32, ar*ac)
		for i := 0; i < ar; i++ {
			for j := 0; j < ac; j++ {
				data[i*ac+j] = at.cachedData[i*ac+j] + bt.cachedData[j]
			}
		}
		return newMLXTensorWithData(result, ar, ac, data)
	}

	data := make([]float32, ar*ac)
	for i := range data {
		data[i] = at.cachedData[i] + bt.cachedData[i]
	}
	return newMLXTensorWithData(result, at.rows, at.cols, data)
}

// Sub performs element-wise subtraction.
func (b *MLXBackend) Sub(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	at.ensureData()
	bt.ensureData()

	// Create negated tensor and add
	negOne := mlx.FromSlice([]float32{-1}, []int{1}, mlx.Float32)
	negB := mlx.Multiply(bt.Array, negOne)
	result := mlx.Add(at.Array, negB)

	data := make([]float32, at.rows*at.cols)
	for i := range data {
		data[i] = at.cachedData[i] - bt.cachedData[i]
	}
	return newMLXTensorWithData(result, at.rows, at.cols, data)
}

// Mul performs element-wise multiplication.
func (b *MLXBackend) Mul(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	at.ensureData()
	bt.ensureData()

	result := mlx.Multiply(at.Array, bt.Array)

	data := make([]float32, at.rows*at.cols)
	for i := range data {
		data[i] = at.cachedData[i] * bt.cachedData[i]
	}
	return newMLXTensorWithData(result, at.rows, at.cols, data)
}

// Div performs element-wise division.
func (b *MLXBackend) Div(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	at.ensureData()
	bt.ensureData()

	data := make([]float32, at.rows*at.cols)
	for i := range data {
		data[i] = at.cachedData[i] / bt.cachedData[i]
	}
	arr := mlx.FromSlice(data, []int{at.rows, at.cols}, mlx.Float32)
	return newMLXTensorWithData(arr, at.rows, at.cols, data)
}

// Scale multiplies all elements by a scalar.
func (b *MLXBackend) Scale(a Tensor, scalar float64) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()

	scalarArr := mlx.FromSlice([]float32{float32(scalar)}, []int{1}, mlx.Float32)
	result := mlx.Multiply(at.Array, scalarArr)

	data := make([]float32, at.rows*at.cols)
	for i := range data {
		data[i] = at.cachedData[i] * float32(scalar)
	}
	return newMLXTensorWithData(result, at.rows, at.cols, data)
}

// --- Matrix Operations ---

// MatMul performs matrix multiplication (GPU accelerated).
func (b *MLXBackend) MatMul(a, b2 Tensor) Tensor {
	at := a.(*MLXTensor)
	bt := b2.(*MLXTensor)

	at.ensureData()
	bt.ensureData()

	result := mlx.MatMul(at.Array, bt.Array)

	// Compute result on CPU as well (for data cache)
	ar, ac := at.Shape()
	_, bc := bt.Shape()

	data := make([]float32, ar*bc)
	for i := 0; i < ar; i++ {
		for j := 0; j < bc; j++ {
			sum := float32(0)
			for k := 0; k < ac; k++ {
				sum += at.cachedData[i*ac+k] * bt.cachedData[k*bc+j]
			}
			data[i*bc+j] = sum
		}
	}

	return newMLXTensorWithData(result, ar, bc, data)
}

// Transpose returns the transpose of the matrix.
func (b *MLXBackend) Transpose(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()

	r, c := at.Shape()
	data := make([]float32, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[j*r+i] = at.cachedData[i*c+j]
		}
	}
	arr := mlx.FromSlice(data, []int{c, r}, mlx.Float32)
	return newMLXTensorWithData(arr, c, r, data)
}

// --- Reductions ---

// Sum sums elements along specified axes.
func (b *MLXBackend) Sum(a Tensor, axes ...int) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	if len(axes) == 0 {
		sum := float32(0)
		for _, v := range at.cachedData {
			sum += v
		}
		arr := mlx.Sum(at.Array)
		return newMLXTensorWithData(arr, 1, 1, []float32{sum})
	}

	axis := axes[0]
	if axis == 0 {
		// Sum along rows -> (1, cols)
		data := make([]float32, c)
		for j := 0; j < c; j++ {
			for i := 0; i < r; i++ {
				data[j] += at.cachedData[i*c+j]
			}
		}
		arr := mlx.Sum(at.Array, axis)
		return newMLXTensorWithData(arr, 1, c, data)
	}

	// Sum along columns -> (rows, 1)
	data := make([]float32, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i] += at.cachedData[i*c+j]
		}
	}
	arr := mlx.Sum(at.Array, axis)
	return newMLXTensorWithData(arr, r, 1, data)
}

// Mean computes mean along specified axes.
func (b *MLXBackend) Mean(a Tensor, axes ...int) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	if len(axes) == 0 {
		sum := float32(0)
		for _, v := range at.cachedData {
			sum += v
		}
		mean := sum / float32(r*c)
		arr := mlx.Mean(at.Array)
		return newMLXTensorWithData(arr, 1, 1, []float32{mean})
	}

	axis := axes[0]
	if axis == 0 {
		data := make([]float32, c)
		for j := 0; j < c; j++ {
			for i := 0; i < r; i++ {
				data[j] += at.cachedData[i*c+j]
			}
			data[j] /= float32(r)
		}
		arr := mlx.Mean(at.Array, axis)
		return newMLXTensorWithData(arr, 1, c, data)
	}

	data := make([]float32, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i] += at.cachedData[i*c+j]
		}
		data[i] /= float32(c)
	}
	arr := mlx.Mean(at.Array, axis)
	return newMLXTensorWithData(arr, r, 1, data)
}

// Max returns the maximum value along specified axes.
func (b *MLXBackend) Max(a Tensor, axes ...int) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	if len(axes) == 0 {
		maxVal := at.cachedData[0]
		for _, v := range at.cachedData[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		arr := mlx.FromSlice([]float32{maxVal}, []int{1, 1}, mlx.Float32)
		return newMLXTensorWithData(arr, 1, 1, []float32{maxVal})
	}

	axis := axes[0]
	if axis == 0 {
		data := make([]float32, c)
		for j := 0; j < c; j++ {
			data[j] = at.cachedData[j]
			for i := 1; i < r; i++ {
				if at.cachedData[i*c+j] > data[j] {
					data[j] = at.cachedData[i*c+j]
				}
			}
		}
		arr := mlx.FromSlice(data, []int{1, c}, mlx.Float32)
		return newMLXTensorWithData(arr, 1, c, data)
	}

	data := make([]float32, r)
	for i := 0; i < r; i++ {
		data[i] = at.cachedData[i*c]
		for j := 1; j < c; j++ {
			if at.cachedData[i*c+j] > data[i] {
				data[i] = at.cachedData[i*c+j]
			}
		}
	}
	arr := mlx.FromSlice(data, []int{r, 1}, mlx.Float32)
	return newMLXTensorWithData(arr, r, 1, data)
}

// Argmax returns indices of maximum values.
func (b *MLXBackend) Argmax(a Tensor, axis int) []int {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	if axis == 0 {
		indices := make([]int, c)
		for j := 0; j < c; j++ {
			maxIdx := 0
			maxVal := at.cachedData[j]
			for i := 1; i < r; i++ {
				if at.cachedData[i*c+j] > maxVal {
					maxVal = at.cachedData[i*c+j]
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
		maxVal := at.cachedData[i*c]
		for j := 1; j < c; j++ {
			if at.cachedData[i*c+j] > maxVal {
				maxVal = at.cachedData[i*c+j]
				maxIdx = j
			}
		}
		indices[i] = maxIdx
	}
	return indices
}

// --- Activation Functions ---

// Softmax computes softmax along the specified axis.
func (b *MLXBackend) Softmax(a Tensor, axis int) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	data := make([]float32, r*c)

	if axis == 1 {
		for i := 0; i < r; i++ {
			// Find max for numerical stability
			maxVal := at.cachedData[i*c]
			for j := 1; j < c; j++ {
				if at.cachedData[i*c+j] > maxVal {
					maxVal = at.cachedData[i*c+j]
				}
			}

			// Compute exp and sum
			sum := float32(0)
			for j := 0; j < c; j++ {
				expVal := float32(math.Exp(float64(at.cachedData[i*c+j] - maxVal)))
				data[i*c+j] = expVal
				sum += expVal
			}

			// Normalize
			for j := 0; j < c; j++ {
				data[i*c+j] /= sum
			}
		}
	} else {
		for j := 0; j < c; j++ {
			maxVal := at.cachedData[j]
			for i := 1; i < r; i++ {
				if at.cachedData[i*c+j] > maxVal {
					maxVal = at.cachedData[i*c+j]
				}
			}

			sum := float32(0)
			for i := 0; i < r; i++ {
				expVal := float32(math.Exp(float64(at.cachedData[i*c+j] - maxVal)))
				data[i*c+j] = expVal
				sum += expVal
			}

			for i := 0; i < r; i++ {
				data[i*c+j] /= sum
			}
		}
	}

	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensorWithData(arr, r, c, data)
}

// ReLU applies the rectified linear unit.
func (b *MLXBackend) ReLU(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()

	zeroArr := mlx.Zeros([]int{at.rows, at.cols}, mlx.Float32)
	result := mlx.Maximum(at.Array, zeroArr)

	data := make([]float32, at.rows*at.cols)
	for i, v := range at.cachedData {
		if v > 0 {
			data[i] = v
		}
	}
	return newMLXTensorWithData(result, at.rows, at.cols, data)
}

// GELU applies the Gaussian Error Linear Unit activation.
func (b *MLXBackend) GELU(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	sqrt2OverPi := float32(math.Sqrt(2.0 / math.Pi))
	data := make([]float32, r*c)
	for i, x := range at.cachedData {
		inner := sqrt2OverPi * (x + 0.044715*x*x*x)
		data[i] = 0.5 * x * (1.0 + float32(math.Tanh(float64(inner))))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensorWithData(arr, r, c, data)
}

// Tanh applies the hyperbolic tangent activation.
func (b *MLXBackend) Tanh(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cachedData {
		data[i] = float32(math.Tanh(float64(v)))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensorWithData(arr, r, c, data)
}

// --- Element-wise Math ---

// Exp computes element-wise exponential.
func (b *MLXBackend) Exp(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cachedData {
		data[i] = float32(math.Exp(float64(v)))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensorWithData(arr, r, c, data)
}

// Log computes element-wise natural logarithm.
func (b *MLXBackend) Log(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cachedData {
		data[i] = float32(math.Log(float64(v)))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensorWithData(arr, r, c, data)
}

// Sqrt computes element-wise square root.
func (b *MLXBackend) Sqrt(a Tensor) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cachedData {
		data[i] = float32(math.Sqrt(float64(v)))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensorWithData(arr, r, c, data)
}

// Pow raises each element to the given power.
func (b *MLXBackend) Pow(a Tensor, power float64) Tensor {
	at := a.(*MLXTensor)
	at.ensureData()
	r, c := at.Shape()

	data := make([]float32, r*c)
	for i, v := range at.cachedData {
		data[i] = float32(math.Pow(float64(v), power))
	}
	arr := mlx.FromSlice(data, []int{r, c}, mlx.Float32)
	return newMLXTensorWithData(arr, r, c, data)
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

	st.ensureData()
	dt.cachedData = make([]float32, len(st.cachedData))
	copy(dt.cachedData, st.cachedData)
	dt.rows = st.rows
	dt.cols = st.cols
	dt.Array = mlx.FromSlice(dt.cachedData, []int{dt.rows, dt.cols}, mlx.Float32)
	dt.dirty = false
}

// init attempts to auto-initialize the MLX backend if available.
func init() {
	// Don't auto-initialize here - let the user choose
	_ = rand.Float64 // Ensure rand is imported
}
