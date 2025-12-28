// GoNum backend implementation using gonum/mat for CPU-based tensor operations.
package backend

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// GoNumTensor wraps gonum's Dense matrix to implement the Tensor interface.
type GoNumTensor struct {
	Dense *mat.Dense
}

// Shape returns the dimensions of the tensor.
func (t *GoNumTensor) Shape() (rows, cols int) {
	return t.Dense.Dims()
}

// At returns the value at position (i, j).
func (t *GoNumTensor) At(i, j int) float64 {
	return t.Dense.At(i, j)
}

// Set sets the value at position (i, j).
func (t *GoNumTensor) Set(i, j int, v float64) {
	t.Dense.Set(i, j, v)
}

// RawData returns the underlying data as a flat slice.
func (t *GoNumTensor) RawData() []float64 {
	r, c := t.Dense.Dims()
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = t.Dense.At(i, j)
		}
	}
	return data
}

// Clone creates a deep copy of the tensor.
func (t *GoNumTensor) Clone() Tensor {
	r, c := t.Dense.Dims()
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = t.Dense.At(i, j)
		}
	}
	return &GoNumTensor{Dense: mat.NewDense(r, c, data)}
}

// GoNumBackend implements the Backend interface using gonum/mat for CPU operations.
type GoNumBackend struct{}

// NewGoNumBackend creates a new CPU backend using gonum/mat.
func NewGoNumBackend() *GoNumBackend {
	return &GoNumBackend{}
}

// Name returns the backend name.
func (b *GoNumBackend) Name() string {
	return "gonum-cpu"
}

// IsGPU returns false since this is a CPU backend.
func (b *GoNumBackend) IsGPU() bool {
	return false
}

// --- Tensor Creation ---

// Zeros creates a tensor filled with zeros.
func (b *GoNumBackend) Zeros(rows, cols int) Tensor {
	return &GoNumTensor{Dense: mat.NewDense(rows, cols, nil)}
}

// Ones creates a tensor filled with ones.
func (b *GoNumBackend) Ones(rows, cols int) Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return &GoNumTensor{Dense: mat.NewDense(rows, cols, data)}
}

// FromSlice creates a tensor from a flat slice.
func (b *GoNumBackend) FromSlice(data []float64, rows, cols int) Tensor {
	// Make a copy to avoid aliasing
	dataCopy := make([]float64, len(data))
	copy(dataCopy, data)
	return &GoNumTensor{Dense: mat.NewDense(rows, cols, dataCopy)}
}

// FromSliceFloat32 creates a tensor from float32 data.
func (b *GoNumBackend) FromSliceFloat32(data []float32, rows, cols int) Tensor {
	data64 := make([]float64, len(data))
	for i, v := range data {
		data64[i] = float64(v)
	}
	return &GoNumTensor{Dense: mat.NewDense(rows, cols, data64)}
}

// Random creates a tensor with random values.
func (b *GoNumBackend) Random(rows, cols int) Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()
	}
	return &GoNumTensor{Dense: mat.NewDense(rows, cols, data)}
}

// --- Arithmetic Operations ---

// Add performs element-wise addition with broadcasting support.
func (b *GoNumBackend) Add(a, b2 Tensor) Tensor {
	at := a.(*GoNumTensor)
	bt := b2.(*GoNumTensor)

	ar, ac := at.Shape()
	br, bc := bt.Shape()

	// Handle broadcasting for bias addition (1, cols) + (rows, cols)
	if br == 1 && bc == ac && ar != 1 {
		result := mat.NewDense(ar, ac, nil)
		for i := 0; i < ar; i++ {
			for j := 0; j < ac; j++ {
				result.Set(i, j, at.At(i, j)+bt.At(0, j))
			}
		}
		return &GoNumTensor{Dense: result}
	}

	// Standard addition
	result := mat.NewDense(ar, ac, nil)
	result.Add(at.Dense, bt.Dense)
	return &GoNumTensor{Dense: result}
}

// Sub performs element-wise subtraction.
func (b *GoNumBackend) Sub(a, b2 Tensor) Tensor {
	at := a.(*GoNumTensor)
	bt := b2.(*GoNumTensor)

	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)
	result.Sub(at.Dense, bt.Dense)
	return &GoNumTensor{Dense: result}
}

// Mul performs element-wise multiplication.
func (b *GoNumBackend) Mul(a, b2 Tensor) Tensor {
	at := a.(*GoNumTensor)
	bt := b2.(*GoNumTensor)

	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)
	result.MulElem(at.Dense, bt.Dense)
	return &GoNumTensor{Dense: result}
}

// Div performs element-wise division.
func (b *GoNumBackend) Div(a, b2 Tensor) Tensor {
	at := a.(*GoNumTensor)
	bt := b2.(*GoNumTensor)

	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)
	result.DivElem(at.Dense, bt.Dense)
	return &GoNumTensor{Dense: result}
}

// Scale multiplies all elements by a scalar.
func (b *GoNumBackend) Scale(a Tensor, scalar float64) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)
	result.Scale(scalar, at.Dense)
	return &GoNumTensor{Dense: result}
}

// --- Matrix Operations ---

// MatMul performs matrix multiplication.
func (b *GoNumBackend) MatMul(a, b2 Tensor) Tensor {
	at := a.(*GoNumTensor)
	bt := b2.(*GoNumTensor)

	ar, _ := at.Shape()
	_, bc := bt.Shape()

	result := mat.NewDense(ar, bc, nil)
	result.Mul(at.Dense, bt.Dense)
	return &GoNumTensor{Dense: result}
}

// Transpose returns the transpose of the matrix.
func (b *GoNumBackend) Transpose(a Tensor) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()
	result := mat.NewDense(c, r, nil)
	result.Copy(at.Dense.T())
	return &GoNumTensor{Dense: result}
}

// --- Reductions ---

// Sum sums elements along specified axes.
func (b *GoNumBackend) Sum(a Tensor, axes ...int) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()

	if len(axes) == 0 {
		// Sum all elements
		sum := 0.0
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				sum += at.At(i, j)
			}
		}
		return &GoNumTensor{Dense: mat.NewDense(1, 1, []float64{sum})}
	}

	axis := axes[0]
	if axis == 0 {
		// Sum along rows -> (1, cols)
		result := mat.NewDense(1, c, nil)
		for j := 0; j < c; j++ {
			sum := 0.0
			for i := 0; i < r; i++ {
				sum += at.At(i, j)
			}
			result.Set(0, j, sum)
		}
		return &GoNumTensor{Dense: result}
	}

	// Sum along columns -> (rows, 1)
	result := mat.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < c; j++ {
			sum += at.At(i, j)
		}
		result.Set(i, 0, sum)
	}
	return &GoNumTensor{Dense: result}
}

// Mean computes mean along specified axes.
func (b *GoNumBackend) Mean(a Tensor, axes ...int) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()

	if len(axes) == 0 {
		// Mean of all elements
		sum := 0.0
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				sum += at.At(i, j)
			}
		}
		return &GoNumTensor{Dense: mat.NewDense(1, 1, []float64{sum / float64(r*c)})}
	}

	axis := axes[0]
	if axis == 0 {
		// Mean along rows -> (1, cols)
		result := mat.NewDense(1, c, nil)
		for j := 0; j < c; j++ {
			sum := 0.0
			for i := 0; i < r; i++ {
				sum += at.At(i, j)
			}
			result.Set(0, j, sum/float64(r))
		}
		return &GoNumTensor{Dense: result}
	}

	// Mean along columns -> (rows, 1)
	result := mat.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < c; j++ {
			sum += at.At(i, j)
		}
		result.Set(i, 0, sum/float64(c))
	}
	return &GoNumTensor{Dense: result}
}

// Max returns the maximum value along specified axes.
func (b *GoNumBackend) Max(a Tensor, axes ...int) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()

	if len(axes) == 0 {
		// Max of all elements
		maxVal := at.At(0, 0)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				if at.At(i, j) > maxVal {
					maxVal = at.At(i, j)
				}
			}
		}
		return &GoNumTensor{Dense: mat.NewDense(1, 1, []float64{maxVal})}
	}

	axis := axes[0]
	if axis == 0 {
		// Max along rows -> (1, cols)
		result := mat.NewDense(1, c, nil)
		for j := 0; j < c; j++ {
			maxVal := at.At(0, j)
			for i := 1; i < r; i++ {
				if at.At(i, j) > maxVal {
					maxVal = at.At(i, j)
				}
			}
			result.Set(0, j, maxVal)
		}
		return &GoNumTensor{Dense: result}
	}

	// Max along columns -> (rows, 1)
	result := mat.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		maxVal := at.At(i, 0)
		for j := 1; j < c; j++ {
			if at.At(i, j) > maxVal {
				maxVal = at.At(i, j)
			}
		}
		result.Set(i, 0, maxVal)
	}
	return &GoNumTensor{Dense: result}
}

// Argmax returns indices of maximum values along specified axis.
func (b *GoNumBackend) Argmax(a Tensor, axis int) []int {
	at := a.(*GoNumTensor)
	r, c := at.Shape()

	if axis == 0 {
		// Argmax along rows
		indices := make([]int, c)
		for j := 0; j < c; j++ {
			maxIdx := 0
			maxVal := at.At(0, j)
			for i := 1; i < r; i++ {
				if at.At(i, j) > maxVal {
					maxVal = at.At(i, j)
					maxIdx = i
				}
			}
			indices[j] = maxIdx
		}
		return indices
	}

	// Argmax along columns
	indices := make([]int, r)
	for i := 0; i < r; i++ {
		maxIdx := 0
		maxVal := at.At(i, 0)
		for j := 1; j < c; j++ {
			if at.At(i, j) > maxVal {
				maxVal = at.At(i, j)
				maxIdx = j
			}
		}
		indices[i] = maxIdx
	}
	return indices
}

// --- Activation Functions ---

// Softmax computes softmax along the specified axis.
func (b *GoNumBackend) Softmax(a Tensor, axis int) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()

	result := mat.NewDense(r, c, nil)

	if axis == 1 {
		// Softmax along rows (most common for neural networks)
		for i := 0; i < r; i++ {
			// Find max for numerical stability
			maxVal := at.At(i, 0)
			for j := 1; j < c; j++ {
				if at.At(i, j) > maxVal {
					maxVal = at.At(i, j)
				}
			}

			// Compute exp and sum
			sum := 0.0
			for j := 0; j < c; j++ {
				expVal := math.Exp(at.At(i, j) - maxVal)
				result.Set(i, j, expVal)
				sum += expVal
			}

			// Normalize
			for j := 0; j < c; j++ {
				result.Set(i, j, result.At(i, j)/sum)
			}
		}
	} else {
		// Softmax along columns
		for j := 0; j < c; j++ {
			maxVal := at.At(0, j)
			for i := 1; i < r; i++ {
				if at.At(i, j) > maxVal {
					maxVal = at.At(i, j)
				}
			}

			sum := 0.0
			for i := 0; i < r; i++ {
				expVal := math.Exp(at.At(i, j) - maxVal)
				result.Set(i, j, expVal)
				sum += expVal
			}

			for i := 0; i < r; i++ {
				result.Set(i, j, result.At(i, j)/sum)
			}
		}
	}

	return &GoNumTensor{Dense: result}
}

// ReLU applies the rectified linear unit.
func (b *GoNumBackend) ReLU(a Tensor) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := at.At(i, j)
			if val > 0 {
				result.Set(i, j, val)
			}
		}
	}
	return &GoNumTensor{Dense: result}
}

// GELU applies the Gaussian Error Linear Unit activation.
func (b *GoNumBackend) GELU(a Tensor) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)

	// Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	sqrt2OverPi := math.Sqrt(2.0 / math.Pi)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			x := at.At(i, j)
			inner := sqrt2OverPi * (x + 0.044715*x*x*x)
			result.Set(i, j, 0.5*x*(1.0+math.Tanh(inner)))
		}
	}
	return &GoNumTensor{Dense: result}
}

// Tanh applies the hyperbolic tangent activation.
func (b *GoNumBackend) Tanh(a Tensor) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, math.Tanh(at.At(i, j)))
		}
	}
	return &GoNumTensor{Dense: result}
}

// --- Element-wise Math ---

// Exp computes element-wise exponential.
func (b *GoNumBackend) Exp(a Tensor) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, math.Exp(at.At(i, j)))
		}
	}
	return &GoNumTensor{Dense: result}
}

// Log computes element-wise natural logarithm.
func (b *GoNumBackend) Log(a Tensor) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, math.Log(at.At(i, j)))
		}
	}
	return &GoNumTensor{Dense: result}
}

// Sqrt computes element-wise square root.
func (b *GoNumBackend) Sqrt(a Tensor) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, math.Sqrt(at.At(i, j)))
		}
	}
	return &GoNumTensor{Dense: result}
}

// Pow raises each element to the given power.
func (b *GoNumBackend) Pow(a Tensor, power float64) Tensor {
	at := a.(*GoNumTensor)
	r, c := at.Shape()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, math.Pow(at.At(i, j), power))
		}
	}
	return &GoNumTensor{Dense: result}
}

// --- Utilities ---

// Synchronize is a no-op for CPU backend.
func (b *GoNumBackend) Synchronize() {
	// No-op for CPU
}

// Copy copies data from src to dst.
func (b *GoNumBackend) Copy(dst, src Tensor) {
	dt := dst.(*GoNumTensor)
	st := src.(*GoNumTensor)
	dt.Dense.Copy(st.Dense)
}

// --- Compatibility helpers ---

// ToDense converts a Tensor to a gonum mat.Dense (for compatibility with existing code).
func ToDense(t Tensor) *mat.Dense {
	if gt, ok := t.(*GoNumTensor); ok {
		return gt.Dense
	}

	// Convert from other tensor types
	r, c := t.Shape()
	data := t.RawData()
	return mat.NewDense(r, c, data)
}

// FromDense creates a Tensor from a gonum mat.Dense.
func FromDense(d *mat.Dense) Tensor {
	return &GoNumTensor{Dense: d}
}
