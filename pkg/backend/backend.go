// Package backend provides an abstraction layer for tensor operations,
// allowing the use of different computation backends (CPU via gonum, GPU via MLX).
package backend

import (
	"sync"
)

// Tensor represents a 2D tensor (matrix) that can be backed by different implementations.
type Tensor interface {
	// Shape returns the dimensions of the tensor
	Shape() (rows, cols int)

	// At returns the value at position (i, j)
	At(i, j int) float64

	// Set sets the value at position (i, j)
	Set(i, j int, v float64)

	// RawData returns the underlying data as a flat slice (row-major order)
	RawData() []float64

	// Clone creates a deep copy of the tensor
	Clone() Tensor
}

// Backend defines the interface for tensor computation backends.
// Implementations can target CPU (gonum) or GPU (MLX on Apple Silicon).
type Backend interface {
	// Name returns a human-readable name for the backend
	Name() string

	// IsGPU returns true if this backend uses GPU acceleration
	IsGPU() bool

	// --- Tensor Creation ---

	// Zeros creates a tensor filled with zeros
	Zeros(rows, cols int) Tensor

	// Ones creates a tensor filled with ones
	Ones(rows, cols int) Tensor

	// FromSlice creates a tensor from a flat slice (row-major order)
	FromSlice(data []float64, rows, cols int) Tensor

	// FromSliceFloat32 creates a tensor from float32 data (for MLX compatibility)
	FromSliceFloat32(data []float32, rows, cols int) Tensor

	// Random creates a tensor with random values from uniform distribution [0, 1)
	Random(rows, cols int) Tensor

	// --- Arithmetic Operations ---

	// Add performs element-wise addition: c = a + b (with broadcasting)
	Add(a, b Tensor) Tensor

	// Sub performs element-wise subtraction: c = a - b
	Sub(a, b Tensor) Tensor

	// Mul performs element-wise multiplication (Hadamard product): c = a * b
	Mul(a, b Tensor) Tensor

	// Div performs element-wise division: c = a / b
	Div(a, b Tensor) Tensor

	// Scale multiplies all elements by a scalar: c = a * scalar
	Scale(a Tensor, scalar float64) Tensor

	// --- Matrix Operations ---

	// MatMul performs matrix multiplication: c = a @ b
	MatMul(a, b Tensor) Tensor

	// Transpose returns the transpose of the matrix
	Transpose(a Tensor) Tensor

	// --- Reductions ---

	// Sum sums elements along specified axes (empty axes = sum all)
	Sum(a Tensor, axes ...int) Tensor

	// Mean computes mean along specified axes (empty axes = mean all)
	Mean(a Tensor, axes ...int) Tensor

	// Max returns the maximum value along specified axes
	Max(a Tensor, axes ...int) Tensor

	// Argmax returns indices of maximum values along specified axis
	Argmax(a Tensor, axis int) []int

	// --- Activation Functions (fused for GPU efficiency) ---

	// Softmax computes softmax along the specified axis (typically axis=1 for rows)
	Softmax(a Tensor, axis int) Tensor

	// ReLU applies the rectified linear unit: max(0, x)
	ReLU(a Tensor) Tensor

	// GELU applies the Gaussian Error Linear Unit activation
	GELU(a Tensor) Tensor

	// Tanh applies the hyperbolic tangent activation
	Tanh(a Tensor) Tensor

	// --- Element-wise Math ---

	// Exp computes element-wise exponential
	Exp(a Tensor) Tensor

	// Log computes element-wise natural logarithm
	Log(a Tensor) Tensor

	// Sqrt computes element-wise square root
	Sqrt(a Tensor) Tensor

	// Pow raises each element to the given power
	Pow(a Tensor, power float64) Tensor

	// --- Utilities ---

	// Synchronize waits for all pending operations to complete (important for GPU)
	Synchronize()

	// Copy copies data from src to dst
	Copy(dst, src Tensor)
}

// Global default backend
var (
	defaultBackend Backend
	backendMu      sync.RWMutex
)

// SetDefault sets the default backend for all operations
func SetDefault(b Backend) {
	backendMu.Lock()
	defer backendMu.Unlock()
	defaultBackend = b
}

// Default returns the current default backend.
// If no backend has been set, it initializes and returns a GoNum (CPU) backend.
func Default() Backend {
	backendMu.RLock()
	if defaultBackend != nil {
		defer backendMu.RUnlock()
		return defaultBackend
	}
	backendMu.RUnlock()

	// Initialize default backend (gonum CPU)
	backendMu.Lock()
	defer backendMu.Unlock()

	// Double-check after acquiring write lock
	if defaultBackend == nil {
		defaultBackend = NewGoNumBackend()
	}
	return defaultBackend
}

// AutoSelectBackend tries to initialize the best available backend.
// On Apple Silicon, it tries MLX (GPU) first, falling back to GoNum (CPU).
func AutoSelectBackend() (Backend, error) {
	// Try MLX first on Apple Silicon
	mlxBackend, err := NewMLXBackend()
	if err == nil {
		return mlxBackend, nil
	}

	// Fallback to CPU
	return NewGoNumBackend(), nil
}
