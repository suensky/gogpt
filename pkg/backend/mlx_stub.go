//go:build !mlx

// Stub MLX backend for when MLX is not available.
// Returns an error when trying to create an MLX backend.
package backend

import "errors"

// ErrMLXNotAvailable is returned when MLX is not available.
var ErrMLXNotAvailable = errors.New("MLX backend not available: build with -tags=mlx and ensure MLX library is installed")

// MLXBackend is a stub for when MLX is not available.
type MLXBackend struct{}

// NewMLXBackend returns an error when MLX is not available.
func NewMLXBackend() (*MLXBackend, error) {
	return nil, ErrMLXNotAvailable
}

// MLXTensor is a stub for when MLX is not available.
type MLXTensor struct{}

// Shape returns 0, 0 for the stub.
func (t *MLXTensor) Shape() (rows, cols int) { return 0, 0 }

// At returns 0 for the stub.
func (t *MLXTensor) At(i, j int) float64 { return 0 }

// Set is a no-op for the stub.
func (t *MLXTensor) Set(i, j int, v float64) {}

// RawData returns nil for the stub.
func (t *MLXTensor) RawData() []float64 { return nil }

// Clone returns nil for the stub.
func (t *MLXTensor) Clone() Tensor { return nil }

// The following methods implement the Backend interface but panic if called.
// They should never be called since NewMLXBackend returns an error.

func (b *MLXBackend) Name() string                                    { return "mlx-unavailable" }
func (b *MLXBackend) IsGPU() bool                                     { return false }
func (b *MLXBackend) Zeros(rows, cols int) Tensor                     { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Ones(rows, cols int) Tensor                      { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) FromSlice(data []float64, rows, cols int) Tensor { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) FromSliceFloat32(data []float32, rows, cols int) Tensor {
	panic(ErrMLXNotAvailable)
}
func (b *MLXBackend) Random(rows, cols int) Tensor          { panic(ErrMLXNotAvailable) }
func (be *MLXBackend) Add(a, b2 Tensor) Tensor              { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Sub(a, bt Tensor) Tensor               { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Mul(a, bt Tensor) Tensor               { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Div(a, bt Tensor) Tensor               { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Scale(a Tensor, scalar float64) Tensor { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) MatMul(a, bt Tensor) Tensor            { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Transpose(a Tensor) Tensor             { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Sum(a Tensor, axes ...int) Tensor      { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Mean(a Tensor, axes ...int) Tensor     { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Max(a Tensor, axes ...int) Tensor      { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Argmax(a Tensor, axis int) []int       { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Softmax(a Tensor, axis int) Tensor     { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) ReLU(a Tensor) Tensor                  { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) GELU(a Tensor) Tensor                  { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Tanh(a Tensor) Tensor                  { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Exp(a Tensor) Tensor                   { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Log(a Tensor) Tensor                   { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Sqrt(a Tensor) Tensor                  { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Pow(a Tensor, power float64) Tensor    { panic(ErrMLXNotAvailable) }
func (b *MLXBackend) Synchronize()                          {}
func (b *MLXBackend) Copy(dst, src Tensor)                  { panic(ErrMLXNotAvailable) }
