//go:build darwin && arm64 && mlx

// Package autograd provides GPU-accelerated automatic differentiation using MLX backend.
// This file implements autograd.Value using backend.Tensor for GPU operations.
package autograd

import (
	"fmt"
	"math"

	"github.com/suensky/gogpt/pkg/backend"
)

// GPUValue represents a tensor with gradient tracking for GPU computation.
type GPUValue struct {
	Data    backend.Tensor  // The underlying GPU tensor
	Grad    backend.Tensor  // Gradient tensor (same shape as Data)
	op      GPUOp           // Operation that created this value (nil for leaf nodes)
	parents []*GPUValue     // Parent values in computation graph
	name    string          // Optional name for debugging
	be      backend.Backend // Backend reference
}

// GPUOp defines operations for GPU autograd
type GPUOp interface {
	Name() string
	Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend)
}

// NewGPUVariable creates a new leaf GPU tensor
func NewGPUVariable(rows, cols int, data []float64, be backend.Backend) *GPUValue {
	var tensor backend.Tensor
	if data != nil {
		tensor = be.FromSlice(data, rows, cols)
	} else {
		tensor = be.Zeros(rows, cols)
	}
	return &GPUValue{
		Data: tensor,
		Grad: be.Zeros(rows, cols),
		be:   be,
	}
}

// NewGPUVariableRandom creates a GPU tensor with random initialization
func NewGPUVariableRandom(rows, cols int, scale float64, be backend.Backend) *GPUValue {
	tensor := be.Random(rows, cols)
	if scale != 1.0 {
		tensor = be.Scale(tensor, scale)
	}
	return &GPUValue{
		Data: tensor,
		Grad: be.Zeros(rows, cols),
		be:   be,
	}
}

// Shape returns the dimensions
func (v *GPUValue) Shape() (rows, cols int) {
	return v.Data.Shape()
}

// ZeroGrad resets gradients to zero
func (v *GPUValue) ZeroGrad() {
	r, c := v.Shape()
	v.Grad = v.be.Zeros(r, c)
}

// SetName sets the name for debugging
func (v *GPUValue) SetName(name string) *GPUValue {
	v.name = name
	return v
}

// Name returns the name
func (v *GPUValue) Name() string {
	return v.name
}

// IsLeaf returns true if this is a leaf node
func (v *GPUValue) IsLeaf() bool {
	return v.op == nil
}

// Backend returns the backend
func (v *GPUValue) Backend() backend.Backend {
	return v.be
}

// String returns a string representation
func (v *GPUValue) String() string {
	r, c := v.Shape()
	name := v.name
	if name == "" {
		name = "unnamed"
	}
	opName := "leaf"
	if v.op != nil {
		opName = v.op.Name()
	}
	return fmt.Sprintf("GPUValue(%s, shape=%dx%d, op=%s)", name, r, c, opName)
}

// newGPUResult creates a result GPUValue from an operation
func newGPUResult(data backend.Tensor, op GPUOp, be backend.Backend, parents ...*GPUValue) *GPUValue {
	r, c := data.Shape()
	return &GPUValue{
		Data:    data,
		Grad:    be.Zeros(r, c),
		op:      op,
		parents: parents,
		be:      be,
	}
}

// =============================================================================
// GPU Operations
// =============================================================================

// --- Add Operation ---
type gpuAddOp struct{}

func (o *gpuAddOp) Name() string { return "GPUAdd" }

func (o *gpuAddOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a, b := inputs[0], inputs[1]
	ar, ac := a.Shape()
	br, bc := b.Shape()
	gr, gc := grad.Shape()

	// Gradient for a
	if ar == gr && ac == gc {
		a.Grad = be.Add(a.Grad, grad)
	}

	// Gradient for b (may need sum if broadcasted)
	if br == 1 && bc == gc && gr > 1 {
		// Sum along rows for broadcast case
		sumGrad := be.Sum(grad, 0) // Sum along axis 0
		b.Grad = be.Add(b.Grad, sumGrad)
	} else if br == gr && bc == gc {
		b.Grad = be.Add(b.Grad, grad)
	}
}

// GPUAdd performs GPU element-wise addition
func GPUAdd(a, b *GPUValue) *GPUValue {
	be := a.be
	result := be.Add(a.Data, b.Data)
	return newGPUResult(result, &gpuAddOp{}, be, a, b)
}

// --- MatMul Operation ---
type gpuMatMulOp struct{}

func (o *gpuMatMulOp) Name() string { return "GPUMatMul" }

func (o *gpuMatMulOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a, b := inputs[0], inputs[1]

	// da = grad @ b.T
	bT := be.Transpose(b.Data)
	daContrib := be.MatMul(grad, bT)
	a.Grad = be.Add(a.Grad, daContrib)

	// db = a.T @ grad
	aT := be.Transpose(a.Data)
	dbContrib := be.MatMul(aT, grad)
	b.Grad = be.Add(b.Grad, dbContrib)
}

// GPUMatMul performs GPU matrix multiplication
func GPUMatMul(a, b *GPUValue) *GPUValue {
	be := a.be
	result := be.MatMul(a.Data, b.Data)
	return newGPUResult(result, &gpuMatMulOp{}, be, a, b)
}

// --- Scale Operation ---
type gpuScaleOp struct {
	scalar float64
}

func (o *gpuScaleOp) Name() string { return "GPUScale" }

func (o *gpuScaleOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a := inputs[0]
	daContrib := be.Scale(grad, o.scalar)
	a.Grad = be.Add(a.Grad, daContrib)
}

// GPUScale multiplies by scalar
func GPUScale(a *GPUValue, scalar float64) *GPUValue {
	be := a.be
	result := be.Scale(a.Data, scalar)
	return newGPUResult(result, &gpuScaleOp{scalar: scalar}, be, a)
}

// --- Mul (Hadamard) Operation ---
type gpuMulOp struct{}

func (o *gpuMulOp) Name() string { return "GPUMul" }

func (o *gpuMulOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a, b := inputs[0], inputs[1]
	// da = grad * b
	daContrib := be.Mul(grad, b.Data)
	a.Grad = be.Add(a.Grad, daContrib)
	// db = grad * a
	dbContrib := be.Mul(grad, a.Data)
	b.Grad = be.Add(b.Grad, dbContrib)
}

// GPUMul performs element-wise multiplication
func GPUMul(a, b *GPUValue) *GPUValue {
	be := a.be
	result := be.Mul(a.Data, b.Data)
	return newGPUResult(result, &gpuMulOp{}, be, a, b)
}

// --- Sub Operation ---
type gpuSubOp struct{}

func (o *gpuSubOp) Name() string { return "GPUSub" }

func (o *gpuSubOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a, b := inputs[0], inputs[1]
	a.Grad = be.Add(a.Grad, grad)
	negGrad := be.Scale(grad, -1)
	b.Grad = be.Add(b.Grad, negGrad)
}

// GPUSub performs element-wise subtraction
func GPUSub(a, b *GPUValue) *GPUValue {
	be := a.be
	result := be.Sub(a.Data, b.Data)
	return newGPUResult(result, &gpuSubOp{}, be, a, b)
}

// --- ReLU Operation ---
type gpuReLUOp struct{}

func (o *gpuReLUOp) Name() string { return "GPUReLU" }

func (o *gpuReLUOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a := inputs[0]
	r, c := a.Shape()
	// ReLU gradient: 1 where input > 0, else 0
	// Using element-wise comparison via accessor (CPU fallback for now)
	maskData := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if a.Data.At(i, j) > 0 {
				maskData[i*c+j] = 1.0
			}
		}
	}
	mask := be.FromSlice(maskData, r, c)
	daContrib := be.Mul(grad, mask)
	a.Grad = be.Add(a.Grad, daContrib)
}

// GPUReLU applies ReLU activation
func GPUReLU(a *GPUValue) *GPUValue {
	be := a.be
	result := be.ReLU(a.Data)
	return newGPUResult(result, &gpuReLUOp{}, be, a)
}

// --- GELU Operation ---
type gpuGELUOp struct{}

func (o *gpuGELUOp) Name() string { return "GPUGELU" }

func (o *gpuGELUOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a := inputs[0]
	// GELU gradient approximation
	r, c := a.Shape()
	gradData := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			x := a.Data.At(i, j)
			// Approximate GELU derivative
			// d/dx GELU(x) ≈ sigmoid(1.702 * x) * (1 + 1.702 * x * (1 - sigmoid(1.702 * x)))
			s := 1.0 / (1.0 + exp(-1.702*x))
			gradData[i*c+j] = s * (1 + 1.702*x*(1-s))
		}
	}
	geluGrad := be.FromSlice(gradData, r, c)
	daContrib := be.Mul(grad, geluGrad)
	a.Grad = be.Add(a.Grad, daContrib)
}

func exp(x float64) float64 {
	if x > 700 {
		return 1e308
	}
	if x < -700 {
		return 0
	}
	return expBuiltin(x)
}

// GPUGelu applies GELU activation
func GPUGELU(a *GPUValue) *GPUValue {
	be := a.be
	result := be.GELU(a.Data)
	return newGPUResult(result, &gpuGELUOp{}, be, a)
}

// --- Softmax Operation ---
type gpuSoftmaxOp struct{}

func (o *gpuSoftmaxOp) Name() string { return "GPUSoftmax" }

func (o *gpuSoftmaxOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a := inputs[0]
	r, c := a.Shape()

	// Softmax backward: d_i = s_i * (grad_i - sum_j(grad_j * s_j))
	gradData := make([]float64, r*c)
	for i := 0; i < r; i++ {
		dotProd := 0.0
		for j := 0; j < c; j++ {
			dotProd += grad.At(i, j) * output.Data.At(i, j)
		}
		for j := 0; j < c; j++ {
			s := output.Data.At(i, j)
			gradData[i*c+j] = s * (grad.At(i, j) - dotProd)
		}
	}
	da := be.FromSlice(gradData, r, c)
	a.Grad = be.Add(a.Grad, da)
}

// GPUSoftmax applies softmax activation row-wise
func GPUSoftmax(a *GPUValue) *GPUValue {
	be := a.be
	result := be.Softmax(a.Data, 1) // Row-wise softmax
	return newGPUResult(result, &gpuSoftmaxOp{}, be, a)
}

// --- Transpose Operation ---
type gpuTransposeOp struct{}

func (o *gpuTransposeOp) Name() string { return "GPUTranspose" }

func (o *gpuTransposeOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a := inputs[0]
	daContrib := be.Transpose(grad)
	a.Grad = be.Add(a.Grad, daContrib)
}

// GPUTranspose returns transpose
func GPUTranspose(a *GPUValue) *GPUValue {
	be := a.be
	result := be.Transpose(a.Data)
	return newGPUResult(result, &gpuTransposeOp{}, be, a)
}

// --- Sum Operation ---
type gpuSumOp struct {
	size float64
}

func (o *gpuSumOp) Name() string { return "GPUSum" }

func (o *gpuSumOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a := inputs[0]
	r, c := a.Shape()
	gradVal := grad.At(0, 0)
	gradData := make([]float64, r*c)
	for i := range gradData {
		gradData[i] = gradVal
	}
	da := be.FromSlice(gradData, r, c)
	a.Grad = be.Add(a.Grad, da)
}

// GPUSum returns sum of all elements
func GPUSum(a *GPUValue) *GPUValue {
	be := a.be
	result := be.Sum(a.Data)
	r, c := a.Shape()
	return newGPUResult(result, &gpuSumOp{size: float64(r * c)}, be, a)
}

// --- Mean Operation ---
type gpuMeanOp struct {
	size float64
}

func (o *gpuMeanOp) Name() string { return "GPUMean" }

func (o *gpuMeanOp) Backward(grad backend.Tensor, inputs []*GPUValue, output *GPUValue, be backend.Backend) {
	a := inputs[0]
	r, c := a.Shape()
	gradVal := grad.At(0, 0) / o.size
	gradData := make([]float64, r*c)
	for i := range gradData {
		gradData[i] = gradVal
	}
	da := be.FromSlice(gradData, r, c)
	a.Grad = be.Add(a.Grad, da)
}

// GPUMean returns mean of all elements
func GPUMean(a *GPUValue) *GPUValue {
	be := a.be
	result := be.Mean(a.Data)
	r, c := a.Shape()
	return newGPUResult(result, &gpuMeanOp{size: float64(r * c)}, be, a)
}

// =============================================================================
// Backward Pass
// =============================================================================

// Backward performs reverse-mode automatic differentiation on GPU
func (v *GPUValue) Backward() {
	order := gpuTopologicalSort(v)
	be := v.be

	// Initialize output gradient to 1
	r, c := v.Shape()
	onesData := make([]float64, r*c)
	for i := range onesData {
		onesData[i] = 1.0
	}
	v.Grad = be.FromSlice(onesData, r, c)

	// Backpropagate in reverse order
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.op != nil && len(node.parents) > 0 {
			node.op.Backward(node.Grad, node.parents, node, be)
		}
	}

	// Sync to ensure all gradients computed
	be.Synchronize()
}

func gpuTopologicalSort(root *GPUValue) []*GPUValue {
	visited := make(map[*GPUValue]bool)
	order := make([]*GPUValue, 0)

	var visit func(node *GPUValue)
	visit = func(node *GPUValue) {
		if visited[node] {
			return
		}
		visited[node] = true
		for _, parent := range node.parents {
			visit(parent)
		}
		order = append(order, node)
	}

	visit(root)
	return order
}

// expBuiltin wraps math.Exp properly
func expBuiltin(x float64) float64 {
	return math.Exp(x)
}
