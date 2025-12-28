// Package autograd implements reverse-mode automatic differentiation.
// It provides a computation graph for tensors with automatic gradient computation.
package autograd

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Value represents a node in the computation graph.
// It holds tensor data, gradients, and references to the operation that created it.
type Value struct {
	Data    *mat.Dense // The underlying tensor data
	Grad    *mat.Dense // Gradient tensor (same shape as Data)
	op      Operation  // Operation that created this value (nil for leaf nodes)
	parents []*Value   // Parent values in the computation graph
	name    string     // Optional name for debugging
}

// Operation defines the interface for computational operations.
// Each operation knows how to compute forward and backward passes.
type Operation interface {
	// Forward computes the output given inputs
	Forward(inputs ...*Value) *Value
	// Backward propagates gradients from output to inputs
	Backward(grad *mat.Dense, inputs []*Value, output *Value)
	// Name returns the operation name for debugging
	Name() string
}

// NewVariable creates a new leaf node (variable) in the computation graph.
// Leaf nodes have no parents and no operation.
func NewVariable(rows, cols int, data []float64) *Value {
	var d *mat.Dense
	if data != nil {
		if len(data) != rows*cols {
			panic(fmt.Sprintf("data length %d does not match dimensions %dx%d", len(data), rows, cols))
		}
		d = mat.NewDense(rows, cols, data)
	} else {
		d = mat.NewDense(rows, cols, nil)
	}
	return &Value{
		Data:    d,
		Grad:    mat.NewDense(rows, cols, nil), // Initialize gradient to zero
		parents: nil,
		op:      nil,
	}
}

// NewVariableFromDense creates a Value from an existing Dense matrix.
func NewVariableFromDense(d *mat.Dense) *Value {
	r, c := d.Dims()
	return &Value{
		Data:    d,
		Grad:    mat.NewDense(r, c, nil),
		parents: nil,
		op:      nil,
	}
}

// Zeros creates a Value filled with zeros.
func Zeros(rows, cols int) *Value {
	return NewVariable(rows, cols, nil)
}

// Ones creates a Value filled with ones.
func Ones(rows, cols int) *Value {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return NewVariable(rows, cols, data)
}

// Scalar creates a 1x1 Value containing a single scalar.
func Scalar(val float64) *Value {
	return NewVariable(1, 1, []float64{val})
}

// Shape returns the dimensions of the Value.
func (v *Value) Shape() (rows, cols int) {
	return v.Data.Dims()
}

// SetName sets a name for debugging purposes.
func (v *Value) SetName(name string) *Value {
	v.name = name
	return v
}

// Name returns the name of the Value.
func (v *Value) Name() string {
	return v.name
}

// ScalarValue returns the scalar value (for 1x1 tensors).
func (v *Value) ScalarValue() float64 {
	r, c := v.Shape()
	if r != 1 || c != 1 {
		panic(fmt.Sprintf("ScalarValue called on non-scalar tensor of shape %dx%d", r, c))
	}
	return v.Data.At(0, 0)
}

// Clone creates a deep copy of the Value (data only, not graph structure).
func (v *Value) Clone() *Value {
	r, c := v.Shape()
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = v.Data.At(i, j)
		}
	}
	return NewVariable(r, c, data)
}

// ZeroGrad resets the gradient to zero.
func (v *Value) ZeroGrad() {
	r, c := v.Shape()
	v.Grad = mat.NewDense(r, c, nil)
}

// Parents returns the parent Values in the computation graph.
func (v *Value) Parents() []*Value {
	return v.parents
}

// IsLeaf returns true if this is a leaf node (no operation created it).
func (v *Value) IsLeaf() bool {
	return v.op == nil
}

// String returns a string representation of the Value.
func (v *Value) String() string {
	r, c := v.Shape()
	name := v.name
	if name == "" {
		name = "unnamed"
	}
	if v.op != nil {
		return fmt.Sprintf("Value(%s, shape=%dx%d, op=%s)", name, r, c, v.op.Name())
	}
	return fmt.Sprintf("Value(%s, shape=%dx%d, leaf)", name, r, c)
}
