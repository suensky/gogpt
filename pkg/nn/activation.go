package nn

import (
	"math"

	"github.com/suensky/gogpt/pkg/autograd"
	"gonum.org/v1/gonum/mat"
)

// ReLU applies the rectified linear unit activation.
// Delegates to autograd.ReLU for proper gradient tracking.
func ReLU(x *autograd.Value) *autograd.Value {
	return autograd.ReLU(x)
}

// Softmax applies softmax activation row-wise.
// Delegates to autograd.Softmax for proper gradient tracking.
func Softmax(x *autograd.Value) *autograd.Value {
	return autograd.Softmax(x)
}

// LogSoftmax applies log-softmax activation row-wise.
// More numerically stable for cross-entropy loss.
func LogSoftmax(x *autograd.Value) *autograd.Value {
	return autograd.LogSoftmax(x)
}

// GELU implements the Gaussian Error Linear Unit activation.
// GELU(x) = x * Φ(x) where Φ is the CDF of the standard normal distribution.
// Approximate: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
type geluOp struct{}

func (o *geluOp) Name() string { return "GELU" }

func (o *geluOp) Forward(inputs ...*autograd.Value) *autograd.Value {
	x := inputs[0]
	r, c := x.Shape()

	sqrt2OverPi := math.Sqrt(2.0 / math.Pi)
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := x.Data.At(i, j)
			inner := sqrt2OverPi * (val + 0.044715*math.Pow(val, 3))
			result.Set(i, j, 0.5*val*(1+math.Tanh(inner)))
		}
	}

	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(r, c, nil),
	}
}

func (o *geluOp) Backward(grad *mat.Dense, inputs []*autograd.Value, output *autograd.Value) {
	x := inputs[0]
	r, c := x.Shape()

	sqrt2OverPi := math.Sqrt(2.0 / math.Pi)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := x.Data.At(i, j)

			// GELU derivative:
			// φ(x) = sqrt(2/π) * (1 + 3*0.044715*x^2)
			// cdf = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x^3)))
			// pdf = 0.5 * sqrt(2/π) * (1 + 3*0.044715*x^2) * sech^2(...)
			// GELU'(x) = cdf + x * pdf

			inner := sqrt2OverPi * (val + 0.044715*math.Pow(val, 3))
			tanhInner := math.Tanh(inner)
			cdf := 0.5 * (1 + tanhInner)
			sech2 := 1 - tanhInner*tanhInner
			innerDeriv := sqrt2OverPi * (1 + 3*0.044715*val*val)
			pdf := 0.5 * sech2 * innerDeriv

			deriv := cdf + val*pdf
			x.Grad.Set(i, j, x.Grad.At(i, j)+grad.At(i, j)*deriv)
		}
	}
}

// GELU applies Gaussian Error Linear Unit activation
func GELU(x *autograd.Value) *autograd.Value {
	op := &geluOp{}
	return op.Forward(x)
}

// Tanh applies hyperbolic tangent activation
type tanhOp struct{}

func (o *tanhOp) Name() string { return "Tanh" }

func (o *tanhOp) Forward(inputs ...*autograd.Value) *autograd.Value {
	x := inputs[0]
	r, c := x.Shape()

	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, math.Tanh(x.Data.At(i, j)))
		}
	}

	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(r, c, nil),
	}
}

func (o *tanhOp) Backward(grad *mat.Dense, inputs []*autograd.Value, output *autograd.Value) {
	x := inputs[0]
	r, c := x.Shape()

	// d/dx tanh(x) = 1 - tanh(x)^2
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tanhVal := output.Data.At(i, j)
			deriv := 1 - tanhVal*tanhVal
			x.Grad.Set(i, j, x.Grad.At(i, j)+grad.At(i, j)*deriv)
		}
	}
}

// Tanh applies the hyperbolic tangent activation
func Tanh(x *autograd.Value) *autograd.Value {
	op := &tanhOp{}
	return op.Forward(x)
}

// Sigmoid applies the sigmoid activation: σ(x) = 1 / (1 + exp(-x))
type sigmoidOp struct{}

func (o *sigmoidOp) Name() string { return "Sigmoid" }

func (o *sigmoidOp) Forward(inputs ...*autograd.Value) *autograd.Value {
	x := inputs[0]
	r, c := x.Shape()

	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, 1.0/(1.0+math.Exp(-x.Data.At(i, j))))
		}
	}

	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(r, c, nil),
	}
}

func (o *sigmoidOp) Backward(grad *mat.Dense, inputs []*autograd.Value, output *autograd.Value) {
	x := inputs[0]
	r, c := x.Shape()

	// d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sigVal := output.Data.At(i, j)
			deriv := sigVal * (1 - sigVal)
			x.Grad.Set(i, j, x.Grad.At(i, j)+grad.At(i, j)*deriv)
		}
	}
}

// Sigmoid applies the sigmoid activation
func Sigmoid(x *autograd.Value) *autograd.Value {
	op := &sigmoidOp{}
	return op.Forward(x)
}
