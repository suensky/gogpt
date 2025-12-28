package autograd

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// =============================================================================
// Helper function to create a result Value with operation tracking
// =============================================================================

func newResult(data *mat.Dense, op Operation, parents ...*Value) *Value {
	r, c := data.Dims()
	return &Value{
		Data:    data,
		Grad:    mat.NewDense(r, c, nil),
		op:      op,
		parents: parents,
	}
}

// =============================================================================
// Add Operation: Element-wise addition
// =============================================================================

type addOp struct{}

func (o *addOp) Name() string { return "Add" }

func (o *addOp) Forward(inputs ...*Value) *Value {
	a, b := inputs[0], inputs[1]
	ar, ac := a.Shape()
	br, bc := b.Shape()

	result := mat.NewDense(ar, ac, nil)

	// Handle broadcasting for bias addition (1, c) + (r, c)
	if br == 1 && bc == ac && ar > 1 {
		for i := 0; i < ar; i++ {
			for j := 0; j < ac; j++ {
				result.Set(i, j, a.Data.At(i, j)+b.Data.At(0, j))
			}
		}
	} else if ar == br && ac == bc {
		result.Add(a.Data, b.Data)
	} else {
		panic("Add: incompatible shapes for broadcasting")
	}

	return newResult(result, o, a, b)
}

func (o *addOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a, b := inputs[0], inputs[1]
	ar, ac := a.Shape()
	br, bc := b.Shape()
	gr, gc := grad.Dims()

	// Gradient for a: accumulate grad
	if ar == gr && ac == gc {
		a.Grad.Add(a.Grad, grad)
	}

	// Gradient for b: may need to sum if broadcasting was applied
	if br == 1 && bc == gc && gr > 1 {
		// Sum along rows for broadcast case
		for j := 0; j < bc; j++ {
			sum := 0.0
			for i := 0; i < gr; i++ {
				sum += grad.At(i, j)
			}
			b.Grad.Set(0, j, b.Grad.At(0, j)+sum)
		}
	} else if br == gr && bc == gc {
		b.Grad.Add(b.Grad, grad)
	}
}

// Add performs element-wise addition: c = a + b
func Add(a, b *Value) *Value {
	op := &addOp{}
	return op.Forward(a, b)
}

// =============================================================================
// Sub Operation: Element-wise subtraction
// =============================================================================

type subOp struct{}

func (o *subOp) Name() string { return "Sub" }

func (o *subOp) Forward(inputs ...*Value) *Value {
	a, b := inputs[0], inputs[1]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	result.Sub(a.Data, b.Data)

	return newResult(result, o, a, b)
}

func (o *subOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a, b := inputs[0], inputs[1]
	a.Grad.Add(a.Grad, grad)

	// b's gradient is negative
	br, bc := b.Shape()
	negGrad := mat.NewDense(br, bc, nil)
	negGrad.Scale(-1, grad)
	b.Grad.Add(b.Grad, negGrad)
}

// Sub performs element-wise subtraction: c = a - b
func Sub(a, b *Value) *Value {
	op := &subOp{}
	return op.Forward(a, b)
}

// =============================================================================
// Mul Operation: Element-wise multiplication (Hadamard product)
// =============================================================================

type mulOp struct{}

func (o *mulOp) Name() string { return "Mul" }

func (o *mulOp) Forward(inputs ...*Value) *Value {
	a, b := inputs[0], inputs[1]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	result.MulElem(a.Data, b.Data)

	return newResult(result, o, a, b)
}

func (o *mulOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a, b := inputs[0], inputs[1]

	// da = grad * b
	ar, ac := a.Shape()
	daContrib := mat.NewDense(ar, ac, nil)
	daContrib.MulElem(grad, b.Data)
	a.Grad.Add(a.Grad, daContrib)

	// db = grad * a
	br, bc := b.Shape()
	dbContrib := mat.NewDense(br, bc, nil)
	dbContrib.MulElem(grad, a.Data)
	b.Grad.Add(b.Grad, dbContrib)
}

// Mul performs element-wise multiplication: c = a * b
func Mul(a, b *Value) *Value {
	op := &mulOp{}
	return op.Forward(a, b)
}

// =============================================================================
// MatMul Operation: Matrix multiplication
// =============================================================================

type matMulOp struct{}

func (o *matMulOp) Name() string { return "MatMul" }

func (o *matMulOp) Forward(inputs ...*Value) *Value {
	a, b := inputs[0], inputs[1]
	ar, _ := a.Shape()
	_, bc := b.Shape()

	result := mat.NewDense(ar, bc, nil)
	result.Mul(a.Data, b.Data)

	return newResult(result, o, a, b)
}

func (o *matMulOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a, b := inputs[0], inputs[1]

	// da = grad @ b.T
	ar, ac := a.Shape()
	daContrib := mat.NewDense(ar, ac, nil)
	daContrib.Mul(grad, b.Data.T())
	a.Grad.Add(a.Grad, daContrib)

	// db = a.T @ grad
	br, bc := b.Shape()
	dbContrib := mat.NewDense(br, bc, nil)
	dbContrib.Mul(a.Data.T(), grad)
	b.Grad.Add(b.Grad, dbContrib)
}

// MatMul performs matrix multiplication: c = a @ b
func MatMul(a, b *Value) *Value {
	op := &matMulOp{}
	return op.Forward(a, b)
}

// =============================================================================
// Transpose Operation
// =============================================================================

type transposeOp struct{}

func (o *transposeOp) Name() string { return "Transpose" }

func (o *transposeOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ac, ar, nil)
	result.Copy(a.Data.T())

	return newResult(result, o, a)
}

func (o *transposeOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	daContrib := mat.NewDense(ar, ac, nil)
	daContrib.Copy(grad.T())
	a.Grad.Add(a.Grad, daContrib)
}

// Transpose returns the transpose of the matrix
func Transpose(a *Value) *Value {
	op := &transposeOp{}
	return op.Forward(a)
}

// =============================================================================
// Scale Operation: Multiply by scalar
// =============================================================================

type scaleOp struct {
	scalar float64
}

func (o *scaleOp) Name() string { return "Scale" }

func (o *scaleOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	result.Scale(o.scalar, a.Data)

	return newResult(result, o, a)
}

func (o *scaleOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	daContrib := mat.NewDense(ar, ac, nil)
	daContrib.Scale(o.scalar, grad)
	a.Grad.Add(a.Grad, daContrib)
}

// Scale multiplies a Value by a scalar: c = a * scalar
func Scale(a *Value, scalar float64) *Value {
	op := &scaleOp{scalar: scalar}
	return op.Forward(a)
}

// =============================================================================
// Neg Operation: Negation
// =============================================================================

type negOp struct{}

func (o *negOp) Name() string { return "Neg" }

func (o *negOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	result.Scale(-1, a.Data)

	return newResult(result, o, a)
}

func (o *negOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	daContrib := mat.NewDense(ar, ac, nil)
	daContrib.Scale(-1, grad)
	a.Grad.Add(a.Grad, daContrib)
}

// Neg negates a Value: c = -a
func Neg(a *Value) *Value {
	op := &negOp{}
	return op.Forward(a)
}

// =============================================================================
// Sum Operation: Sum all elements
// =============================================================================

type sumOp struct{}

func (o *sumOp) Name() string { return "Sum" }

func (o *sumOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	sum := 0.0
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			sum += a.Data.At(i, j)
		}
	}

	result := mat.NewDense(1, 1, []float64{sum})
	return newResult(result, o, a)
}

func (o *sumOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()
	gradVal := grad.At(0, 0)

	// Gradient flows to all elements
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			a.Grad.Set(i, j, a.Grad.At(i, j)+gradVal)
		}
	}
}

// Sum returns the sum of all elements as a scalar Value
func Sum(a *Value) *Value {
	op := &sumOp{}
	return op.Forward(a)
}

// =============================================================================
// Mean Operation: Mean of all elements
// =============================================================================

type meanOp struct {
	size float64
}

func (o *meanOp) Name() string { return "Mean" }

func (o *meanOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()
	o.size = float64(ar * ac)

	sum := 0.0
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			sum += a.Data.At(i, j)
		}
	}

	result := mat.NewDense(1, 1, []float64{sum / o.size})
	return newResult(result, o, a)
}

func (o *meanOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()
	gradVal := grad.At(0, 0) / o.size

	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			a.Grad.Set(i, j, a.Grad.At(i, j)+gradVal)
		}
	}
}

// Mean returns the mean of all elements as a scalar Value
func Mean(a *Value) *Value {
	op := &meanOp{}
	return op.Forward(a)
}

// =============================================================================
// Exp Operation: Element-wise exponential
// =============================================================================

type expOp struct{}

func (o *expOp) Name() string { return "Exp" }

func (o *expOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			result.Set(i, j, math.Exp(a.Data.At(i, j)))
		}
	}

	return newResult(result, o, a)
}

func (o *expOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	// d/dx exp(x) = exp(x)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			a.Grad.Set(i, j, a.Grad.At(i, j)+grad.At(i, j)*output.Data.At(i, j))
		}
	}
}

// Exp computes element-wise exponential
func Exp(a *Value) *Value {
	op := &expOp{}
	return op.Forward(a)
}

// =============================================================================
// Log Operation: Element-wise natural logarithm
// =============================================================================

type logOp struct{}

func (o *logOp) Name() string { return "Log" }

func (o *logOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			val := a.Data.At(i, j)
			if val <= 0 {
				result.Set(i, j, math.Inf(-1))
			} else {
				result.Set(i, j, math.Log(val))
			}
		}
	}

	return newResult(result, o, a)
}

func (o *logOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	// d/dx log(x) = 1/x
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			val := a.Data.At(i, j)
			if val != 0 {
				a.Grad.Set(i, j, a.Grad.At(i, j)+grad.At(i, j)/val)
			}
		}
	}
}

// Log computes element-wise natural logarithm
func Log(a *Value) *Value {
	op := &logOp{}
	return op.Forward(a)
}

// =============================================================================
// Pow Operation: Element-wise power
// =============================================================================

type powOp struct {
	n float64
}

func (o *powOp) Name() string { return "Pow" }

func (o *powOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			result.Set(i, j, math.Pow(a.Data.At(i, j), o.n))
		}
	}

	return newResult(result, o, a)
}

func (o *powOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	// d/dx x^n = n * x^(n-1)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			val := a.Data.At(i, j)
			deriv := o.n * math.Pow(val, o.n-1)
			a.Grad.Set(i, j, a.Grad.At(i, j)+grad.At(i, j)*deriv)
		}
	}
}

// Pow computes element-wise power: a^n
func Pow(a *Value, n float64) *Value {
	op := &powOp{n: n}
	return op.Forward(a)
}

// =============================================================================
// ReLU Operation: Rectified Linear Unit
// =============================================================================

type reluOp struct{}

func (o *reluOp) Name() string { return "ReLU" }

func (o *reluOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			val := a.Data.At(i, j)
			if val > 0 {
				result.Set(i, j, val)
			}
		}
	}

	return newResult(result, o, a)
}

func (o *reluOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	// Gradient is 1 where input > 0, else 0
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if a.Data.At(i, j) > 0 {
				a.Grad.Set(i, j, a.Grad.At(i, j)+grad.At(i, j))
			}
		}
	}
}

// ReLU applies the Rectified Linear Unit activation: max(0, x)
func ReLU(a *Value) *Value {
	op := &reluOp{}
	return op.Forward(a)
}

// =============================================================================
// Softmax Operation (row-wise)
// =============================================================================

type softmaxOp struct{}

func (o *softmaxOp) Name() string { return "Softmax" }

func (o *softmaxOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)

	// Apply softmax row-wise with numerical stability
	for i := 0; i < ar; i++ {
		// Find max for numerical stability
		maxVal := a.Data.At(i, 0)
		for j := 1; j < ac; j++ {
			if a.Data.At(i, j) > maxVal {
				maxVal = a.Data.At(i, j)
			}
		}

		// Compute exp(x - max) and sum
		sum := 0.0
		expVals := make([]float64, ac)
		for j := 0; j < ac; j++ {
			expVals[j] = math.Exp(a.Data.At(i, j) - maxVal)
			sum += expVals[j]
		}

		// Normalize
		for j := 0; j < ac; j++ {
			result.Set(i, j, expVals[j]/sum)
		}
	}

	return newResult(result, o, a)
}

func (o *softmaxOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	// Softmax backward: for each row, compute Jacobian-vector product
	// d_i = s_i * (grad_i - sum_j(grad_j * s_j))
	for i := 0; i < ar; i++ {
		// Compute sum_j(grad_j * s_j) for this row
		dotProd := 0.0
		for j := 0; j < ac; j++ {
			dotProd += grad.At(i, j) * output.Data.At(i, j)
		}

		// Compute gradient for each element
		for j := 0; j < ac; j++ {
			s := output.Data.At(i, j)
			d := s * (grad.At(i, j) - dotProd)
			a.Grad.Set(i, j, a.Grad.At(i, j)+d)
		}
	}
}

// Softmax applies softmax activation row-wise
func Softmax(a *Value) *Value {
	op := &softmaxOp{}
	return op.Forward(a)
}

// =============================================================================
// LogSoftmax Operation (row-wise) - more numerically stable for CrossEntropy
// =============================================================================

type logSoftmaxOp struct{}

func (o *logSoftmaxOp) Name() string { return "LogSoftmax" }

func (o *logSoftmaxOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)

	// Apply log-softmax row-wise: log(softmax(x)) = x - max - log(sum(exp(x - max)))
	for i := 0; i < ar; i++ {
		// Find max for numerical stability
		maxVal := a.Data.At(i, 0)
		for j := 1; j < ac; j++ {
			if a.Data.At(i, j) > maxVal {
				maxVal = a.Data.At(i, j)
			}
		}

		// Compute log-sum-exp
		sumExp := 0.0
		for j := 0; j < ac; j++ {
			sumExp += math.Exp(a.Data.At(i, j) - maxVal)
		}
		logSumExp := maxVal + math.Log(sumExp)

		// log-softmax = x - log-sum-exp
		for j := 0; j < ac; j++ {
			result.Set(i, j, a.Data.At(i, j)-logSumExp)
		}
	}

	return newResult(result, o, a)
}

func (o *logSoftmaxOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	// LogSoftmax backward: d_i = grad_i - softmax_i * sum_j(grad_j)
	for i := 0; i < ar; i++ {
		// Compute sum of gradients for this row
		gradSum := 0.0
		for j := 0; j < ac; j++ {
			gradSum += grad.At(i, j)
		}

		// Compute gradient: grad - softmax * gradSum
		for j := 0; j < ac; j++ {
			softmax := math.Exp(output.Data.At(i, j)) // exp(log_softmax) = softmax
			d := grad.At(i, j) - softmax*gradSum
			a.Grad.Set(i, j, a.Grad.At(i, j)+d)
		}
	}
}

// LogSoftmax applies log-softmax activation row-wise (numerically stable)
func LogSoftmax(a *Value) *Value {
	op := &logSoftmaxOp{}
	return op.Forward(a)
}

// =============================================================================
// Div Operation: Element-wise division
// =============================================================================

type divOp struct{}

func (o *divOp) Name() string { return "Div" }

func (o *divOp) Forward(inputs ...*Value) *Value {
	a, b := inputs[0], inputs[1]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	result.DivElem(a.Data, b.Data)

	return newResult(result, o, a, b)
}

func (o *divOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a, b := inputs[0], inputs[1]
	ar, ac := a.Shape()

	// da = grad / b
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			bVal := b.Data.At(i, j)
			a.Grad.Set(i, j, a.Grad.At(i, j)+grad.At(i, j)/bVal)

			// db = -grad * a / b^2
			aVal := a.Data.At(i, j)
			b.Grad.Set(i, j, b.Grad.At(i, j)-grad.At(i, j)*aVal/(bVal*bVal))
		}
	}
}

// Div performs element-wise division: c = a / b
func Div(a, b *Value) *Value {
	op := &divOp{}
	return op.Forward(a, b)
}

// =============================================================================
// Sqrt Operation: Element-wise square root
// =============================================================================

type sqrtOp struct{}

func (o *sqrtOp) Name() string { return "Sqrt" }

func (o *sqrtOp) Forward(inputs ...*Value) *Value {
	a := inputs[0]
	ar, ac := a.Shape()

	result := mat.NewDense(ar, ac, nil)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			result.Set(i, j, math.Sqrt(a.Data.At(i, j)))
		}
	}

	return newResult(result, o, a)
}

func (o *sqrtOp) Backward(grad *mat.Dense, inputs []*Value, output *Value) {
	a := inputs[0]
	ar, ac := a.Shape()

	// d/dx sqrt(x) = 1 / (2 * sqrt(x))
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			sqrtVal := output.Data.At(i, j)
			if sqrtVal != 0 {
				a.Grad.Set(i, j, a.Grad.At(i, j)+grad.At(i, j)/(2*sqrtVal))
			}
		}
	}
}

// Sqrt computes element-wise square root
func Sqrt(a *Value) *Value {
	op := &sqrtOp{}
	return op.Forward(a)
}

// =============================================================================
// Variance Operation (row-wise for LayerNorm)
// =============================================================================

// Variance computes the variance along the last axis (column variance per row)
// Returns a column vector of shape [rows, 1]
func Variance(a *Value) *Value {
	ar, ac := a.Shape()

	// First compute mean per row
	means := make([]float64, ar)
	for i := 0; i < ar; i++ {
		sum := 0.0
		for j := 0; j < ac; j++ {
			sum += a.Data.At(i, j)
		}
		means[i] = sum / float64(ac)
	}

	// Compute variance per row
	variances := make([]float64, ar)
	for i := 0; i < ar; i++ {
		sum := 0.0
		for j := 0; j < ac; j++ {
			diff := a.Data.At(i, j) - means[i]
			sum += diff * diff
		}
		variances[i] = sum / float64(ac)
	}

	return NewVariable(ar, 1, variances)
}

// RowMean computes the mean along each row
// Returns a column vector of shape [rows, 1]
func RowMean(a *Value) *Value {
	ar, ac := a.Shape()

	means := make([]float64, ar)
	for i := 0; i < ar; i++ {
		sum := 0.0
		for j := 0; j < ac; j++ {
			sum += a.Data.At(i, j)
		}
		means[i] = sum / float64(ac)
	}

	return NewVariable(ar, 1, means)
}
