package autograd

import (
	"math"
	"testing"
)

// Helper to check if two floats are approximately equal
func approxEqual(a, b, eps float64) bool {
	return math.Abs(a-b) < eps
}

// TestBasicAddMul tests the example: c = a * b + d
// Verifies that gradients are computed correctly.
func TestBasicAddMul(t *testing.T) {
	// Create leaf nodes
	a := NewVariable(1, 1, []float64{2.0})
	a.SetName("a")
	b := NewVariable(1, 1, []float64{3.0})
	b.SetName("b")
	d := NewVariable(1, 1, []float64{4.0})
	d.SetName("d")

	// c = a * b + d = 2 * 3 + 4 = 10
	ab := Mul(a, b) // a * b = 6
	c := Add(ab, d) // 6 + 4 = 10

	// Verify forward pass
	if !approxEqual(c.ScalarValue(), 10.0, 1e-6) {
		t.Errorf("Expected c = 10, got %f", c.ScalarValue())
	}

	// Backward pass
	c.Backward()

	// Expected gradients:
	// dc/dc = 1
	// dc/d(ab) = 1, dc/dd = 1
	// dc/da = dc/d(ab) * d(ab)/da = 1 * b = 3
	// dc/db = dc/d(ab) * d(ab)/db = 1 * a = 2

	if !approxEqual(a.Grad.At(0, 0), 3.0, 1e-6) {
		t.Errorf("Expected da = 3.0, got %f", a.Grad.At(0, 0))
	}
	if !approxEqual(b.Grad.At(0, 0), 2.0, 1e-6) {
		t.Errorf("Expected db = 2.0, got %f", b.Grad.At(0, 0))
	}
	if !approxEqual(d.Grad.At(0, 0), 1.0, 1e-6) {
		t.Errorf("Expected dd = 1.0, got %f", d.Grad.At(0, 0))
	}
}

// TestMatMulGradient tests matrix multiplication gradients
func TestMatMulGradient(t *testing.T) {
	// A: 2x3 matrix
	a := NewVariable(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	a.SetName("A")

	// B: 3x2 matrix
	b := NewVariable(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})
	b.SetName("B")

	// C = A @ B: 2x2 matrix
	c := MatMul(a, b)

	// Expected C:
	// [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
	// [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
	if !approxEqual(c.Data.At(0, 0), 22.0, 1e-6) {
		t.Errorf("Expected C[0,0] = 22, got %f", c.Data.At(0, 0))
	}
	if !approxEqual(c.Data.At(1, 1), 64.0, 1e-6) {
		t.Errorf("Expected C[1,1] = 64, got %f", c.Data.At(1, 1))
	}

	// Sum to get scalar loss
	loss := Sum(c)

	// Backward
	loss.Backward()

	// Gradient of sum is 1 for all elements
	// dL/dC = ones(2, 2)
	// dL/dA = dL/dC @ B.T
	// dL/dB = A.T @ dL/dC

	// Expected dL/dA (2x3):
	// ones(2,2) @ B.T = [[1,1],[1,1]] @ [[1,3,5],[2,4,6]] = [[3,7,11],[3,7,11]]
	if !approxEqual(a.Grad.At(0, 0), 3.0, 1e-6) {
		t.Errorf("Expected dA[0,0] = 3, got %f", a.Grad.At(0, 0))
	}
	if !approxEqual(a.Grad.At(0, 2), 11.0, 1e-6) {
		t.Errorf("Expected dA[0,2] = 11, got %f", a.Grad.At(0, 2))
	}
}

// TestReLUGradient tests ReLU activation gradient
func TestReLUGradient(t *testing.T) {
	a := NewVariable(2, 3, []float64{
		-1, 0, 1,
		2, -3, 4,
	})

	out := ReLU(a)

	// Check forward
	expected := []float64{0, 0, 1, 2, 0, 4}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if !approxEqual(out.Data.At(i, j), expected[i*3+j], 1e-6) {
				t.Errorf("ReLU: expected %f at (%d,%d), got %f", expected[i*3+j], i, j, out.Data.At(i, j))
			}
		}
	}

	// Backward with sum loss
	loss := Sum(out)
	loss.Backward()

	// Gradient is 1 where input > 0, else 0
	expectedGrad := []float64{0, 0, 1, 1, 0, 1}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if !approxEqual(a.Grad.At(i, j), expectedGrad[i*3+j], 1e-6) {
				t.Errorf("ReLU grad: expected %f at (%d,%d), got %f", expectedGrad[i*3+j], i, j, a.Grad.At(i, j))
			}
		}
	}
}

// TestSoftmaxGradient tests softmax activation gradient
func TestSoftmaxGradient(t *testing.T) {
	// Single row
	a := NewVariable(1, 3, []float64{1.0, 2.0, 3.0})

	out := Softmax(a)

	// Check that softmax sums to 1
	sum := out.Data.At(0, 0) + out.Data.At(0, 1) + out.Data.At(0, 2)
	if !approxEqual(sum, 1.0, 1e-6) {
		t.Errorf("Softmax should sum to 1, got %f", sum)
	}

	// Check relative ordering (larger input -> larger output)
	if out.Data.At(0, 0) >= out.Data.At(0, 1) || out.Data.At(0, 1) >= out.Data.At(0, 2) {
		t.Error("Softmax ordering incorrect")
	}
}

// TestLogSoftmaxGradient tests log-softmax gradient
func TestLogSoftmaxGradient(t *testing.T) {
	a := NewVariable(1, 3, []float64{1.0, 2.0, 3.0})

	out := LogSoftmax(a)

	// log-softmax values should be negative (log of probabilities < 1)
	for j := 0; j < 3; j++ {
		if out.Data.At(0, j) > 0 {
			t.Errorf("Log-softmax should be negative, got %f at %d", out.Data.At(0, j), j)
		}
	}

	// exp(log-softmax) should sum to 1
	sum := math.Exp(out.Data.At(0, 0)) + math.Exp(out.Data.At(0, 1)) + math.Exp(out.Data.At(0, 2))
	if !approxEqual(sum, 1.0, 1e-6) {
		t.Errorf("exp(LogSoftmax) should sum to 1, got %f", sum)
	}
}

// TestTransposeGradient tests transpose gradient
func TestTransposeGradient(t *testing.T) {
	a := NewVariable(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})

	aT := Transpose(a)

	// Check shape
	r, c := aT.Shape()
	if r != 3 || c != 2 {
		t.Errorf("Transpose shape should be 3x2, got %dx%d", r, c)
	}

	// Check values
	if !approxEqual(aT.Data.At(0, 1), 4.0, 1e-6) {
		t.Errorf("Transpose: expected A.T[0,1] = 4, got %f", aT.Data.At(0, 1))
	}

	// Backward
	loss := Sum(aT)
	loss.Backward()

	// All gradients should be 1 (sum gradient)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if !approxEqual(a.Grad.At(i, j), 1.0, 1e-6) {
				t.Errorf("Transpose grad should be 1 at (%d,%d), got %f", i, j, a.Grad.At(i, j))
			}
		}
	}
}

// TestChainedOperations tests a more complex computation graph
func TestChainedOperations(t *testing.T) {
	// y = (x^2 + 3x + 1) where x = 2
	// dy/dx = 2x + 3 = 7

	x := NewVariable(1, 1, []float64{2.0})
	x.SetName("x")

	// x^2
	xSquared := Mul(x, x)

	// 3x
	three := Scalar(3.0)
	threeX := Mul(three, x)

	// x^2 + 3x
	sum1 := Add(xSquared, threeX)

	// + 1
	one := Scalar(1.0)
	y := Add(sum1, one)

	// y = 4 + 6 + 1 = 11
	if !approxEqual(y.ScalarValue(), 11.0, 1e-6) {
		t.Errorf("Expected y = 11, got %f", y.ScalarValue())
	}

	y.Backward()

	// dy/dx = 2x + 3 = 7
	// But x is used twice in x^2 and once in 3x
	// In x^2 = x * x, gradient accumulates: 2x = 4
	// In 3x, gradient is 3
	// Total: 4 + 3 = 7
	if !approxEqual(x.Grad.At(0, 0), 7.0, 1e-6) {
		t.Errorf("Expected dx = 7.0, got %f", x.Grad.At(0, 0))
	}
}

// TestMeanGradient tests mean operation gradient
func TestMeanGradient(t *testing.T) {
	a := NewVariable(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})

	mean := Mean(a)

	// Mean = 21/6 = 3.5
	if !approxEqual(mean.ScalarValue(), 3.5, 1e-6) {
		t.Errorf("Expected mean = 3.5, got %f", mean.ScalarValue())
	}

	mean.Backward()

	// Gradient of mean is 1/n for all elements
	expectedGrad := 1.0 / 6.0
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if !approxEqual(a.Grad.At(i, j), expectedGrad, 1e-6) {
				t.Errorf("Mean grad should be %f at (%d,%d), got %f", expectedGrad, i, j, a.Grad.At(i, j))
			}
		}
	}
}

// TestExpLogGradient tests exp and log gradients
func TestExpLogGradient(t *testing.T) {
	a := NewVariable(1, 2, []float64{1.0, 2.0})

	// exp then log should give back the original
	expA := Exp(a)
	logExpA := Log(expA)

	if !approxEqual(logExpA.Data.At(0, 0), 1.0, 1e-6) {
		t.Errorf("log(exp(1)) should be 1, got %f", logExpA.Data.At(0, 0))
	}

	loss := Sum(logExpA)
	loss.Backward()

	// Gradient of log(exp(x)) with respect to x is 1
	if !approxEqual(a.Grad.At(0, 0), 1.0, 1e-6) {
		t.Errorf("d/dx log(exp(x)) should be 1, got %f", a.Grad.At(0, 0))
	}
}

// TestBroadcastAdd tests broadcasting in addition (bias addition)
func TestBroadcastAdd(t *testing.T) {
	// A is 3x4, B is 1x4 (bias)
	a := NewVariable(3, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	b := NewVariable(1, 4, []float64{1, 1, 1, 1})

	c := Add(a, b)

	// Check shape
	r, cols := c.Shape()
	if r != 3 || cols != 4 {
		t.Errorf("Expected shape 3x4, got %dx%d", r, cols)
	}

	// Check values (each element increased by 1)
	if !approxEqual(c.Data.At(0, 0), 2.0, 1e-6) {
		t.Errorf("Expected c[0,0] = 2, got %f", c.Data.At(0, 0))
	}

	// Backward
	loss := Sum(c)
	loss.Backward()

	// Gradient for a should be 1 everywhere
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if !approxEqual(a.Grad.At(i, j), 1.0, 1e-6) {
				t.Errorf("Expected da[%d,%d] = 1, got %f", i, j, a.Grad.At(i, j))
			}
		}
	}

	// Gradient for b should sum across rows: 3 for each column
	for j := 0; j < 4; j++ {
		if !approxEqual(b.Grad.At(0, j), 3.0, 1e-6) {
			t.Errorf("Expected db[0,%d] = 3, got %f", j, b.Grad.At(0, j))
		}
	}
}

// TestNumericalGradient tests gradients against numerical differentiation
func TestNumericalGradient(t *testing.T) {
	eps := 1e-5

	// Test function: f(x) = x^2
	x := NewVariable(1, 1, []float64{3.0})
	y := Mul(x, x)
	y.Backward()

	// Numerical gradient
	xPlus := NewVariable(1, 1, []float64{3.0 + eps})
	xMinus := NewVariable(1, 1, []float64{3.0 - eps})
	yPlus := Mul(xPlus, xPlus)
	yMinus := Mul(xMinus, xMinus)
	numGrad := (yPlus.ScalarValue() - yMinus.ScalarValue()) / (2 * eps)

	// Analytical gradient should be 2x = 6
	analyticalGrad := x.Grad.At(0, 0)

	if !approxEqual(analyticalGrad, numGrad, 1e-4) {
		t.Errorf("Gradient mismatch: analytical=%f, numerical=%f", analyticalGrad, numGrad)
	}
}
