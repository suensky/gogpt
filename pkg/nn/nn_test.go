package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/suensky/gogpt/pkg/autograd"
)

func approxEqual(a, b, eps float64) bool {
	return math.Abs(a-b) < eps
}

func TestLinearForward(t *testing.T) {
	// Set seed for reproducibility
	rand.Seed(42)

	linear := NewLinear(3, 2)

	// Set known weights for testing
	linear.Weights = autograd.NewVariable(2, 3, []float64{
		1, 2, 3, // First output neuron weights
		4, 5, 6, // Second output neuron weights
	})
	linear.Bias = autograd.NewVariable(1, 2, []float64{0.5, 0.5})

	// Input: 2 samples, 3 features
	input := autograd.NewVariable(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})

	output := linear.Forward(input)

	// Expected:
	// y[0] = [1*1+2*2+3*3+0.5, 1*4+2*5+3*6+0.5] = [14.5, 32.5]
	// y[1] = [4*1+5*2+6*3+0.5, 4*4+5*5+6*6+0.5] = [32.5, 77.5]
	if !approxEqual(output.Data.At(0, 0), 14.5, 1e-6) {
		t.Errorf("Expected output[0,0] = 14.5, got %f", output.Data.At(0, 0))
	}
	if !approxEqual(output.Data.At(1, 1), 77.5, 1e-6) {
		t.Errorf("Expected output[1,1] = 77.5, got %f", output.Data.At(1, 1))
	}
}

func TestLinearBackward(t *testing.T) {
	rand.Seed(42)

	linear := NewLinear(2, 1)
	linear.Weights = autograd.NewVariable(1, 2, []float64{1, 2})
	linear.Bias = autograd.NewVariable(1, 1, []float64{0})

	input := autograd.NewVariable(1, 2, []float64{3, 4})

	output := linear.Forward(input)
	// y = 3*1 + 4*2 = 11

	if !approxEqual(output.ScalarValue(), 11.0, 1e-6) {
		t.Errorf("Expected output = 11, got %f", output.ScalarValue())
	}

	// Backward
	output.Backward()

	// Gradient of output w.r.t. weights should be input values
	// dL/dW = x (transposed gradient flow)
	if !approxEqual(linear.Weights.Grad.At(0, 0), 3.0, 1e-6) {
		t.Errorf("Expected weight grad[0,0] = 3, got %f", linear.Weights.Grad.At(0, 0))
	}
	if !approxEqual(linear.Weights.Grad.At(0, 1), 4.0, 1e-6) {
		t.Errorf("Expected weight grad[0,1] = 4, got %f", linear.Weights.Grad.At(0, 1))
	}
}

func TestEmbedding(t *testing.T) {
	embed := NewEmbedding(5, 3) // 5 tokens, dimension 3

	// Set known weights
	embed.Weight = autograd.NewVariable(5, 3, []float64{
		0.1, 0.2, 0.3, // Token 0
		0.4, 0.5, 0.6, // Token 1
		0.7, 0.8, 0.9, // Token 2
		1.0, 1.1, 1.2, // Token 3
		1.3, 1.4, 1.5, // Token 4
	})

	// Look up tokens 1, 3
	output := embed.Forward([]int{1, 3})

	r, c := output.Shape()
	if r != 2 || c != 3 {
		t.Errorf("Expected shape 2x3, got %dx%d", r, c)
	}

	// Check token 1 embedding
	if !approxEqual(output.Data.At(0, 0), 0.4, 1e-6) {
		t.Errorf("Expected embed[0,0] = 0.4, got %f", output.Data.At(0, 0))
	}

	// Check token 3 embedding
	if !approxEqual(output.Data.At(1, 2), 1.2, 1e-6) {
		t.Errorf("Expected embed[1,2] = 1.2, got %f", output.Data.At(1, 2))
	}
}

func TestSGDOptimizer(t *testing.T) {
	// Simple gradient descent test
	// Minimize f(x) = x^2, starting at x=4
	// After one step with lr=0.5: x = 4 - 0.5*2*4 = 0

	x := autograd.NewVariable(1, 1, []float64{4.0})

	opt := NewSGD([]*autograd.Value{x}, 0.5)

	// Forward: f(x) = x^2
	y := autograd.Mul(x, x)

	// Backward
	y.Backward()

	// Gradient should be 2x = 8
	if !approxEqual(x.Grad.At(0, 0), 8.0, 1e-6) {
		t.Errorf("Expected gradient = 8, got %f", x.Grad.At(0, 0))
	}

	// Step
	opt.Step()

	// x should be 4 - 0.5*8 = 0
	if !approxEqual(x.Data.At(0, 0), 0.0, 1e-6) {
		t.Errorf("Expected x = 0 after step, got %f", x.Data.At(0, 0))
	}
}

func TestAdamOptimizer(t *testing.T) {
	// Test Adam with simple quadratic
	x := autograd.NewVariable(1, 1, []float64{4.0})
	opt := NewAdam([]*autograd.Value{x}, 0.1)

	initialX := x.Data.At(0, 0)

	for i := 0; i < 10; i++ {
		opt.ZeroGrad()
		y := autograd.Mul(x, x)
		y.Backward()
		opt.Step()
	}

	// x should decrease after optimization
	if x.Data.At(0, 0) >= initialX {
		t.Errorf("Expected x to decrease, got %f (was %f)", x.Data.At(0, 0), initialX)
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	// Logits for 2 samples, 3 classes
	logits := autograd.NewVariable(2, 3, []float64{
		1.0, 2.0, 3.0, // Sample 0: softmax will favor class 2
		3.0, 2.0, 1.0, // Sample 1: softmax will favor class 0
	})

	targets := []int{2, 0} // Correct predictions

	loss := CrossEntropyLoss(logits, targets)

	// Loss should be relatively small since predictions match targets
	lossVal := loss.ScalarValue()
	if lossVal < 0 || lossVal > 2.0 {
		t.Errorf("Unexpected loss value: %f", lossVal)
	}

	// Now test with wrong predictions
	wrongTargets := []int{0, 2}
	lossWrong := CrossEntropyLoss(logits, wrongTargets)

	// Loss should be higher for wrong predictions
	if lossWrong.ScalarValue() <= lossVal {
		t.Errorf("Expected higher loss for wrong predictions")
	}
}

func TestSinusoidalPositionalEncoding(t *testing.T) {
	pe := SinusoidalPositionalEncoding(10, 4)

	r, c := pe.Shape()
	if r != 10 || c != 4 {
		t.Errorf("Expected shape 10x4, got %dx%d", r, c)
	}

	// Position 0 should have sin(0)=0 for even indices
	if !approxEqual(pe.Data.At(0, 0), 0.0, 1e-6) {
		t.Errorf("Expected PE[0,0] = 0, got %f", pe.Data.At(0, 0))
	}

	// Position 0 should have cos(0)=1 for odd indices
	if !approxEqual(pe.Data.At(0, 1), 1.0, 1e-6) {
		t.Errorf("Expected PE[0,1] = 1, got %f", pe.Data.At(0, 1))
	}
}

func TestMSELoss(t *testing.T) {
	predictions := autograd.NewVariable(2, 2, []float64{
		1.0, 2.0,
		3.0, 4.0,
	})
	targets := autograd.NewVariable(2, 2, []float64{
		1.0, 2.0,
		3.0, 4.0,
	})

	loss := MSELoss(predictions, targets)

	// Perfect predictions should have zero loss
	if !approxEqual(loss.ScalarValue(), 0.0, 1e-6) {
		t.Errorf("Expected MSE = 0 for perfect predictions, got %f", loss.ScalarValue())
	}

	// Test with errors
	predictions2 := autograd.NewVariable(2, 2, []float64{
		2.0, 3.0,
		4.0, 5.0,
	})
	loss2 := MSELoss(predictions2, targets)

	// Each element has error of 1, MSE = mean(1^2) = 1
	if !approxEqual(loss2.ScalarValue(), 1.0, 1e-6) {
		t.Errorf("Expected MSE = 1, got %f", loss2.ScalarValue())
	}
}
