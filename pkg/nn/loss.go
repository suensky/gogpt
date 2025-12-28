package nn

import (
	"math"

	"github.com/suensky/gogpt/pkg/autograd"
	"gonum.org/v1/gonum/mat"
)

// CrossEntropyLoss computes the cross-entropy loss between logits and target indices.
// logits: [batchSize, numClasses] - raw scores (before softmax)
// targets: slice of target class indices for each sample
// Returns: scalar loss value
func CrossEntropyLoss(logits *autograd.Value, targets []int) *autograd.Value {
	op := &crossEntropyOp{targets: targets}
	return op.Forward(logits)
}

type crossEntropyOp struct {
	targets   []int
	softmax   *mat.Dense // Cache for backward pass
	batchSize int
}

func (o *crossEntropyOp) Name() string { return "CrossEntropyLoss" }

func (o *crossEntropyOp) Forward(inputs ...*autograd.Value) *autograd.Value {
	logits := inputs[0]
	batchSize, numClasses := logits.Shape()
	o.batchSize = batchSize

	if len(o.targets) != batchSize {
		panic("targets length must match batch size")
	}

	// Compute softmax with numerical stability
	softmax := mat.NewDense(batchSize, numClasses, nil)
	loss := 0.0

	for i := range batchSize {
		// Find max for numerical stability
		maxVal := logits.Data.At(i, 0)
		for j := 1; j < numClasses; j++ {
			if logits.Data.At(i, j) > maxVal {
				maxVal = logits.Data.At(i, j)
			}
		}

		// Compute exp(x - max) and sum
		sumExp := 0.0
		for j := range numClasses {
			expVal := math.Exp(logits.Data.At(i, j) - maxVal)
			softmax.Set(i, j, expVal)
			sumExp += expVal
		}

		// Normalize to get softmax probabilities
		for j := range numClasses {
			softmax.Set(i, j, softmax.At(i, j)/sumExp)
		}

		// Cross-entropy: -log(softmax[target])
		targetIdx := o.targets[i]
		prob := softmax.At(i, targetIdx)
		if prob < 1e-10 {
			prob = 1e-10 // Avoid log(0)
		}
		loss -= math.Log(prob)
	}

	// Average loss
	loss /= float64(batchSize)

	// Cache softmax for backward pass
	o.softmax = softmax

	// Create result with this operation attached
	result := mat.NewDense(1, 1, []float64{loss})
	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(1, 1, nil),
	}
}

func (o *crossEntropyOp) Backward(grad *mat.Dense, inputs []*autograd.Value, output *autograd.Value) {
	logits := inputs[0]
	batchSize, numClasses := logits.Shape()
	gradScale := grad.At(0, 0) / float64(batchSize)

	// Gradient of cross-entropy with softmax is: softmax - one_hot(target)
	for i := 0; i < batchSize; i++ {
		targetIdx := o.targets[i]
		for j := 0; j < numClasses; j++ {
			g := o.softmax.At(i, j)
			if j == targetIdx {
				g -= 1.0
			}
			g *= gradScale
			logits.Grad.Set(i, j, logits.Grad.At(i, j)+g)
		}
	}
}

// CrossEntropyLossWithLogits is an alias that makes it clear logits are expected
func CrossEntropyLossWithLogits(logits *autograd.Value, targets []int) *autograd.Value {
	return CrossEntropyLoss(logits, targets)
}

// CrossEntropyLossWithSmoothing computes cross-entropy loss with label smoothing.
// Label smoothing is a regularization technique that prevents the model from
// becoming too confident in its predictions.
// smoothing: typically 0.1, meaning 10% probability mass is redistributed
func CrossEntropyLossWithSmoothing(logits *autograd.Value, targets []int, smoothing float64) *autograd.Value {
	op := &smoothedCrossEntropyOp{targets: targets, smoothing: smoothing}
	return op.Forward(logits)
}

type smoothedCrossEntropyOp struct {
	targets   []int
	smoothing float64
	softmax   *mat.Dense
	batchSize int
}

func (o *smoothedCrossEntropyOp) Name() string { return "SmoothedCrossEntropyLoss" }

func (o *smoothedCrossEntropyOp) Forward(inputs ...*autograd.Value) *autograd.Value {
	logits := inputs[0]
	batchSize, numClasses := logits.Shape()
	o.batchSize = batchSize

	if len(o.targets) != batchSize {
		panic("targets length must match batch size")
	}

	// Compute softmax with numerical stability
	softmax := mat.NewDense(batchSize, numClasses, nil)
	loss := 0.0

	// Label smoothing: smooth_target = (1 - smoothing) * one_hot + smoothing / num_classes
	smoothedTarget := o.smoothing / float64(numClasses)
	trueTarget := 1.0 - o.smoothing + smoothedTarget

	for i := range batchSize {
		// Find max for numerical stability
		maxVal := logits.Data.At(i, 0)
		for j := 1; j < numClasses; j++ {
			if logits.Data.At(i, j) > maxVal {
				maxVal = logits.Data.At(i, j)
			}
		}

		// Compute exp(x - max) and sum
		sumExp := 0.0
		for j := range numClasses {
			expVal := math.Exp(logits.Data.At(i, j) - maxVal)
			softmax.Set(i, j, expVal)
			sumExp += expVal
		}

		// Normalize to get softmax probabilities
		for j := range numClasses {
			softmax.Set(i, j, softmax.At(i, j)/sumExp)
		}

		// Cross-entropy with smoothed labels
		targetIdx := o.targets[i]
		for j := range numClasses {
			prob := softmax.At(i, j)
			if prob < 1e-10 {
				prob = 1e-10
			}
			if j == targetIdx {
				loss -= trueTarget * math.Log(prob)
			} else {
				loss -= smoothedTarget * math.Log(prob)
			}
		}
	}

	// Average loss
	loss /= float64(batchSize)

	// Cache softmax for backward pass
	o.softmax = softmax

	result := mat.NewDense(1, 1, []float64{loss})
	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(1, 1, nil),
	}
}

func (o *smoothedCrossEntropyOp) Backward(grad *mat.Dense, inputs []*autograd.Value, output *autograd.Value) {
	logits := inputs[0]
	batchSize, numClasses := logits.Shape()
	gradScale := grad.At(0, 0) / float64(batchSize)

	// Label smoothing values
	smoothedTarget := o.smoothing / float64(numClasses)
	trueTarget := 1.0 - o.smoothing + smoothedTarget

	// Gradient of smoothed cross-entropy with softmax
	for i := 0; i < batchSize; i++ {
		targetIdx := o.targets[i]
		for j := 0; j < numClasses; j++ {
			softmax := o.softmax.At(i, j)
			var targetProb float64
			if j == targetIdx {
				targetProb = trueTarget
			} else {
				targetProb = smoothedTarget
			}
			g := (softmax - targetProb) * gradScale
			logits.Grad.Set(i, j, logits.Grad.At(i, j)+g)
		}
	}
}

// MSELoss computes Mean Squared Error loss.
// predictions and targets should have the same shape.
func MSELoss(predictions, targets *autograd.Value) *autograd.Value {
	diff := autograd.Sub(predictions, targets)
	squared := autograd.Mul(diff, diff)
	return autograd.Mean(squared)
}

// NLLLoss computes Negative Log-Likelihood loss.
// Expects log-probabilities as input (after LogSoftmax).
// logProbs: [batchSize, numClasses]
// targets: slice of target indices
func NLLLoss(logProbs *autograd.Value, targets []int) *autograd.Value {
	op := &nllOp{targets: targets}
	return op.Forward(logProbs)
}

type nllOp struct {
	targets   []int
	batchSize int
}

func (o *nllOp) Name() string { return "NLLLoss" }

func (o *nllOp) Forward(inputs ...*autograd.Value) *autograd.Value {
	logProbs := inputs[0]
	batchSize, _ := logProbs.Shape()
	o.batchSize = batchSize

	loss := 0.0
	for i := 0; i < batchSize; i++ {
		targetIdx := o.targets[i]
		loss -= logProbs.Data.At(i, targetIdx)
	}
	loss /= float64(batchSize)

	result := mat.NewDense(1, 1, []float64{loss})
	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(1, 1, nil),
	}
}

func (o *nllOp) Backward(grad *mat.Dense, inputs []*autograd.Value, output *autograd.Value) {
	logProbs := inputs[0]
	batchSize, numClasses := logProbs.Shape()
	gradScale := grad.At(0, 0) / float64(batchSize)

	// Gradient is -1/batchSize for target indices, 0 elsewhere
	for i := 0; i < batchSize; i++ {
		targetIdx := o.targets[i]
		for j := 0; j < numClasses; j++ {
			if j == targetIdx {
				logProbs.Grad.Set(i, j, logProbs.Grad.At(i, j)-gradScale)
			}
		}
	}
}
