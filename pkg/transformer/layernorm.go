// Package transformer implements GPT-style transformer decoder components.
package transformer

import (
	"math"

	"github.com/suensky/gogpt/pkg/autograd"
	"gonum.org/v1/gonum/mat"
)

// LayerNorm implements Layer Normalization.
// Normalizes across the last dimension (features) for each sample.
// Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
type LayerNorm struct {
	Gamma   *autograd.Value // Scale parameter [1, features]
	Beta    *autograd.Value // Shift parameter [1, features]
	Epsilon float64
}

// NewLayerNorm creates a new LayerNorm layer.
// features: number of features to normalize over
func NewLayerNorm(features int) *LayerNorm {
	// Initialize gamma to ones and beta to zeros
	gammaData := make([]float64, features)
	for i := range gammaData {
		gammaData[i] = 1.0
	}

	return &LayerNorm{
		Gamma:   autograd.NewVariable(1, features, gammaData).SetName("LayerNorm.gamma"),
		Beta:    autograd.NewVariable(1, features, nil).SetName("LayerNorm.beta"),
		Epsilon: 1e-5,
	}
}

// Forward applies layer normalization.
// Input shape: [batchSize, features]
// Output shape: [batchSize, features]
func (ln *LayerNorm) Forward(x *autograd.Value) *autograd.Value {
	op := &layerNormOp{
		ln: ln,
	}
	return op.Forward(x)
}

// Parameters returns the learnable parameters.
func (ln *LayerNorm) Parameters() []*autograd.Value {
	return []*autograd.Value{ln.Gamma, ln.Beta}
}

// layerNormOp implements LayerNorm as an autograd Operation
type layerNormOp struct {
	ln       *LayerNorm
	mean     *mat.Dense // Cache for backward
	variance *mat.Dense // Cache for backward
	xNorm    *mat.Dense // Cache for backward
}

func (o *layerNormOp) Name() string { return "LayerNorm" }

func (o *layerNormOp) Forward(inputs ...*autograd.Value) *autograd.Value {
	x := inputs[0]
	batchSize, features := x.Shape()

	// Compute mean and variance per sample
	o.mean = mat.NewDense(batchSize, 1, nil)
	o.variance = mat.NewDense(batchSize, 1, nil)
	o.xNorm = mat.NewDense(batchSize, features, nil)

	result := mat.NewDense(batchSize, features, nil)

	for i := 0; i < batchSize; i++ {
		// Compute mean
		sum := 0.0
		for j := 0; j < features; j++ {
			sum += x.Data.At(i, j)
		}
		mean := sum / float64(features)
		o.mean.Set(i, 0, mean)

		// Compute variance
		varSum := 0.0
		for j := 0; j < features; j++ {
			diff := x.Data.At(i, j) - mean
			varSum += diff * diff
		}
		variance := varSum / float64(features)
		o.variance.Set(i, 0, variance)

		// Normalize and scale
		std := math.Sqrt(variance + o.ln.Epsilon)
		for j := 0; j < features; j++ {
			xNorm := (x.Data.At(i, j) - mean) / std
			o.xNorm.Set(i, j, xNorm)
			// Apply gamma and beta (broadcasting from [1, features])
			result.Set(i, j, xNorm*o.ln.Gamma.Data.At(0, j)+o.ln.Beta.Data.At(0, j))
		}
	}

	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(batchSize, features, nil),
	}
}

func (o *layerNormOp) Backward(grad *mat.Dense, inputs []*autograd.Value, output *autograd.Value) {
	x := inputs[0]
	batchSize, features := x.Shape()
	n := float64(features)

	// Gradient for gamma: sum_batch(grad * xNorm)
	for j := 0; j < features; j++ {
		sum := 0.0
		for i := 0; i < batchSize; i++ {
			sum += grad.At(i, j) * o.xNorm.At(i, j)
		}
		o.ln.Gamma.Grad.Set(0, j, o.ln.Gamma.Grad.At(0, j)+sum)
	}

	// Gradient for beta: sum_batch(grad)
	for j := 0; j < features; j++ {
		sum := 0.0
		for i := 0; i < batchSize; i++ {
			sum += grad.At(i, j)
		}
		o.ln.Beta.Grad.Set(0, j, o.ln.Beta.Grad.At(0, j)+sum)
	}

	// Gradient for input x
	for i := 0; i < batchSize; i++ {
		variance := o.variance.At(i, 0)
		std := math.Sqrt(variance + o.ln.Epsilon)

		// Compute intermediate sums for this sample
		sumGradGamma := 0.0
		sumGradGammaXNorm := 0.0
		for j := 0; j < features; j++ {
			gradGamma := grad.At(i, j) * o.ln.Gamma.Data.At(0, j)
			sumGradGamma += gradGamma
			sumGradGammaXNorm += gradGamma * o.xNorm.At(i, j)
		}

		// Compute gradient for each feature
		for j := 0; j < features; j++ {
			gradGamma := grad.At(i, j) * o.ln.Gamma.Data.At(0, j)
			dx := (gradGamma - sumGradGamma/n - o.xNorm.At(i, j)*sumGradGammaXNorm/n) / std
			x.Grad.Set(i, j, x.Grad.At(i, j)+dx)
		}
	}
}
