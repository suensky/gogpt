package nn

import (
	"math"
	"math/rand"

	"github.com/suensky/gogpt/pkg/autograd"
)

// Linear implements a fully connected layer: y = x @ W.T + b
type Linear struct {
	Weights *autograd.Value // [outFeatures, inFeatures]
	Bias    *autograd.Value // [1, outFeatures]
	UseBias bool
}

// NewLinear creates a new Linear layer with Xavier/Glorot initialization.
// inFeatures: number of input features
// outFeatures: number of output features
func NewLinear(inFeatures, outFeatures int) *Linear {
	// Xavier/Glorot initialization: stddev = sqrt(2 / (fan_in + fan_out))
	limit := math.Sqrt(6.0 / float64(inFeatures+outFeatures))

	// Initialize weights uniformly in [-limit, limit]
	weightsData := make([]float64, outFeatures*inFeatures)
	for i := range weightsData {
		weightsData[i] = (rand.Float64()*2 - 1) * limit
	}

	// Initialize bias to zero
	biasData := make([]float64, outFeatures)

	return &Linear{
		Weights: autograd.NewVariable(outFeatures, inFeatures, weightsData).SetName("Weights"),
		Bias:    autograd.NewVariable(1, outFeatures, biasData).SetName("Bias"),
		UseBias: true,
	}
}

// NewLinearNoBias creates a Linear layer without bias
func NewLinearNoBias(inFeatures, outFeatures int) *Linear {
	limit := math.Sqrt(6.0 / float64(inFeatures+outFeatures))

	weightsData := make([]float64, outFeatures*inFeatures)
	for i := range weightsData {
		weightsData[i] = (rand.Float64()*2 - 1) * limit
	}

	return &Linear{
		Weights: autograd.NewVariable(outFeatures, inFeatures, weightsData).SetName("Weights"),
		Bias:    nil,
		UseBias: false,
	}
}

// Forward computes y = x @ W.T + b
// x: [batchSize, inFeatures]
// returns: [batchSize, outFeatures]
func (l *Linear) Forward(x *autograd.Value) *autograd.Value {
	// y = x @ W.T
	wT := autograd.Transpose(l.Weights)
	y := autograd.MatMul(x, wT)

	if l.UseBias {
		y = autograd.Add(y, l.Bias)
	}

	return y
}

// Parameters returns the learnable parameters
func (l *Linear) Parameters() []*autograd.Value {
	if l.UseBias {
		return []*autograd.Value{l.Weights, l.Bias}
	}
	return []*autograd.Value{l.Weights}
}
