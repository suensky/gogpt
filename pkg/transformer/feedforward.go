package transformer

import (
	"github.com/suensky/gogpt/pkg/autograd"
	"github.com/suensky/gogpt/pkg/nn"
)

// FeedForward implements the position-wise feed-forward network.
// FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
// Typically, the hidden dimension is 4x the embedding dimension.
type FeedForward struct {
	FC1       *nn.Linear
	FC2       *nn.Linear
	HiddenDim int
}

// NewFeedForward creates a new feed-forward layer.
// embedDim: input/output dimension
// hiddenDim: hidden dimension (typically 4 * embedDim)
func NewFeedForward(embedDim, hiddenDim int) *FeedForward {
	return &FeedForward{
		FC1:       nn.NewLinear(embedDim, hiddenDim),
		FC2:       nn.NewLinear(hiddenDim, embedDim),
		HiddenDim: hiddenDim,
	}
}

// Forward applies the feed-forward transformation.
// Input shape: [seqLen, embedDim]
// Output shape: [seqLen, embedDim]
func (ff *FeedForward) Forward(x *autograd.Value) *autograd.Value {
	// FFN(x) = ReLU(Linear1(x)) -> Linear2
	h := ff.FC1.Forward(x)
	h = autograd.ReLU(h)
	return ff.FC2.Forward(h)
}

// ForwardGELU applies the feed-forward transformation with GELU activation.
// Used in newer transformer variants (GPT-2, BERT).
func (ff *FeedForward) ForwardGELU(x *autograd.Value) *autograd.Value {
	h := ff.FC1.Forward(x)
	h = nn.GELU(h)
	return ff.FC2.Forward(h)
}

// Parameters returns the learnable parameters.
func (ff *FeedForward) Parameters() []*autograd.Value {
	params := make([]*autograd.Value, 0)
	params = append(params, ff.FC1.Parameters()...)
	params = append(params, ff.FC2.Parameters()...)
	return params
}
