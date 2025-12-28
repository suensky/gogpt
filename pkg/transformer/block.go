package transformer

import (
	"github.com/suensky/gogpt/pkg/autograd"
)

// DecoderBlock implements a single transformer decoder block.
// It consists of:
// 1. Multi-head self-attention with residual connection
// 2. Feed-forward network with residual connection
// Both use layer normalization (Pre-LN style like GPT-2).
type DecoderBlock struct {
	Attention *MultiHeadAttention
	FeedFwd   *FeedForward
	LN1       *LayerNorm
	LN2       *LayerNorm
}

// NewDecoderBlock creates a new decoder block.
// embedDim: embedding dimension
// numHeads: number of attention heads
// ffHiddenDim: hidden dimension for feed-forward network (typically 4 * embedDim)
func NewDecoderBlock(embedDim, numHeads, ffHiddenDim int) *DecoderBlock {
	return &DecoderBlock{
		Attention: NewMultiHeadAttention(embedDim, numHeads),
		FeedFwd:   NewFeedForward(embedDim, ffHiddenDim),
		LN1:       NewLayerNorm(embedDim),
		LN2:       NewLayerNorm(embedDim),
	}
}

// Forward applies the decoder block transformation.
// Uses Pre-LN (layer norm before attention/FFN) architecture:
//
//	x = x + Attention(LN1(x))
//	x = x + FFN(LN2(x))
func (b *DecoderBlock) Forward(x *autograd.Value) *autograd.Value {
	// Self-attention with residual connection
	normed := b.LN1.Forward(x)
	attnOut := b.Attention.Forward(normed)
	x = autograd.Add(x, attnOut)

	// Feed-forward with residual connection
	normed = b.LN2.Forward(x)
	ffOut := b.FeedFwd.Forward(normed)
	x = autograd.Add(x, ffOut)

	return x
}

// Parameters returns all learnable parameters.
func (b *DecoderBlock) Parameters() []*autograd.Value {
	params := make([]*autograd.Value, 0)
	params = append(params, b.Attention.Parameters()...)
	params = append(params, b.FeedFwd.Parameters()...)
	params = append(params, b.LN1.Parameters()...)
	params = append(params, b.LN2.Parameters()...)
	return params
}
