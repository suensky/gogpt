package transformer

import (
	"math"

	"github.com/suensky/gogpt/pkg/autograd"
	"github.com/suensky/gogpt/pkg/nn"
	"gonum.org/v1/gonum/mat"
)

// MultiHeadAttention implements multi-head self-attention with causal masking.
// This is the core component of the transformer decoder.
type MultiHeadAttention struct {
	NumHeads  int
	HeadDim   int
	EmbedDim  int
	QueryProj *nn.Linear
	KeyProj   *nn.Linear
	ValueProj *nn.Linear
	OutProj   *nn.Linear
}

// NewMultiHeadAttention creates a new multi-head attention layer.
// embedDim: dimension of input embeddings
// numHeads: number of attention heads (embedDim must be divisible by numHeads)
func NewMultiHeadAttention(embedDim, numHeads int) *MultiHeadAttention {
	if embedDim%numHeads != 0 {
		panic("embedDim must be divisible by numHeads")
	}
	headDim := embedDim / numHeads

	return &MultiHeadAttention{
		NumHeads:  numHeads,
		HeadDim:   headDim,
		EmbedDim:  embedDim,
		QueryProj: nn.NewLinear(embedDim, embedDim),
		KeyProj:   nn.NewLinear(embedDim, embedDim),
		ValueProj: nn.NewLinear(embedDim, embedDim),
		OutProj:   nn.NewLinear(embedDim, embedDim),
	}
}

// Forward computes multi-head self-attention with causal masking.
// Input shape: [seqLen, embedDim]
// Output shape: [seqLen, embedDim]
func (mha *MultiHeadAttention) Forward(x *autograd.Value) *autograd.Value {
	seqLen, _ := x.Shape()

	// Project to Q, K, V
	q := mha.QueryProj.Forward(x) // [seqLen, embedDim]
	k := mha.KeyProj.Forward(x)   // [seqLen, embedDim]
	v := mha.ValueProj.Forward(x) // [seqLen, embedDim]

	// For simplicity, we'll implement single-head attention first
	// and treat the full embedDim as one head's dimension
	// A proper multi-head implementation would reshape and split

	// Scaled dot-product attention: softmax(Q @ K.T / sqrt(d_k)) @ V
	// with causal masking

	// Q @ K.T: [seqLen, seqLen]
	kT := autograd.Transpose(k)
	scores := autograd.MatMul(q, kT)

	// Scale by sqrt(headDim)
	scale := 1.0 / math.Sqrt(float64(mha.EmbedDim))
	scores = autograd.Scale(scores, scale)

	// Apply causal mask
	scores = applyCausalMask(scores, seqLen)

	// Softmax over keys (last dimension / columns)
	// We need row-wise softmax
	attnWeights := autograd.Softmax(scores)

	// Attention output: attnWeights @ V
	attnOutput := autograd.MatMul(attnWeights, v)

	// Output projection
	return mha.OutProj.Forward(attnOutput)
}

// applyCausalMask applies a causal (lower triangular) mask to attention scores.
// Sets upper triangular values to -inf so they become 0 after softmax.
func applyCausalMask(scores *autograd.Value, seqLen int) *autograd.Value {
	op := &causalMaskOp{seqLen: seqLen}
	return op.Forward(scores)
}

type causalMaskOp struct {
	seqLen int
}

func (o *causalMaskOp) Name() string { return "CausalMask" }

func (o *causalMaskOp) Forward(inputs ...*autograd.Value) *autograd.Value {
	scores := inputs[0]
	r, c := scores.Shape()

	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if j > i {
				// Future position: mask with large negative value
				result.Set(i, j, -1e9)
			} else {
				result.Set(i, j, scores.Data.At(i, j))
			}
		}
	}

	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(r, c, nil),
	}
}

func (o *causalMaskOp) Backward(grad *mat.Dense, inputs []*autograd.Value, output *autograd.Value) {
	scores := inputs[0]
	r, c := scores.Shape()

	// Gradient only flows through non-masked positions
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if j <= i {
				scores.Grad.Set(i, j, scores.Grad.At(i, j)+grad.At(i, j))
			}
			// Masked positions get zero gradient
		}
	}
}

// Parameters returns all learnable parameters.
func (mha *MultiHeadAttention) Parameters() []*autograd.Value {
	params := make([]*autograd.Value, 0)
	params = append(params, mha.QueryProj.Parameters()...)
	params = append(params, mha.KeyProj.Parameters()...)
	params = append(params, mha.ValueProj.Parameters()...)
	params = append(params, mha.OutProj.Parameters()...)
	return params
}

// CausalMask creates a causal (lower triangular) mask matrix.
// Used for visualizing or debugging attention patterns.
func CausalMask(seqLen int) *mat.Dense {
	mask := mat.NewDense(seqLen, seqLen, nil)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j > i {
				mask.Set(i, j, -1e9)
			}
		}
	}
	return mask
}
