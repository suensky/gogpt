package nn

import (
	"math"
	"math/rand"

	"github.com/suensky/gogpt/pkg/autograd"
	"gonum.org/v1/gonum/mat"
)

// Embedding implements a lookup table for embeddings.
// Given indices, it returns the corresponding embedding vectors.
type Embedding struct {
	Weight    *autograd.Value // [vocabSize, embedDim]
	VocabSize int
	EmbedDim  int
}

// NewEmbedding creates a new Embedding layer with normal initialization.
// vocabSize: size of the vocabulary
// embedDim: dimension of embedding vectors
func NewEmbedding(vocabSize, embedDim int) *Embedding {
	// Initialize with small random values (normal distribution)
	stddev := 1.0 / math.Sqrt(float64(embedDim))
	data := make([]float64, vocabSize*embedDim)
	for i := range data {
		data[i] = rand.NormFloat64() * stddev
	}

	return &Embedding{
		Weight:    autograd.NewVariable(vocabSize, embedDim, data).SetName("EmbeddingWeight"),
		VocabSize: vocabSize,
		EmbedDim:  embedDim,
	}
}

// Forward looks up embeddings for the given indices.
// indices: list of token indices
// returns: [len(indices), embedDim] tensor
func (e *Embedding) Forward(indices []int) *autograd.Value {
	seqLen := len(indices)

	// Create output matrix by selecting rows from Weight
	outputData := make([]float64, seqLen*e.EmbedDim)
	for i, idx := range indices {
		if idx < 0 || idx >= e.VocabSize {
			panic("embedding index out of range")
		}
		for j := 0; j < e.EmbedDim; j++ {
			outputData[i*e.EmbedDim+j] = e.Weight.Data.At(idx, j)
		}
	}

	result := mat.NewDense(seqLen, e.EmbedDim, outputData)

	// Simple forward without gradient tracking (use ForwardWithGrad for training)
	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(seqLen, e.EmbedDim, nil),
	}
}

// ForwardWithGrad looks up embeddings and builds computation graph for gradients.
// This creates proper backprop through the embedding lookup.
func (e *Embedding) ForwardWithGrad(indices []int) *autograd.Value {
	op := &embeddingOp{
		embedding: e,
		indices:   indices,
	}
	return op.Forward(e.Weight)
}

// embeddingOp implements the embedding lookup as an autograd Operation
type embeddingOp struct {
	embedding *Embedding
	indices   []int
}

func (o *embeddingOp) Name() string { return "Embedding" }

func (o *embeddingOp) Forward(inputs ...*autograd.Value) *autograd.Value {
	weight := inputs[0]
	seqLen := len(o.indices)
	_, embedDim := weight.Shape()

	outputData := make([]float64, seqLen*embedDim)
	for i, idx := range o.indices {
		for j := 0; j < embedDim; j++ {
			outputData[i*embedDim+j] = weight.Data.At(idx, j)
		}
	}

	result := mat.NewDense(seqLen, embedDim, outputData)
	return &autograd.Value{
		Data: result,
		Grad: mat.NewDense(seqLen, embedDim, nil),
	}
}

func (o *embeddingOp) Backward(grad *mat.Dense, inputs []*autograd.Value, output *autograd.Value) {
	weight := inputs[0]
	_, embedDim := weight.Shape()

	// Accumulate gradients for each used index
	for i, idx := range o.indices {
		for j := 0; j < embedDim; j++ {
			current := weight.Grad.At(idx, j)
			weight.Grad.Set(idx, j, current+grad.At(i, j))
		}
	}
}

// Parameters returns the embedding weight
func (e *Embedding) Parameters() []*autograd.Value {
	return []*autograd.Value{e.Weight}
}

// PositionalEmbedding creates sinusoidal positional embeddings (non-learnable)
// This is the original transformer positional encoding.
func SinusoidalPositionalEncoding(maxLen, embedDim int) *autograd.Value {
	data := make([]float64, maxLen*embedDim)

	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < embedDim; i++ {
			angle := float64(pos) / math.Pow(10000, 2*float64(i/2)/float64(embedDim))
			if i%2 == 0 {
				data[pos*embedDim+i] = math.Sin(angle)
			} else {
				data[pos*embedDim+i] = math.Cos(angle)
			}
		}
	}

	return autograd.NewVariable(maxLen, embedDim, data).SetName("PositionalEncoding")
}
