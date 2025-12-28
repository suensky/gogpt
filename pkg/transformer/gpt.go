package transformer

import (
	"math"
	"math/rand"

	"github.com/suensky/gogpt/pkg/autograd"
	"github.com/suensky/gogpt/pkg/nn"
	"gonum.org/v1/gonum/mat"
)

// GPTConfig holds the configuration for a GPT model.
type GPTConfig struct {
	VocabSize     int // Size of vocabulary
	EmbedDim      int // Embedding dimension
	NumHeads      int // Number of attention heads
	NumLayers     int // Number of decoder blocks
	ContextWindow int // Maximum sequence length
	FFHiddenDim   int // Feed-forward hidden dimension (0 = 4 * EmbedDim)
}

// DefaultGPTConfig returns a default configuration with sensible defaults.
func DefaultGPTConfig(vocabSize int) GPTConfig {
	return GPTConfig{
		VocabSize:     vocabSize,
		EmbedDim:      64,
		NumHeads:      4,
		NumLayers:     4,
		ContextWindow: 32,
		FFHiddenDim:   0, // Will be set to 4 * EmbedDim
	}
}

// GPT implements a GPT-style transformer decoder.
type GPT struct {
	Config     GPTConfig
	TokenEmbed *nn.Embedding // Token embeddings
	PosEmbed   *nn.Embedding // Position embeddings (learnable)
	Blocks     []*DecoderBlock
	LNFinal    *LayerNorm
	Head       *nn.Linear // Output projection to vocabulary
}

// NewGPT creates a new GPT model.
func NewGPT(config GPTConfig) *GPT {
	// Set default FFN hidden dim if not specified
	ffHiddenDim := config.FFHiddenDim
	if ffHiddenDim == 0 {
		ffHiddenDim = 4 * config.EmbedDim
	}

	// Create decoder blocks
	blocks := make([]*DecoderBlock, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		blocks[i] = NewDecoderBlock(config.EmbedDim, config.NumHeads, ffHiddenDim)
	}

	return &GPT{
		Config:     config,
		TokenEmbed: nn.NewEmbedding(config.VocabSize, config.EmbedDim),
		PosEmbed:   nn.NewEmbedding(config.ContextWindow, config.EmbedDim),
		Blocks:     blocks,
		LNFinal:    NewLayerNorm(config.EmbedDim),
		Head:       nn.NewLinear(config.EmbedDim, config.VocabSize),
	}
}

// Forward computes the forward pass of the GPT model.
// tokens: list of token indices [seqLen]
// Returns: logits [seqLen, vocabSize]
func (g *GPT) Forward(tokens []int) *autograd.Value {
	seqLen := len(tokens)
	if seqLen > g.Config.ContextWindow {
		panic("sequence length exceeds context window")
	}

	// Get token embeddings
	tokEmb := g.TokenEmbed.Forward(tokens) // [seqLen, embedDim]

	// Get position embeddings
	positions := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		positions[i] = i
	}
	posEmb := g.PosEmbed.Forward(positions) // [seqLen, embedDim]

	// Combine token and position embeddings
	x := autograd.Add(tokEmb, posEmb)

	// Pass through decoder blocks
	for _, block := range g.Blocks {
		x = block.Forward(x)
	}

	// Final layer norm
	x = g.LNFinal.Forward(x)

	// Project to vocabulary
	logits := g.Head.Forward(x) // [seqLen, vocabSize]

	return logits
}

// ForwardWithEmbedding is like Forward but takes pre-computed embeddings.
// Useful for gradient-based training where we need the computation graph.
func (g *GPT) ForwardWithEmbedding(tokenEmb, posEmb *autograd.Value) *autograd.Value {
	// Combine token and position embeddings
	x := autograd.Add(tokenEmb, posEmb)

	// Pass through decoder blocks
	for _, block := range g.Blocks {
		x = block.Forward(x)
	}

	// Final layer norm
	x = g.LNFinal.Forward(x)

	// Project to vocabulary
	logits := g.Head.Forward(x)

	return logits
}

// Parameters returns all learnable parameters of the model.
func (g *GPT) Parameters() []*autograd.Value {
	params := make([]*autograd.Value, 0)

	// Embeddings
	params = append(params, g.TokenEmbed.Parameters()...)
	params = append(params, g.PosEmbed.Parameters()...)

	// Decoder blocks
	for _, block := range g.Blocks {
		params = append(params, block.Parameters()...)
	}

	// Final layer norm and head
	params = append(params, g.LNFinal.Parameters()...)
	params = append(params, g.Head.Parameters()...)

	return params
}

// NumParameters returns the total number of parameters.
func (g *GPT) NumParameters() int {
	total := 0
	for _, p := range g.Parameters() {
		r, c := p.Shape()
		total += r * c
	}
	return total
}

// Generate generates tokens autoregressively.
// startTokens: initial tokens to seed the generation
// maxLen: maximum number of tokens to generate
// temperature: sampling temperature (1.0 = normal, <1 = more deterministic, >1 = more random)
func (g *GPT) Generate(startTokens []int, maxLen int, temperature float64) []int {
	tokens := make([]int, len(startTokens))
	copy(tokens, startTokens)

	for len(tokens) < maxLen {
		// Truncate to context window if necessary
		contextTokens := tokens
		if len(tokens) > g.Config.ContextWindow {
			contextTokens = tokens[len(tokens)-g.Config.ContextWindow:]
		}

		// Forward pass
		logits := g.Forward(contextTokens)

		// Get logits for last position
		seqLen, vocabSize := logits.Shape()
		lastLogits := make([]float64, vocabSize)
		for j := 0; j < vocabSize; j++ {
			lastLogits[j] = logits.Data.At(seqLen-1, j)
		}

		// Apply temperature
		if temperature != 1.0 {
			for j := range lastLogits {
				lastLogits[j] /= temperature
			}
		}

		// Convert to probabilities (softmax)
		probs := softmax(lastLogits)

		// Sample from distribution
		nextToken := sampleFromProbs(probs)
		tokens = append(tokens, nextToken)
	}

	return tokens
}

// softmax computes softmax of a slice
func softmax(logits []float64) []float64 {
	// Find max for numerical stability
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp(x - max) and sum
	probs := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		probs[i] = math.Exp(v - maxVal)
		sum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

// sampleFromProbs samples an index from a probability distribution
func sampleFromProbs(probs []float64) int {
	r := rand.Float64()
	cumsum := 0.0
	for i, p := range probs {
		cumsum += p
		if r < cumsum {
			return i
		}
	}
	return len(probs) - 1
}

// argmax returns the index of the maximum value
func argmax(values []float64) int {
	maxIdx := 0
	maxVal := values[0]
	for i, v := range values[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx
}

// GetTokenEmbedding retrieves the embedding for a single token.
func (g *GPT) GetTokenEmbedding(token int) []float64 {
	embedDim := g.Config.EmbedDim
	result := make([]float64, embedDim)
	for j := 0; j < embedDim; j++ {
		result[j] = g.TokenEmbed.Weight.Data.At(token, j)
	}
	return result
}

// CreateEmbedding creates a Value from token and position embeddings for training.
// This properly tracks gradients through the embedding lookup.
func (g *GPT) CreateEmbedding(tokens []int) (*autograd.Value, *autograd.Value) {
	seqLen := len(tokens)
	embedDim := g.Config.EmbedDim

	// Create token embedding data
	tokData := make([]float64, seqLen*embedDim)
	for i, idx := range tokens {
		for j := 0; j < embedDim; j++ {
			tokData[i*embedDim+j] = g.TokenEmbed.Weight.Data.At(idx, j)
		}
	}
	tokEmb := &autograd.Value{
		Data: mat.NewDense(seqLen, embedDim, tokData),
		Grad: mat.NewDense(seqLen, embedDim, nil),
	}

	// Create position embedding data
	posData := make([]float64, seqLen*embedDim)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < embedDim; j++ {
			posData[i*embedDim+j] = g.PosEmbed.Weight.Data.At(i, j)
		}
	}
	posEmb := &autograd.Value{
		Data: mat.NewDense(seqLen, embedDim, posData),
		Grad: mat.NewDense(seqLen, embedDim, nil),
	}

	return tokEmb, posEmb
}
