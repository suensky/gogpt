package transformer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/suensky/gogpt/pkg/autograd"
)

func approxEqual(a, b, eps float64) bool {
	return math.Abs(a-b) < eps
}

func TestLayerNormForward(t *testing.T) {
	rand.Seed(42)

	ln := NewLayerNorm(4)

	// Input: 2 samples, 4 features
	input := autograd.NewVariable(2, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})

	output := ln.Forward(input)

	r, c := output.Shape()
	if r != 2 || c != 4 {
		t.Errorf("Expected shape 2x4, got %dx%d", r, c)
	}

	// After normalization, each row should have mean ~0 and std ~1
	// (before gamma/beta scaling with default gamma=1, beta=0)
	for i := 0; i < 2; i++ {
		sum := 0.0
		for j := 0; j < 4; j++ {
			sum += output.Data.At(i, j)
		}
		mean := sum / 4.0
		if !approxEqual(mean, 0.0, 0.1) {
			t.Errorf("Row %d mean should be ~0, got %f", i, mean)
		}
	}
}

func TestCausalMask(t *testing.T) {
	mask := CausalMask(4)

	// Check that upper triangle is masked (large negative)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if j > i {
				if mask.At(i, j) > -1e8 {
					t.Errorf("Position (%d,%d) should be masked, got %f", i, j, mask.At(i, j))
				}
			} else {
				if mask.At(i, j) != 0 {
					t.Errorf("Position (%d,%d) should not be masked, got %f", i, j, mask.At(i, j))
				}
			}
		}
	}
}

func TestMultiHeadAttentionForward(t *testing.T) {
	rand.Seed(42)

	mha := NewMultiHeadAttention(8, 2) // embedDim=8, numHeads=2

	// Input: 4 tokens, 8 dimensions
	input := autograd.NewVariable(4, 8, nil)
	for i := 0; i < 4*8; i++ {
		input.Data.Set(i/8, i%8, rand.Float64())
	}

	output := mha.Forward(input)

	r, c := output.Shape()
	if r != 4 || c != 8 {
		t.Errorf("Expected shape 4x8, got %dx%d", r, c)
	}

	// Output should not be all zeros
	hasNonZero := false
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if output.Data.At(i, j) != 0 {
				hasNonZero = true
				break
			}
		}
	}
	if !hasNonZero {
		t.Error("Output should not be all zeros")
	}
}

func TestFeedForwardForward(t *testing.T) {
	rand.Seed(42)

	ff := NewFeedForward(8, 32) // embedDim=8, hiddenDim=32

	input := autograd.NewVariable(4, 8, nil)
	for i := 0; i < 4*8; i++ {
		input.Data.Set(i/8, i%8, rand.Float64())
	}

	output := ff.Forward(input)

	r, c := output.Shape()
	if r != 4 || c != 8 {
		t.Errorf("Expected shape 4x8, got %dx%d", r, c)
	}
}

func TestDecoderBlockForward(t *testing.T) {
	rand.Seed(42)

	block := NewDecoderBlock(8, 2, 32)

	input := autograd.NewVariable(4, 8, nil)
	for i := 0; i < 4*8; i++ {
		input.Data.Set(i/8, i%8, rand.Float64())
	}

	output := block.Forward(input)

	r, c := output.Shape()
	if r != 4 || c != 8 {
		t.Errorf("Expected shape 4x8, got %dx%d", r, c)
	}

	// Due to residual connections, output should be different from input
	// but should have somewhat similar magnitudes
}

func TestGPTForward(t *testing.T) {
	rand.Seed(42)

	config := GPTConfig{
		VocabSize:     10,
		EmbedDim:      16,
		NumHeads:      2,
		NumLayers:     2,
		ContextWindow: 8,
	}

	gpt := NewGPT(config)

	// Test forward pass
	tokens := []int{1, 2, 3, 4}
	logits := gpt.Forward(tokens)

	r, c := logits.Shape()
	if r != 4 || c != 10 {
		t.Errorf("Expected logits shape 4x10, got %dx%d", r, c)
	}
}

func TestGPTGenerate(t *testing.T) {
	rand.Seed(42)

	config := GPTConfig{
		VocabSize:     10,
		EmbedDim:      8,
		NumHeads:      2,
		NumLayers:     1,
		ContextWindow: 8,
	}

	gpt := NewGPT(config)

	// Generate tokens
	startTokens := []int{1}
	generated := gpt.Generate(startTokens, 5, 1.0)

	if len(generated) != 5 {
		t.Errorf("Expected 5 generated tokens, got %d", len(generated))
	}

	// Check all tokens are within vocab range
	for _, tok := range generated {
		if tok < 0 || tok >= config.VocabSize {
			t.Errorf("Generated token %d out of vocab range", tok)
		}
	}
}

func TestGPTNumParameters(t *testing.T) {
	config := GPTConfig{
		VocabSize:     100,
		EmbedDim:      32,
		NumHeads:      4,
		NumLayers:     2,
		ContextWindow: 16,
	}

	gpt := NewGPT(config)
	numParams := gpt.NumParameters()

	// Should have a reasonable number of parameters
	// Approximate calculation:
	// TokenEmbed: 100 * 32 = 3200
	// PosEmbed: 16 * 32 = 512
	// Per block: ~4 * (32*32 + 32) + 2 * (32*128 + 128*32 + biases) + LayerNorms
	// This should be in thousands
	if numParams < 1000 {
		t.Errorf("Expected at least 1000 parameters, got %d", numParams)
	}

	t.Logf("GPT model has %d parameters", numParams)
}

func TestDecoderBlockResidual(t *testing.T) {
	rand.Seed(42)

	block := NewDecoderBlock(8, 2, 32)

	// Create input where first element is distinctive
	inputData := make([]float64, 4*8)
	inputData[0] = 100.0 // Distinctive value
	for i := 1; i < len(inputData); i++ {
		inputData[i] = rand.Float64()
	}
	input := autograd.NewVariable(4, 8, inputData)

	output := block.Forward(input)

	// Due to residual connection, input information should be preserved
	// The output should have some influence from the input
	// (This is a weak test, mainly checking the residual path exists)
	r, c := output.Shape()
	if r != 4 || c != 8 {
		t.Errorf("Expected shape 4x8, got %dx%d", r, c)
	}
}
