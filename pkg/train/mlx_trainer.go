//go:build darwin && arm64 && mlx

// Package train provides GPU-accelerated training using MLX backend.
// Uses GPU for MatMul (the bottleneck) with robust CPU-GPU coordination.
package train

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/suensky/gogpt/pkg/backend"
)

// MLXTrainer provides GPU-accelerated training for language models.
// Uses GPU MatMul for the forward pass with CPU storage for reliability.
type MLXTrainer struct {
	be backend.Backend

	// Model parameters stored as float64 for CPU safety
	embedWeight []float64 // [vocabSize * embedDim]
	outWeight   []float64 // [embedDim * vocabSize]
	outBias     []float64 // [vocabSize]

	// Gradient accumulators
	embedGrad []float64
	outWGrad  []float64
	outBGrad  []float64

	// Model dimensions
	VocabSize  int
	EmbedDim   int
	ContextLen int

	// Training config
	BaseLR       float64
	NumEpochs    int
	WarmupEpochs int
	PrintEvery   int
	NumBatches   int
}

// NewMLXTrainer creates a new GPU-accelerated trainer.
func NewMLXTrainer(vocabSize, embedDim, contextLen int) (*MLXTrainer, error) {
	be := backend.Default()
	if !be.IsGPU() {
		return nil, fmt.Errorf("MLX GPU backend not available")
	}

	// Initialize weights
	embedWeight := make([]float64, vocabSize*embedDim)
	for i := range embedWeight {
		embedWeight[i] = (rand.Float64() - 0.5) * 0.1
	}

	outWeight := make([]float64, embedDim*vocabSize)
	for i := range outWeight {
		outWeight[i] = (rand.Float64() - 0.5) * 0.1
	}

	outBias := make([]float64, vocabSize)

	return &MLXTrainer{
		be:           be,
		embedWeight:  embedWeight,
		outWeight:    outWeight,
		outBias:      outBias,
		embedGrad:    make([]float64, vocabSize*embedDim),
		outWGrad:     make([]float64, embedDim*vocabSize),
		outBGrad:     make([]float64, vocabSize),
		VocabSize:    vocabSize,
		EmbedDim:     embedDim,
		ContextLen:   contextLen,
		BaseLR:       0.05,
		NumEpochs:    1000,
		WarmupEpochs: 100,
		PrintEvery:   10, // Print every 10 epochs
		NumBatches:   200,
	}, nil
}

// Train trains the model using GPU-accelerated MatMul.
func (t *MLXTrainer) Train(tokens []int) {
	fmt.Println("=== MLX GPU Training (Batched MatMul) ===")
	fmt.Printf("Backend: %s (GPU: %v)\n", t.be.Name(), t.be.IsGPU())
	fmt.Printf("Vocab: %d, Embed: %d, Context: %d\n", t.VocabSize, t.EmbedDim, t.ContextLen)
	fmt.Println()

	startTime := time.Now()

	for epoch := 0; epoch < t.NumEpochs; epoch++ {
		totalLoss := 0.0
		numSamples := 0

		lr := t.computeLR(epoch)
		numBatches := min(t.NumBatches, len(tokens)-t.ContextLen-1)

		for batch := 0; batch < numBatches; batch++ {
			i := rand.Intn(len(tokens) - t.ContextLen - 1)
			inputTokens := tokens[i : i+t.ContextLen]
			targetTokens := tokens[i+1 : i+t.ContextLen+1]

			t.zeroGrad()
			batchLoss := t.forwardBackward(inputTokens, targetTokens)
			t.updateParams(lr)

			totalLoss += batchLoss
			numSamples++
		}

		avgLoss := totalLoss / float64(numSamples)

		if epoch%t.PrintEvery == 0 || epoch == t.NumEpochs-1 {
			elapsed := time.Since(startTime)
			fmt.Printf("Epoch %3d | Loss: %.4f | LR: %.6f | Time: %v\n",
				epoch, avgLoss, lr, elapsed.Round(time.Millisecond))
		}
	}

	fmt.Println()
	fmt.Println("Training complete!")
	fmt.Printf("Total time: %v\n", time.Since(startTime).Round(time.Millisecond))
}

// forwardBackward uses GPU MatMul for forward, CPU for gradients.
func (t *MLXTrainer) forwardBackward(inputTokens, targetTokens []int) float64 {
	be := t.be

	// Build embedding matrix [contextLen, embedDim]
	embedData := make([]float64, t.ContextLen*t.EmbedDim)
	for pos := 0; pos < t.ContextLen; pos++ {
		tokenIdx := inputTokens[pos]
		if tokenIdx < t.VocabSize {
			copy(embedData[pos*t.EmbedDim:(pos+1)*t.EmbedDim],
				t.embedWeight[tokenIdx*t.EmbedDim:(tokenIdx+1)*t.EmbedDim])
		}
	}

	// GPU MatMul: [contextLen, embedDim] @ [embedDim, vocabSize] = [contextLen, vocabSize]
	embedTensor := be.FromSlice(embedData, t.ContextLen, t.EmbedDim)
	outWeightTensor := be.FromSlice(t.outWeight, t.EmbedDim, t.VocabSize)
	logitsTensor := be.MatMul(embedTensor, outWeightTensor)
	be.Synchronize() // Ensure GPU computation completes

	// Extract logits to CPU, add bias, compute softmax and loss
	batchLoss := 0.0
	validPositions := 0

	for pos := 0; pos < t.ContextLen; pos++ {
		inputIdx := inputTokens[pos]
		targetIdx := targetTokens[pos]

		if inputIdx >= t.VocabSize || targetIdx >= t.VocabSize {
			continue
		}
		validPositions++

		// Get logits and add bias
		logits := make([]float64, t.VocabSize)
		for j := 0; j < t.VocabSize; j++ {
			logits[j] = logitsTensor.At(pos, j) + t.outBias[j]
		}

		// Softmax
		probs := softmaxVec(logits)

		// Cross-entropy loss
		loss := -math.Log(probs[targetIdx] + 1e-10)
		batchLoss += loss

		// Compute dL/dLogits = probs - one_hot(target)
		dLogits := make([]float64, t.VocabSize)
		for j := 0; j < t.VocabSize; j++ {
			dLogits[j] = probs[j]
			if j == targetIdx {
				dLogits[j] -= 1.0
			}
		}

		// Accumulate gradients
		embedding := embedData[pos*t.EmbedDim : (pos+1)*t.EmbedDim]
		t.accumulateGradients(inputIdx, embedding, dLogits)
	}

	if validPositions > 0 {
		return batchLoss / float64(validPositions)
	}
	return 0
}

func softmaxVec(logits []float64) []float64 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	probs := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		probs[i] = math.Exp(v - maxVal)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func (t *MLXTrainer) accumulateGradients(inputIdx int, embedding, dLogits []float64) {
	// dL/dOutBias = dLogits
	for j := 0; j < t.VocabSize; j++ {
		t.outBGrad[j] += dLogits[j]
	}

	// dL/dOutWeight[d, j] = embedding[d] * dLogits[j]
	for d := 0; d < t.EmbedDim; d++ {
		for j := 0; j < t.VocabSize; j++ {
			t.outWGrad[d*t.VocabSize+j] += embedding[d] * dLogits[j]
		}
	}

	// dL/dEmbedding[d] = sum_j(outWeight[d,j] * dLogits[j])
	for d := 0; d < t.EmbedDim; d++ {
		grad := 0.0
		for j := 0; j < t.VocabSize; j++ {
			grad += t.outWeight[d*t.VocabSize+j] * dLogits[j]
		}
		t.embedGrad[inputIdx*t.EmbedDim+d] += grad
	}
}

func (t *MLXTrainer) zeroGrad() {
	for i := range t.embedGrad {
		t.embedGrad[i] = 0
	}
	for i := range t.outWGrad {
		t.outWGrad[i] = 0
	}
	for i := range t.outBGrad {
		t.outBGrad[i] = 0
	}
}

func (t *MLXTrainer) updateParams(lr float64) {
	maxGrad := 1.0
	scale := lr / float64(t.ContextLen)

	for j := 0; j < t.VocabSize; j++ {
		g := t.outBGrad[j] * scale
		if g > maxGrad {
			g = maxGrad
		} else if g < -maxGrad {
			g = -maxGrad
		}
		t.outBias[j] -= g
	}

	for i := range t.outWeight {
		g := t.outWGrad[i] * scale
		if g > maxGrad {
			g = maxGrad
		} else if g < -maxGrad {
			g = -maxGrad
		}
		t.outWeight[i] -= g
	}

	for i := range t.embedWeight {
		g := t.embedGrad[i] * scale
		if g > maxGrad {
			g = maxGrad
		} else if g < -maxGrad {
			g = -maxGrad
		}
		t.embedWeight[i] -= g
	}
}

func (t *MLXTrainer) computeLR(epoch int) float64 {
	if epoch < t.WarmupEpochs {
		return t.BaseLR * float64(epoch+1) / float64(t.WarmupEpochs)
	}
	progress := float64(epoch-t.WarmupEpochs) / float64(t.NumEpochs-t.WarmupEpochs)
	return t.BaseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
}

// TestPredictions tests predictions after training.
func (t *MLXTrainer) TestPredictions(tokenToString func(int) string, n int) {
	fmt.Println("=== Sample Token Predictions ===")

	outWeightTensor := t.be.FromSlice(t.outWeight, t.EmbedDim, t.VocabSize)

	for i := 0; i < min(n, t.VocabSize); i++ {
		embedding := t.embedWeight[i*t.EmbedDim : (i+1)*t.EmbedDim]
		embTensor := t.be.FromSlice(embedding, 1, t.EmbedDim)
		logitsTensor := t.be.MatMul(embTensor, outWeightTensor)
		t.be.Synchronize()

		logits := make([]float64, t.VocabSize)
		for j := 0; j < t.VocabSize; j++ {
			logits[j] = logitsTensor.At(0, j) + t.outBias[j]
		}

		probs := softmaxVec(logits)

		topIdx := 0
		topProb := probs[0]
		for j := 1; j < t.VocabSize; j++ {
			if probs[j] > topProb {
				topProb = probs[j]
				topIdx = j
			}
		}

		fromStr := tokenToString(i)
		toStr := tokenToString(topIdx)
		fmt.Printf("  '%s' -> '%s' (prob: %.2f)\n", fromStr, toStr, topProb)
	}
}
