// GoGPT Training Script with BPE Tokenizer
// This trains a small GPT-style transformer on Jules Verne's "The Mysterious Island"
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/suensky/gogpt/internal/tokenizer"
	"github.com/suensky/gogpt/pkg/autograd"
	"github.com/suensky/gogpt/pkg/transformer"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Set random seed for reproducibility
	rand.Seed(42)

	fmt.Println("=== GoGPT: Training a Small Transformer ===")
	fmt.Println("=== Using BPE Tokenizer with Jules Verne ===")
	fmt.Println()

	// Load training data
	dataPath := "data/jules_verne.txt"
	textBytes, err := os.ReadFile(dataPath)
	if err != nil {
		fmt.Printf("Could not load %s: %v\n", dataPath, err)
		fmt.Println("Falling back to simple demo text...")
		textBytes = []byte("hello world hello go hello transformer hello world hello go hello transformer")
	}
	text := string(textBytes)

	// Limit text size for faster training
	maxChars := 20000
	if len(text) > maxChars {
		text = text[:maxChars]
	}

	fmt.Printf("Training text: %d characters\n", len(text))
	fmt.Printf("First 200 chars: %s...\n\n", text[:min(200, len(text))])

	// Create BPE tokenizer
	fmt.Println("Building BPE tokenizer...")
	tok := tokenizer.NewCharTokenizer(text)
	charVocabSize := tok.VocabSize

	// Train BPE merges
	numMerges := 100 // Learn 100 merge rules
	rules := tok.TrainBPE(text, numMerges)
	fmt.Printf("Character vocabulary: %d tokens\n", charVocabSize)
	fmt.Printf("After BPE training: %d tokens (added %d merges)\n", tok.VocabSize, tok.VocabSize-charVocabSize)

	// Show some learned merge rules
	rulesLines := splitLines(rules)
	if len(rulesLines) > 5 {
		fmt.Println("Sample merge rules:")
		for i := 0; i < 5; i++ {
			fmt.Printf("  %s\n", rulesLines[i])
		}
		fmt.Println("  ...")
	}
	fmt.Println()

	// Encode the text
	tokens := tok.Encode(text)
	fmt.Printf("Text encoded to %d tokens (%.1fx compression)\n", len(tokens), float64(len(text))/float64(len(tokens)))
	fmt.Printf("Sample tokens: %v\n\n", tokens[:min(20, len(tokens))])

	// Hyperparameters
	config := transformer.GPTConfig{
		VocabSize:     tok.VocabSize,
		EmbedDim:      32,
		NumHeads:      4,
		NumLayers:     2,
		ContextWindow: 16,
		FFHiddenDim:   128,
	}

	fmt.Println("Model Configuration:")
	fmt.Printf("  Vocab Size: %d\n", config.VocabSize)
	fmt.Printf("  Embed Dim: %d\n", config.EmbedDim)
	fmt.Printf("  Num Heads: %d\n", config.NumHeads)
	fmt.Printf("  Num Layers: %d\n", config.NumLayers)
	fmt.Printf("  Context Window: %d\n", config.ContextWindow)
	fmt.Printf("  FF Hidden Dim: %d\n", config.FFHiddenDim)
	fmt.Println()

	// Train models
	trainSimpleModel(tok, tokens, config)
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i, ch := range s {
		if ch == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

// trainSimpleModel trains a simple embedding + linear model
func trainSimpleModel(tok *tokenizer.BPETokenizer, tokens []int, config transformer.GPTConfig) {
	fmt.Println("=== Training Simple Embedding + Linear Model ===")
	fmt.Println()

	vocabSize := tok.VocabSize
	embedDim := config.EmbedDim
	contextLen := config.ContextWindow

	// Simple model: embeddings + linear projection
	embedWeight := autograd.NewVariable(vocabSize, embedDim, nil)
	for i := 0; i < vocabSize; i++ {
		for j := 0; j < embedDim; j++ {
			embedWeight.Data.Set(i, j, (rand.Float64()-0.5)*0.1)
		}
	}

	outWeight := autograd.NewVariable(embedDim, vocabSize, nil)
	for i := 0; i < embedDim; i++ {
		for j := 0; j < vocabSize; j++ {
			outWeight.Data.Set(i, j, (rand.Float64()-0.5)*0.1)
		}
	}

	outBias := autograd.NewVariable(1, vocabSize, nil)

	params := []*autograd.Value{embedWeight, outWeight, outBias}
	lr := 0.3
	numEpochs := 200
	printEvery := 20

	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0
		numSamples := 0

		// Random sampling instead of iterating all
		numBatches := min(50, len(tokens)-contextLen-1)

		for batch := 0; batch < numBatches; batch++ {
			i := rand.Intn(len(tokens) - contextLen - 1)
			inputTokens := tokens[i : i+contextLen]
			targetTokens := tokens[i+1 : i+contextLen+1]

			// Zero gradients
			for _, p := range params {
				p.ZeroGrad()
			}

			batchLoss := 0.0

			for pos := 0; pos < contextLen; pos++ {
				inputIdx := inputTokens[pos]
				targetIdx := targetTokens[pos]

				// Skip if out of vocab range
				if inputIdx >= vocabSize || targetIdx >= vocabSize {
					continue
				}

				embedding := getRow(embedWeight, inputIdx)
				logits := matVecMul(outWeight, embedding, outBias)
				probs := softmaxVec(logits)
				loss := -math.Log(probs[targetIdx] + 1e-10)
				batchLoss += loss

				// Gradients
				dLogits := make([]float64, vocabSize)
				for j := 0; j < vocabSize; j++ {
					dLogits[j] = probs[j]
					if j == targetIdx {
						dLogits[j] -= 1.0
					}
				}

				for j := 0; j < vocabSize; j++ {
					outBias.Grad.Set(0, j, outBias.Grad.At(0, j)+dLogits[j])
				}

				for d := 0; d < embedDim; d++ {
					for j := 0; j < vocabSize; j++ {
						outWeight.Grad.Set(d, j, outWeight.Grad.At(d, j)+embedding[d]*dLogits[j])
					}
				}

				dEmbed := make([]float64, embedDim)
				for d := 0; d < embedDim; d++ {
					for j := 0; j < vocabSize; j++ {
						dEmbed[d] += outWeight.Data.At(d, j) * dLogits[j]
					}
				}

				for d := 0; d < embedDim; d++ {
					embedWeight.Grad.Set(inputIdx, d, embedWeight.Grad.At(inputIdx, d)+dEmbed[d])
				}
			}

			// Update parameters
			for _, p := range params {
				r, c := p.Shape()
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						newVal := p.Data.At(i, j) - lr*p.Grad.At(i, j)/float64(contextLen)
						p.Data.Set(i, j, newVal)
					}
				}
			}

			totalLoss += batchLoss / float64(contextLen)
			numSamples++
		}

		avgLoss := totalLoss / float64(numSamples)

		if epoch%printEvery == 0 || epoch == numEpochs-1 {
			elapsed := time.Since(startTime)
			fmt.Printf("Epoch %3d | Loss: %.4f | Time: %v\n", epoch, avgLoss, elapsed.Round(time.Millisecond))
		}
	}

	fmt.Println()
	fmt.Println("Training complete!")
	fmt.Printf("Total time: %v\n", time.Since(startTime).Round(time.Millisecond))
	fmt.Println()

	// Test predictions
	fmt.Println("=== Sample Token Predictions ===")
	testPredictions(embedWeight, outWeight, outBias, tok, 10)

	// Train full GPT
	fmt.Println()
	trainFullGPT(tok, tokens, config)
}

func testPredictions(embedWeight, outWeight, outBias *autograd.Value, tok *tokenizer.BPETokenizer, n int) {
	_, cols := outWeight.Shape()
	vocabSize := cols

	// Test most common tokens
	vocab := tok.GetVocab()
	count := 0
	for i := 0; i < min(20, len(vocab)) && count < n; i++ {
		token := vocab[i]

		id, ok := tok.TokenID(token)
		if !ok || id >= vocabSize {
			continue
		}

		embedding := getRow(embedWeight, id)
		logits := matVecMul(outWeight, embedding, outBias)
		probs := softmaxVec(logits)
		topIdx := argmax(probs)
		topProb := probs[topIdx]
		topToken, _ := tok.TokenString(topIdx)

		// Format for display
		displayFrom := formatToken(token)
		displayTo := formatToken(topToken)
		fmt.Printf("  '%s' -> '%s' (prob: %.2f)\n", displayFrom, displayTo, topProb)
		count++
	}
}

func formatToken(s string) string {
	if s == " " {
		return "<space>"
	}
	if s == "\n" {
		return "\\n"
	}
	if s == "\t" {
		return "\\t"
	}
	if len(s) > 8 {
		return s[:8] + "..."
	}
	return s
}

func trainFullGPT(tok *tokenizer.BPETokenizer, tokens []int, config transformer.GPTConfig) {
	fmt.Println("=== Training Full GPT Model ===")
	fmt.Println()

	model := transformer.NewGPT(config)
	fmt.Printf("Model has %d parameters\n", model.NumParameters())

	params := model.Parameters()
	lr := 0.0005
	numEpochs := 100
	printEvery := 10
	contextLen := config.ContextWindow
	numBatches := min(30, len(tokens)-contextLen-1)

	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0
		numSamples := 0

		for batch := 0; batch < numBatches; batch++ {
			i := rand.Intn(len(tokens) - contextLen - 1)
			inputTokens := tokens[i : i+contextLen]
			targetTokens := tokens[i+1 : i+contextLen+1]

			for _, p := range params {
				p.ZeroGrad()
			}

			logits := model.Forward(inputTokens)
			seqLen, vocabSize := logits.Shape()

			loss := 0.0
			dLogits := mat.NewDense(seqLen, vocabSize, nil)

			for pos := 0; pos < seqLen; pos++ {
				maxVal := logits.Data.At(pos, 0)
				for j := 1; j < vocabSize; j++ {
					if logits.Data.At(pos, j) > maxVal {
						maxVal = logits.Data.At(pos, j)
					}
				}

				probs := make([]float64, vocabSize)
				sumExp := 0.0
				for j := 0; j < vocabSize; j++ {
					probs[j] = math.Exp(logits.Data.At(pos, j) - maxVal)
					sumExp += probs[j]
				}
				for j := 0; j < vocabSize; j++ {
					probs[j] /= sumExp
				}

				targetIdx := targetTokens[pos]
				if targetIdx < vocabSize {
					loss -= math.Log(probs[targetIdx] + 1e-10)
				}

				for j := 0; j < vocabSize; j++ {
					grad := probs[j]
					if j == targetIdx {
						grad -= 1.0
					}
					dLogits.Set(pos, j, grad)
				}
			}

			loss /= float64(seqLen)
			updateModelEmbeddings(model, inputTokens, config, dLogits, lr)

			totalLoss += loss
			numSamples++
		}

		avgLoss := totalLoss / float64(numSamples)

		if epoch%printEvery == 0 || epoch == numEpochs-1 {
			elapsed := time.Since(startTime)
			fmt.Printf("Epoch %3d | Loss: %.4f | Time: %v\n", epoch, avgLoss, elapsed.Round(time.Millisecond))
		}
	}

	fmt.Println()
	fmt.Println("GPT training complete!")
	fmt.Println()

	// Generation
	fmt.Println("=== Text Generation ===")
	prompts := []string{"The ", "Captain ", "mysterious ", "island "}
	for _, prompt := range prompts {
		startTokens := tok.Encode(prompt)
		if len(startTokens) == 0 {
			continue
		}
		generated := model.Generate(startTokens, 30, 0.8)
		text := tok.Decode(generated)
		fmt.Printf("Prompt: '%s'\n", prompt)
		fmt.Printf("Generated: %s\n\n", text)
	}
}

func updateModelEmbeddings(model *transformer.GPT, inputTokens []int, config transformer.GPTConfig, dLogits *mat.Dense, lr float64) {
	seqLen, vocabSize := dLogits.Dims()
	embedDim := config.EmbedDim

	for pos := 0; pos < seqLen; pos++ {
		tokIdx := inputTokens[pos]
		if tokIdx >= config.VocabSize {
			continue
		}

		for d := 0; d < embedDim; d++ {
			embVal := model.TokenEmbed.Weight.Data.At(tokIdx, d)
			for j := 0; j < vocabSize; j++ {
				dL := dLogits.At(pos, j)
				curr := model.Head.Weights.Data.At(j, d)
				model.Head.Weights.Data.Set(j, d, curr-lr*embVal*dL/float64(seqLen))
			}
		}

		for j := 0; j < vocabSize; j++ {
			dL := dLogits.At(pos, j)
			curr := model.Head.Bias.Data.At(0, j)
			model.Head.Bias.Data.Set(0, j, curr-lr*dL/float64(seqLen))
		}

		for d := 0; d < embedDim; d++ {
			grad := 0.0
			for j := 0; j < vocabSize; j++ {
				grad += model.Head.Weights.Data.At(j, d) * dLogits.At(pos, j)
			}
			curr := model.TokenEmbed.Weight.Data.At(tokIdx, d)
			model.TokenEmbed.Weight.Data.Set(tokIdx, d, curr-lr*grad/float64(seqLen))
		}
	}
}

// Helper functions

func getRow(m *autograd.Value, row int) []float64 {
	_, cols := m.Shape()
	result := make([]float64, cols)
	for j := 0; j < cols; j++ {
		result[j] = m.Data.At(row, j)
	}
	return result
}

func matVecMul(w *autograd.Value, v []float64, bias *autograd.Value) []float64 {
	rows, cols := w.Shape()
	result := make([]float64, cols)

	for j := 0; j < cols; j++ {
		sum := 0.0
		for i := 0; i < rows && i < len(v); i++ {
			sum += w.Data.At(i, j) * v[i]
		}
		result[j] = sum + bias.Data.At(0, j)
	}
	return result
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
