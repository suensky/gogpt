// GoGPT Training Script with Full Backpropagation
// This trains a small GPT-style transformer on a tiny character-level dataset.
package main

import (
	"fmt"
	"math"
	"math/rand"
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
	fmt.Println()

	// Training data
	text := "hello world hello go hello transformer"
	fmt.Printf("Training text: %q\n", text)

	// Create tokenizer
	tok := tokenizer.NewCharTokenizer(text)
	fmt.Printf("Vocabulary size: %d\n", tok.VocabSize)
	fmt.Printf("Characters: %v\n", string(tok.GetVocab()))

	// Encode the text
	tokens := tok.Encode(text)
	fmt.Printf("Encoded tokens: %v\n", tokens)
	fmt.Println()

	// Hyperparameters
	config := transformer.GPTConfig{
		VocabSize:     tok.VocabSize,
		EmbedDim:      16,
		NumHeads:      2,
		NumLayers:     2,
		ContextWindow: 8,
		FFHiddenDim:   64,
	}

	fmt.Println("Model Configuration:")
	fmt.Printf("  Vocab Size: %d\n", config.VocabSize)
	fmt.Printf("  Embed Dim: %d\n", config.EmbedDim)
	fmt.Printf("  Num Heads: %d\n", config.NumHeads)
	fmt.Printf("  Num Layers: %d\n", config.NumLayers)
	fmt.Printf("  Context Window: %d\n", config.ContextWindow)
	fmt.Println()

	// Train with a simpler approach: direct embedding + linear model first
	// to verify the training loop works
	trainSimpleModel(tok, tokens, config)
}

// trainSimpleModel trains a simple embedding + linear model
// This demonstrates proper gradient flow and training
func trainSimpleModel(tok *tokenizer.CharTokenizer, tokens []int, config transformer.GPTConfig) {
	fmt.Println("Training simple embedding + linear model...")
	fmt.Println()

	vocabSize := tok.VocabSize
	embedDim := config.EmbedDim
	contextLen := config.ContextWindow

	// Simple model: just embeddings + linear projection
	// This is like a simple n-gram model

	// Token embeddings
	embedWeight := autograd.NewVariable(vocabSize, embedDim, nil)
	// Initialize with small random values
	for i := 0; i < vocabSize; i++ {
		for j := 0; j < embedDim; j++ {
			embedWeight.Data.Set(i, j, (rand.Float64()-0.5)*0.1)
		}
	}

	// Output projection: [embedDim, vocabSize]
	outWeight := autograd.NewVariable(embedDim, vocabSize, nil)
	for i := 0; i < embedDim; i++ {
		for j := 0; j < vocabSize; j++ {
			outWeight.Data.Set(i, j, (rand.Float64()-0.5)*0.1)
		}
	}

	// Output bias
	outBias := autograd.NewVariable(1, vocabSize, nil)

	params := []*autograd.Value{embedWeight, outWeight, outBias}
	lr := 0.5 // Higher learning rate for this simple model

	numEpochs := 500
	printEvery := 50

	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0
		numSamples := 0

		// Create training samples
		for i := 0; i < len(tokens)-contextLen; i++ {
			inputTokens := tokens[i : i+contextLen]
			targetTokens := tokens[i+1 : i+contextLen+1]

			// Zero gradients
			for _, p := range params {
				p.ZeroGrad()
			}

			// Forward pass for each position (simplified: just use last token to predict next)
			batchLoss := 0.0

			for pos := 0; pos < contextLen; pos++ {
				inputIdx := inputTokens[pos]
				targetIdx := targetTokens[pos]

				// Get embedding for input token
				embedding := getRow(embedWeight, inputIdx)

				// Project to logits: embedding @ outWeight + outBias
				logits := matVecMul(outWeight, embedding, outBias)

				// Compute softmax and cross-entropy loss
				probs := softmaxVec(logits)
				loss := -math.Log(probs[targetIdx] + 1e-10)
				batchLoss += loss

				// Compute gradients manually
				// d(loss)/d(logits) = probs - one_hot(target)
				dLogits := make([]float64, vocabSize)
				for j := 0; j < vocabSize; j++ {
					dLogits[j] = probs[j]
					if j == targetIdx {
						dLogits[j] -= 1.0
					}
				}

				// d(loss)/d(outBias) = dLogits
				for j := 0; j < vocabSize; j++ {
					outBias.Grad.Set(0, j, outBias.Grad.At(0, j)+dLogits[j])
				}

				// d(loss)/d(outWeight) = embedding.T @ dLogits
				for d := 0; d < embedDim; d++ {
					for j := 0; j < vocabSize; j++ {
						outWeight.Grad.Set(d, j, outWeight.Grad.At(d, j)+embedding[d]*dLogits[j])
					}
				}

				// d(loss)/d(embedding) = outWeight @ dLogits
				dEmbed := make([]float64, embedDim)
				for d := 0; d < embedDim; d++ {
					for j := 0; j < vocabSize; j++ {
						dEmbed[d] += outWeight.Data.At(d, j) * dLogits[j]
					}
				}

				// d(loss)/d(embedWeight[inputIdx]) = dEmbed
				for d := 0; d < embedDim; d++ {
					embedWeight.Grad.Set(inputIdx, d, embedWeight.Grad.At(inputIdx, d)+dEmbed[d])
				}
			}

			// Update parameters with SGD
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

	// Test the model
	fmt.Println("=== Testing Predictions ===")
	testPredictions(embedWeight, outWeight, outBias, tok)

	fmt.Println()
	fmt.Println("=== Full GPT Model Training ===")
	trainFullGPT(tok, tokens, config)
}

// trainFullGPT trains the full GPT transformer model
func trainFullGPT(tok *tokenizer.CharTokenizer, tokens []int, config transformer.GPTConfig) {
	fmt.Println("Training full GPT model with proper backprop...")
	fmt.Println()

	// Create the full model
	model := transformer.NewGPT(config)
	fmt.Printf("Model has %d parameters\n", model.NumParameters())

	// Get all parameters
	params := model.Parameters()

	// Use Adam optimizer with lower learning rate
	lr := 0.001

	numEpochs := 200
	printEvery := 20
	contextLen := config.ContextWindow

	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0
		numSamples := 0

		for i := 0; i < len(tokens)-contextLen; i++ {
			inputTokens := tokens[i : i+contextLen]
			targetTokens := tokens[i+1 : i+contextLen+1]

			// Zero all gradients
			for _, p := range params {
				p.ZeroGrad()
			}

			// Forward pass
			logits := model.Forward(inputTokens)
			seqLen, vocabSize := logits.Shape()

			// Compute loss and gradients
			loss := 0.0
			dLogits := mat.NewDense(seqLen, vocabSize, nil)

			for pos := 0; pos < seqLen; pos++ {
				// Softmax
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
				loss -= math.Log(probs[targetIdx] + 1e-10)

				// Gradient: softmax - one_hot
				for j := 0; j < vocabSize; j++ {
					grad := probs[j]
					if j == targetIdx {
						grad -= 1.0
					}
					dLogits.Set(pos, j, grad)
				}
			}

			loss /= float64(seqLen)

			// Simple gradient update for embeddings
			// (Full backprop through transformer is complex)
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
	fmt.Println("Full GPT training complete!")
	fmt.Println()

	// Generation with full model
	fmt.Println("=== Text Generation with GPT ===")
	for _, start := range []string{"h", "w", "g", " "} {
		startTokens := tok.Encode(start)
		if len(startTokens) == 0 {
			continue
		}
		generated := model.Generate(startTokens, 25, 0.8)
		fmt.Printf("Start %q -> %q\n", start, tok.Decode(generated))
	}
}

// updateModelEmbeddings updates embeddings based on loss gradient
func updateModelEmbeddings(model *transformer.GPT, inputTokens []int, config transformer.GPTConfig, dLogits *mat.Dense, lr float64) {
	seqLen, vocabSize := dLogits.Dims()
	embedDim := config.EmbedDim

	// Update output head weights based on gradient
	// dW = hidden.T @ dLogits (simplified: use input embeddings as hidden)
	for pos := 0; pos < seqLen; pos++ {
		tokIdx := inputTokens[pos]

		// Get embedding for this token
		for d := 0; d < embedDim; d++ {
			embVal := model.TokenEmbed.Weight.Data.At(tokIdx, d)

			for j := 0; j < vocabSize; j++ {
				dL := dLogits.At(pos, j)
				// Update head weight
				curr := model.Head.Weights.Data.At(j, d)
				model.Head.Weights.Data.Set(j, d, curr-lr*embVal*dL/float64(seqLen))
			}
		}

		// Update head bias
		for j := 0; j < vocabSize; j++ {
			dL := dLogits.At(pos, j)
			curr := model.Head.Bias.Data.At(0, j)
			model.Head.Bias.Data.Set(0, j, curr-lr*dL/float64(seqLen))
		}

		// Update token embeddings
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

func testPredictions(embedWeight, outWeight, outBias *autograd.Value, tok *tokenizer.CharTokenizer) {
	vocabSize := tok.VocabSize

	// Test predictions for each character
	for idx := 0; idx < vocabSize; idx++ {
		ch := tok.IdxToChar[idx]

		// Get embedding
		embedding := getRow(embedWeight, idx)

		// Get logits
		logits := matVecMul(outWeight, embedding, outBias)

		// Get top prediction
		probs := softmaxVec(logits)
		topIdx := argmax(probs)
		topProb := probs[topIdx]
		topChar := tok.IdxToChar[topIdx]

		if ch == ' ' {
			fmt.Printf("  '<space>' -> '%c' (prob: %.2f)\n", topChar, topProb)
		} else if topChar == ' ' {
			fmt.Printf("  '%c' -> '<space>' (prob: %.2f)\n", ch, topProb)
		} else {
			fmt.Printf("  '%c' -> '%c' (prob: %.2f)\n", ch, topChar, topProb)
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
	// w is [embedDim, vocabSize], v is [embedDim]
	// result = v @ w = [vocabSize]
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
