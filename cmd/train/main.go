// GoGPT Training Script with Tiktoken Tokenizer
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
	"github.com/suensky/gogpt/pkg/backend"
	"github.com/suensky/gogpt/pkg/train"
	"github.com/suensky/gogpt/pkg/transformer"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Set random seed for reproducibility
	rand.Seed(42)

	fmt.Println("=== GoGPT: Training a Small Transformer ===")
	fmt.Println("=== Using Tiktoken (GPT-4 compatible) ===")
	fmt.Println()

	// Initialize and display backend
	be, _ := backend.AutoSelectBackend()
	backend.SetDefault(be)
	fmt.Printf("Compute Backend: %s (GPU: %v)\n", be.Name(), be.IsGPU())
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

	// Use full training text (no character cap for industry-standard training)
	fmt.Printf("Training text: %d characters\n", len(text))
	fmt.Printf("First 200 chars: %s...\n\n", text[:min(200, len(text))])

	// Create tiktoken tokenizer (industry-standard, GPT-4 compatible)
	fmt.Println("Initializing Tiktoken tokenizer (cl100k_base)...")
	tok, err := tokenizer.NewTiktokenizer()
	if err != nil {
		fmt.Printf("Failed to create tiktoken: %v\n", err)
		fmt.Println("Falling back to custom BPE tokenizer...")
		// Fallback to custom BPE
		bpeTok := tokenizer.NewCharTokenizer(text)
		bpeTok.TrainBPE(text, 500)
		trainWithBPE(bpeTok, text)
		return
	}

	fmt.Printf("Tokenizer: cl100k_base (vocab size: %d)\n", tok.VocabSize)

	// Encode the text
	tokens := tok.Encode(text)
	fmt.Printf("Text encoded to %d tokens (%.1fx compression)\n", len(tokens), float64(len(text))/float64(len(tokens)))
	fmt.Printf("Sample tokens: %v\n\n", tokens[:min(20, len(tokens))])

	// For tiktoken, we use a subset of the vocab that appears in training data
	// This creates a mapping from tiktoken IDs to contiguous IDs for efficient training
	tokenSet := make(map[int]bool)
	for _, t := range tokens {
		tokenSet[t] = true
	}
	activeVocabSize := len(tokenSet)

	// Create mapping from tiktoken IDs to training IDs
	tiktokenToTrainID := make(map[int]int)
	trainIDToTiktoken := make(map[int]int)
	idx := 0
	for t := range tokenSet {
		tiktokenToTrainID[t] = idx
		trainIDToTiktoken[idx] = t
		idx++
	}

	// Remap tokens to contiguous IDs for training
	trainTokens := make([]int, len(tokens))
	for i, t := range tokens {
		trainTokens[i] = tiktokenToTrainID[t]
	}

	fmt.Printf("Active vocabulary: %d unique tokens (out of %d total)\n\n", activeVocabSize, tok.VocabSize)

	// Hyperparameters
	config := transformer.GPTConfig{
		VocabSize:     activeVocabSize, // Only train on tokens we've seen
		EmbedDim:      128,
		NumHeads:      4,
		NumLayers:     4,
		ContextWindow: 64,
		FFHiddenDim:   512,
	}

	fmt.Println("Model Configuration:")
	fmt.Printf("  Vocab Size: %d\n", config.VocabSize)
	fmt.Printf("  Embed Dim: %d\n", config.EmbedDim)
	fmt.Printf("  Num Heads: %d\n", config.NumHeads)
	fmt.Printf("  Num Layers: %d\n", config.NumLayers)
	fmt.Printf("  Context Window: %d\n", config.ContextWindow)
	fmt.Printf("  FF Hidden Dim: %d\n", config.FFHiddenDim)
	fmt.Println()

	// Train with tiktoken
	trainWithTiktoken(tok, trainTokens, trainIDToTiktoken, config)
}

// trainWithBPE is the fallback training path using custom BPE
func trainWithBPE(tok *tokenizer.BPETokenizer, text string) {
	tokens := tok.Encode(text)
	config := transformer.GPTConfig{
		VocabSize:     tok.VocabSize,
		EmbedDim:      128,
		NumHeads:      4,
		NumLayers:     4,
		ContextWindow: 64,
		FFHiddenDim:   512,
	}
	trainSimpleModelBPE(tok, tokens, config)
}

// trainWithTiktoken trains using the tiktoken tokenizer
func trainWithTiktoken(tok *tokenizer.TiktokenWrapper, tokens []int, trainIDToTiktoken map[int]int, config transformer.GPTConfig) {
	// Try MLX GPU training first
	be := backend.Default()
	if be.IsGPU() {
		fmt.Println("=== GPU Training Mode (MLX) ===")
		mlxTrainer, err := train.NewMLXTrainer(config.VocabSize, config.EmbedDim, config.ContextWindow)
		if err == nil {
			mlxTrainer.Train(tokens)

			// Test predictions
			tokenToString := func(trainID int) string {
				if tiktokenID, ok := trainIDToTiktoken[trainID]; ok {
					s, _ := tok.TokenString(tiktokenID)
					return formatToken(s)
				}
				return fmt.Sprintf("<%d>", trainID)
			}
			mlxTrainer.TestPredictions(tokenToString, 10)

			// Continue to full GPT training
			fmt.Println()
			trainFullGPTTiktoken(tok, tokens, trainIDToTiktoken, config)
			return
		}
		fmt.Printf("MLX trainer init failed: %v, falling back to CPU...\n", err)
	}

	// CPU training fallback
	fmt.Println("=== Training Simple Embedding + Linear Model (CPU) ===")
	fmt.Println()

	vocabSize := config.VocabSize
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
	baseLR := 0.05
	numEpochs := 1000
	printEvery := 100
	warmupEpochs := 100

	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0
		numSamples := 0

		numBatches := min(200, len(tokens)-contextLen-1)

		for batch := 0; batch < numBatches; batch++ {
			i := rand.Intn(len(tokens) - contextLen - 1)
			inputTokens := tokens[i : i+contextLen]
			targetTokens := tokens[i+1 : i+contextLen+1]

			for _, p := range params {
				p.ZeroGrad()
			}

			batchLoss := 0.0

			for pos := 0; pos < contextLen; pos++ {
				inputIdx := inputTokens[pos]
				targetIdx := targetTokens[pos]

				if inputIdx >= vocabSize || targetIdx >= vocabSize {
					continue
				}

				embedding := getRow(embedWeight, inputIdx)
				logits := matVecMul(outWeight, embedding, outBias)
				probs := softmaxVec(logits)
				loss := -math.Log(probs[targetIdx] + 1e-10)
				batchLoss += loss

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

			var lr float64
			if epoch < warmupEpochs {
				lr = baseLR * float64(epoch+1) / float64(warmupEpochs)
			} else {
				progress := float64(epoch-warmupEpochs) / float64(numEpochs-warmupEpochs)
				lr = baseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
			}

			maxGrad := 1.0
			scale := lr / float64(contextLen)
			for _, p := range params {
				r, c := p.Shape()
				gradRaw := p.Grad.RawMatrix().Data
				dataRaw := p.Data.RawMatrix().Data
				for idx := 0; idx < r*c; idx++ {
					grad := gradRaw[idx] * scale
					if grad > maxGrad {
						grad = maxGrad
					} else if grad < -maxGrad {
						grad = -maxGrad
					}
					dataRaw[idx] -= grad
				}
			}

			totalLoss += batchLoss / float64(contextLen)
			numSamples++
		}

		avgLoss := totalLoss / float64(numSamples)

		if epoch%printEvery == 0 || epoch == numEpochs-1 {
			elapsed := time.Since(startTime)
			var currentLR float64
			if epoch < warmupEpochs {
				currentLR = baseLR * float64(epoch+1) / float64(warmupEpochs)
			} else {
				progress := float64(epoch-warmupEpochs) / float64(numEpochs-warmupEpochs)
				currentLR = baseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
			}
			fmt.Printf("Epoch %3d | Loss: %.4f | LR: %.6f | Time: %v\n", epoch, avgLoss, currentLR, elapsed.Round(time.Millisecond))
		}
	}

	fmt.Println()
	fmt.Println("Training complete!")
	fmt.Printf("Total time: %v\n", time.Since(startTime).Round(time.Millisecond))
	fmt.Println()

	// Test predictions with tiktoken
	fmt.Println("=== Sample Token Predictions ===")
	testPredictionsTiktoken(embedWeight, outWeight, outBias, tok, trainIDToTiktoken, 10)

	// Train full GPT
	fmt.Println()
	trainFullGPTTiktoken(tok, tokens, trainIDToTiktoken, config)
}

// trainSimpleModelBPE trains using custom BPE tokenizer (fallback)
func trainSimpleModelBPE(tok *tokenizer.BPETokenizer, tokens []int, config transformer.GPTConfig) {
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
	baseLR := 0.05    // Lower base LR for stability
	numEpochs := 1000 // More epochs for industry-standard loss
	printEvery := 100
	warmupEpochs := 100 // Longer warmup period

	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0
		numSamples := 0

		// Random sampling - more batches for better gradient estimates
		numBatches := min(200, len(tokens)-contextLen-1)

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

			// Compute learning rate with warmup and cosine decay
			var lr float64
			if epoch < warmupEpochs {
				// Linear warmup
				lr = baseLR * float64(epoch+1) / float64(warmupEpochs)
			} else {
				// Cosine decay
				progress := float64(epoch-warmupEpochs) / float64(numEpochs-warmupEpochs)
				lr = baseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
			}

			// Update parameters with gradient clipping - batched for efficiency
			maxGrad := 1.0 // Gradient clipping threshold
			scale := lr / float64(contextLen)
			for _, p := range params {
				r, c := p.Shape()
				// Get raw slice access for faster updates
				gradRaw := p.Grad.RawMatrix().Data
				dataRaw := p.Data.RawMatrix().Data
				for idx := 0; idx < r*c; idx++ {
					grad := gradRaw[idx] * scale
					// Clip gradient
					if grad > maxGrad {
						grad = maxGrad
					} else if grad < -maxGrad {
						grad = -maxGrad
					}
					dataRaw[idx] -= grad
				}
			}

			totalLoss += batchLoss / float64(contextLen)
			numSamples++
		}

		avgLoss := totalLoss / float64(numSamples)

		if epoch%printEvery == 0 || epoch == numEpochs-1 {
			elapsed := time.Since(startTime)
			// Compute current LR for display
			var currentLR float64
			if epoch < warmupEpochs {
				currentLR = baseLR * float64(epoch+1) / float64(warmupEpochs)
			} else {
				progress := float64(epoch-warmupEpochs) / float64(numEpochs-warmupEpochs)
				currentLR = baseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
			}
			fmt.Printf("Epoch %3d | Loss: %.4f | LR: %.6f | Time: %v\n", epoch, avgLoss, currentLR, elapsed.Round(time.Millisecond))
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

// testPredictionsTiktoken tests predictions with tiktoken tokenizer
func testPredictionsTiktoken(embedWeight, outWeight, outBias *autograd.Value, tok *tokenizer.TiktokenWrapper, trainIDToTiktoken map[int]int, n int) {
	_, cols := outWeight.Shape()
	vocabSize := cols

	// Test first n training IDs
	count := 0
	for trainID := 0; trainID < vocabSize && count < n; trainID++ {
		tiktokenID := trainIDToTiktoken[trainID]

		embedding := getRow(embedWeight, trainID)
		logits := matVecMul(outWeight, embedding, outBias)
		probs := softmaxVec(logits)
		topIdx := argmax(probs)
		topProb := probs[topIdx]

		// Decode tokens for display
		inputToken, _ := tok.TokenString(tiktokenID)
		outputTiktoken := trainIDToTiktoken[topIdx]
		outputToken, _ := tok.TokenString(outputTiktoken)

		displayFrom := formatToken(inputToken)
		displayTo := formatToken(outputToken)
		fmt.Printf("  '%s' -> '%s' (prob: %.2f)\n", displayFrom, displayTo, topProb)
		count++
	}
}

// trainFullGPTTiktoken trains the full GPT model with tiktoken
func trainFullGPTTiktoken(tok *tokenizer.TiktokenWrapper, tokens []int, trainIDToTiktoken map[int]int, config transformer.GPTConfig) {
	fmt.Println("=== Training Full GPT Model ===")
	fmt.Println()

	model := transformer.NewGPT(config)
	fmt.Printf("Model has %d parameters\n", model.NumParameters())

	params := model.Parameters()
	baseLR := 0.0005
	numEpochs := 500
	printEvery := 50
	warmupEpochs := 50
	contextLen := config.ContextWindow
	numBatches := min(200, len(tokens)-contextLen-1)

	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0
		numSamples := 0

		var lr float64
		if epoch < warmupEpochs {
			lr = baseLR * float64(epoch+1) / float64(warmupEpochs)
		} else {
			progress := float64(epoch-warmupEpochs) / float64(numEpochs-warmupEpochs)
			lr = baseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
		}

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
			fmt.Printf("Epoch %3d | Loss: %.4f | LR: %.6f | Time: %v\n", epoch, avgLoss, lr, elapsed.Round(time.Millisecond))
		}
	}

	fmt.Println()
	fmt.Println("GPT training complete!")
	fmt.Println()

	// Generation with tiktoken (skip for now as we need reverse mapping)
	fmt.Println("=== Text Generation ===")
	fmt.Println("(Generation with remapped tokens - using first tokens from training set)")
	for i := 0; i < 3 && i < len(tokens)-30; i++ {
		startTokens := tokens[i*10 : i*10+3]
		generated := model.Generate(startTokens, 30, 0.8)

		// Convert back to tiktoken IDs for decoding
		tiktokenIDs := make([]int, len(generated))
		for j, trainID := range generated {
			if tid, ok := trainIDToTiktoken[trainID]; ok {
				tiktokenIDs[j] = tid
			}
		}
		text := tok.Decode(tiktokenIDs)
		fmt.Printf("Generated: %s\n\n", text)
	}
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
	baseLR := 0.0005 // Lower LR for larger model stability
	numEpochs := 500 // More epochs for industry-standard loss (target < 2.0)
	printEvery := 50
	warmupEpochs := 50 // Longer warmup for stability
	contextLen := config.ContextWindow
	numBatches := min(200, len(tokens)-contextLen-1) // More batches per epoch

	startTime := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0
		numSamples := 0

		// Compute learning rate with warmup and cosine decay
		var lr float64
		if epoch < warmupEpochs {
			// Linear warmup
			lr = baseLR * float64(epoch+1) / float64(warmupEpochs)
		} else {
			// Cosine decay
			progress := float64(epoch-warmupEpochs) / float64(numEpochs-warmupEpochs)
			lr = baseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
		}

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
			fmt.Printf("Epoch %3d | Loss: %.4f | LR: %.6f | Time: %v\n", epoch, avgLoss, lr, elapsed.Round(time.Millisecond))
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
	scale := lr / float64(seqLen)

	// Get raw slice access for faster updates
	dLogitsRaw := dLogits.RawMatrix().Data
	headWeightsRaw := model.Head.Weights.Data.RawMatrix().Data
	headBiasRaw := model.Head.Bias.Data.RawMatrix().Data
	tokenEmbedRaw := model.TokenEmbed.Weight.Data.RawMatrix().Data

	for pos := 0; pos < seqLen; pos++ {
		tokIdx := inputTokens[pos]
		if tokIdx >= config.VocabSize {
			continue
		}

		dLogitsRow := dLogitsRaw[pos*vocabSize : (pos+1)*vocabSize]

		// Update head weights: headWeights[j, d] -= lr * embVal * dL / seqLen
		for d := 0; d < embedDim; d++ {
			embVal := tokenEmbedRaw[tokIdx*embedDim+d]
			for j := 0; j < vocabSize; j++ {
				headWeightsRaw[j*embedDim+d] -= scale * embVal * dLogitsRow[j]
			}
		}

		// Update head bias: headBias[0, j] -= lr * dL / seqLen
		for j := 0; j < vocabSize; j++ {
			headBiasRaw[j] -= scale * dLogitsRow[j]
		}

		// Update token embeddings: tokenEmbed[tokIdx, d] -= lr * grad / seqLen
		for d := 0; d < embedDim; d++ {
			grad := 0.0
			for j := 0; j < vocabSize; j++ {
				grad += headWeightsRaw[j*embedDim+d] * dLogitsRow[j]
			}
			tokenEmbedRaw[tokIdx*embedDim+d] -= scale * grad
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
