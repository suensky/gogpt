// Package tokenizer provides text tokenization utilities.
// This provides a tiktoken-based tokenizer using OpenAI's cl100k_base encoding,
// with a fallback to custom BPE for training on custom vocabularies.
package tokenizer

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tiktoken-go/tokenizer"
)

// TiktokenWrapper wraps the tiktoken-go library with a compatible interface.
// Uses cl100k_base encoding (GPT-4/ChatGPT compatible).
type TiktokenWrapper struct {
	codec     tokenizer.Codec
	VocabSize int
}

// NewTiktokenizer creates a new tiktoken-based tokenizer.
// Uses cl100k_base encoding which is compatible with GPT-4 and ChatGPT.
func NewTiktokenizer() (*TiktokenWrapper, error) {
	// Use cl100k_base encoding (GPT-4, ChatGPT, text-embedding-ada-002)
	codec, err := tokenizer.Get(tokenizer.Cl100kBase)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize tiktoken: %w", err)
	}

	// cl100k_base has ~100,000 tokens
	return &TiktokenWrapper{
		codec:     codec,
		VocabSize: 100277, // Actual cl100k_base vocab size
	}, nil
}

// NewGPT2Tokenizer creates a tokenizer with GPT-2 encoding.
// This is useful for smaller models or research purposes.
func NewGPT2Tokenizer() (*TiktokenWrapper, error) {
	codec, err := tokenizer.Get(tokenizer.R50kBase)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize GPT-2 tokenizer: %w", err)
	}

	return &TiktokenWrapper{
		codec:     codec,
		VocabSize: 50257, // GPT-2 vocab size
	}, nil
}

// Encode converts a string to a slice of token indices.
func (t *TiktokenWrapper) Encode(text string) []int {
	ids, _, err := t.codec.Encode(text)
	if err != nil {
		// Return empty on error
		return nil
	}

	// Convert uint to int
	result := make([]int, len(ids))
	for i, id := range ids {
		result[i] = int(id)
	}
	return result
}

// Decode converts a slice of token indices back to a string.
func (t *TiktokenWrapper) Decode(indices []int) string {
	// Convert int to uint
	uintIndices := make([]uint, len(indices))
	for i, idx := range indices {
		uintIndices[i] = uint(idx)
	}

	text, err := t.codec.Decode(uintIndices)
	if err != nil {
		return ""
	}
	return text
}

// TokenID returns the ID for a given token string.
func (t *TiktokenWrapper) TokenID(token string) (int, bool) {
	ids, _, err := t.codec.Encode(token)
	if err != nil || len(ids) != 1 {
		return 0, false
	}
	return int(ids[0]), true
}

// TokenString returns the string for a given token ID.
func (t *TiktokenWrapper) TokenString(id int) (string, bool) {
	text, err := t.codec.Decode([]uint{uint(id)})
	if err != nil {
		return "", false
	}
	return text, true
}

// GetVocab returns a sample of common tokens (not full vocab due to size).
func (t *TiktokenWrapper) GetVocab() []string {
	// Return some common tokens for compatibility
	common := []string{" ", "the", "a", "is", "of", "and", "to", "in", "that", "it"}
	return common
}

// EncodeToFloat converts tokens to float64 slice.
func (t *TiktokenWrapper) EncodeToFloat(text string) []float64 {
	tokens := t.Encode(text)
	result := make([]float64, len(tokens))
	for i, tok := range tokens {
		result[i] = float64(tok)
	}
	return result
}

// DecodeFromFloat converts float64 tokens back to string.
func (t *TiktokenWrapper) DecodeFromFloat(tokens []float64) string {
	intTokens := make([]int, len(tokens))
	for i, tok := range tokens {
		intTokens[i] = int(tok)
	}
	return t.Decode(intTokens)
}

// CountTokens returns the number of tokens in a string (useful for API limits).
func (t *TiktokenWrapper) CountTokens(text string) int {
	return len(t.Encode(text))
}

// --- Backward Compatibility: Keep BPETokenizer for custom training ---

// BPETokenizer implements Byte Pair Encoding tokenization.
// Kept for backward compatibility and custom vocabulary training.
type BPETokenizer struct {
	tokenToID  map[string]int
	idToToken  map[int]string
	mergeRules map[int64]int
	rulesOrder []int64
	VocabSize  int
}

// NewCharTokenizer creates a simple character-level tokenizer.
func NewCharTokenizer(text string) *BPETokenizer {
	t := &BPETokenizer{
		tokenToID:  make(map[string]int),
		idToToken:  make(map[int]string),
		mergeRules: make(map[int64]int),
		rulesOrder: nil,
	}

	text = normalizeNewlines(text)
	t.addCharsToVocab(text)
	t.VocabSize = len(t.tokenToID)
	return t
}

// Encode converts a string to a slice of token indices.
func (t *BPETokenizer) Encode(text string) []int {
	text = normalizeNewlines(text)

	var tokens []int
	for _, ch := range text {
		tok, ok := t.tokenToID[string(ch)]
		if !ok {
			continue
		}
		tokens = append(tokens, tok)
	}

	// Apply merge rules in order
	for _, rule := range t.rulesOrder {
		var newTokens []int
		tok1, tok2 := unzip(rule)

		for i := 0; i < len(tokens); {
			hasNextToken := i+1 < len(tokens)
			shouldMerge := hasNextToken && tokens[i] == tok1 && tokens[i+1] == tok2
			if shouldMerge {
				newTokens = append(newTokens, t.mergeRules[rule])
				i += 2
			} else {
				newTokens = append(newTokens, tokens[i])
				i++
			}
		}
		tokens = newTokens
	}

	return tokens
}

// Decode converts a slice of token indices back to a string.
func (t *BPETokenizer) Decode(indices []int) string {
	var result strings.Builder
	for _, idx := range indices {
		if token, ok := t.idToToken[idx]; ok {
			result.WriteString(token)
		}
	}
	return result.String()
}

// GetVocab returns the vocabulary as a sorted list of tokens.
func (t *BPETokenizer) GetVocab() []string {
	tokens := make([]string, 0, len(t.tokenToID))
	for token := range t.tokenToID {
		tokens = append(tokens, token)
	}
	sort.Strings(tokens)
	return tokens
}

// TrainBPE learns BPE merge rules from the given text.
func (t *BPETokenizer) TrainBPE(text string, numMerges int) string {
	text = normalizeNewlines(text)

	tokens := make([]int, 0, len(text))
	for _, ch := range text {
		if tok, ok := t.tokenToID[string(ch)]; ok {
			tokens = append(tokens, tok)
		}
	}

	var rules []string

	for m := 0; m < numMerges; m++ {
		pairCounts := make(map[int64]int)
		for i := 0; i < len(tokens)-1; i++ {
			key := zip(tokens[i], tokens[i+1])
			pairCounts[key]++
		}

		if len(pairCounts) == 0 {
			break
		}

		var bestPair int64
		bestCount := 0
		for pair, count := range pairCounts {
			if count > bestCount {
				bestCount = count
				bestPair = pair
			}
		}

		if bestCount < 2 {
			break
		}

		tok1, tok2 := unzip(bestPair)
		newToken := t.idToToken[tok1] + t.idToToken[tok2]
		newID := t.addToken(newToken)

		left := strings.ReplaceAll(t.idToToken[tok1], "\n", "\\n")
		right := strings.ReplaceAll(t.idToToken[tok2], "\n", "\\n")
		merged := strings.ReplaceAll(newToken, "\n", "\\n")
		rules = append(rules, fmt.Sprintf("[%s][%s] -> [%s]", left, right, merged))

		t.mergeRules[bestPair] = newID
		t.rulesOrder = append(t.rulesOrder, bestPair)

		var newTokens []int
		for i := 0; i < len(tokens); {
			if i+1 < len(tokens) && tokens[i] == tok1 && tokens[i+1] == tok2 {
				newTokens = append(newTokens, newID)
				i += 2
			} else {
				newTokens = append(newTokens, tokens[i])
				i++
			}
		}
		tokens = newTokens
	}

	t.VocabSize = len(t.tokenToID)
	return strings.Join(rules, "\n")
}

// TokenID returns the ID for a given token string.
func (t *BPETokenizer) TokenID(token string) (int, bool) {
	id, ok := t.tokenToID[token]
	return id, ok
}

// TokenString returns the string for a given token ID.
func (t *BPETokenizer) TokenString(id int) (string, bool) {
	token, ok := t.idToToken[id]
	return token, ok
}

// addCharsToVocab adds all unique characters from text to vocabulary.
func (t *BPETokenizer) addCharsToVocab(text string) {
	charSet := make(map[rune]bool)
	for _, ch := range text {
		charSet[ch] = true
	}

	chars := make([]rune, 0, len(charSet))
	for ch := range charSet {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool {
		return chars[i] < chars[j]
	})

	for _, ch := range chars {
		t.addToken(string(ch))
	}
}

// addToken adds a token to the vocabulary if not already present.
func (t *BPETokenizer) addToken(token string) int {
	if id, ok := t.tokenToID[token]; ok {
		return id
	}
	id := len(t.tokenToID)
	t.tokenToID[token] = id
	t.idToToken[id] = token
	return id
}

// Helper functions

func normalizeNewlines(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	return strings.ReplaceAll(text, "\r", "\n")
}

func zip(tok1, tok2 int) int64 {
	return int64(tok1)<<32 | int64(tok2&0xFFFFFFFF)
}

func unzip(key int64) (int, int) {
	tok1 := int(key >> 32)
	tok2 := int(key & 0xFFFFFFFF)
	return tok1, tok2
}
