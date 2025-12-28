// Package tokenizer provides text tokenization utilities.
package tokenizer

import (
	"sort"
	"strings"
)

// CharTokenizer implements a simple character-level tokenizer.
// Each unique character in the vocabulary is mapped to an integer index.
type CharTokenizer struct {
	CharToIdx map[rune]int
	IdxToChar map[int]rune
	VocabSize int
}

// NewCharTokenizer creates a character tokenizer from the given text.
// It builds a vocabulary from all unique characters in the text.
func NewCharTokenizer(text string) *CharTokenizer {
	// Find unique characters
	charSet := make(map[rune]bool)
	for _, ch := range text {
		charSet[ch] = true
	}

	// Sort characters for deterministic ordering
	chars := make([]rune, 0, len(charSet))
	for ch := range charSet {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool {
		return chars[i] < chars[j]
	})

	// Create mappings
	charToIdx := make(map[rune]int)
	idxToChar := make(map[int]rune)
	for i, ch := range chars {
		charToIdx[ch] = i
		idxToChar[i] = ch
	}

	return &CharTokenizer{
		CharToIdx: charToIdx,
		IdxToChar: idxToChar,
		VocabSize: len(chars),
	}
}

// Encode converts a string to a slice of token indices.
func (t *CharTokenizer) Encode(text string) []int {
	tokens := make([]int, 0, len(text))
	for _, ch := range text {
		if idx, ok := t.CharToIdx[ch]; ok {
			tokens = append(tokens, idx)
		}
		// Skip unknown characters
	}
	return tokens
}

// Decode converts a slice of token indices back to a string.
func (t *CharTokenizer) Decode(tokens []int) string {
	var sb strings.Builder
	for _, idx := range tokens {
		if ch, ok := t.IdxToChar[idx]; ok {
			sb.WriteRune(ch)
		}
	}
	return sb.String()
}

// GetVocab returns the vocabulary as a sorted list of characters.
func (t *CharTokenizer) GetVocab() []rune {
	chars := make([]rune, t.VocabSize)
	for idx, ch := range t.IdxToChar {
		chars[idx] = ch
	}
	return chars
}

// PrintVocab prints the vocabulary for debugging.
func (t *CharTokenizer) PrintVocab() string {
	var sb strings.Builder
	sb.WriteString("Vocabulary:\n")
	for idx := 0; idx < t.VocabSize; idx++ {
		ch := t.IdxToChar[idx]
		if ch == '\n' {
			sb.WriteString("  " + string(rune('0'+idx)) + ": \\n\n")
		} else if ch == ' ' {
			sb.WriteString("  " + string(rune('0'+idx)) + ": <space>\n")
		} else {
			sb.WriteString("  " + string(rune('0'+idx)) + ": " + string(ch) + "\n")
		}
	}
	return sb.String()
}

// EncodeWithPadding encodes text and pads to a fixed length.
func (t *CharTokenizer) EncodeWithPadding(text string, length int, padToken int) []int {
	tokens := t.Encode(text)
	if len(tokens) >= length {
		return tokens[:length]
	}

	// Pad with padToken
	padded := make([]int, length)
	copy(padded, tokens)
	for i := len(tokens); i < length; i++ {
		padded[i] = padToken
	}
	return padded
}
