// Package tokenizer provides text tokenization utilities.
// This implements a BPE (Byte Pair Encoding) tokenizer that:
// 1. Starts with character-level tokens
// 2. Applies merge rules to combine frequently occurring token pairs
package tokenizer

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
)

// BPETokenizer implements Byte Pair Encoding tokenization.
// It starts with character-level tokens and applies merge rules
// to combine frequently occurring pairs.
type BPETokenizer struct {
	tokenToID  map[string]int
	idToToken  map[int]string
	mergeRules map[int64]int
	rulesOrder []int64
	VocabSize  int
}

// NewBPETokenizer creates a new BPE tokenizer.
// text: the training text to build character vocabulary from
// vocabFile: path to the BPE vocab/merge rules file (optional, empty for character-level only)
// numMerges: how many merge rules to use from the vocab file
func NewBPETokenizer(text string, vocabFile string, numMerges int) *BPETokenizer {
	t := &BPETokenizer{
		tokenToID:  make(map[string]int),
		idToToken:  make(map[int]string),
		mergeRules: make(map[int64]int),
		rulesOrder: nil,
	}

	// Normalize newlines
	text = normalizeNewlines(text)

	// Add all unique characters to vocabulary
	t.addCharsToVocab(text)

	// Load merge rules if vocab file provided
	if vocabFile != "" && numMerges > 0 {
		t.loadMergeRules(vocabFile, numMerges)
	}

	t.VocabSize = len(t.tokenToID)
	return t
}

// NewCharTokenizer creates a simple character-level tokenizer (backward compatible).
func NewCharTokenizer(text string) *BPETokenizer {
	return NewBPETokenizer(text, "", 0)
}

// Encode converts a string to a slice of token indices.
func (t *BPETokenizer) Encode(text string) []int {
	text = normalizeNewlines(text)

	var tokens []int

	// First, tokenize at character level
	for _, ch := range text {
		tok, ok := t.tokenToID[string(ch)]
		if !ok {
			// Skip unknown characters
			continue
		}
		tokens = append(tokens, tok)
	}

	// Apply merge rules in order
	for _, rule := range t.rulesOrder {
		var newTokens []int
		tok1, tok2 := unzip(rule)

		// Try to apply rule on every pair of tokens
		for i := 0; i < len(tokens); {
			hasNextToken := i+1 < len(tokens)
			shouldMerge := hasNextToken && tokens[i] == tok1 && tokens[i+1] == tok2
			if shouldMerge {
				newTokens = append(newTokens, t.mergeRules[rule])
				i += 2 // eat two tokens
			} else {
				newTokens = append(newTokens, tokens[i])
				i++ // eat one token
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

// GetChars returns just the single-character tokens.
func (t *BPETokenizer) GetChars() string {
	var chars []string
	for token := range t.tokenToID {
		if len([]rune(token)) == 1 {
			chars = append(chars, token)
		}
	}
	sort.Strings(chars)
	return strings.Join(chars, "")
}

// addCharsToVocab adds all unique characters from text to vocabulary.
func (t *BPETokenizer) addCharsToVocab(text string) {
	charSet := make(map[rune]bool)
	for _, ch := range text {
		charSet[ch] = true
	}

	// Sort for deterministic ordering
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

// loadMergeRules loads BPE merge rules from a vocab file.
// Format: [token1][token2] -> [merged_token]
func (t *BPETokenizer) loadMergeRules(vocabFile string, numMerges int) {
	file, err := os.Open(vocabFile)
	if err != nil {
		fmt.Printf("Warning: Could not open vocab file %s: %v\n", vocabFile, err)
		return
	}
	defer file.Close()

	re := regexp.MustCompile(`\[(.*?)\]\[(.*?)\] -> \[(.*?)\]`)
	scanner := bufio.NewScanner(file)
	count := 0

	for scanner.Scan() && count < numMerges {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		matches := re.FindStringSubmatch(line)
		if len(matches) != 4 {
			continue
		}

		// Process escape sequences
		left := strings.ReplaceAll(matches[1], "\\n", "\n")
		right := strings.ReplaceAll(matches[2], "\\n", "\n")
		merged := strings.ReplaceAll(matches[3], "\\n", "\n")

		// Add merged token to vocabulary
		t.addToken(merged)

		// Verify all tokens exist
		leftID, okL := t.tokenToID[left]
		rightID, okR := t.tokenToID[right]
		mergedID, okM := t.tokenToID[merged]

		if !okL || !okR || !okM {
			// Skip invalid rules (tokens not in our vocabulary)
			continue
		}

		// Add merge rule
		key := zip(leftID, rightID)
		t.mergeRules[key] = mergedID
		t.rulesOrder = append(t.rulesOrder, key)
		count++
	}
}

// TrainBPE learns BPE merge rules from the given text.
// Returns the vocabulary file content that can be saved.
func (t *BPETokenizer) TrainBPE(text string, numMerges int) string {
	text = normalizeNewlines(text)

	// Start with character tokens
	tokens := make([]int, 0, len(text))
	for _, ch := range text {
		if tok, ok := t.tokenToID[string(ch)]; ok {
			tokens = append(tokens, tok)
		}
	}

	var rules []string

	for m := 0; m < numMerges; m++ {
		// Count pair frequencies
		pairCounts := make(map[int64]int)
		for i := 0; i < len(tokens)-1; i++ {
			key := zip(tokens[i], tokens[i+1])
			pairCounts[key]++
		}

		if len(pairCounts) == 0 {
			break
		}

		// Find most frequent pair
		var bestPair int64
		bestCount := 0
		for pair, count := range pairCounts {
			if count > bestCount {
				bestCount = count
				bestPair = pair
			}
		}

		if bestCount < 2 {
			break // No more useful merges
		}

		// Create new token
		tok1, tok2 := unzip(bestPair)
		newToken := t.idToToken[tok1] + t.idToToken[tok2]
		newID := t.addToken(newToken)

		// Record rule
		left := strings.ReplaceAll(t.idToToken[tok1], "\n", "\\n")
		right := strings.ReplaceAll(t.idToToken[tok2], "\n", "\\n")
		merged := strings.ReplaceAll(newToken, "\n", "\\n")
		rules = append(rules, fmt.Sprintf("[%s][%s] -> [%s]", left, right, merged))

		// Add to merge rules
		t.mergeRules[bestPair] = newID
		t.rulesOrder = append(t.rulesOrder, bestPair)

		// Apply merge to tokens
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

// SaveVocab saves the learned BPE rules to a file.
func (t *BPETokenizer) SaveVocab(filename string) error {
	var rules []string
	for _, key := range t.rulesOrder {
		tok1, tok2 := unzip(key)
		mergedID := t.mergeRules[key]

		left := strings.ReplaceAll(t.idToToken[tok1], "\n", "\\n")
		right := strings.ReplaceAll(t.idToToken[tok2], "\n", "\\n")
		merged := strings.ReplaceAll(t.idToToken[mergedID], "\n", "\\n")

		rules = append(rules, fmt.Sprintf("[%s][%s] -> [%s]", left, right, merged))
	}

	return os.WriteFile(filename, []byte(strings.Join(rules, "\n")), 0644)
}

// Helper functions

func normalizeNewlines(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	return strings.ReplaceAll(text, "\r", "\n")
}

// zip combines two token IDs into a single int64 key.
func zip(tok1, tok2 int) int64 {
	return int64(tok1)<<32 | int64(tok2&0xFFFFFFFF)
}

// unzip splits an int64 key back into two token IDs.
func unzip(key int64) (int, int) {
	tok1 := int(key >> 32)
	tok2 := int(key & 0xFFFFFFFF)
	return tok1, tok2
}

// EncodeToFloat converts tokens to float64 slice (for compatibility with some APIs)
func (t *BPETokenizer) EncodeToFloat(text string) []float64 {
	tokens := t.Encode(text)
	result := make([]float64, len(tokens))
	for i, tok := range tokens {
		result[i] = float64(tok)
	}
	return result
}

// DecodeFromFloat converts float64 tokens back to string
func (t *BPETokenizer) DecodeFromFloat(tokens []float64) string {
	intTokens := make([]int, len(tokens))
	for i, tok := range tokens {
		intTokens[i] = int(tok)
	}
	return t.Decode(intTokens)
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
