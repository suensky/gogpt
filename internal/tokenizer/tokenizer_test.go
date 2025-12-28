package tokenizer

import (
	"strings"
	"testing"
)

func TestTiktokenWrapper(t *testing.T) {
	tok, err := NewTiktokenizer()
	if err != nil {
		t.Fatalf("Failed to create tiktoken: %v", err)
	}

	t.Run("VocabSize", func(t *testing.T) {
		if tok.VocabSize != 100277 {
			t.Errorf("Expected vocab size 100277, got %d", tok.VocabSize)
		}
	})

	t.Run("EncodeDecode", func(t *testing.T) {
		text := "Hello, World!"
		encoded := tok.Encode(text)
		if len(encoded) == 0 {
			t.Error("Expected non-empty encoding")
		}

		decoded := tok.Decode(encoded)
		if decoded != text {
			t.Errorf("Expected '%s', got '%s'", text, decoded)
		}
	})

	t.Run("CountTokens", func(t *testing.T) {
		text := "The quick brown fox jumps over the lazy dog."
		count := tok.CountTokens(text)
		if count <= 0 {
			t.Error("Expected positive token count")
		}
		t.Logf("Token count for '%s': %d", text, count)
	})
}

func TestGPT2Tokenizer(t *testing.T) {
	tok, err := NewGPT2Tokenizer()
	if err != nil {
		t.Fatalf("Failed to create GPT-2 tokenizer: %v", err)
	}

	if tok.VocabSize != 50257 {
		t.Errorf("Expected vocab size 50257, got %d", tok.VocabSize)
	}

	text := "Hello, World!"
	encoded := tok.Encode(text)
	decoded := tok.Decode(encoded)

	if decoded != text {
		t.Errorf("GPT-2 tokenizer round trip failed: got '%s'", decoded)
	}
}

func TestCharTokenizer(t *testing.T) {
	text := "hello world"
	tok := NewCharTokenizer(text)

	if tok.VocabSize != 8 { // ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'
		t.Errorf("Expected vocab size 8, got %d", tok.VocabSize)
	}

	encoded := tok.Encode("hello")
	if len(encoded) != 5 {
		t.Errorf("Expected 5 tokens, got %d", len(encoded))
	}

	decoded := tok.Decode(encoded)
	if decoded != "hello" {
		t.Errorf("Expected 'hello', got '%s'", decoded)
	}
}

func TestBPETokenizer(t *testing.T) {
	text := "hello hello hello world world"
	tok := NewCharTokenizer(text)

	// Train BPE with 2 merges
	rules := tok.TrainBPE(text, 2)

	// Should have learned some merges
	if len(rules) == 0 {
		t.Error("Expected some merge rules to be learned")
	}

	// Vocab size should have increased
	if tok.VocabSize <= 8 {
		t.Errorf("Expected vocab size > 8 after BPE, got %d", tok.VocabSize)
	}

	t.Logf("Learned BPE rules:\n%s", rules)
	t.Logf("Final vocab size: %d", tok.VocabSize)

	// Test encode/decode still works
	encoded := tok.Encode("hello")
	decoded := tok.Decode(encoded)
	if decoded != "hello" {
		t.Errorf("Expected 'hello', got '%s' after BPE encoding/decoding", decoded)
	}
}

func TestBPECompression(t *testing.T) {
	text := strings.Repeat("hello world ", 100)
	tok := NewCharTokenizer(text)

	origEncoded := tok.Encode(text)
	origLen := len(origEncoded)

	tok.TrainBPE(text, 20)

	bpeEncoded := tok.Encode(text)
	bpeLen := len(bpeEncoded)

	t.Logf("Original length: %d, BPE length: %d, compression: %.2fx",
		origLen, bpeLen, float64(origLen)/float64(bpeLen))

	if bpeLen >= origLen {
		t.Error("Expected BPE to compress the text")
	}
}

func TestEncodeDecode(t *testing.T) {
	testCases := []string{
		"hello",
		"hello world",
		"The quick brown fox",
		"Testing 123",
		"Line1\nLine2",
	}

	for _, tc := range testCases {
		tok := NewCharTokenizer(tc)
		encoded := tok.Encode(tc)
		decoded := tok.Decode(encoded)

		if decoded != tc {
			t.Errorf("Round trip failed for '%s': got '%s'", tc, decoded)
		}
	}
}

func TestNewlineNormalization(t *testing.T) {
	tok := NewCharTokenizer("a\nb\nc")

	testCases := []struct {
		input    string
		expected string
	}{
		{"a\nb\nc", "a\nb\nc"},     // Unix
		{"a\r\nb\r\nc", "a\nb\nc"}, // Windows
		{"a\rb\rc", "a\nb\nc"},     // Old Mac
	}

	for _, tc := range testCases {
		encoded := tok.Encode(tc.input)
		decoded := tok.Decode(encoded)
		if decoded != tc.expected {
			t.Errorf("Newline normalization failed: input='%q', expected='%q', got='%q'",
				tc.input, tc.expected, decoded)
		}
	}
}

func TestZipUnzip(t *testing.T) {
	testCases := [][2]int{
		{0, 0},
		{1, 2},
		{100, 200},
		{65535, 65535},
	}

	for _, tc := range testCases {
		key := zip(tc[0], tc[1])
		tok1, tok2 := unzip(key)
		if tok1 != tc[0] || tok2 != tc[1] {
			t.Errorf("Zip/Unzip failed: expected (%d, %d), got (%d, %d)",
				tc[0], tc[1], tok1, tok2)
		}
	}
}
