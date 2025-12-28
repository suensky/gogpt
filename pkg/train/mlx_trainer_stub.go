//go:build !mlx || !darwin || !arm64

// Package train provides training utilities.
// This is a stub for non-MLX builds.
package train

import (
	"fmt"
)

// MLXTrainer is a stub - not available without MLX.
type MLXTrainer struct{}

// NewMLXTrainer returns an error on non-MLX builds.
func NewMLXTrainer(vocabSize, embedDim, contextLen int) (*MLXTrainer, error) {
	return nil, fmt.Errorf("MLX training not available: build with -tags mlx on Apple Silicon")
}

// Train is a stub.
func (t *MLXTrainer) Train(tokens []int) {}

// TestPredictions is a stub.
func (t *MLXTrainer) TestPredictions(tokenToString func(int) string, n int) {}
