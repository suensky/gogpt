# GoGPT

A from-scratch implementation of automatic differentiation and GPT-style transformer decoder in pure Go.

## Features

- **Autograd Engine**: Reverse-mode automatic differentiation with support for:
  - Basic operations: Add, Sub, Mul, MatMul, Transpose
  - Activations: ReLU, Softmax, LogSoftmax, GELU, Tanh, Sigmoid
  - Math functions: Exp, Log, Pow, Sqrt, Sum, Mean

- **Neural Network Layers**:
  - Linear (fully connected) with Xavier initialization
  - Embedding (token lookup)
  - LayerNorm

- **Optimizers**:
  - SGD
  - Adam
  - SGD with Momentum

- **Loss Functions**:
  - CrossEntropyLoss
  - MSELoss
  - NLLLoss

- **Transformer Components**:
  - Multi-head self-attention with causal masking
  - Position-wise feed-forward network
  - Decoder blocks with residual connections
  - Full GPT model with text generation

## Project Structure

```
gogpt/
├── cmd/
│   └── train/
│       └── main.go           # Training script
├── pkg/
│   ├── autograd/             # Autodiff engine
│   ├── nn/                   # Neural network layers
│   └── transformer/          # Transformer components
├── internal/
│   └── tokenizer/            # Character-level tokenizer
└── docs/
    └── devlog.md             # Development log
```

## Quick Start

```bash
# Run tests
go test -v ./...

# Train the model
go run ./cmd/train/main.go
```

## Example Output

```
=== GoGPT: Training a Small Transformer ===

Training text: "hello world hello go hello transformer"
Vocabulary size: 15

Model Configuration:
  Vocab Size: 15
  Embed Dim: 16
  Num Heads: 2
  Num Layers: 2
  Context Window: 8

Model has 7215 parameters

Epoch   0 | Loss: 2.5219
Epoch 500 | Loss: 0.6147

=== Testing Predictions ===
  'h' -> 'e' (prob: 1.00)
  'e' -> 'l' (prob: 0.77)
  'l' -> 'o' (prob: 0.61)
  'w' -> 'o' (prob: 1.00)
```

## Dependencies

- [gonum](https://gonum.org/) - Numerical library for matrix operations

## References

- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

MIT