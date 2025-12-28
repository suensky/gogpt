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
  - SGD / SGD with Momentum
  - Adam
  - **AdamW** (recommended for transformers)

- **Loss Functions**:
  - CrossEntropyLoss
  - CrossEntropyLossWithSmoothing (label smoothing)
  - MSELoss
  - NLLLoss

- **Transformer Components**:
  - Multi-head self-attention with causal masking
  - Position-wise feed-forward network
  - Decoder blocks with residual connections
  - Full GPT model with text generation

- **Backend Abstraction** (CPU/GPU):
  - GoNum backend (CPU, default)
  - MLX backend (Apple Silicon GPU)

## Project Structure

```
gogpt/
├── cmd/
│   └── train/
│       └── main.go           # Training script
├── pkg/
│   ├── autograd/             # Autodiff engine
│   ├── backend/              # CPU/GPU backend abstraction
│   ├── nn/                   # Neural network layers
│   └── transformer/          # Transformer components
├── internal/
│   └── tokenizer/            # BPE tokenizer
└── docs/
    └── devlog.md             # Development log
```

## Quick Start

```bash
# Run tests
go test -v ./...

# Train the model (CPU)
go run ./cmd/train/main.go
```

## Dev Environment

- Go 1.21+
- CGO enabled (needed for MLX): `export CGO_ENABLED=1`
- macOS only: Xcode Command Line Tools (`xcode-select --install`)
- Git LFS (for `lib/mlx.metallib`): `git lfs install && git lfs pull --include lib/mlx.metallib`

## GPU Acceleration (Apple Silicon)

GoGPT supports GPU acceleration on Apple Silicon Macs using Apple's MLX framework.
MLX is pulled via Go modules; no local clone or CMake build is required.

### Prerequisites

1. **macOS on Apple Silicon** (M1/M2/M3/M4)
2. **Xcode Command Line Tools**: `xcode-select --install`
3. **CGO enabled**: `export CGO_ENABLED=1`

### Setup (Recommended)

```bash
./scripts/setup_mlx.sh
```

This downloads the prebuilt `libmlx.a` into `./lib` and (if present) copies
`mlx.metallib` into `./lib` for runtime loading.
If `lib/mlx.metallib` is a Git LFS pointer, run:

```bash
git lfs pull --include lib/mlx.metallib
```

If you already have a local MLX build, you can point the setup script at it:

```bash
MLX_METALLIB_SRC=/path/to/mlx.metallib ./scripts/setup_mlx.sh
```

Optional, if you want to force the backend:

```bash
export MLX_BACKEND=metal  # or auto (default), cpu
```

### Verify MLX Works

```bash
# Defaults to Metal on macOS ARM64
make mlx-check

# If you downloaded libmlx.a elsewhere
MLX_LIB_DIR=/path/to/lib make mlx-check
```

### Building with GPU Support

```bash
# Build the train binary with MLX support
make build-train

# Run training with the built binary
make run-train

# If you downloaded libmlx.a elsewhere
MLX_LIB_DIR=/path/to/lib make build-train
```

### Verifying GPU Usage

When running, the output will show which backend is being used:

```
=== GoGPT: Training a Small Transformer ===
Compute Backend: mlx-gpu (GPU: true)    # GPU acceleration active!
```

Without MLX built (or without `-tags=mlx`), it falls back to CPU:

```
Compute Backend: gonum-cpu (GPU: false)  # CPU mode
```

Troubleshooting:
- If you see `ld: library 'mlx' not found`, ensure `lib/libmlx.a` exists (run `./scripts/setup_mlx.sh`) or set `MLX_LIB_DIR` to the directory containing `libmlx.a`.
- If you see `SIGSEGV` right after "Using backend: Metal", ensure `lib/mlx.metallib` is a real binary (not a Git LFS pointer) and re-run `make mlx-check`.
- If it still shows `gonum-cpu`, rebuild with `-tags=mlx` and confirm `MLX_BACKEND=metal` (or `auto`) is set.

## Example Output

```
=== GoGPT: Training a Small Transformer ===
Compute Backend: gonum-cpu (GPU: false)

Training text: 100000 characters
Vocabulary: 556 tokens (500 BPE merges)

Model Configuration:
  Embed Dim: 128
  Num Heads: 4
  Num Layers: 4
  Context Window: 64

Epoch   0 | Loss: 6.3207 | LR: 0.000500
Epoch 100 | Loss: 4.5123 | LR: 0.001000
Epoch 500 | Loss: 2.8456 | LR: 0.000750
```

## Dependencies

- [gonum](https://gonum.org/) - Numerical library for matrix operations (CPU backend)
- [luxfi/mlx](https://github.com/luxfi/mlx) - Go bindings for Apple's MLX (optional, GPU backend)

## References

- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Apple MLX Framework](https://github.com/ml-explore/mlx)

## License

MIT
