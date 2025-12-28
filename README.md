# GoGPT

A from-scratch implementation of automatic differentiation and GPT-style transformer decoder in pure Go. No Python, no PyTorch—just Go and matrix math.

## Features

- **Autograd Engine** — Reverse-mode automatic differentiation supporting Add, Sub, Mul, MatMul, ReLU, Softmax, GELU, Tanh, Sigmoid, and more
- **Neural Network Layers** — Linear, Embedding, LayerNorm with Xavier initialization
- **Optimizers** — SGD, Adam, AdamW (recommended for transformers)
- **Loss Functions** — CrossEntropyLoss, CrossEntropyLossWithSmoothing, MSELoss, NLLLoss
- **Transformer Components** — Multi-head self-attention with causal masking, FFN, residual connections, full GPT model with text generation
- **BPE Tokenizer** — Byte Pair Encoding tokenizer for subword tokenization
- **Backend Abstraction** — GoNum (CPU) and MLX (Apple Silicon GPU)

---

## Development Environment Setup

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Go 1.21+** | Required |
| **CGO** | Enable with `export CGO_ENABLED=1` |
| **macOS** | Xcode Command Line Tools: `xcode-select --install` |
| **Git LFS** | For `lib/mlx.metallib`: `git lfs install && git lfs pull --include lib/mlx.metallib` |

### Build & Run

```bash
# Clone the repository
git clone https://github.com/suensky/gogpt.git
cd gogpt

# Run tests
go test -v ./...

# Train on CPU (simple)
go run ./cmd/train/main.go
```

### GPU Acceleration (Apple Silicon)

GoGPT supports GPU acceleration on Apple Silicon (M1/M2/M3/M4) via Apple's MLX framework.

```bash
# 1. Setup MLX (downloads prebuilt libmlx.a)
./scripts/setup_mlx.sh

# 2. Pull the Metal library if using Git LFS
git lfs pull --include lib/mlx.metallib

# 3. Build with MLX support
make build-train

# 4. Run training
make run-train
```

**Verify GPU is active:**

```
Compute Backend: mlx-gpu (GPU: true)    # GPU acceleration active!
```

If you see `gonum-cpu (GPU: false)`, rebuild with `-tags=mlx` and confirm `MLX_BACKEND=metal` is set.

### Project Structure

```
gogpt/
├── cmd/train/          # Training script
├── pkg/
│   ├── autograd/       # Autodiff engine
│   ├── backend/        # CPU/GPU backend abstraction
│   ├── nn/             # Neural network layers
│   └── transformer/    # Transformer components (attention, GPT)
├── internal/tokenizer/ # BPE tokenizer
├── data/               # Training data (jules_verne.txt)
├── lib/                # MLX libraries (libmlx.a, mlx.metallib)
└── scripts/            # Setup scripts
```

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make build-train` | Build training binary with MLX support |
| `make run-train` | Run the training binary |
| `make mlx-check` | Verify MLX backend is working |

---

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

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ld: library 'mlx' not found` | Run `./scripts/setup_mlx.sh` or set `MLX_LIB_DIR` |
| `SIGSEGV` after "Using backend: Metal" | Ensure `lib/mlx.metallib` is real binary (not LFS pointer) |
| Still shows `gonum-cpu` | Rebuild with `-tags=mlx`, set `MLX_BACKEND=metal` |

---

## Dependencies

- [gonum](https://gonum.org/) — Numerical library for matrix operations (CPU backend)
- [luxfi/mlx](https://github.com/luxfi/mlx) — Go bindings for Apple's MLX (optional GPU backend)

## References

- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Apple MLX Framework](https://github.com/ml-explore/mlx)

## License

MIT
