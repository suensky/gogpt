#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_LIB_DIR="${MLX_LIB_DIR:-$REPO_ROOT/lib}"
MLX_METALLIB_SRC="${MLX_METALLIB_SRC:-$HOME/src/mlx/build/mlx/backend/metal/kernels/mlx.metallib}"
MLX_LIB_URL="https://github.com/luxfi/mlx/releases/latest/download/libmlx-macos-arm64.tar.gz"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: MLX Metal backend is supported on macOS only."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: MLX Metal backend requires Apple Silicon (arm64)."
  exit 1
fi

if ! command -v go >/dev/null 2>&1; then
  echo "ERROR: Go not found. Install Go 1.21+."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl not found."
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "ERROR: tar not found."
  exit 1
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "ERROR: Xcode tools not found (missing 'metal')."
  echo "Install Xcode Command Line Tools:"
  echo "  xcode-select --install"
  echo "If Xcode is installed, select it:"
  echo "  sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer"
  exit 1
fi

cgo_enabled="$(go env CGO_ENABLED)"
if [[ "$cgo_enabled" != "1" ]]; then
  echo "WARN: CGO is disabled. Enable it for MLX builds:"
  echo "  export CGO_ENABLED=1"
fi

echo "==> Downloading prebuilt MLX library..."
mkdir -p "$MLX_LIB_DIR"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
archive="$tmp_dir/libmlx-macos-arm64.tar.gz"
curl -fsSL "$MLX_LIB_URL" -o "$archive"
tar -xzf "$archive" -C "$MLX_LIB_DIR"

if [[ ! -f "$MLX_LIB_DIR/libmlx.a" ]]; then
  echo "ERROR: libmlx.a not found in $MLX_LIB_DIR"
  exit 1
fi

if [[ ! -f "$MLX_LIB_DIR/mlx.metallib" && -f "$MLX_METALLIB_SRC" ]]; then
  echo "==> Copying mlx.metallib from local MLX build..."
  cp "$MLX_METALLIB_SRC" "$MLX_LIB_DIR/mlx.metallib"
fi

if [[ ! -f "$MLX_LIB_DIR/mlx.metallib" ]]; then
  echo "WARN: mlx.metallib not found in $MLX_LIB_DIR."
  echo "      If you have a local MLX build, set:"
  echo "        MLX_METALLIB_SRC=/path/to/mlx.metallib ./scripts/setup_mlx.sh"
elif head -n 1 "$MLX_LIB_DIR/mlx.metallib" | grep -q "git-lfs"; then
  echo "WARN: mlx.metallib is a Git LFS pointer."
  echo "      Run: git lfs pull --include lib/mlx.metallib"
fi

echo "==> MLX Go bindings use the module cache; no local MLX clone or CMake build required."
echo "==> Recommended environment:"
echo "  export CGO_ENABLED=1"
echo "  export MLX_BACKEND=metal  # or auto (default)"
echo "==> Smoke test:"
echo "  MLX_LIB_DIR=\"$MLX_LIB_DIR\" make mlx-check"
echo "==> Training with GPU:"
echo "  MLX_LIB_DIR=\"$MLX_LIB_DIR\" make build-train"
echo "  MLX_BACKEND=metal ./train"
