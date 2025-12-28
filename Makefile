.PHONY: check-metallib mlx-check build-train run-train

MLX_BACKEND ?= metal
MLX_LIB_DIR ?= $(CURDIR)/lib
MLX_METALLIB ?= $(MLX_LIB_DIR)/mlx.metallib
TRAIN_TAGS ?= mlx
TRAIN_BIN ?= train
MLX_CHECK_BIN ?= bin/mlx_smoke

check-metallib:
	if [ "$(MLX_BACKEND)" = "metal" ]; then \
		if [ ! -f "$(MLX_METALLIB)" ]; then \
			echo "ERROR: $(MLX_METALLIB) not found. Run: git lfs pull --include $(MLX_METALLIB)"; \
			exit 1; \
		fi; \
		if head -n 1 "$(MLX_METALLIB)" | grep -q "git-lfs"; then \
			echo "ERROR: $(MLX_METALLIB) is a Git LFS pointer. Run: git lfs pull --include $(MLX_METALLIB)"; \
			exit 1; \
		fi; \
	fi

mlx-check: check-metallib
	mkdir -p $(dir $(MLX_CHECK_BIN))
	CGO_ENABLED=1 CGO_LDFLAGS="-L$(MLX_LIB_DIR)" go build -o $(MLX_CHECK_BIN) ./cmd/mlx_smoke
	if [ -f "$(MLX_METALLIB)" ]; then cp "$(MLX_METALLIB)" "$(dir $(MLX_CHECK_BIN))"; fi
	MLX_BACKEND=$(MLX_BACKEND) ./$(MLX_CHECK_BIN)

build-train:
	CGO_ENABLED=1 CGO_LDFLAGS="-L$(MLX_LIB_DIR)" go build -tags=$(TRAIN_TAGS) -o $(TRAIN_BIN) ./cmd/train
	if [ -f "$(MLX_METALLIB)" ]; then cp "$(MLX_METALLIB)" "$(dir $(TRAIN_BIN))"; fi

run-train: check-metallib
	MLX_BACKEND=$(MLX_BACKEND) ./$(TRAIN_BIN)
