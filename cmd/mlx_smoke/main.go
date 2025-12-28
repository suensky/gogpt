package main

import (
	"fmt"

	"github.com/luxfi/mlx"
)

func main() {
	// Auto-detect best backend (Metal on macOS ARM64).
	fmt.Printf("Using backend: %s\n", mlx.GetBackend())

	a := mlx.Random([]int{100, 100}, mlx.Float32)
	b := mlx.Random([]int{100, 100}, mlx.Float32)
	c := mlx.MatMul(a, b)

	mlx.Eval(c)
	mlx.Synchronize()

	fmt.Println("Matrix multiplication completed!")
}
