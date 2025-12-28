package backend

import (
	"testing"
)

func TestGoNumBackendBasicOps(t *testing.T) {
	be := NewGoNumBackend()

	t.Run("Name", func(t *testing.T) {
		if be.Name() != "gonum-cpu" {
			t.Errorf("expected 'gonum-cpu', got %s", be.Name())
		}
		if be.IsGPU() {
			t.Error("gonum backend should not report as GPU")
		}
	})

	t.Run("Zeros", func(t *testing.T) {
		z := be.Zeros(3, 4)
		r, c := z.Shape()
		if r != 3 || c != 4 {
			t.Errorf("expected 3x4, got %dx%d", r, c)
		}
		if z.At(1, 2) != 0 {
			t.Error("zeros should be 0")
		}
	})

	t.Run("Ones", func(t *testing.T) {
		o := be.Ones(2, 3)
		if o.At(1, 1) != 1 {
			t.Error("ones should be 1")
		}
	})

	t.Run("FromSlice", func(t *testing.T) {
		data := []float64{1, 2, 3, 4, 5, 6}
		m := be.FromSlice(data, 2, 3)
		if m.At(0, 0) != 1 || m.At(1, 2) != 6 {
			t.Error("FromSlice data mismatch")
		}
	})
}

func TestGoNumBackendArithmetic(t *testing.T) {
	be := NewGoNumBackend()

	a := be.FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	b := be.FromSlice([]float64{5, 6, 7, 8}, 2, 2)

	t.Run("Add", func(t *testing.T) {
		c := be.Add(a, b)
		if c.At(0, 0) != 6 || c.At(1, 1) != 12 {
			t.Error("Add result mismatch")
		}
	})

	t.Run("Sub", func(t *testing.T) {
		c := be.Sub(b, a)
		if c.At(0, 0) != 4 || c.At(1, 1) != 4 {
			t.Error("Sub result mismatch")
		}
	})

	t.Run("Mul", func(t *testing.T) {
		c := be.Mul(a, b)
		if c.At(0, 0) != 5 || c.At(1, 1) != 32 {
			t.Error("Mul result mismatch")
		}
	})

	t.Run("Scale", func(t *testing.T) {
		c := be.Scale(a, 2)
		if c.At(0, 0) != 2 || c.At(1, 1) != 8 {
			t.Error("Scale result mismatch")
		}
	})
}

func TestGoNumBackendMatMul(t *testing.T) {
	be := NewGoNumBackend()

	// 2x3 @ 3x2 = 2x2
	a := be.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := be.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 3, 2)

	c := be.MatMul(a, b)
	r, cols := c.Shape()
	if r != 2 || cols != 2 {
		t.Errorf("expected 2x2, got %dx%d", r, cols)
	}

	// [1,2,3] @ [1,2; 3,4; 5,6] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
	if c.At(0, 0) != 22 || c.At(0, 1) != 28 {
		t.Errorf("MatMul result mismatch: got [%.0f, %.0f]", c.At(0, 0), c.At(0, 1))
	}
}

func TestGoNumBackendTranspose(t *testing.T) {
	be := NewGoNumBackend()

	a := be.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	aT := be.Transpose(a)

	r, c := aT.Shape()
	if r != 3 || c != 2 {
		t.Errorf("expected 3x2, got %dx%d", r, c)
	}

	if aT.At(0, 0) != 1 || aT.At(0, 1) != 4 || aT.At(2, 0) != 3 {
		t.Error("Transpose values mismatch")
	}
}

func TestGoNumBackendReductions(t *testing.T) {
	be := NewGoNumBackend()

	a := be.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)

	t.Run("Sum all", func(t *testing.T) {
		s := be.Sum(a)
		if s.At(0, 0) != 21 {
			t.Errorf("expected 21, got %.0f", s.At(0, 0))
		}
	})

	t.Run("Sum axis 0", func(t *testing.T) {
		s := be.Sum(a, 0)
		// Sum down rows: [1+4, 2+5, 3+6] = [5, 7, 9]
		if s.At(0, 0) != 5 || s.At(0, 2) != 9 {
			t.Error("Sum axis 0 mismatch")
		}
	})

	t.Run("Sum axis 1", func(t *testing.T) {
		s := be.Sum(a, 1)
		// Sum across cols: [1+2+3, 4+5+6] = [6, 15]
		if s.At(0, 0) != 6 || s.At(1, 0) != 15 {
			t.Error("Sum axis 1 mismatch")
		}
	})

	t.Run("Mean all", func(t *testing.T) {
		m := be.Mean(a)
		if m.At(0, 0) != 3.5 {
			t.Errorf("expected 3.5, got %.2f", m.At(0, 0))
		}
	})

	t.Run("Argmax", func(t *testing.T) {
		indices := be.Argmax(a, 1)
		// Row 0: max at col 2 (value 3)
		// Row 1: max at col 2 (value 6)
		if indices[0] != 2 || indices[1] != 2 {
			t.Errorf("expected [2, 2], got %v", indices)
		}
	})
}

func TestGoNumBackendSoftmax(t *testing.T) {
	be := NewGoNumBackend()

	a := be.FromSlice([]float64{1, 2, 3, 1, 2, 3}, 2, 3)
	s := be.Softmax(a, 1)

	// Each row should sum to 1
	for i := 0; i < 2; i++ {
		sum := s.At(i, 0) + s.At(i, 1) + s.At(i, 2)
		if sum < 0.999 || sum > 1.001 {
			t.Errorf("row %d sum: expected 1, got %.4f", i, sum)
		}
	}

	// Larger values should have higher probabilities
	if s.At(0, 2) <= s.At(0, 1) || s.At(0, 1) <= s.At(0, 0) {
		t.Error("softmax should preserve ordering")
	}
}

func TestGoNumBackendActivations(t *testing.T) {
	be := NewGoNumBackend()

	t.Run("ReLU", func(t *testing.T) {
		a := be.FromSlice([]float64{-2, -1, 0, 1, 2, 3}, 2, 3)
		r := be.ReLU(a)
		if r.At(0, 0) != 0 || r.At(0, 2) != 0 || r.At(1, 2) != 3 {
			t.Error("ReLU mismatch")
		}
	})

	t.Run("GELU", func(t *testing.T) {
		a := be.FromSlice([]float64{0, 1, 2}, 1, 3)
		g := be.GELU(a)
		// GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.954
		if g.At(0, 0) > 0.01 {
			t.Errorf("GELU(0) should be ~0, got %.4f", g.At(0, 0))
		}
		if g.At(0, 1) < 0.8 || g.At(0, 1) > 0.9 {
			t.Errorf("GELU(1) should be ~0.84, got %.4f", g.At(0, 1))
		}
	})
}

func TestGoNumBackendClone(t *testing.T) {
	be := NewGoNumBackend()

	a := be.FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	b := a.Clone()

	// Modify original
	a.Set(0, 0, 100)

	// Clone should be unchanged
	if b.At(0, 0) != 1 {
		t.Error("Clone should be a deep copy")
	}
}

func TestAutoSelectBackend(t *testing.T) {
	be, err := AutoSelectBackend()
	if err != nil {
		t.Errorf("AutoSelectBackend should always succeed: %v", err)
	}

	// Should return some backend
	if be == nil {
		t.Error("AutoSelectBackend returned nil backend")
	}

	// Name should be non-empty
	if be.Name() == "" {
		t.Error("backend name should not be empty")
	}
}

func TestDefaultBackend(t *testing.T) {
	be := Default()
	if be == nil {
		t.Error("Default() should never return nil")
	}

	// Should be gonum by default
	if be.Name() != "gonum-cpu" {
		t.Logf("Default backend is %s (may be MLX on Apple Silicon)", be.Name())
	}
}
