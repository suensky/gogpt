package nn

import (
	"math"

	"github.com/suensky/gogpt/pkg/autograd"
	"gonum.org/v1/gonum/mat"
)

// AdamW optimizer implements Adam with decoupled weight decay.
// This is the preferred optimizer for transformer training.
// Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
type AdamW struct {
	LR          float64 // Learning rate
	Beta1       float64 // Exponential decay rate for first moment (default: 0.9)
	Beta2       float64 // Exponential decay rate for second moment (default: 0.999)
	Eps         float64 // Small constant for numerical stability (default: 1e-8)
	WeightDecay float64 // Weight decay coefficient (default: 0.01)

	// State
	m map[*autograd.Value]*mat.Dense // First moment estimates
	v map[*autograd.Value]*mat.Dense // Second moment estimates
	t int                            // Timestep
}

// NewAdamW creates a new AdamW optimizer with the given learning rate.
func NewAdamW(lr float64) *AdamW {
	return &AdamW{
		LR:          lr,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 0.01,
		m:           make(map[*autograd.Value]*mat.Dense),
		v:           make(map[*autograd.Value]*mat.Dense),
		t:           0,
	}
}

// NewAdamWWithParams creates a new AdamW optimizer with custom parameters.
func NewAdamWWithParams(lr, beta1, beta2, eps, weightDecay float64) *AdamW {
	return &AdamW{
		LR:          lr,
		Beta1:       beta1,
		Beta2:       beta2,
		Eps:         eps,
		WeightDecay: weightDecay,
		m:           make(map[*autograd.Value]*mat.Dense),
		v:           make(map[*autograd.Value]*mat.Dense),
		t:           0,
	}
}

// Step performs a single optimization step.
func (opt *AdamW) Step(params []*autograd.Value) {
	opt.t++

	for _, p := range params {
		r, c := p.Shape()

		// Initialize moment estimates if needed
		if opt.m[p] == nil {
			opt.m[p] = mat.NewDense(r, c, nil)
			opt.v[p] = mat.NewDense(r, c, nil)
		}

		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				grad := p.Grad.At(i, j)
				param := p.Data.At(i, j)

				// Update biased first moment estimate
				m := opt.Beta1*opt.m[p].At(i, j) + (1-opt.Beta1)*grad
				opt.m[p].Set(i, j, m)

				// Update biased second raw moment estimate
				v := opt.Beta2*opt.v[p].At(i, j) + (1-opt.Beta2)*grad*grad
				opt.v[p].Set(i, j, v)

				// Compute bias-corrected first moment estimate
				mHat := m / (1 - math.Pow(opt.Beta1, float64(opt.t)))

				// Compute bias-corrected second raw moment estimate
				vHat := v / (1 - math.Pow(opt.Beta2, float64(opt.t)))

				// AdamW: decoupled weight decay (applied to param, not gradient)
				// Update parameter
				update := opt.LR * (mHat/(math.Sqrt(vHat)+opt.Eps) + opt.WeightDecay*param)
				p.Data.Set(i, j, param-update)
			}
		}
	}
}

// ZeroGrad resets gradients for all parameters.
func (opt *AdamW) ZeroGrad(params []*autograd.Value) {
	for _, p := range params {
		p.ZeroGrad()
	}
}

// SetLR sets the learning rate (useful for learning rate scheduling).
func (opt *AdamW) SetLR(lr float64) {
	opt.LR = lr
}

// GetLR returns the current learning rate.
func (opt *AdamW) GetLR() float64 {
	return opt.LR
}

// Reset resets the optimizer state (useful when restarting training).
func (opt *AdamW) Reset() {
	opt.m = make(map[*autograd.Value]*mat.Dense)
	opt.v = make(map[*autograd.Value]*mat.Dense)
	opt.t = 0
}

// LRScheduler provides learning rate scheduling functions.
type LRScheduler struct {
	BaseLR      float64
	WarmupSteps int
	TotalSteps  int
	MinLR       float64
	CurrentStep int
}

// NewCosineScheduler creates a cosine annealing scheduler with warmup.
func NewCosineScheduler(baseLR float64, warmupSteps, totalSteps int, minLR float64) *LRScheduler {
	return &LRScheduler{
		BaseLR:      baseLR,
		WarmupSteps: warmupSteps,
		TotalSteps:  totalSteps,
		MinLR:       minLR,
		CurrentStep: 0,
	}
}

// Step advances the scheduler and returns the new learning rate.
func (s *LRScheduler) Step() float64 {
	s.CurrentStep++
	return s.GetLR()
}

// GetLR computes the learning rate for the current step.
func (s *LRScheduler) GetLR() float64 {
	if s.CurrentStep < s.WarmupSteps {
		// Linear warmup
		return s.BaseLR * float64(s.CurrentStep) / float64(s.WarmupSteps)
	}

	// Cosine annealing
	progress := float64(s.CurrentStep-s.WarmupSteps) / float64(s.TotalSteps-s.WarmupSteps)
	if progress > 1 {
		progress = 1
	}

	return s.MinLR + 0.5*(s.BaseLR-s.MinLR)*(1+math.Cos(math.Pi*progress))
}

// Reset resets the scheduler to the beginning.
func (s *LRScheduler) Reset() {
	s.CurrentStep = 0
}
