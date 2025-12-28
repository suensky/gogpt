package nn

import (
	"math"

	"github.com/suensky/gogpt/pkg/autograd"
	"gonum.org/v1/gonum/mat"
)

// Optimizer is the interface for all optimizers.
type Optimizer interface {
	// Step performs a single optimization step
	Step()
	// ZeroGrad sets all parameter gradients to zero
	ZeroGrad()
}

// SGD implements Stochastic Gradient Descent optimizer.
type SGD struct {
	Params       []*autograd.Value
	LearningRate float64
}

// NewSGD creates a new SGD optimizer.
func NewSGD(params []*autograd.Value, lr float64) *SGD {
	return &SGD{
		Params:       params,
		LearningRate: lr,
	}
}

// Step performs a single optimization step: param = param - lr * grad
func (o *SGD) Step() {
	for _, p := range o.Params {
		r, c := p.Shape()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				newVal := p.Data.At(i, j) - o.LearningRate*p.Grad.At(i, j)
				p.Data.Set(i, j, newVal)
			}
		}
	}
}

// ZeroGrad resets all parameter gradients to zero.
func (o *SGD) ZeroGrad() {
	for _, p := range o.Params {
		p.ZeroGrad()
	}
}

// Adam implements the Adam optimizer.
// Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)
type Adam struct {
	Params       []*autograd.Value
	LearningRate float64
	Beta1        float64 // Exponential decay rate for first moment
	Beta2        float64 // Exponential decay rate for second moment
	Epsilon      float64 // Small constant for numerical stability

	// State
	m []*mat.Dense // First moment estimates
	v []*mat.Dense // Second moment estimates
	t int          // Timestep
}

// NewAdam creates a new Adam optimizer with default hyperparameters.
// lr: learning rate (default 0.001)
// Defaults: beta1=0.9, beta2=0.999, epsilon=1e-8
func NewAdam(params []*autograd.Value, lr float64) *Adam {
	return NewAdamWithParams(params, lr, 0.9, 0.999, 1e-8)
}

// NewAdamWithParams creates an Adam optimizer with custom hyperparameters.
func NewAdamWithParams(params []*autograd.Value, lr, beta1, beta2, epsilon float64) *Adam {
	m := make([]*mat.Dense, len(params))
	v := make([]*mat.Dense, len(params))

	for i, p := range params {
		r, c := p.Shape()
		m[i] = mat.NewDense(r, c, nil)
		v[i] = mat.NewDense(r, c, nil)
	}

	return &Adam{
		Params:       params,
		LearningRate: lr,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
		m:            m,
		v:            v,
		t:            0,
	}
}

// Step performs a single Adam optimization step.
func (o *Adam) Step() {
	o.t++

	for i, p := range o.Params {
		r, c := p.Shape()

		for row := 0; row < r; row++ {
			for col := 0; col < c; col++ {
				g := p.Grad.At(row, col)

				// Update biased first moment: m = beta1 * m + (1 - beta1) * g
				mVal := o.Beta1*o.m[i].At(row, col) + (1-o.Beta1)*g
				o.m[i].Set(row, col, mVal)

				// Update biased second moment: v = beta2 * v + (1 - beta2) * g^2
				vVal := o.Beta2*o.v[i].At(row, col) + (1-o.Beta2)*g*g
				o.v[i].Set(row, col, vVal)

				// Bias correction
				mHat := mVal / (1 - math.Pow(o.Beta1, float64(o.t)))
				vHat := vVal / (1 - math.Pow(o.Beta2, float64(o.t)))

				// Update parameter
				newVal := p.Data.At(row, col) - o.LearningRate*mHat/(math.Sqrt(vHat)+o.Epsilon)
				p.Data.Set(row, col, newVal)
			}
		}
	}
}

// ZeroGrad resets all parameter gradients to zero.
func (o *Adam) ZeroGrad() {
	for _, p := range o.Params {
		p.ZeroGrad()
	}
}

// SGDMomentum implements SGD with momentum.
type SGDMomentum struct {
	Params       []*autograd.Value
	LearningRate float64
	Momentum     float64

	// State
	velocity []*mat.Dense
}

// NewSGDMomentum creates SGD with momentum.
func NewSGDMomentum(params []*autograd.Value, lr, momentum float64) *SGDMomentum {
	velocity := make([]*mat.Dense, len(params))
	for i, p := range params {
		r, c := p.Shape()
		velocity[i] = mat.NewDense(r, c, nil)
	}

	return &SGDMomentum{
		Params:       params,
		LearningRate: lr,
		Momentum:     momentum,
		velocity:     velocity,
	}
}

// Step performs optimization with momentum.
func (o *SGDMomentum) Step() {
	for i, p := range o.Params {
		r, c := p.Shape()
		for row := 0; row < r; row++ {
			for col := 0; col < c; col++ {
				g := p.Grad.At(row, col)

				// v = momentum * v - lr * grad
				v := o.Momentum*o.velocity[i].At(row, col) - o.LearningRate*g
				o.velocity[i].Set(row, col, v)

				// param = param + v
				p.Data.Set(row, col, p.Data.At(row, col)+v)
			}
		}
	}
}

// ZeroGrad resets all gradients to zero.
func (o *SGDMomentum) ZeroGrad() {
	for _, p := range o.Params {
		p.ZeroGrad()
	}
}
