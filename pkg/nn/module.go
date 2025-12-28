// Package nn provides neural network layer implementations built on autograd.
package nn

import (
	"github.com/suensky/gogpt/pkg/autograd"
)

// Module is the interface for all neural network layers.
// Every layer must implement Forward and return its learnable parameters.
type Module interface {
	// Forward computes the output given input
	Forward(input *autograd.Value) *autograd.Value
	// Parameters returns all learnable parameters of this module
	Parameters() []*autograd.Value
}

// ModuleList is a container for multiple modules
type ModuleList struct {
	Modules []Module
}

// NewModuleList creates a new ModuleList
func NewModuleList(modules ...Module) *ModuleList {
	return &ModuleList{Modules: modules}
}

// Parameters returns all parameters from all modules
func (ml *ModuleList) Parameters() []*autograd.Value {
	params := make([]*autograd.Value, 0)
	for _, m := range ml.Modules {
		params = append(params, m.Parameters()...)
	}
	return params
}

// Add appends a module to the list
func (ml *ModuleList) Add(m Module) {
	ml.Modules = append(ml.Modules, m)
}

// Get returns the module at index i
func (ml *ModuleList) Get(i int) Module {
	return ml.Modules[i]
}

// Len returns the number of modules
func (ml *ModuleList) Len() int {
	return len(ml.Modules)
}
