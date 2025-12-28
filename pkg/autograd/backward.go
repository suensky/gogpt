package autograd

// Backward performs reverse-mode automatic differentiation.
// It computes gradients for all nodes in the computation graph
// by traversing from this output node back to the input leaves.
func (v *Value) Backward() {
	// Build topological ordering
	order := topologicalSort(v)

	// Initialize output gradient to 1 (or appropriate shape)
	r, c := v.Shape()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v.Grad.Set(i, j, 1.0)
		}
	}

	// Backpropagate in reverse topological order
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.op != nil && len(node.parents) > 0 {
			node.op.Backward(node.Grad, node.parents, node)
		}
	}
}

// topologicalSort returns nodes in topological order using DFS.
// The output node will be last in the returned slice.
func topologicalSort(root *Value) []*Value {
	visited := make(map[*Value]bool)
	order := make([]*Value, 0)

	var visit func(node *Value)
	visit = func(node *Value) {
		if visited[node] {
			return
		}
		visited[node] = true

		// Visit all parents first
		for _, parent := range node.parents {
			visit(parent)
		}

		// Then add this node
		order = append(order, node)
	}

	visit(root)
	return order
}

// BackwardWithGrad performs backward pass with a custom initial gradient.
// Useful when this Value is not the final loss (e.g., for chained backward).
func (v *Value) BackwardWithGrad(initGrad *Value) {
	// Copy initial gradient
	r, c := v.Shape()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v.Grad.Set(i, j, initGrad.Data.At(i, j))
		}
	}

	// Build topological ordering
	order := topologicalSort(v)

	// Backpropagate in reverse topological order
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.op != nil && len(node.parents) > 0 {
			node.op.Backward(node.Grad, node.parents, node)
		}
	}
}
