package gguf

import (
	"fmt"
	"path/filepath"
)

// MatchPattern returns true if name matches any of the given glob patterns.
// A pattern is a glob-style pattern with * and ? wildcards.
func MatchPattern(name string, patterns []string) bool {
	for _, pattern := range patterns {
		if match, _ := filepath.Match(pattern, name); match {
			return true
		}
	}
	return false
}

// DataForTensor returns raw data for a tensor by name.
func (g *GGUF) DataForTensor(name string) ([]byte, error) {
	tensors, err := g.Tensors()
	if err != nil {
		return nil, err
	}
	for _, t := range tensors {
		if t.Info().Name == name {
			data, err := t.Bytes()
			if err != nil {
				return nil, fmt.Errorf("gguf: read tensor %s: %w", name, err)
			}
			return data, nil
		}
	}
	return nil, fmt.Errorf("gguf: tensor %q not found", name)
}

// FindTensor returns the tensor with the given name, or an error if not found.
func (g *GGUF) FindTensor(name string) (*Tensor, error) {
	tensors, err := g.Tensors()
	if err != nil {
		return nil, err
	}
	for _, t := range tensors {
		if t.Info().Name == name {
			return t, nil
		}
	}
	return nil, fmt.Errorf("gguf: tensor %q not found", name)
}
