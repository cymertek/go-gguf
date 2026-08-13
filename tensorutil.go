package gguf

import (
	"fmt"
	"io"
	"path/filepath"
)

// MatchPattern returns true if [name] matches any of the provided glob-style [patterns]. Each pattern
// supports * (match any characters) and ? (match single character), following filepath.Match semantics.
// Returns false if patterns is empty or no pattern matches. Used internally by [StreamOptions.Include]/[Exclude]
// and [CopyMetadataLazy] for tensor/metadata name filtering.
func MatchPattern(name string, patterns []string) bool {
	for _, pattern := range patterns {
		if match, _ := filepath.Match(pattern, name); match {
			return true
		}
	}
	return false
}

// DataForTensor looks up the tensor named [name] in this [*GGUF], reads its entire raw data from file
// via streaming (no full allocation), and returns it as a byte slice. Returns an error if the tensor
// is not found or cannot be read. For partial reads use [Tensor.ReadAt] instead to avoid loading the
// full tensor into memory.
func (g *GGUF) DataForTensor(name string) ([]byte, error) {
	tensors, err := g.Tensors()
	if err != nil {
		return nil, err
	}
	for _, t := range tensors {
		if t.Info().Name == name {
			r := t.Reader()
			data, err := io.ReadAll(r)
			if err != nil {
				return nil, fmt.Errorf("gguf: read tensor %s: %w", name, err)
			}
			return data, nil
		}
	}
	return nil, fmt.Errorf("gguf: tensor %q not found", name)
}

// FindTensor looks up and returns the [*Tensor] handle for the tensor named [name]. Returns an error
// if no tensor with that name exists in this GGUF file. The returned *Tensor is valid until [GGUF.Close]
// or [Tensor.Close]; use [Tensor.Info] to inspect its metadata, [Tensor.ReadAt] / [Tensor.Reader] for data access.
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
