package gguf

import (
	"fmt"
	"os"
)

// NewReader opens a GGUF file by path and returns a lazy reader.
// This is the recommended entry point for most use cases.
func NewReader(path string) (*GGUF, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open %q: %w", path, err)
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: stat %q: %w", path, err)
	}

	g, err := OpenFromReader(f, info.Size())
	if err != nil {
		f.Close()
		return nil, err
	}
	return g, nil
}
