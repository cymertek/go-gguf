package gguf

import (
	"bytes"
	"fmt"
)

// StreamOptions configures streaming operations.
type StreamOptions struct {
	// Filter tensors by name pattern. Empty = include all.
	Include []string
	Exclude []string

	// Requantize tensors to target type. Empty = passthrough (stream copy).
	TargetType GgmlType

	// Custom per-tensor hook called before writing. Return nil to write,
	// non-nil to skip. Called with dequantized data so researchers can
	// implement custom logic.
	Transform func(data []byte, info TensorInfo) ([]byte, error)
}

// StreamCopy reads tensors from srcGGUF and writes them to dstWriter.
// Supports filtering, skipping, and stream passthrough.
func StreamCopy(dst *GGUFWriter, src *GGUF, opts StreamOptions) error {
	tensors, err := src.Tensors()
	if err != nil {
		return fmt.Errorf("gguf: get tensors: %w", err)
	}

	for _, t := range tensors {
		info := t.Info()

		// Check if tensor should be included
		match := len(opts.Include) == 0 || matchTensorName(info.Name, opts.Include)
		if !match && len(opts.Exclude) > 0 {
			match = !matchTensorName(info.Name, opts.Exclude)
		}
		if !match {
			continue
		}

		// Get raw tensor data
		data, err := t.Bytes()
		if err != nil {
			return fmt.Errorf("gguf: read tensor %s: %w", info.Name, err)
		}

		// Apply transformations
		if opts.Transform != nil {
			var transformErr error
			data, transformErr = opts.Transform(data, info)
			if transformErr != nil {
				return transformErr
			}
			if data == nil {
				continue // Skip tensor
			}
		}

		// Apply requantization
		if opts.TargetType.IsSupported() && opts.TargetType != info.GgmlType {
			// Dequantize data
			floats, err := Dequant(data, info.GgmlType)
			if err != nil {
				return fmt.Errorf("gguf: dequant %s: %w", info.Name, err)
			}
			// Requantize to target type
			newData, err := Requant(floats, opts.TargetType)
			if err != nil {
				return fmt.Errorf("gguf: requant %s: %w", info.Name, err)
			}
			data = newData
		}

		// Add tensor to writer
		tinfo := info
		if opts.TargetType != info.GgmlType {
			tinfo.GgmlType = opts.TargetType
		}
		idx := dst.AddTensor(info.Name, tinfo.Shape, tinfo.GgmlType)

		// Create reader from data bytes
		reader := bytes.NewReader(data)
		if err := dst.WriteTensorData(idx, reader); err != nil {
			return fmt.Errorf("gguf: write tensor %s data: %w", info.Name, err)
		}
	}

	return nil
}

// StreamRequantize reads tensors from srcGGUF and writes them to dstWriter with requantization.
func StreamRequantize(dst *GGUFWriter, src *GGUF, targetType GgmlType) error {
	return StreamCopy(dst, src, StreamOptions{
		TargetType: targetType,
	})
}

// StreamMerge merges tensors from multiple sources into a single dstWriter.
// KV metadata from first source is used; later sources' KV entries are skipped to avoid conflicts.
func StreamMerge(dst *GGUFWriter, sources []*GGUF, opts StreamOptions) error {
	// Copy metadata from first source
	if len(sources) > 0 {
		metaEntries, err := sources[0].Metadata()
		if err != nil {
			return fmt.Errorf("gguf: get metadata: %w", err)
		}
		for _, e := range metaEntries {
			v, err := e.Value()
			if err != nil {
				continue // Skip entries that fail to load
			}
			dst.SetKV(e.Name(), v)
		}
	}

	// Stream tensors from all sources
	for _, src := range sources {
		if err := StreamCopy(dst, src, opts); err != nil {
			return err
		}
	}

	return nil
}

// matchTensorName checks if a tensor name matches any of the given patterns.
func matchTensorName(name string, patterns []string) bool {
	for _, pattern := range patterns {
		if MatchPattern(name, []string{pattern}) {
			return true
		}
	}
	return false
}
