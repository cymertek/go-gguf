package gguf

import (
	"bytes"
	"fmt"
)

// StreamOptions configures streaming operations ([StreamCopy], [StreamMerge]). All fields are optional;
// zero values mean "no filtering" or "passthrough".
//
// Include/Exclude use glob-style patterns (via [MatchPattern]) to select which tensors to copy.
// TargetType triggers on-the-fly requantization (dequantize then requantize). Transform allows
// custom per-tensor processing before writing.
type StreamOptions struct {
	// Include filters tensors: only tensors whose names match at least one pattern are copied.
	// Empty means include all tensors. Patterns use glob-style wildcards (* and ?).
	Include []string

	// Exclude filters tensors: tensors matching any pattern are skipped even if they passed Include.
	// Evaluated after Include; a tensor must pass both to be written.
	Exclude []string

	// TargetType, when set to a supported [GgmlType] different from the source type, triggers
	// on-the-fly requantization: each tensor is dequantized to float32 then re-quantized to
	// TargetType before being written. Empty (zero value) means passthrough -- no conversion.
	TargetType GgmlType

	// Transform is an optional per-tensor hook called with the raw tensor bytes and its [TensorInfo]
	// before writing. Return nil to skip the tensor entirely; return a non-nil []byte to replace
	// the data (e.g., for custom pruning or mixing). Called after any requantization if TargetType is set.
	Transform func(data []byte, info TensorInfo) ([]byte, error)
}

// StreamCopy reads all (optionally filtered) tensors from a source [*GGUF] and writes them to a
// [GGUFWriter]. It supports name-based filtering via [StreamOptions.Include]/[StreamOptions.Exclude],
// on-the-fly requantization via [StreamOptions.TargetType], and custom per-tensor transforms via
// [StreamOptions.Transform]. Returns the first error encountered during reading or writing.
//
// Example -- copy all tensors from a Q4_0 model to a new F32 file:
//
//	src, err := gguf.Open("model-q4.gguf")
//	if err != nil { log.Fatal(err) }
//	defer src.Close()
//	dst := gguf.Create(os.Stdout)
//	err = gguf.StreamCopy(dst, src, gguf.StreamOptions{TargetType: gguf.GgmlF32})
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

// StreamRequantize is a convenience wrapper around [StreamCopy] that converts every tensor from its
// source quantization to the given TargetType. Equivalent to calling StreamCopy with
// StreamOptions{TargetType: targetType}. Useful for downgrading model precision (e.g., Q8_0 -> Q4_K)
// or upgrading backends to a preferred format.
func StreamRequantize(dst *GGUFWriter, src *GGUF, targetType GgmlType) error {
	return StreamCopy(dst, src, StreamOptions{
		TargetType: targetType,
	})
}

// StreamMerge merges tensors from multiple [*GGUF] source files into a single [GGUFWriter].
// Metadata (KV pairs) is copied only from the first source; later sources' KV entries are skipped
// to avoid key conflicts. Tensor data from all sources is streamed sequentially with optional
// filtering and requantization via opts. Returns the first error encountered during reading or writing.
//
// Example -- merge two split shards into one file:
//
//	s1, _ := gguf.Open("shard-00001-of-00002.gguf")
//	s2, _ := gguf.Open("shard-00002-of-00002.gguf")
//	defer s1.Close(); defer s2.Close()
//	dst := gguf.Create(os.Stdout)
//	err := gguf.StreamMerge(dst, []*gguf.GGUF{s1, s2}, gguf.StreamOptions{})
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
