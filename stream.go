package gguf

import (
	"bytes"
	"fmt"
	"io"
)

// StreamOptions configures streaming operations ([StreamCopy], [StreamMerge]). All fields are optional;
// zero values mean "no filtering" or "passthrough".
//
// Include/Exclude use glob-style patterns (via [MatchPattern]) to select which tensors to copy.
// Transform allows custom per-tensor processing before writing. For quantization changes, use the
// go-quant library ([github.com/cymertek/go-quant]) with [Tensor.Reader] to stream data through
// dequantize/requantize pipelines without loading full tensors into memory.
type StreamOptions struct {
	// Include filters tensors: only tensors whose names match at least one pattern are copied.
	// Empty means include all tensors. Patterns use glob-style wildcards (* and ?).
	Include []string

	// Exclude filters tensors: tensors matching any pattern are skipped even if they passed Include.
	// Evaluated after Include; a tensor must pass both to be written.
	Exclude []string

	// Transform is an optional per-tensor hook called with the raw tensor bytes and its [TensorInfo]
	// before writing. Return nil to skip the tensor entirely; return a non-nil []byte to replace
	// the data (e.g., for custom pruning or mixing). Called after any requantization if TargetType is set.
	Transform func(data []byte, info TensorInfo) ([]byte, error)
}

// StreamCopy reads all (optionally filtered) tensors from a source [*GGUF] and writes them to a
// [GGUFWriter]. It supports name-based filtering via [StreamOptions.Include]/[StreamOptions.Exclude]
// and custom per-tensor transforms via [StreamOptions.Transform]. Returns the first error encountered
// during reading or writing.
//
// For passthrough copies (no transform), data streams directly from source to writer through a 256 KB
// pooled buffer -- tensors larger than available RAM are handled without loading the full tensor into
// memory. Transforms require full data in memory since they must process every byte before writing.
//
// Example -- copy all weight tensors with filtering:
//
//	src, err := gguf.Open("model.gguf")
//	if err != nil { log.Fatal(err) }
//	defer src.Close()
//	dst := gguf.Create(os.Stdout)
//	err = gguf.StreamCopy(dst, src, gguf.StreamOptions{Include: []string{"*.weight"}})
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

		idx := dst.AddTensor(info.Name, info.Shape, info.GgmlType)

		// Apply transformations if needed (requires full data in memory)
		if opts.Transform != nil {
			reader := t.Reader()
			data, err := io.ReadAll(reader)
			if err != nil {
				return fmt.Errorf("gguf: read tensor %s for transform: %w", info.Name, err)
			}

			newData, err := opts.Transform(data, info)
			if err != nil {
				return err
			}
			if newData == nil {
				continue // Skip tensor
			}

			if err := dst.WriteTensorData(idx, bytes.NewReader(newData)); err != nil {
				return fmt.Errorf("gguf: write transformed tensor %s data: %w", info.Name, err)
			}
			continue
		}

		// Zero-copy passthrough: pass source reader directly to writer for deferred streaming at Close()
		if err := dst.WriteTensorData(idx, t.Reader()); err != nil {
			return fmt.Errorf("gguf: write tensor %s data: %w", info.Name, err)
		}
	}

	return nil
}

// StreamMerge merges tensors from multiple [*GGUF] source files into a single [GGUFWriter].
// Metadata (KV pairs) is copied only from the first source; later sources' KV entries are skipped
// to avoid key conflicts. Tensor data from all sources is streamed sequentially with optional
// filtering via opts. Returns the first error encountered during reading or writing.
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
