package gguf

import "fmt"

// CopyMetadataLazy copies metadata (KV pairs) from a lazy [*GGUF] reader to a [GGUFWriter], applying
// optional glob-style filters via Include and Exclude patterns. Entries that fail to load are silently
// skipped. Use this when you need to transfer model configuration without copying tensor data.
//
// Example -- copy only architecture-related metadata:
//
//	err := gguf.CopyMetadataLazy(dst, src, []string{"general.*"}, nil)
func CopyMetadataLazy(dst *GGUFWriter, src *GGUF, include, exclude []string) error {
	metaEntries, err := src.Metadata()
	if err != nil {
		return fmt.Errorf("gguf: get metadata: %w", err)
	}

	for _, e := range metaEntries {
		var match bool

		// Check include pattern
		if len(include) > 0 {
			for _, pattern := range include {
				if match = MatchPattern(e.Name(), []string{pattern}); match {
					break
				}
			}
			if !match {
				continue
			}
		}

		// Check exclude pattern
		if len(exclude) > 0 {
			for _, pattern := range exclude {
				if match = MatchPattern(e.Name(), []string{pattern}); match {
					match = false
					break
				}
			}
			if !match {
				continue
			}
		}

		v, err := e.Value()
		if err != nil {
			continue // Skip entries that fail to load
		}
		dst.SetKV(e.Name(), v)
	}

	return nil
}

// FilterMetadataEntries returns a new slice containing only [KVEntry] items whose Key matches the
// given glob-style pattern (via [MatchPattern]). The original entries slice is not modified.
// Returns nil if no entries match.
//
// Example -- keep only general.* keys:
//
//	keep := gguf.FilterMetadataEntries(allMeta, "general.*")
func FilterMetadataEntries(entries []KVEntry, pattern string) []KVEntry {
	var keep []KVEntry
	for _, kv := range entries {
		if MatchPattern(kv.Key, []string{pattern}) {
			keep = append(keep, kv)
		}
	}
	return keep
}

// MergeMetadataEntries copies all [KVEntry] pairs from src into the writer dst via [GGUFWriter.SetKV].
// If dst already contains a key with the same name, its value is overwritten by src's. Call before
// [GGUFWriter.Close]; the entries are not validated beyond being set on the writer.
func MergeMetadataEntries(dst *GGUFWriter, src []KVEntry) {
	for _, e := range src {
		dst.SetKV(e.Key, e.Value)
	}
}
