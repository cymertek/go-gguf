package gguf

import "fmt"

// CopyMetadataLazy copies KV entries from a lazy GGUF reader to a writer.
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

// FilterMetadataEntries returns a subset of KVEntry slice matching the given pattern.
func FilterMetadataEntries(entries []KVEntry, pattern string) []KVEntry {
	var keep []KVEntry
	for _, kv := range entries {
		if MatchPattern(kv.Key, []string{pattern}) {
			keep = append(keep, kv)
		}
	}
	return keep
}

// MergeMetadataEntries merges KV entries from src into dst.
// If dst already has a key, it is overwritten by src's value.
func MergeMetadataEntries(dst *GGUFWriter, src []KVEntry) {
	for _, e := range src {
		dst.SetKV(e.Key, e.Value)
	}
}
