package gguf

import "path/filepath"

// MatchPattern returns true if [name] matches any of the provided glob-style [patterns]. Each pattern
// supports * (match any characters) and ? (match single character), following filepath.Match semantics.
// Returns false if patterns is empty or no pattern matches. Used internally by [StreamOptions.Include]/[Exclude]
// for tensor name filtering.
func MatchPattern(name string, patterns []string) bool {
	for _, pattern := range patterns {
		if match, _ := filepath.Match(pattern, name); match {
			return true
		}
	}
	return false
}
