package gguf

import (
	"fmt"
	"os"
	"testing"
)

// TestMultiShardSplit verifies that multi-shard GGUF files can be opened and read correctly.
func TestMultiShardSplit(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping multi-shard test in short mode")
	}

	// Use TestModel-V4 split files as test data — pass all shards explicitly (no auto-detection)
	basePath := "/workdir/TestModel-V4-Flash-0731-UD-IQ1_S"
	shardPaths := []string{
		fmt.Sprintf("%s-00001-of-00003.gguf", basePath),
		fmt.Sprintf("%s-00002-of-00003.gguf", basePath),
		fmt.Sprintf("%s-00003-of-00003.gguf", basePath),
	}

	// Verify all shard files exist
	for _, path := range shardPaths {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Skipf("Split file not found: %s, skipping test", path)
		}
	}

	g, err := NewReaderFile(shardPaths...)
	if err != nil {
		t.Fatalf("Failed to open split files: %v", err)
	}
	defer g.Close()

	// Verify it's detected as a split file
	if !g.IsSplit() {
		t.Fatal("Expected multi-shard GGUF, but got single-file")
	}

	splitInfo := g.SplitInfo()
	if splitInfo == nil {
		t.Fatal("SplitInfo should not be nil for split files")
	}

	if splitInfo.Count != 3 {
		t.Errorf("Expected 3 shards, got %d", splitInfo.Count)
	}

	// Verify tensor counts per shard match expected values
	expectedTensors := []uint64{0, 812, 516}
	for i, expected := range expectedTensors {
		if splitInfo.TensorsPerShard[i] != expected {
			t.Errorf("Shard %d: expected %d tensors in header, got %d", i+1, expected, splitInfo.TensorsPerShard[i])
		}
	}

	// Read metadata from shard 0 (metadata-only shard)
	if len(splitInfo.Shards) == 0 {
		t.Fatal("No shards found")
	}

	metadata, err := splitInfo.Shards[0].GetMetadata()
	if err != nil {
		t.Fatalf("Failed to read metadata from shard 1: %v", err)
	}

	// Verify we got the expected KV entries
	if len(metadata) == 0 {
		t.Fatal("Expected at least one metadata entry from shard 1")
	}

	// Check for split indicators in metadata
	hasSplitNo := false
	hasSplitCount := false
	for _, entry := range metadata {
		switch entry.Name() {
		case "split.no":
			hasSplitNo = true
		case "split.count":
			hasSplitCount = true
		}
	}

	if !hasSplitNo {
		t.Error("Missing 'split.no' in metadata")
	}
	if !hasSplitCount {
		t.Error("Missing 'split.count' in metadata")
	}

	// Read tensors from shards 2 and 3
	totalTensors := 0
	for i, shard := range splitInfo.Shards[1:] { // Skip shard 0 (metadata-only)
		tensors, err := shard.Tensors()
		if err != nil {
			t.Errorf("Failed to read tensors from shard %d: %v", i+2, err)
			continue
		}

		expectedCount := expectedTensors[i+1]
		if uint64(len(tensors)) != expectedCount {
			t.Errorf("Shard %d: expected %d tensors, got %d", i+2, expectedCount, len(tensors))
		}

		totalTensors += len(tensors)

		// Verify first and last tensor names for sanity
		if len(tensors) > 0 {
			t.Logf("Shard %d: first=%s, last=%s", i+2, tensors[0].Info().Name, tensors[len(tensors)-1].Info().Name)
		}
	}

	expectedTotal := uint64(0)
	for _, count := range expectedTensors {
		expectedTotal += count
	}

	if uint64(totalTensors) != expectedTotal {
		t.Errorf("Total tensors mismatch: expected %d, got %d", expectedTotal, totalTensors)
	}

	fmt.Printf("\n=== Multi-shard GGUF Test Results ===\n")
	fmt.Printf("Shards: %d\n", splitInfo.Count)
	fmt.Printf("Tensors per shard: %v\n", splitInfo.TensorsPerShard)
	fmt.Printf("Total tensors: %d\n", totalTensors)

	for i, shard := range splitInfo.Shards {
		fmt.Printf("\nShard %d:\n", i+1)
		fmt.Printf("  Path: %s\n", shard.Path)
		fmt.Printf("  Size: %.3f GB\n", float64(shard.Size)/1e9)
		fmt.Printf("  Tensors: %d\n", shard.NumTensors)
	}
}
