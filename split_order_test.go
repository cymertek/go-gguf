package gguf

import (
	"os"
	"path/filepath"
	"testing"
)

// TestMultiShardOrderValidation verifies that out-of-order shards are correctly sorted.
func TestMultiShardOrderValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping multi-shard test in short mode")
	}

	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "gguf-split-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Copy the real TestModel-V4 split files to temp directory with reversed order
	shard1 := "/workdir/TestModel-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf"
	shard2 := "/workdir/TestModel-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf"
	shard3 := "/workdir/TestModel-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf"

	// Create files in reverse order (3, 2, 1) to test sorting
	destShard1 := filepath.Join(tmpDir, "model-00003-of-00003.gguf")
	destShard2 := filepath.Join(tmpDir, "model-00002-of-00003.gguf")
	destShard3 := filepath.Join(tmpDir, "model-00001-of-00003.gguf")

	if err := copyFile(shard3, destShard1); err != nil {
		t.Fatalf("Failed to copy shard 3: %v", err)
	}
	if err := copyFile(shard2, destShard2); err != nil {
		t.Fatalf("Failed to copy shard 2: %v", err)
	}
	if err := copyFile(shard1, destShard3); err != nil {
		t.Fatalf("Failed to copy shard 1: %v", err)
	}

	// Pass shards in reverse order (3, 2, 1) to test that they get sorted correctly
	g, err := NewReaderFile(destShard1, destShard2, destShard3)
	if err != nil {
		t.Fatalf("Failed to open split files: %v", err)
	}
	defer g.Close()

	if !g.IsSplit() {
		t.Fatal("Expected multi-shard GGUF")
	}

	splitInfo := g.SplitInfo()
	if splitInfo == nil {
		t.Fatal("SplitInfo should not be nil for split files")
	}

	// Verify shards are in correct order (0, 1, 2) after sorting by split.no
	expectedOrder := []int64{0, 1, 2}
	for i, shard := range splitInfo.Shards {
		if int64(shard.Index) != expectedOrder[i] {
			t.Errorf("Shard %d has incorrect index: got %d, want %d", i+1, shard.Index, expectedOrder[i])
		}
	}

	// Verify we can still read tensors correctly after sorting
	totalTensors := 0
	for _, shard := range splitInfo.Shards {
		tensors, err := shard.Tensors()
		if err != nil {
			t.Errorf("Failed to get tensors from shard %d: %v", shard.Index+1, err)
			continue
		}
		totalTensors += len(tensors)
	}

	expectedTotal := uint64(0)
	for _, count := range splitInfo.TensorsPerShard {
		expectedTotal += count
	}

	if uint64(totalTensors) != expectedTotal {
		t.Errorf("Total tensors mismatch: expected %d, got %d", expectedTotal, totalTensors)
	}
}

// TestMultiShardMismatchedCount verifies that shards with different split.count values are rejected.
func TestMultiShardMismatchedCount(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping multi-shard test in short mode")
	}

	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "gguf-split-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Copy shard 1 (metadata-only with split.count=3) as both files
	shard1 := "/workdir/TestModel-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf"
	destShard1 := filepath.Join(tmpDir, "model-00001-of-00002.gguf")
	destShard2 := filepath.Join(tmpDir, "model-00002-of-00002.gguf")

	if err := copyFile(shard1, destShard1); err != nil {
		t.Fatalf("Failed to copy shard 1: %v", err)
	}
	if err := copyFile(shard1, destShard2); err != nil {
		t.Fatalf("Failed to copy shard 1: %v", err)
	}

	// Try to open with mismatched split counts (one says 3, other says 2)
	_, err = NewReaderFile(destShard1, destShard2)
	if err == nil {
		t.Error("Expected error for mismatched shard counts, got nil")
	} else if !contains(err.Error(), "split.count") && !contains(err.Error(), "mismatch") {
		t.Errorf("Error should mention split count mismatch, got: %v", err)
	}
}

// TestMultiShardMissingShard verifies that incomplete shard sets are rejected.
func TestMultiShardMissingShard(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping multi-shard test in short mode")
	}

	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "gguf-split-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Copy only 2 out of 3 shards — user must pass all explicitly
	shard1 := "/workdir/TestModel-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf"
	destShard1 := filepath.Join(tmpDir, "model-00001-of-00003.gguf")
	destShard2 := filepath.Join(tmpDir, "model-00002-of-00003.gguf")

	if err := copyFile(shard1, destShard1); err != nil {
		t.Fatalf("Failed to copy shard 1: %v", err)
	}
	if err := copyFile(shard1, destShard2); err != nil {
		t.Fatalf("Failed to copy shard 1: %v", err)
	}

	// Try to open with incomplete set (only 2 of 3 shards passed explicitly)
	_, err = NewReaderFile(destShard1, destShard2)
	if err == nil {
		t.Error("Expected error for incomplete shard set, got nil")
	} else if !contains(err.Error(), "split.count") && !contains(err.Error(), "expected 3 shards") && !contains(err.Error(), "mismatched filename") {
		t.Errorf("Error should mention missing/incomplete shards or filename mismatch, got: %v", err)
	}
}

// Helper function to copy a file
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = destFile.ReadFrom(sourceFile)
	return err
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
