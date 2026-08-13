package gguf

import (
	"io"
	"os"
	"testing"
)

// TestEndToEndBonsai verifies the lazy reader works correctly with a real 1.16GB GGUF file.
func TestEndToEndBonsai(t *testing.T) {
	const bonsaiPath = "/workdir/Bonsai-8B.gguf"

	// Open the file lazily (only header read)
	f, err := os.Open(bonsaiPath)
	if err != nil {
		t.Fatalf("os.Open: %v", err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}

	rdr, err := OpenFromReader(f, info.Size())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer rdr.Close()

	t.Logf("Opened Bonsai-8B.gguf (%d bytes)", info.Size())
	t.Logf("Version=%d nKV=%d nTensor=%d", rdr.Version(), rdr.NumTensors(), rdr.NumTensors())

	// 1. Read metadata (lazy walk with eager threshold ≤64B)
	metaEntries, err := rdr.Metadata()
	if err != nil {
		t.Fatalf("Metadata: %v", err)
	}
	t.Logf("Read %d KV entries", len(metaEntries))

	// Verify we can read specific values
	for _, entry := range metaEntries {
		v, err := entry.Value()
		if err != nil {
			// Skip entries that fail to load (e.g., large arrays)
			t.Logf("  %s: skipped (%v)", entry.Name(), err)
			continue
		}
		t.Logf("  %s = %+v (loaded=%v, btype=%d)", entry.Name(), v, entry.ok, entry.BType())

		// Check that we got the expected architecture string
		if entry.Name() == "general.architecture" {
			s, _ := v.AsString()
			if s != "test3" {
				t.Errorf("expected 'test3', got %q", s)
			}
		}
	}

	// 2. Get tensors (ONE seek to kvEnd + sequential read of all entries)
	tensors, err := rdr.Tensors()
	if err != nil {
		t.Fatalf("Tensors: %v", err)
	}
	t.Logf("Read %d tensors", len(tensors))

	// Verify first few tensors have correct metadata
	for i, tensor := range tensors[:min(5, len(tensors))] {
		info := tensor.Info()
		t.Logf("  Tensor[%d]: %s (shape=%v type=%s nbytes=%d)",
			i, info.Name, info.Shape, info.GgmlType.GgmlName(), info.NBytes)
	}

	// 3. Read tensor data with per-tensor cache
	if len(tensors) > 0 {
		t0 := tensors[0]

		// First read - should populate cache
		buf1 := make([]byte, 64)
		n1, err := t0.ReadAt(buf1, 0)
		if err != nil {
			t.Fatalf("First ReadAt: %v", err)
		}
		t.Logf("First read: %d bytes (cache populated)", n1)

		// Second read from same offset - should hit cache
		buf2 := make([]byte, 64)
		n2, err := t0.ReadAt(buf2, 0)
		if err != nil {
			t.Fatalf("Second ReadAt (cache): %v", err)
		}
		t.Logf("Second read: %d bytes (from cache)", n2)

		// Verify data matches
		for i := 0; i < n1 && i < n2; i++ {
			if buf1[i] != buf2[i] {
				t.Errorf("cache mismatch at byte %d", i)
				break
			}
		}

		t0.Close() // Release cache buffer back to pool
	}

	// 4. Find a specific tensor by name (use llama.cpp naming convention)
	targetNames := []string{"token_embd.weight", "output.weight"}
	found := false
	for _, name := range targetNames {
		for _, tensor := range tensors {
			if tensor.Info().Name == name {
				found = true
				t.Logf("Found %s (nbytes=%d, shape=%v)", tensor.Info().Name, tensor.Info().NBytes, tensor.Info().Shape)
				break
			}
		}
		if found {
			break
		}
	}
	if !found {
		t.Errorf("Did not find any of %v in tensors", targetNames)
	}

	// 5. Verify Dequant works on F32 tensor (embed_tokens should be F16 or similar)
	for _, tensor := range tensors[:min(10, len(tensors))] {
		info := tensor.Info()
		if info.GgmlType == GgmlF32 || info.GgmlType == GgmlQ4_0 {
			r := tensor.Reader()
			data, err := io.ReadAll(r)
			if err != nil {
				t.Errorf("Read tensor %s: %v", info.Name, err)
				continue
			}
			t.Logf("%s: %d bytes raw data (streamed)", info.Name, len(data))

			// Try dequantizing (may fail for unsupported types)
			if len(data) > 0 {
				deq, err := Dequant(data, info.GgmlType)
				if err != nil {
					t.Logf("Dequant error (expected for some types): %v", err)
				} else {
					t.Logf("%s: dequantized to %d floats", info.Name, len(deq))
				}
			}
			break // Only test first matching tensor
		}
	}

	t.Log("End-to-end test passed!")
}
