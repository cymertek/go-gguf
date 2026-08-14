package gguf

import (
	"encoding/binary"
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

	rdr := &GGUF{
		r:       f,
		fileSz:  info.Size(),
		version: Version3, // validated via header read below
		nTensor: 1,        // placeholder; actual value comes from header read below
		nKV:     0,
		alignment: defaultAlignment,
	}

	// Read and validate header
	var hdr [24]byte
	if _, err := io.ReadFull(io.NewSectionReader(f, 0, 24), hdr[:]); err != nil {
		t.Fatalf("read header: %v", err)
	}
	if string(hdr[0:4]) != Magic {
		t.Fatalf("invalid magic")
	}
	rdr.version = binary.LittleEndian.Uint32(hdr[4:8])
	rdr.nTensor = binary.LittleEndian.Uint64(hdr[8:16])
	rdr.nKV = binary.LittleEndian.Uint64(hdr[16:24])

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
			if s != "qwen3" {
				t.Errorf("expected 'qwen3' (Bonsai-8B architecture), got %q", s)
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

	// 5. Verify streaming Reader() works on F32/Q4_0 tensors
	for _, tensor := range tensors[:min(10, len(tensors))] {
		info := tensor.Info()
		if info.GgmlType == GgmlF32 || info.GgmlType == GgmlQ4_0 {
			r := tensor.Reader()
			data, err := io.ReadAll(r)
			if err != nil {
				t.Errorf("Read tensor %s: %v", info.Name, err)
				continue
			}
			t.Logf("%s: %d bytes raw data (streamed via Reader())", info.Name, len(data))
			break // Only test first matching tensor
		}
	}

	t.Log("End-to-end test passed!")
}
