package gguf

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"testing"
)

// TestRoundTripFile verifies that writing a GGUF file and reading it back produces identical data.
func TestRoundTripFile(t *testing.T) {
	tmpFile := "/tmp/test-roundtrip.gguf"
	defer os.Remove(tmpFile)

	// Create test data
	metaEntries := []KVEntry{
		{Key: "general.architecture", Value: Value{Str: "llama", BType: BTypeString}},
		{Key: "general.type", Value: Value{Str: "model", BType: BTypeString}},
		{Key: "general.name", Value: Value{Str: "test-model", BType: BTypeString}},
		{Key: "llama.context_length", Value: Value{Int: 2048, BType: BTypeUint32}},
		{Key: "llama.embedding_length", Value: Value{Int: 512, BType: BTypeUint32}},
	}

	tensors := []struct {
		name     string
		shape    []uint64
		ggmlType GgmlType
		data     []byte
	}{
		{
			name:     "tok_embeddings.weight",
			shape:    []uint64{512, 32000},
			ggmlType: GgmlF32,
			data:     makeTestData(512 * 32000 * 4), // F32 = 4 bytes per element
		},
		{
			name:     "output.weight",
			shape:    []uint64{32000, 512},
			ggmlType: GgmlF32,
			data:     makeTestData(32000 * 512 * 4),
		},
	}

	// Write GGUF file
	gw, err := OpenForWrite(tmpFile)
	if err != nil {
		t.Fatalf("OpenForWrite: %v", err)
	}

	for _, e := range metaEntries {
		if err := gw.SetKV(e.Key, e.Value); err != nil {
			t.Fatalf("SetKV(%s): %v", e.Key, err)
		}
	}

	var tensorIndices []uint64
	for _, tn := range tensors {
		idx := gw.AddTensor(tn.name, tn.shape, tn.ggmlType)
		tensorIndices = append(tensorIndices, idx)
		if err := gw.WriteTensorData(idx, bytes.NewReader(tn.data)); err != nil {
			t.Fatalf("WriteTensorData(%s): %v", tn.name, err)
		}
	}

	writtenBytes, err := gw.Close()
	if err != nil {
		t.Fatalf("Close: %v", err)
	}
	t.Logf("Wrote %d bytes to %s", writtenBytes, tmpFile)

	// Read back and verify
	f, err := os.Open(tmpFile)
	if err != nil {
		t.Fatalf("os.Open: %v", err)
	}
	defer f.Close()

	info, _ := f.Stat()
	rdr, err := OpenFromReader(f, info.Size())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer rdr.Close()

	// Verify version and counts
	if rdr.Version() != Version3 {
		t.Errorf("version = %d, want %d", rdr.Version(), Version3)
	}
	if rdr.NumTensors() != len(tensors) {
		t.Fatalf("NumTensors() = %d, want %d", rdr.NumTensors(), len(tensors))
	}

	// Verify metadata round-trip
	metaEntriesRead, err := rdr.Metadata()
	if err != nil {
		t.Fatalf("Metadata: %v", err)
	}

	for _, expected := range metaEntries {
		var found bool
		for _, entry := range metaEntriesRead {
			if entry.Name() == expected.Key {
				found = true
				v, err := entry.Value()
				if err != nil {
					t.Errorf("Value(%s): %v", expected.Key, err)
					continue
				}

				// Verify value matches
				switch expected.Value.BType {
				case BTypeString:
					s, ok := v.AsString()
					if !ok || s != expected.Value.Str {
						t.Errorf("KV %s: got string %q, want %q", expected.Key, s, expected.Value.Str)
					}
				case BTypeUint32:
					i, ok := v.AsInt()
					if !ok || i != expected.Value.Int {
						t.Errorf("KV %s: got int %d, want %d", expected.Key, i, expected.Value.Int)
					}
				}
			}
		}
		if !found {
			t.Errorf("Missing KV entry: %s", expected.Key)
		}
	}

	// Verify tensor metadata round-trip
	tensorsRead, err := rdr.Tensors()
	if err != nil {
		t.Fatalf("Tensors: %v", err)
	}

	for i, tn := range tensors {
		tr := tensorsRead[i]
		ti := tr.Info()

		if ti.Name != tn.name {
			t.Errorf("Tensor[%d].Name = %q, want %q", i, ti.Name, tn.name)
		}
		if len(ti.Shape) != len(tn.shape) {
			t.Errorf("Tensor[%d] shape length mismatch: got %d, want %d", i, len(ti.Shape), len(tn.shape))
		} else {
			for j := range tn.shape {
				if ti.Shape[j] != tn.shape[j] {
					t.Errorf("Tensor[%d].Shape[%d] = %d, want %d", i, j, ti.Shape[j], tn.shape[j])
				}
			}
		}
		if ti.GgmlType != tn.ggmlType {
			t.Errorf("Tensor[%d].GgmlType = %v, want %v", i, ti.GgmlType, tn.ggmlType)
		}

		// Verify tensor data matches byte-for-byte
		dataRead, err := tr.Bytes()
		if err != nil {
			t.Fatalf("Bytes(%s): %v", tn.name, err)
		}
		if len(dataRead) != len(tn.data) {
			t.Errorf("Tensor[%d] data size mismatch: got %d bytes, want %d", i, len(dataRead), len(tn.data))
			continue
		}

		hash1 := sha256.Sum256(tn.data)
		hash2 := sha256.Sum256(dataRead)
		if hash1 != hash2 {
			t.Errorf("Tensor[%d] data mismatch: %s", i, tn.name)
			// Find first differing byte
			for j := range dataRead {
				if dataRead[j] != tn.data[j] {
					t.Errorf("  First diff at byte %d: got %02x, want %02x", j, dataRead[j], tn.data[j])
					break
				}
			}
		} else {
			t.Logf("Tensor[%d] (%s): %d bytes verified ✓", i, tn.name, len(dataRead))
		}
	}

	fmt.Printf("\n=== Round-trip test passed: wrote and read back %d tensors, %d KV entries ===\n", len(tensors), len(metaEntries))
}

func makeTestData(size int) []byte {
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i % 256)
	}
	return data
}
