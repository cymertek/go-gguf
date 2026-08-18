package gguf

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"testing"
	"time"
)

func TestRoundTrip(t *testing.T) {
	// Create a temporary file for writing
	tmpFile := fmt.Sprintf("/tmp/test-roundtrip-%d.gguf", time.Now().UnixNano())
	defer os.Remove(tmpFile)

	// Write using new API
	gw, err := OpenForWrite(tmpFile)
	if err != nil {
		t.Fatalf("OpenForWrite: %v", err)
	}

	gw.SetKV("general.architecture", Value{Str: "test35", BType: BTypeString})
	gw.SetKV("general.file_type", Value{Int: 2, BType: BTypeUint32})

	shape := []uint64{4, 8}
	data := make([]byte, 4*4*8) // 4 bytes per float32
	for i := range data {
		data[i] = byte(i)
	}

	gw.AddTensor("test.weight", shape, GgmlF32)
	gw.WriteTensorData(0, bytes.NewReader(data))

	writtenBytes, err := gw.Close()
	if err != nil {
		t.Fatalf("Close: %v", err)
	}
	if writtenBytes == 0 {
		t.Fatal("no bytes written")
	}

	// Read back using new API
	g, err := NewReaderFile(tmpFile)
	if err != nil {
		t.Fatalf("NewReaderFile: %v", err)
	}
	defer g.Close()

	rdr := g

	if rdr.Version() != Version3 {
		t.Errorf("version = %d, want %d", rdr.Version(), Version3)
	}

	// Check metadata
	metaEntries, err := rdr.Metadata()
	if err != nil {
		t.Fatalf("Metadata: %v", err)
	}

	for _, e := range metaEntries {
		v, err := e.Value()
		if err != nil {
			t.Fatalf("Value for %s: %v", e.Name(), err)
		}
		t.Logf("  %s = %+v", e.Name(), v)
	}

	// Check tensor count
	if rdr.NumTensors() != 1 {
		t.Fatalf("NumTensors() = %d, want 1", rdr.NumTensors())
	}

	// Read tensors
	tensors, err := rdr.Tensors()
	if err != nil {
		t.Fatalf("Tensors: %v", err)
	}
	if len(tensors) != 1 {
		t.Fatalf("expected 1 tensor, got %d", len(tensors))
	}

	ti := tensors[0].Info()
	if ti.Name != "test.weight" {
		t.Errorf("tensor name = %q, want test.weight", ti.Name)
	}
	if len(ti.Shape) != 2 || ti.Shape[0] != 4 || ti.Shape[1] != 8 {
		t.Errorf("shape = %v, want [4 8]", ti.Shape)
	}
	if ti.GgmlType != GgmlF32 {
		t.Errorf("ggml type = %v, want F32", ti.GgmlType)
	}

	r := tensors[0].Reader()
	readData, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read tensor data: %v", err)
	}
	if !bytes.Equal(data, readData) {
		t.Errorf("tensor data mismatch - first 8 bytes written: %v, read: %v", data[:8], readData[:8])
	}

	t.Logf("OK: wrote %d bytes, read %d tensors", writtenBytes, len(tensors))
}

func TestConvertName(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"output.weight", "lm_head.weight"},
		{"token_embd.weight", "model.embed_tokens.weight"},
		{"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"},
		{"blk.1.ffn_down.weight", "model.layers.1.mlp.down_proj.weight"},
		{"blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"},
	}
	for _, tc := range tests {
		got := ConvertName(tc.in)
		if got != tc.want {
			t.Errorf("ConvertName(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// rwBuffer is a bytes.Buffer that implements io.ReadSeeker.
type rwBuffer struct {
	buf []byte
	pos int64
}

func (r *rwBuffer) Read(p []byte) (int, error) {
	if r.pos >= int64(len(r.buf)) {
		return 0, io.EOF
	}
	n := copy(p, r.buf[r.pos:])
	r.pos += int64(n)
	return n, nil
}

func (r *rwBuffer) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(r.buf)) {
		return 0, io.EOF
	}
	n := copy(p, r.buf[off:])
	return n, nil
}

func (r *rwBuffer) Seek(offset int64, whence int) (int64, error) {
	var abs int64
	switch whence {
	case 0: // SeekStart
		abs = offset
	case 1: // SeekCurrent
		abs = r.pos + offset
	case 2: // SeekEnd
		abs = int64(len(r.buf)) + offset
	default:
		return 0, fmt.Errorf("gguf: invalid whence")
	}
	if abs < 0 {
		return 0, fmt.Errorf("gguf: negative seek")
	}
	r.pos = abs
	return abs, nil
}

func (r *rwBuffer) Write(p []byte) (int, error) {
	return 0, fmt.Errorf("write not supported")
}
