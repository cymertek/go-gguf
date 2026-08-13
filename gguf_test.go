package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
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
	f, err := os.Open(tmpFile)
	if err != nil {
		t.Fatalf("os.Open: %v", err)
	}
	info, _ := f.Stat()
	rdr, err := OpenFromReader(f, info.Size())
	if err != nil {
		f.Close()
		t.Fatalf("Open: %v", err)
	}
	defer rdr.Close()

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

func TestF32Dequant(t *testing.T) {
	inputs := []float32{0.0, 1.0, -1.0, 0.5, -0.5, 1.234}
	data := make([]byte, len(inputs)*4)
	for i, v := range inputs {
		binary.LittleEndian.PutUint32(data[i*4:i*4+4], math.Float32bits(v))
	}

	deq, err := Dequant(data, GgmlF32)
	if err != nil {
		t.Fatalf("Dequant: %v", err)
	}
	if len(deq) != len(inputs) {
		t.Fatalf("dequant len = %d, want %d", len(deq), len(inputs))
	}
	for i, v := range inputs {
		if deq[i] != v {
			t.Errorf("dequant[%d] = %f, want %f", i, deq[i], v)
		}
	}
}

func TestNVFP4Dequant(t *testing.T) {
	inputs := []float32{
		0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
		-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
		0.25, 0.75, 1.25, 2.5, 3.5, 5.0, 0.0, -0.25,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	}
	if len(inputs) != 64 {
		t.Fatalf("inputs len = %d, want 64", len(inputs))
	}

	requant, err := Requantize(inputs, GgmlNVFP4)
	if err != nil {
		t.Fatalf("Requantize: %v", err)
	}
	if len(requant) != 40 {
		t.Fatalf("requant size = %d, want 40", len(requant))
	}

	deq, err := Dequant(requant, GgmlNVFP4)
	if err != nil {
		t.Fatalf("Dequant: %v", err)
	}
	if len(deq) != len(inputs) {
		t.Fatalf("dequant len = %d, want %d", len(deq), len(inputs))
	}

	for i := range inputs {
		if inputs[i] == 0 {
			continue
		}
		got := deq[i]
		if got == 0 {
			if inputs[i] != 0.25 && inputs[i] != 0.75 && inputs[i] != 1.25 && inputs[i] != 2.5 && inputs[i] != 3.5 && inputs[i] != 5.0 {
				t.Errorf("dequant[%d] = 0, want approx %f", i, inputs[i])
			}
		}
	}

	if GgmlNVFP4.GgmlName() != "NVFP4" {
		t.Errorf("GgmlName = %q, want NVFP4", GgmlNVFP4.GgmlName())
	}
	if GgmlNVFP4.ElementsPerBlock() != 64 {
		t.Errorf("ElementsPerBlock = %d, want 64", GgmlNVFP4.ElementsPerBlock())
	}
	if GgmlNVFP4.BlockBytes() != 40 {
		t.Errorf("BlockBytes = %d, want 40", GgmlNVFP4.BlockBytes())
	}
	t.Logf("NVFP4: 64 elements, block=40B, name=%s", GgmlNVFP4.GgmlName())
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
