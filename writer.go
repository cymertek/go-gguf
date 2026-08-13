package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// ---------------------------------------------------------------------------
// StreamWriter — low-memory backend that writes directly to io.WriterAt
// ---------------------------------------------------------------------------

// StreamWriter writes GGUF files directly to an [io.WriterAt], avoiding full in-memory
// accumulation of the entire file. It is the low-level backend used by [GGUFWriter] and
// [Create]. Metadata (KV pairs) are kept in memory since they are small; tensor data is
// written sequentially at [StreamWriter.Close] time after all tensors have been queued.
type StreamWriter struct {
	w        io.WriterAt     // underlying writer (file, network buffer, etc.)
	meta     *writerMeta     // accumulated KV entries (in memory — always small)
	buf8     [8]byte         // reusable 8-byte buffer for writes
	tensors  []tensorBuf     // queued tensors with pre-computed offsets
	dataStart uint64         // aligned start of tensor data section
	totalSize uint64         // total file size after all tensors written

	alignment    uint64
	headerWritten bool
	initialized  bool
}

// writerMeta holds KV entries in memory for writing. Metadata is always small relative to tensors.
type writerMeta struct {
	pairs     []KVEntry
	alignment uint64
}

func newWriterMeta() *writerMeta {
	return &writerMeta{
		pairs:     make([]KVEntry, 0, 32),
		alignment: defaultAlignment,
	}
}

// tensorBuf holds a queued tensor's definition and its deferred reader reference. The reader is NOT
// consumed at AddTensor/WriteTensorData/NewTensor time -- bytes stream from it through a small pooled
// buffer only when [StreamWriter.Close] writes to disk, keeping RAM usage O(1) regardless of tensor
// size (tensors larger than available RAM work without allocation pressure).
type tensorBuf struct {
	info   TensorInfo
	reader io.Reader
}

// ---------------------------------------------------------------------------
// Create — high-level builder for writing GGUF files to any io.Writer
// ---------------------------------------------------------------------------

// Create creates a new [GGUFWriter] wrapping the given [io.Writer]. If w implements [io.WriterAt],
// data is written directly to it; otherwise it is wrapped in an adapter that buffers writes.
// Use [GGUFWriter.SetKV] to add metadata, [GGUFWriter.AddTensor] + [GGUFWriter.WriteTensorData]
// to queue and stream tensor data, then [GGUFWriter.Close] to flush the header and finalize.
func Create(w io.Writer) *GGUFWriter {
	var sw *StreamWriter
	if wa, ok := w.(io.WriterAt); ok {
		sw = NewStreamWriter(wa)
	} else {
		sw = NewStreamWriter(&writerAdapter{w: w})
	}
	return &GGUFWriter{sw: sw, meta: sw.meta}
}

// OpenForWrite creates a [GGUFWriter] that writes to a new file at the given path. The file is
// created with [os.Create] (truncating any existing content). Equivalent to Create(os.File) but
// manages the file handle internally until Close. Returns an error if the file cannot be opened.
func OpenForWrite(path string) (*GGUFWriter, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: create %q: %w", path, err)
	}
	w := Create(f)
	return w, nil
}

// ConvertName translates tensor names from the llama.cpp naming convention to the Hugging Face
// Transformers convention (and vice versa for symmetric keys). Handles special cases like
// "output.weight" -> "lm_head.weight", and replaces prefix patterns like "blk." -> "model.layers.".
// Returns the original name unchanged if no conversion applies.
func ConvertName(name string) string {
	conv := map[string]string{
		"output.weight":      "lm_head.weight",
		"token_embd.weight":  "model.embed_tokens.weight",
		"output_norm.weight": "model.norm.weight",
	}
	if r, ok := conv[name]; ok {
		return r
	}

	return strings.NewReplacer(
		"blk.", "model.layers.",
		".attn_norm.weight", ".input_layernorm.weight",
		".ffn_down.weight", ".mlp.down_proj.weight",
		".ffn_gate.weight", ".mlp.gate_proj.weight",
		".ffn_up.weight", ".mlp.up_proj.weight",
		".ffn_norm.weight", ".post_attention_layernorm.weight",
		".attn_q.weight", ".self_attn.q_proj.weight",
		".attn_k.weight", ".self_attn.k_proj.weight",
		".attn_v.weight", ".self_attn.v_proj.weight",
		".attn_output.weight", ".self_attn.o_proj.weight",
	).Replace(name)
}

// writerAdapter wraps io.Writer to implement io.WriterAt by buffering.
type writerAdapter struct {
	w   io.Writer
	buf []byte
	pos int64
}

func (a *writerAdapter) Write(p []byte) (int, error) {
	n, err := a.w.Write(p)
	a.buf = append(a.buf[:a.pos], p...)
	a.pos += int64(n)
	return n, err
}

func (a *writerAdapter) WriteAt(p []byte, off int64) (int, error) {
	if off+int64(len(p)) > int64(len(a.buf)) {
		newBuf := make([]byte, off+int64(len(p)))
		copy(newBuf, a.buf)
		a.buf = newBuf
	}
	n := copy(a.buf[off:], p)
	return n, nil
}

// GGUFWriter is a high-level builder for creating new GGUF files. It wraps [StreamWriter] and
// provides a simple API: call [GGUFWriter.SetKV] to add metadata, [GGUFWriter.AddTensor] +
// [GGUFWriter.WriteTensorData] to queue tensors with their data, then [GGUFWriter.Close] to
// finalize the file. GGUFWriter is NOT safe for concurrent use; serialize calls from a single goroutine.
type GGUFWriter struct {
	sw    *StreamWriter
	meta  *writerMeta // alias to StreamWriter's internal meta
}

// SetKV adds or replaces a key-value pair in the metadata section. If the key already exists,
// its value is overwritten. Call before [GGUFWriter.AddTensor] or [GGUFWriter.Close].
func (w *GGUFWriter) SetKV(key string, v Value) error {
	return w.sw.SetMetadataEntry(KVEntry{Key: key, Value: v})
}

// GetMetadata returns deep copies of all [KVEntry] pairs currently queued for writing. The returned
// slice is safe to read and modify without affecting the writer's state. Useful for inspecting
// or transforming metadata before finalizing with [GGUFWriter.Close].
func (w *GGUFWriter) GetMetadata() []KVEntry {
	result := make([]KVEntry, len(w.meta.pairs))
	copy(result, w.meta.pairs)
	return result
}

// NewTensor queues a tensor definition and immediately consumes raw bytes from r to back it. This
// is equivalent to calling [GGUFWriter.AddTensor] followed by [GGUFWriter.WriteTensorData], but
// accepts any [io.Reader] directly (including network streams, file handles, or byte buffers) so the
// caller does not need to manage an intermediate []byte slice. The reader is consumed fully via
// [io.ReadFull]; the tensor's data size must match what is predicted by shape × GgmlType. Call
// before [GGUFWriter.Close].
//
// Example -- stream a tensor directly from a network connection:
//
//	w := gguf.Create(file)
//	err := w.NewTensor("model.layers.0.attn_q.weight", []uint64{4096, 12288}, gguf.GgmlQ4_0, conn)
//	if err != nil { log.Fatal(err) }
func (w *GGUFWriter) NewTensor(name string, shape []uint64, ggmlType GgmlType, r io.Reader) error {
	return w.sw.NewTensor(name, shape, ggmlType, r)
}

// AddTensor queues a tensor definition (name, shape, quantization type) to be written later.
// Returns an index that must be passed to [GGUFWriter.WriteTensorData] or [GGUFWriter.NewTensor]
// to stream the raw bytes. Call in any order; data is written sequentially at [GGUFWriter.Close].
// The shape slice is copied internally so the caller may reuse it after this call returns. Prefer
// [GGUFWriter.NewTensor] when you already have an io.Reader for the data -- it combines definition
// and streaming into a single call.
func (w *GGUFWriter) AddTensor(name string, shape []uint64, ggmlType GgmlType) uint64 {
	idx := w.sw.AddTensor(name, shape, ggmlType)
	return idx
}

// NumTensors returns the number of tensors currently queued for writing via [GGUFWriter.AddTensor].
// Useful for tracking progress during large model writes.
func (w *GGUFWriter) NumTensors() int {
	return len(w.sw.tensors)
}

// WriteTensorData reads all bytes from r and associates them with the tensor at index idx
// (returned by [GGUFWriter.AddTensor]). The data must match exactly the size predicted by
// the tensor's shape and GgmlType; a short or long read returns an error. Call in the same
// order as AddTensor indices.
func (w *GGUFWriter) WriteTensorData(idx uint64, r io.Reader) error {
	return w.sw.WriteTensorData(idx, r)
}

// Close writes the GGUF header, KV section, tensor metadata section, alignment padding, and all
// queued tensor data to the underlying writer in a single pass. After calling Close no further
// [GGUFWriter.SetKV], [GGUFWriter.AddTensor], or [GGUFWriter.WriteTensorData] calls are valid.
// Returns the total number of bytes written and any error encountered during finalization.
func (w *GGUFWriter) Close() (int64, error) {
	return w.sw.Close()
}

// ---------------------------------------------------------------------------
// StreamWriter methods
// ---------------------------------------------------------------------------

// NewStreamWriter creates a new [StreamWriter] that writes directly to the given [io.WriterAt].
// Metadata entries can be added with [StreamWriter.SetMetadataEntry], tensors queued with
// [StreamWriter.AddTensor] + [StreamWriter.WriteTensorData], and the file finalized with
// [StreamWriter.Close]. Prefer this over [Create] when you already have an io.WriterAt (e.g., a file).
func NewStreamWriter(w io.WriterAt) *StreamWriter {
	return &StreamWriter{
		w:           w,
		meta:        newWriterMeta(),
		tensors:     make([]tensorBuf, 0, 16),
		alignment:   defaultAlignment,
	}
}

// SetMetadataEntry adds or replaces a [KVEntry] in the writer's metadata section. If an entry with
// the same key already exists, it is overwritten. Call before [StreamWriter.Close]. Safe to call
// multiple times for the same key -- last write wins.
func (w *StreamWriter) SetMetadataEntry(e KVEntry) error {
	for i := range w.meta.pairs {
		if w.meta.pairs[i].Key == e.Key {
			w.meta.pairs[i] = e
			return nil
		}
	}
	w.meta.pairs = append(w.meta.pairs, e)

	// Auto-initialize when first metadata entry is added (in case no tensors are written)
	if !w.initialized && len(w.meta.pairs) > 0 {
		w.initialized = true
	}

	return nil
}

// NewTensor queues a tensor definition and associates it with an [io.Reader] whose bytes will be
// streamed to disk at [StreamWriter.Close]. The reader is NOT consumed here -- only stored for
// deferred reading through a small pooled buffer during final write. This keeps memory usage O(1)
// regardless of tensor size, so tensors larger than available RAM work without allocation pressure.
// Call before [StreamWriter.Close]; the shape slice and reader reference are retained until Close.
func (w *StreamWriter) NewTensor(name string, shape []uint64, ggmlType GgmlType, r io.Reader) error {
	idx := w.AddTensor(name, shape, ggmlType)
	return w.queueReader(idx, r)
}

// AddTensor queues a tensor definition (name, shape, quantization type) for later writing. Returns
// an index usable in [StreamWriter.WriteTensorData] or [StreamWriter.NewTensor]. Call before Close;
// the shape slice is copied internally so the caller may reuse it after this call returns. Prefer
// [StreamWriter.NewTensor] when you already have an io.Reader -- it avoids holding tensor data in
// memory before Close writes to disk.
func (w *StreamWriter) AddTensor(name string, shape []uint64, ggmlType GgmlType) uint64 {
	info := TensorInfo{
		Name:     name,
		Shape:    shape,
		GgmlType: ggmlType,
	}

	idx := uint64(len(w.tensors))
	w.tensors = append(w.tensors, tensorBuf{info: info})

	// Auto-initialize when first tensor is added
	if !w.initialized {
		w.initialized = true
	}

	return idx
}

// WriteTensorData associates an [io.Reader] with the tensor at index idx (returned by AddTensor). The
// reader's bytes are NOT consumed here -- they stream to disk at Close via a small pooled buffer, so
// tensors larger than available RAM work without allocation pressure. Call in the same order as
// AddTensor indices, before [StreamWriter.Close].
func (w *StreamWriter) WriteTensorData(idx uint64, r io.Reader) error {
	return w.queueReader(idx, r)
}

// queueReader stores a deferred reader for the tensor at idx and validates that computeDataSize
// returns a non-zero expected size (otherwise NBytes is unknown and we cannot stream safely).
func (w *StreamWriter) queueReader(idx uint64, r io.Reader) error {
	if int(idx) >= len(w.tensors) {
		return fmt.Errorf("gguf: tensor index %d out of range", idx)
	}

	tb := &w.tensors[idx]
	expectedSz := computeDataSize(tb.info)
	if expectedSz == 0 {
		return fmt.Errorf("gguf: tensor %s: shape/type combination has no known data size (GgmlType=%d)", tb.info.Name, tb.info.GgmlType)
	}

	tb.reader = r
	return nil
}

// SetAlignment sets the byte alignment for the start of the tensor-data section. Must be a
// power-of-two multiple of 8 (default: 32). Higher alignment may improve sequential read
// performance on some storage devices but wastes more space in small files. Call before any
// [StreamWriter.AddTensor] or [StreamWriter.SetMetadataEntry].
func (w *StreamWriter) SetAlignment(a uint64) {
	if a == 0 {
		a = defaultAlignment
	}
	w.alignment = a
	w.meta.alignment = a
}

// Close writes the GGUF header, KV section, tensor metadata section, alignment padding, and all
// queued tensor data to the underlying [io.WriterAt] in a single pass. After calling Close no further
// [StreamWriter.SetMetadataEntry], [StreamWriter.AddTensor], or [StreamWriter.WriteTensorData] calls are valid.
// Returns the total number of bytes written and any error encountered during finalization.
func (w *StreamWriter) Close() (int64, error) {
	if !w.initialized {
		return 0, fmt.Errorf("gguf: stream not initialized")
	}

	var totalWritten int64

	// --- Write header (24 bytes) ---
	if err := w.writeHeader(); err != nil {
		return totalWritten, err
	}
	totalWritten += 24

	// --- Compute KV section size and write ---
	kvSize := computeKVSize(w.meta.pairs)
	tensorMetaSize := computeTensorMetaSize(w.tensors)

	w.dataStart = uint64(24) + kvSize + tensorMetaSize
	if rem := w.dataStart % defaultAlignment; rem != 0 {
		w.dataStart += defaultAlignment - rem
	}

	// Write KV section at offset 24
	kvOff := int64(24)
	for _, e := range w.meta.pairs {
		nw, err := w.writeKVEntry(kvOff, e.Key, e.Value)
		if err != nil {
			return totalWritten, fmt.Errorf("gguf: write KV %q: %w", e.Key, err)
		}
		kvOff += nw
		totalWritten += nw
	}

	// --- Pre-compute data offsets for all tensors (sizes derived from shape × GgmlType, no data read yet) ---
	dataPos := w.dataStart
	for i := range w.tensors {
		tb := &w.tensors[i]
		if rem := dataPos % defaultAlignment; rem != 0 {
			padLen := int64(defaultAlignment - rem)
			dataPos += uint64(padLen) // Skip padding before this tensor
		}
		nBytes := computeDataSize(tb.info)
		tb.info.Offset = dataPos - w.dataStart // Store offset relative to dataStart
		if nBytes == 0 {
			return totalWritten, fmt.Errorf("gguf: tensor %s has unknown NBytes (shape/type combination unsupported)", tb.info.Name)
		}
		dataPos += nBytes
	}

	// --- Write tensor metadata section at offset kvEnd (with correct offsets now) ---
	tensorMetaPos := int64(kvOff)
	for _, tb := range w.tensors {
		if err := writeTensorMeta(w.w, tensorMetaPos, tb.info); err != nil {
			return totalWritten, fmt.Errorf("gguf: write tensor meta %s: %w", tb.info.Name, err)
		}
		entrySize := int64(8 + len(tb.info.Name) + 4 + len(tb.info.Shape)*8 + 4 + 8)
		tensorMetaPos += entrySize
	}

	dataEnd := tensorMetaPos
	w.totalSize = uint64(dataEnd)

	// --- Write alignment padding between metadata end and data start ---
	if w.dataStart > uint64(dataEnd) {
		padLen := int64(w.dataStart - uint64(dataEnd))
		padBuf := make([]byte, padLen)
		nw, err := w.w.WriteAt(padBuf, dataEnd) // write at dataEnd (which is dataStart-padLen)
		if err != nil {
			return totalWritten, fmt.Errorf("gguf: write alignment padding: %w", err)
		}
		totalWritten += int64(nw)
	}

	// --- Stream tensor data from deferred readers through a small pooled buffer ---
	const streamBufSize = 256 << 10 // 256 KB streaming window; keeps RAM usage O(1) per tensor
	pos := w.dataStart
	for i := range w.tensors {
		tb := &w.tensors[i]

		// Skip padding to align this tensor's data
		if rem := pos % defaultAlignment; rem != 0 {
			padLen := int64(defaultAlignment - rem)
			padBuf := make([]byte, padLen)
			nw, err := w.w.WriteAt(padBuf, int64(pos))
			if err != nil {
				return totalWritten, fmt.Errorf("gguf: write tensor alignment padding: %w", err)
			}
			totalWritten += int64(nw)
			pos += uint64(padLen)
		}

		r := tb.reader
		if r == nil {
			return totalWritten, fmt.Errorf("gguf: tensor %s has no reader (did you call WriteTensorData/NewTensor?)", tb.info.Name)
		}

		buf := getBuffer(streamBufSize)[:streamBufSize]
		expectedSz := computeDataSize(tb.info)
		var wrote uint64
		for wrote < expectedSz {
			n, err := r.Read(buf)
			if n > 0 {
				wrote += uint64(n)
				nw, err2 := w.w.WriteAt(buf[:n], int64(pos))
				if err2 != nil {
					return totalWritten, fmt.Errorf("gguf: write tensor %s data: %w", tb.info.Name, err2)
				}
				totalWritten += int64(nw)
				pos += uint64(nw)
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				return totalWritten, fmt.Errorf("gguf: read tensor %s data: %w", tb.info.Name, err)
			}
		}

		if wrote != expectedSz {
			return totalWritten, fmt.Errorf("gguf: tensor %s: short stream %d/%d bytes (reader ended early)", tb.info.Name, wrote, expectedSz)
		}
		putBuffer(buf)
	}

	return totalWritten, nil
}

// --- Internal helpers ---

func (w *StreamWriter) writeHeader() error {
	offset := int64(0)

	if _, err := w.w.WriteAt([]byte(Magic), offset); err != nil {
		return fmt.Errorf("gguf: write magic: %w", err)
	}
	offset += 4

	binary.LittleEndian.PutUint32(w.buf8[:], Version3)
	if _, err := w.w.WriteAt(w.buf8[:4], offset); err != nil {
		return fmt.Errorf("gguf: write version: %w", err)
	}
	offset += 4

	binary.LittleEndian.PutUint64(w.buf8[:], uint64(len(w.tensors)))
	if _, err := w.w.WriteAt(w.buf8[:], offset); err != nil {
		return fmt.Errorf("gguf: write nTensors: %w", err)
	}
	offset += 8

	binary.LittleEndian.PutUint64(w.buf8[:], uint64(len(w.meta.pairs)))
	if _, err := w.w.WriteAt(w.buf8[:], offset); err != nil {
		return fmt.Errorf("gguf: write nKV: %w", err)
	}

	return nil
}

func (w *StreamWriter) writeKVEntry(off int64, key string, v Value) (int64, error) {
	var totalWritten int64

	keyBytes := []byte(key)
	binary.LittleEndian.PutUint64(w.buf8[:], uint64(len(keyBytes)))
	if _, err := w.w.WriteAt(w.buf8[:], off); err != nil {
		return 0, err
	}
	totalWritten += 8

	if _, err := w.w.WriteAt(keyBytes, off+8); err != nil {
		return totalWritten + 8, err
	}
	totalWritten += int64(len(keyBytes))

	btypeOff := off + 8 + int64(len(keyBytes))
	binary.LittleEndian.PutUint32(w.buf8[:], uint32(v.BType))
	if _, err := w.w.WriteAt(w.buf8[:4], btypeOff); err != nil {
		return totalWritten, err
	}
	totalWritten += 4

	valOff := btypeOff + 4

	switch v.BType {
	case BTypeBool:
		b := byte(0)
		if v.Int != 0 {
			b = 1
		}
		if _, err := w.w.WriteAt([]byte{b}, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten++

	case BTypeUint8:
		if _, err := w.w.WriteAt([]byte{byte(v.Int)}, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten++

	case BTypeInt8:
		if _, err := w.w.WriteAt([]byte{byte(v.Int)}, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten++

	case BTypeUint16:
		buf := make([]byte, 2)
		binary.LittleEndian.PutUint16(buf, uint16(v.Int))
		if _, err := w.w.WriteAt(buf, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 2

	case BTypeInt16:
		buf := make([]byte, 2)
		binary.LittleEndian.PutUint16(buf, uint16(v.Int))
		if _, err := w.w.WriteAt(buf, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 2

	case BTypeUint32:
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, uint32(v.Int))
		if _, err := w.w.WriteAt(buf, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 4

	case BTypeInt32:
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, uint32(v.Int))
		if _, err := w.w.WriteAt(buf, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 4

	case BTypeFloat32:
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, math.Float32bits(float32(v.Float)))
		if _, err := w.w.WriteAt(buf, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 4

	case BTypeUint64:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, uint64(v.Int))
		if _, err := w.w.WriteAt(buf, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 8

	case BTypeInt64:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, uint64(v.Int))
		if _, err := w.w.WriteAt(buf, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 8

	case BTypeFloat64:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, math.Float64bits(v.Float))
		if _, err := w.w.WriteAt(buf, valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 8

	case BTypeString:
		strBuf := []byte(v.Str)
		binary.LittleEndian.PutUint64(w.buf8[:], uint64(len(strBuf)))
		if _, err := w.w.WriteAt(w.buf8[:], valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 8
		if _, err := w.w.WriteAt(strBuf, valOff+8); err != nil {
			return totalWritten + 8, err
		}
		totalWritten += int64(len(strBuf))

	case BTypeArray:
		binary.LittleEndian.PutUint32(w.buf8[:4], uint32(v.ElemType))
		if _, err := w.w.WriteAt(w.buf8[:4], valOff); err != nil {
			return totalWritten, err
		}
		totalWritten += 4

		binary.LittleEndian.PutUint64(w.buf8[:], uint64(v.Int))
		if _, err := w.w.WriteAt(w.buf8[:], valOff+4); err != nil {
			return totalWritten + 4, err
		}
		totalWritten += 8

		if len(v.Raw) > 0 {
			nw, err := w.w.WriteAt(v.Raw, valOff+12)
			if err != nil {
				return totalWritten + 12, err
			}
			totalWritten += int64(nw)
		}

	default:
		return totalWritten, fmt.Errorf("gguf: unsupported write type %d", v.BType)
	}

	return totalWritten, nil
}

func writeTensorMeta(w io.WriterAt, off int64, info TensorInfo) error {
	buf8 := make([]byte, 8)

	binary.LittleEndian.PutUint64(buf8, uint64(len(info.Name)))
	if _, err := w.WriteAt(buf8, off); err != nil {
		return fmt.Errorf("write tensor name_len: %w", err)
	}

	nameOff := int64(off + 8)
	if _, err := w.WriteAt([]byte(info.Name), nameOff); err != nil {
		return fmt.Errorf("write tensor name: %w", err)
	}

	slOff := int64(nameOff + int64(len(info.Name)))
	binary.LittleEndian.PutUint32(buf8[:4], uint32(len(info.Shape)))
	if _, err := w.WriteAt(buf8[:4], slOff); err != nil {
		return fmt.Errorf("write shape_len: %w", err)
	}

	dimOff := int64(slOff + 4)
	for _, dim := range info.Shape {
		binary.LittleEndian.PutUint64(buf8, dim)
		if _, err := w.WriteAt(buf8, dimOff); err != nil {
			return fmt.Errorf("write dimension: %w", err)
		}
		dimOff += 8
	}

	binary.LittleEndian.PutUint32(buf8[:4], uint32(info.GgmlType))
	if _, err := w.WriteAt(buf8[:4], dimOff); err != nil {
		return fmt.Errorf("write ggml_type: %w", err)
	}

	binary.LittleEndian.PutUint64(buf8, info.Offset)
	if _, err := w.WriteAt(buf8, dimOff+4); err != nil {
		return fmt.Errorf("write tensor offset: %w", err)
	}

	return nil
}

// computeKVSize computes total wire size of all KV entries.
func computeKVSize(entries []KVEntry) uint64 {
	var total uint64
	for _, e := range entries {
		total += 8 + uint64(len(e.Key)) + 4 // key_len(8) + key(N) + btype(4)

		switch e.Value.BType {
		case BTypeString:
			total += 8 + uint64(len(e.Value.Str))
		case BTypeArray:
			total += 12 + uint64(len(e.Value.Raw)) // elem_type(4)+count(8)+raw data
		default:
			typeSizes := map[BType]int{
				BTypeBool:         1,
				BTypeUint8:        1,
				BTypeInt8:         1,
				BTypeUint16:       2,
				BTypeInt16:        2,
				BTypeUint32:       4,
				BTypeInt32:        4,
				BTypeFloat32:      4,
				BTypeUint64:       8,
				BTypeInt64:        8,
				BTypeFloat64:      8,
			}
			total += uint64(typeSizes[e.Value.BType])
		}
	}
	return total
}

// computeTensorMetaSize computes total wire size of tensor metadata section.
func computeTensorMetaSize(tensors []tensorBuf) uint64 {
	var total uint64
	for _, tb := range tensors {
		total += 8 + uint64(len(tb.info.Name)) + 4 + uint64(len(tb.info.Shape))*8 + 4 + 8
	}
	return total
}

// computeDataSize computes expected byte size for a tensor's raw data.
func computeDataSize(info TensorInfo) uint64 {
	var totalElements uint64 = 1
	for _, d := range info.Shape {
		totalElements *= d
	}
	blockBytes := info.GgmlType.BlockBytes()
	elementsPerBlock := info.GgmlType.ElementsPerBlock()
	if elementsPerBlock == 0 || blockBytes == 0 {
		return 0
	}
	return totalElements * uint64(blockBytes) / uint64(elementsPerBlock)
}
