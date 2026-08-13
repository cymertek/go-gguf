package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const defaultAlignment = uint64(32)

// ---------------------------------------------------------------------------
// Open — reads only the 24-byte header, returns a lazy GGUF handle
// ---------------------------------------------------------------------------

// Open opens a GGUF file by path and returns a lazy [*GGUF] reader. If the file is part of
// a multi-shard split (e.g., TestModel-V4 with files named "model-*-of-NNNNN.gguf"), it will
// automatically detect all shards, validate header consistency across them, and combine into
// a single logical [io.ReaderAt]. The returned *GGUF must be closed via [GGUF.Close] when done.
//
// Open reads only the 24-byte GGUF header immediately; KV metadata and tensor info are walked
// lazily on first call to [GGUF.Metadata] or [GGUF.Tensors].
//
// Example:
//
//	g, err := gguf.Open("model.gguf")
//	if err != nil { log.Fatal(err) }
//	defer g.Close()
//	fmt.Printf("Version=%d Tensors=%d KV=%d\n", g.Version(), g.NumTensors(), len(g.Metadata()))
func Open(path string) (*GGUF, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open %q: %w", path, err)
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: stat %q: %w", path, err)
	}

	g, err := OpenFromReader(f, info.Size())
	if err != nil {
		f.Close()
		return nil, err
	}
	g.sourcePath = path

	// Check if this is a split file and combine shards
	splitInfo, err := g.detectSplit(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: detect split: %w", err)
	}

	if splitInfo != nil && len(splitInfo.shards) > 1 {
		g.splitInfo = splitInfo
	}

	return g, nil
}

// OpenFromReader opens a GGUF file from any [io.ReaderAt] (e.g., *os.File, S3 range reader,
// gzip decompressor) and returns a lazy [*GGUF]. The caller must provide the exact total file
// size via fileSz. This is useful for testing with in-memory buffers or for reading over
// network protocols that implement ReaderAt semantics.
//
// Example:
//
//	data := readFile("model.gguf")
//	g, err := gguf.OpenFromReader(bytes.NewReader(data), int64(len(data)))
//	if err != nil { log.Fatal(err) }
//	defer g.Close()
func OpenFromReader(r io.ReaderAt, fileSz int64) (*GGUF, error) {
	if fileSz < 24 {
		return nil, fmt.Errorf("gguf: file size %d too small for header", fileSz)
	}

	var hdr [24]byte
	if _, err := io.ReadFull(io.NewSectionReader(r, 0, 24), hdr[:]); err != nil {
		return nil, fmt.Errorf("gguf: read header: %w", err)
	}

	// Validate magic
	if string(hdr[0:4]) != Magic {
		return nil, fmt.Errorf("gguf: invalid magic %q, want %q", hdr[0:4], Magic)
	}

	version := binary.LittleEndian.Uint32(hdr[4:8])
	if version < Version1 || version > Version3 {
		return nil, fmt.Errorf("gguf: unsupported version %d (only v1–v3 supported)", version)
	}

	nTensor := binary.LittleEndian.Uint64(hdr[8:16])
	nKV := binary.LittleEndian.Uint64(hdr[16:24])

	g := &GGUF{
		r:       r,
		fileSz:  fileSz,
		version: version,
		nTensor: nTensor,
		nKV:     nKV,
		alignment: defaultAlignment,
	}

	return g, nil
}

// Close releases the GGUF reader. If the underlying [io.ReaderAt] also implements io.Closer
// (e.g., [*os.File]), this closes the file handle. For in-memory readers or other non-closable
// types, it is a no-op and returns nil. Always call Close when done reading to avoid leaking
// file descriptors.
func (g *GGUF) Close() error {
	if c, ok := g.r.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

// Version returns the GGUF file version read from the 24-byte header. Valid values are
// [Version1], [Version2], or [Version3]. The library currently only fully supports v3 files;
// older versions will return an error at open time.
func (g *GGUF) Version() uint32 { return g.version }

// NumTensors returns the number of tensors reported in the GGUF header. This is a fast,
// zero-allocation query that does not walk the tensor metadata section -- actual [Tensor]
// handles are only constructed when [GGUF.Tensors] or related methods are called.
func (g *GGUF) NumTensors() int { return int(g.nTensor) }

// OpenForRead opens a GGUF file by path for lazy reading. It is an alias for [Open] with the
// same semantics -- reads only the 24-byte header immediately and walks KV/tensor sections lazily.
// Prefer [Open] for new code; this name exists for backwards compatibility.
func OpenForRead(path string) (*GGUF, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open %q: %w", path, err)
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: stat %q: %w", path, err)
	}

	g, err := OpenFromReader(f, info.Size())
	if err != nil {
		f.Close()
		return nil, err
	}
	return g, nil
}

// ---------------------------------------------------------------------------
// walkKVSection — lazy KV metadata walk with per-entry value caching
// ---------------------------------------------------------------------------

const eagerThreshold = 64 // only eagerly parse values ≤64 bytes during KV walk

// Metadata walks the KV section and returns all [MetadataEntry] handles. Small scalar values
// (≤64 bytes wire size, non-string/array) are parsed eagerly during this walk; large string
// or array values remain file-backed and are only loaded on first call to [MetadataEntry.Value].
// The returned slice is safe for concurrent reads but must not be modified. Subsequent calls
// return the cached result without re-walking the section.
func (g *GGUF) Metadata() ([]*MetadataEntry, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.kvWalk {
		return g.metadataFromEntries(), nil
	}

	if err := g.walkKVSection(); err != nil {
		return nil, err
	}

	return g.metadataFromEntries(), nil
}

func (g *GGUF) metadataFromEntries() []*MetadataEntry {
	result := make([]*MetadataEntry, len(g.kvEntries))
	for i, e := range g.kvEntries {
		me := &MetadataEntry{
			key:    e.key,
			btype:  e.btype,
			wireSz: e.wireSz,
			rawOff: e.rawOff,
			gguf:   g,
		}
		if e.loaded {
			me.val = e.value
			me.ok = true
		}
		result[i] = me
	}
	return result
}

// walkKVSection walks the KV section from offset 24, parsing each entry's key, type, and value.
// Small values are eagerly parsed; large ones keep their file offset for lazy loading.
func (g *GGUF) walkKVSection() error {
	pos := uint64(24)

	g.kvEntries = make([]*kvEntry, 0, g.nKV)
	g.kvNames = make([]string, 0, g.nKV)

	for i := uint64(0); i < g.nKV; i++ {
		entryStart := pos

		// Read key length (8 bytes)
		keyLenBytes := make([]byte, 8)
		if _, err := readFull(g.r, keyLenBytes, int64(pos)); err != nil {
			return fmt.Errorf("gguf: kv[%d] key_len: %w", i, err)
		}
		keyLen := binary.LittleEndian.Uint64(keyLenBytes)

		if keyLen > 1<<20 { // sanity: key longer than 1MB is invalid
			g.kvEnd = pos
			return nil
		}

		pos += 8 + keyLen

		// Read btype (4 bytes)
		btypeBytes := make([]byte, 4)
		if _, err := readFull(g.r, btypeBytes, int64(pos)); err != nil {
			return fmt.Errorf("gguf: kv[%d] btype: %w", i, err)
		}
		btype := BType(binary.LittleEndian.Uint32(btypeBytes))
		pos += 4

		valueStart := pos // absolute file offset where value bytes start

		var wireSz int64
		var rawLen uint64 // for variable-size types (String, Array)

		switch btype {
		case BTypeBool, BTypeUint8, BTypeInt8:
			wireSz = 1
			pos += uint64(wireSz) // skip past value data
		case BTypeUint16, BTypeInt16:
			wireSz = 2
			pos += uint64(wireSz) // skip past value data
		case BTypeUint32, BTypeInt32, BTypeFloat32:
			wireSz = 4
			pos += uint64(wireSz) // skip past value data
		case BTypeUint64, BTypeInt64, BTypeFloat64:
			wireSz = 8
			pos += uint64(wireSz) // skip past value data
		case BTypeString:
			strLenBytes := make([]byte, 8)
			if _, err := readFull(g.r, strLenBytes, int64(pos)); err != nil {
				return fmt.Errorf("gguf: kv[%d] string_len: %w", i, err)
			}
			rawLen = binary.LittleEndian.Uint64(strLenBytes)
			wireSz = int64(rawLen) + 8 // str_len(8) + data(rawLen)
			pos += uint64(wireSz)      // skip past str_len(8) + string data
		case BTypeArray:
			elemTypeBytes := make([]byte, 4)
			countBytes := make([]byte, 8)
			if _, err := readFull(g.r, elemTypeBytes, int64(pos)); err != nil {
				return fmt.Errorf("gguf: kv[%d] array_elem_type: %w", i, err)
			}
			if _, err := readFull(g.r, countBytes, int64(pos)+4); err != nil {
				return fmt.Errorf("gguf: kv[%d] array_count: %w", i, err)
			}
			elemType := BType(binary.LittleEndian.Uint32(elemTypeBytes))
			count := binary.LittleEndian.Uint64(countBytes)

			pos += 12 // elem_type(4) + count(8) already read
			valueStart = pos

			switch elemType {
			case BTypeString:
				// String array: sum str_len(8)+data for each element
				var totalStrData uint64
				for j := uint64(0); j < count; j++ {
					slenBuf := make([]byte, 8)
					if _, err := readFull(g.r, slenBuf, int64(pos)); err != nil {
						return fmt.Errorf("gguf: kv[%d] array string len[%d]: %w", i, j, err)
					}
					slen := binary.LittleEndian.Uint64(slenBuf)
					totalStrData += 8 + slen // str_len(8) + data bytes (sl)
					pos += uint64(8 + slen)  // skip str_len(8) + string data(sl)
				}
				wireSz = int64(totalStrData) - 12 // subtract elem_type+count already read at wire level
			default:
				elemSize := elemType.Size()
				if elemSize == 0 {
					elemSize = 1 // unknown type, assume 1 byte per element
				}
				wireSz = int64(count) * int64(elemSize) + 12 // elem_type(4)+count(8)+data
				pos += uint64(count) * uint64(elemSize)       // skip past array data
			}
		default:
			wireSz = -1 // unsupported
		}

		keyData := make([]byte, keyLen)
		if _, err := readFull(g.r, keyData, int64(entryStart+8)); err != nil {
			return fmt.Errorf("gguf: kv[%d] key data: %w", i, err)
		}

		keyStr := string(keyData)
		e := &kvEntry{
			key:    keyStr,
			btype:  btype,
			wireSz: wireSz,
			rawOff: valueStart,
		}

		// Eagerly parse small scalar values (≤64 bytes threshold per user spec)
		if wireSz > 0 && wireSz <= eagerThreshold && btype != BTypeString && btype != BTypeArray {
			v := readFixedValue(g.r, int64(e.rawOff), btype, int(wireSz))
			e.value = v
			e.loaded = true
		}

		g.kvEntries = append(g.kvEntries, e)
		g.kvNames = append(g.kvNames, keyStr)
	}

	g.kvEnd = pos
	g.kvWalk = true
	return nil
}

// readFixedValue reads a fixed-size GGUF value from the file at rawOff.
func readFixedValue(r io.ReaderAt, off int64, btype BType, wireSz int) Value {
	data := getBuffer(wireSz)[:wireSz]
	defer putBuffer(data)

	if _, err := readFull(r, data, off); err != nil {
		return Value{BType: btype} // return empty on error
	}

	var v Value
	v.BType = btype
	switch btype {
	case BTypeBool, BTypeUint8, BTypeInt8:
		v.Int = int64(data[0])
	case BTypeUint16, BTypeInt16:
		v.Int = int64(binary.LittleEndian.Uint16(data))
		if btype == BTypeInt16 {
			v.Int = int64(int16(v.Int))
		}
	case BTypeUint32, BTypeInt32:
		v.Int = int64(binary.LittleEndian.Uint32(data))
		if btype == BTypeInt32 {
			v.Int = int64(int32(v.Int))
		}
	case BTypeFloat32:
		v.Float = float64(math.Float32frombits(binary.LittleEndian.Uint32(data)))
	case BTypeUint64, BTypeInt64:
		v.Int = int64(binary.LittleEndian.Uint64(data))
	case BTypeFloat64:
		v.Float = math.Float64frombits(binary.LittleEndian.Uint64(data))
	}
	return v
}

// readLazyValue reads a variable-size or scalar value from the file at rawOff.
func readLazyValue(r io.ReaderAt, off int64, btype BType, wireSz int64) (Value, error) {
	// For arrays, we don't eagerly load - just store wire info
	if btype == BTypeArray {
		return Value{BType: BTypeArray}, nil
	}

	data := getBuffer(int(wireSz))[:wireSz]
	defer putBuffer(data)

	if _, err := readFull(r, data, off); err != nil {
		return Value{}, fmt.Errorf("gguf: read value at %d: %w", off, err)
	}

	var v Value
	v.BType = btype

	switch btype {
	case BTypeBool, BTypeUint8, BTypeInt8:
		v.Int = int64(data[0])
	case BTypeUint16, BTypeInt16:
		v.Int = int64(binary.LittleEndian.Uint16(data))
		if btype == BTypeInt16 {
			v.Int = int64(int16(v.Int))
		}
	case BTypeUint32, BTypeInt32:
		v.Int = int64(binary.LittleEndian.Uint32(data))
		if btype == BTypeInt32 {
			v.Int = int64(int32(v.Int))
		}
	case BTypeFloat32:
		v.Float = float64(math.Float32frombits(binary.LittleEndian.Uint32(data)))
	case BTypeUint64, BTypeInt64:
		v.Int = int64(binary.LittleEndian.Uint64(data))
	case BTypeFloat64:
		v.Float = math.Float64frombits(binary.LittleEndian.Uint64(data))
	case BTypeString:
		if len(data) < 8 {
			return Value{}, fmt.Errorf("gguf: string value too short")
		}
		strLen := binary.LittleEndian.Uint64(data[:8])
		if uint64(len(data)) < 8+strLen {
			return Value{}, fmt.Errorf("gguf: string value truncated")
		}
		v.Str = string(data[8 : 8+strLen])
	default:
		return Value{}, fmt.Errorf("gguf: readLazyValue unsupported btype %d", btype)
	}

	return v, nil
}

// ---------------------------------------------------------------------------
// walkTensorSection — sequential tensor metadata walk (ONE seek from kvEnd)
// ---------------------------------------------------------------------------

// Tensors walks the tensor metadata section and returns all [*Tensor] handles.
// It performs exactly ONE seek followed by N sequential reads for nTensor entries, making it
// efficient for large models with thousands of tensors. The returned slice is safe for concurrent
// reads but must not be modified. Subsequent calls return cached results without re-walking.
//
// Example:
//
//	tensors, err := g.Tensors()
//	if err != nil { log.Fatal(err) }
//	for _, t := range tensors[:3] { // inspect first 3 tensors
//	    info := t.Info()
//	    fmt.Printf("%s shape=%v type=%s\n", info.Name, info.Shape, info.GgmlType.GgmlName())
//	}
func (g *GGUF) Tensors() ([]*Tensor, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if len(g.tensorInfos) > 0 {
		return g.buildTensorHandles(), nil
	}

	// Ensure KV section is walked first to compute kvEnd
	if !g.kvWalk {
		if err := g.walkKVSection(); err != nil {
			return nil, fmt.Errorf("gguf: walk KV section before tensors: %w", err)
		}
	}

	if err := g.walkTensorSection(); err != nil {
		return nil, fmt.Errorf("gguf: walk tensor section: %w", err)
	}

	return g.buildTensorHandles(), nil
}

func (g *GGUF) buildTensorHandles() []*Tensor {
	handles := make([]*Tensor, len(g.tensorInfos))
	for i, ti := range g.tensorInfos {
		tensor := &Tensor{
			info:      ti,
			absOffset: uint64(ti.Offset), // will be adjusted in the loop below
			gguf:      g,
		}
		handles[i] = tensor
	}

	// Now compute aligned absOffsets — we need dataStart for this
	for i := range handles {
		absOff := uint64(handles[i].info.Offset)
		if rem := absOff % defaultAlignment; rem != 0 {
			absOff += defaultAlignment - rem
		}
		handles[i].absOffset = g.dataStart + absOff
	}

	return handles
}

// walkTensorSection walks the tensor metadata section starting from kvEnd.
// All nTensor entries are read sequentially with ONE seek to kvEnd position.
func (g *GGUF) walkTensorSection() error {
	pos := int64(g.kvEnd)
	g.tensorInfos = make([]TensorInfo, 0, g.nTensor)

	for i := uint64(0); i < g.nTensor; i++ {
		// Read name length (8 bytes)
		nameLenBytes := make([]byte, 8)
		if _, err := readFull(g.r, nameLenBytes, pos); err != nil {
			return fmt.Errorf("gguf: tensor[%d] name_len: %w", i, err)
		}
		nameLen := binary.LittleEndian.Uint64(nameLenBytes)
		pos += 8

		if nameLen > 1<<20 { // sanity: name longer than 1MB is invalid
			return fmt.Errorf("gguf: tensor[%d] name_len %d too large", i, nameLen)
		}

		// Read name bytes (already at pos after nameLen)
		nameData := make([]byte, nameLen)
		if _, err := readFull(g.r, nameData, pos); err != nil {
			return fmt.Errorf("gguf: tensor[%d] name: %w", i, err)
		}

		pos += int64(nameLen) // advance past name

		// Read shape length (4 bytes)
		shapeLenBytes := make([]byte, 4)
		if _, err := readFull(g.r, shapeLenBytes, pos); err != nil {
			return fmt.Errorf("gguf: tensor[%d] shape_len: %w", i, err)
		}
		shapeLen := binary.LittleEndian.Uint32(shapeLenBytes)
		pos += 4

		if shapeLen > 64 { // sanity check
			return fmt.Errorf("gguf: tensor[%d] shape_len %d too large", i, shapeLen)
		}

		// Read dimensions (shapeLen × uint64)
		dims := make([]uint64, shapeLen)
		for d := uint32(0); d < shapeLen; d++ {
			dimBuf := make([]byte, 8)
			if _, err := readFull(g.r, dimBuf, pos); err != nil {
				return fmt.Errorf("gguf: tensor[%d] dim %d: %w", i, d, err)
			}
			dims[d] = binary.LittleEndian.Uint64(dimBuf)
			pos += 8
		}

		// Read ggml type (4 bytes)
		ggmlBytes := make([]byte, 4)
		if _, err := readFull(g.r, ggmlBytes, pos); err != nil {
			return fmt.Errorf("gguf: tensor[%d] type: %w", i, err)
		}
		ggmlType := GgmlType(binary.LittleEndian.Uint32(ggmlBytes))
		pos += 4

		// Read offset (8 bytes)
		offsetBuf := make([]byte, 8)
		if _, err := readFull(g.r, offsetBuf, pos); err != nil {
			return fmt.Errorf("gguf: tensor[%d] offset: %w", i, err)
		}
		tensorOff := binary.LittleEndian.Uint64(offsetBuf)
		pos += 8

		// Compute NBytes = totalElements × BlockBytes / ElementsPerBlock
		var totalElements uint64 = 1
		for _, dim := range dims {
			totalElements *= dim
		}
		blockBytes := ggmlType.BlockBytes()
		elsPerBlock := ggmlType.ElementsPerBlock()

		var nBytes uint64
		if elsPerBlock != 0 && blockBytes != 0 {
			nBytes = totalElements * uint64(blockBytes) / uint64(elsPerBlock)
		} else if elsPerBlock == 0 && blockBytes == 0 {
			// For unknown types, set NBytes to 0 (will be calculated later from offsets)
			nBytes = 0
		}

		g.tensorInfos = append(g.tensorInfos, TensorInfo{
			Name:     string(nameData),
			Shape:    dims,
			GgmlType: ggmlType,
			Offset:   tensorOff,
			NBytes:   nBytes,
		})
	}

	// After all tensors are processed, calculate NBytes for unknown types
	// by looking at consecutive tensor offsets
	for i := 0; i < len(g.tensorInfos); i++ {
		if g.tensorInfos[i].NBytes == 0 && g.tensorInfos[i].Offset != 0 {
			// Calculate NBytes from next tensor's offset (or end of file for last tensor)
			var nextOffset uint64
			if i+1 < len(g.tensorInfos) {
				nextOffset = g.tensorInfos[i+1].Offset
			} else {
				// Last tensor - use dataStart as approximate end
				nextOffset = g.dataStart
			}

			if nextOffset > g.tensorInfos[i].Offset {
				g.tensorInfos[i].NBytes = nextOffset - g.tensorInfos[i].Offset
			}
		}
	}

	g.dataStart = uint64(pos) // unaligned end of tensor metadata section
	// Align dataStart to alignment boundary
	if rem := g.dataStart % defaultAlignment; rem != 0 {
		g.dataStart += defaultAlignment - rem
	}

	return nil
}

// ---------------------------------------------------------------------------
// Tensor.ReadAt — reads with per-tensor cache overlap optimization
// ---------------------------------------------------------------------------

const tensorCacheMinSize = int64(1 << 20) // minimum cache size: 1MB (alignment-aligned block)

// ReadAt reads up to len(dst) bytes starting at off (relative to tensor data start) into dst.
// On the first call it reads a 1 MB aligned chunk into an internal cache; subsequent calls
// within that cached region serve from memory without disk I/O, significantly speeding up
// repeated partial reads of the same tensor. Off is relative to the tensor's aligned data
// start (not file offset). Returns io.EOF when off is past the end of the tensor.
func (t *Tensor) ReadAt(dst []byte, off int64) (int, error) {
	if len(dst) == 0 {
		return 0, nil
	}

	t.cacheMu.Lock()
	defer t.cacheMu.Unlock()

	srcOff := int64(t.absOffset) + off
	cacheStart := t.cacheOff
	cacheLen := int64(len(t.cache))

	// Check if the requested range is fully within cache
	if t.cacheValid && srcOff >= cacheStart && srcOff+int64(len(dst)) <= cacheStart+cacheLen {
		// Full cache hit — serve from memory
		startInCache := int(srcOff - cacheStart)
		n := copy(dst, t.cache[startInCache:])
		return n, nil
	}

	// Read into cache first (if not already cached or if range overlaps cache region)
	if !t.cacheValid || srcOff < cacheStart || srcOff >= cacheStart+cacheLen {
		// Allocate a fresh aligned cache buffer
		bufSz := int64(tensorCacheMinSize)
		buf := getBuffer(int(bufSz))[:bufSz]

		n, err := readFull(t.gguf.r, buf, srcOff)
		if err != nil || n == 0 {
			putBuffer(buf)
			return 0, err
		}

		// Store cache state
		t.cache = buf
		t.cacheOff = srcOff
		t.cacheEnd = uint64(srcOff + int64(n))
		t.cacheValid = true

		// Now copy from cache to dst
		startInCache := 0 // we just read starting at srcOff, so dst[0] maps to buf[0]
		return copy(dst[startInCache:], buf), nil
	}

	// Partial overlap: need to read more data and merge with existing cache
	// For simplicity, discard old cache and re-read a larger aligned block
	bufSz := int64(tensorCacheMinSize)
	buf := getBuffer(int(bufSz))[:bufSz]
	n, err := readFull(t.gguf.r, buf, srcOff)
	if err != nil || n == 0 {
		putBuffer(buf)
		return 0, err
	}

	t.cache = buf
	t.cacheOff = srcOff
	t.cacheEnd = uint64(srcOff + int64(n))
	t.cacheValid = true

	startInCache := 0
	return copy(dst[startInCache:], buf), nil
}

// ---------------------------------------------------------------------------
// readFull — reads exactly len(buf) bytes from io.ReaderAt at offset off
// ---------------------------------------------------------------------------

func readFull(r io.ReaderAt, buf []byte, off int64) (int, error) {
	var total int
	for total < len(buf) {
		n, err := r.ReadAt(buf[total:], off+int64(total))
		total += n
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

// ---------------------------------------------------------------------------
// Dequant convenience method on Tensor
// ---------------------------------------------------------------------------

// Dequant reads the entire raw tensor data from file via streaming (no full allocation to []byte),
// then calls [Dequant] to convert it into a []float32 slice of de-quantized values. For large tensors
// this still allocates memory for the float slice; prefer reading chunks via [Tensor.ReadAt] or
// [Tensor.Reader] with streaming dequantization for memory-constrained applications. The returned
// slice must not be modified by the caller.
func (t *Tensor) Dequant() ([]float32, error) {
	r := t.Reader()
	data := make([]byte, t.info.NBytes)
	n, err := io.ReadFull(r, data)
	if err != nil || n == 0 {
		return nil, fmt.Errorf("gguf: read tensor %s for dequant: %w", t.info.Name, err)
	}
	return Dequant(data[:n], t.info.GgmlType)
}

// Read reads up to len(dst) bytes from the tensor's raw data starting at offset 0. This is a
// convenience wrapper around [Tensor.ReadAt] that returns an [io.Reader]-style interface for use in
// streaming pipelines (e.g., io.Copy with another writer). The caller owns dst and must not modify it
// while reading continues concurrently.
func (t *Tensor) Read(dst []byte) (int, error) {
	return t.ReadAt(dst, 0)
}

// Data returns an [io.ReaderAt] that reads from this tensor's raw data region in the underlying
// file. The returned reader is positioned at the aligned start of the tensor data and has a
// fixed length equal to NBytes. Useful for passing tensor bytes directly to other libraries
// (e.g., neural network inference backends) without copying into memory first.
func (t *Tensor) Data() (io.ReaderAt, error) {
	return &tensorReaderAt{
		r:  t.gguf.r,
		off: int64(t.absOffset),
		n:   int64(t.info.NBytes),
	}, nil
}

// Reader returns an [io.ReadSeeker] that streams the tensor's raw binary data sequentially from
// the underlying file. This is a [io.LimitedReader]-style wrapper: reads are limited to NBytes and
// no dequantization or transformation is applied -- callers receive exactly the bytes as stored on
// disk. Use this when feeding tensor data directly into another library (e.g., a custom tokenizer,
// inference backend, or quantizer) that manages its own consumption rate and memory layout.
//
// Example -- stream raw Q4_0 bytes to an external quantizer:
//
//	tensors, _ := g.Tensors()
//	for _, t := range tensors {
//	    if t.Info().GgmlType == gguf.GgmlQ4_0 {
//	        r := t.Reader() // io.ReadSeeker, limited to NBytes
//	        externalLib.Feed(r)
//	    }
//	}
func (t *Tensor) Reader() io.ReadSeeker {
	return io.NewSectionReader(t.gguf.r, int64(t.absOffset), int64(t.info.NBytes))
}

// tensorReaderAt wraps an io.ReaderAt with a fixed offset and length.
type tensorReaderAt struct {
	r   io.ReaderAt
	off int64
	n   int64
}

func (t *tensorReaderAt) ReadAt(buf []byte, off int64) (int, error) {
	if off >= t.n || off < 0 {
		return 0, io.EOF
	}
	avail := t.n - off
	n := len(buf)
	if n > int(avail) {
		n = int(avail)
	}
	data := buf[:n]
	read, err := t.r.ReadAt(data, t.off+off)
	return read, err
}
