package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const defaultAlignment = uint64(32)

// ---------------------------------------------------------------------------
// Open — reads only the 24-byte header, returns a lazy GGUF handle
// ---------------------------------------------------------------------------

// NewReader opens one or more io.ReaderAt inputs and returns a lazy [*GGUF] reader. For single-file
// GGUFs pass one reader; for multi-shard splits (e.g., TestModel-V4), pass all shard readers in
// order — the library validates headers match and combines into a single logical reader. The returned
// *GGUF must be closed via [GGUF.Close] when done. NewReader reads only the 24-byte GGUF header
// immediately; KV metadata and tensor info are walked lazily on first call to [GGUF.Metadata] or
// [GGUF.Tensors].
//
// Example — single file:
//
//	f, _ := os.Open("model.gguf")
//	defer f.Close()
//	g, err := gguf.NewReader(f)
//
// Example — multi-shard split (readers passed in order):
//
//	shards := []*os.File{f1, f2, f3}
//	g, _ := gguf.NewReader(shards...)
func NewReader(ras ...io.ReaderAt) (*GGUF, error) {
	if len(ras) == 0 {
		return nil, fmt.Errorf("gguf: no readers provided")
	}

	g := &GGUF{
		version:   Version3, // placeholder; validated per shard below
		nTensor:   1,        // placeholder; summed across shards below
		nKV:       0,        // placeholder; summed across shards below
		alignment: defaultAlignment,
	}

	if len(ras) == 1 {
		return nil, fmt.Errorf("gguf: single reader requires file size info")
	}

	// Multi-reader — validate all headers match and combine into multiReaderAt
	shards := make([]*shardHandle, len(ras))
	var totalTensors uint64
	var totalKV uint64

	for i, ra := range ras {
		// Read header from each reader
		var hdr [24]byte
		if _, err := io.ReadFull(io.NewSectionReader(ra, 0, 24), hdr[:]); err != nil {
			return nil, fmt.Errorf("gguf: read header for shard %d: %w", i+1, err)
		}

		if string(hdr[0:4]) != Magic {
			return nil, fmt.Errorf("gguf: invalid magic in shard %d", i+1)
		}

		version := binary.LittleEndian.Uint32(hdr[4:8])
		if version < Version1 || version > Version3 {
			return nil, fmt.Errorf("gguf: unsupported version %d in shard %d", version, i+1)
		}

		nTensor := binary.LittleEndian.Uint64(hdr[8:16])
		nKV := binary.LittleEndian.Uint64(hdr[16:24])

		// Try to get file size from reader if it's an *os.File
		var fileSz int64 = -1 // unknown by default
		if f, ok := ra.(*os.File); ok {
			info, _ := f.Stat()
			if info != nil {
				fileSz = info.Size()
			}
		}

		shards[i] = &shardHandle{
			r:       ra,
			fileSz:  fileSz,
			version: version,
			nKV:     nKV,
			nTensor: nTensor,
			index:   i,
		}

		totalTensors += nTensor
		totalKV += nKV
	}


	g.r = &multiReaderAt{shards: shards}
	g.fileSz = 0 // computed by multiReaderAt
	g.version = shards[0].version
	g.nTensor = totalTensors
	g.nKV = totalKV

	// panic("TEST PANIC - checking if we get here")

	// Validate split metadata consistency across all shards
	if err := g.validateSplitMetadata(shards, totalTensors); err != nil {
		return nil, err
	}

	return g, nil
}

// validateSplitMetadata checks that all shards have consistent split metadata (split.no, split.count).
func (g *GGUF) validateSplitMetadata(shards []*shardHandle, totalTensors uint64) error {
	if len(shards) < 2 {
		return nil // single file, no split validation needed
	}

	// Find expected split count by scanning all shards for "split.count" metadata
	var expectedCount int64
	for i, shard := range shards {
		tmpG := &GGUF{r: shard.r, alignment: defaultAlignment, fileSz: shard.fileSz, nKV: shard.nKV}
		if err := tmpG.walkKVSection(); err != nil {
			return fmt.Errorf("gguf: read metadata from shard %d: %w", i+1, err)
		}

		var splitNo, splitCount int64
		for _, e := range tmpG.kvEntries {
			if !e.loaded {
				continue
			}
			switch e.key {
			case "split.no":
				splitNo = e.value.Int
			case "split.count":
				splitCount = e.value.Int
			}
		}

		shard.splitNo = splitNo
		shard.splitCount = splitCount

		if expectedCount == 0 {
			expectedCount = splitCount
		} else if splitCount != expectedCount {
			return fmt.Errorf("gguf: shard %d has inconsistent split.count (got %d, expected %d)", i+1, splitCount, expectedCount)
		}
	}

	if expectedCount == 0 {
		return nil // no split metadata found in any shard, treat as single file
	}

	if expectedCount == 0 {
		return nil // no split metadata found in any shard, treat as single file
	}

	// Validate all shards have consistent split.no and split.count
	// Parse each shard's basename independently to extract base name pattern (e.g., "model" from "model-00001-of-00003.gguf")
	var expectedBaseName string
	for i, shard := range shards {
		var actualBaseName string

		tmpG := &GGUF{r: shard.r, alignment: defaultAlignment, fileSz: shard.fileSz, nKV: shard.nKV}
		if err := tmpG.walkKVSection(); err != nil {
			return fmt.Errorf("gguf: read metadata from shard %d: %w", i+1, err)
		}

		var splitNo, splitCount int64
		for _, e := range tmpG.kvEntries {
			if !e.loaded {
				continue
			}
			switch e.key {
			case "split.no":
				splitNo = e.value.Int
			case "split.count":
				splitCount = e.value.Int
			}
		}

		// Extract base name from shard filename (e.g., "model" from "/path/to/model-00001-of-00003.gguf")
		actualBasename := filepath.Base(shard.path)
		if before, _, found := strings.Cut(actualBasename, "-of-"); found {
			// Remove the shard number suffix (e.g., "model-00001" -> "model")
			if dashIdx := strings.LastIndex(before, "-"); dashIdx != -1 {
				actualBaseName = before[:dashIdx]
			} else {
				actualBaseName = before
			}
		}

		if expectedBaseName == "" {
			expectedBaseName = actualBaseName
		} else if actualBaseName != expectedBaseName {
			return fmt.Errorf("gguf: shard %d has mismatched base name (got %q, want %q)", i+1, actualBaseName, expectedBaseName)
		}

		if splitCount != int64(len(shards)) {
			return fmt.Errorf("gguf: shard %d has mismatched split.count (got %d, expected %d)", i+1, splitCount, len(shards))
		}
		if splitNo < 0 || splitNo >= int64(len(shards)) {
			return fmt.Errorf("gguf: shard %d has invalid split.no=%d for count=%d", i+1, splitNo, splitCount)
		}

		shard.splitNo = splitNo
		shard.splitCount = splitCount
	}

	// Sort shards by split.no to ensure correct order
	sort.Slice(shards, func(i, j int) bool {
		return shards[i].splitNo < shards[j].splitNo
	})

	// Re-index after sorting
	for i := range shards {
		shards[i].index = i
	}

	// Populate splitInfo for IsSplit() to return true
	g.splitInfo = &splitInfo{
		shards:       shards,
		totalTensors: totalTensors,
	}

	return nil
}

// NewReaderFile opens one or more GGUF files from paths and returns a lazy [*GGUF] reader. For
// single-file GGUFs pass one path; for multi-shard splits (e.g., TestModel-V4), pass all shard
// paths in order — the library validates headers match and combines into a single logical reader.
// The returned *GGUF must be closed via [GGUF.Close] when done. NewReaderFile reads only the 24-byte
// GGUF header immediately; KV metadata and tensor info are walked lazily on first call to
// [GGUF.Metadata] or [GGUF.Tensors].
//
// Example — single file:
//
//	g, err := gguf.NewReaderFile("model.gguf")
//	if err != nil { log.Fatal(err) }
//	defer g.Close()
//
// Example — multi-shard split (files passed in order):
//
//	g, err := gguf.NewReaderFile(
//	    "shard-00001-of-00003.gguf",
//	    "shard-00002-of-00003.gguf",
//	    "shard-00003-of-00003.gguf",
//	)
func NewReaderFile(files ...string) (*GGUF, error) {
	if len(files) == 0 {
		return nil, fmt.Errorf("gguf: no files provided")
	}

	g := &GGUF{
		version:   Version3, // placeholder; validated per shard below
		nTensor:   1,        // placeholder; summed across shards below
		nKV:       0,        // placeholder; summed across shards below
		alignment: defaultAlignment,
	}


	if len(files) == 1 {
		// Single file — open and validate header
		f, err := os.Open(files[0])
		if err != nil {
			return nil, fmt.Errorf("gguf: open %q: %w", files[0], err)
		}

		info, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: stat %q: %w", files[0], err)
		}

		var hdr [24]byte
		if _, err := io.ReadFull(io.NewSectionReader(f, 0, 24), hdr[:]); err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: read header for %q: %w", files[0], err)
		}

		if string(hdr[0:4]) != Magic {
			f.Close()
			return nil, fmt.Errorf("gguf: invalid magic in %q", files[0])
		}

		version := binary.LittleEndian.Uint32(hdr[4:8])
		if version < Version1 || version > Version3 {
			f.Close()
			return nil, fmt.Errorf("gguf: unsupported version %d in %q", version, files[0])
		}

		g.r = f
		g.fileSz = info.Size()
		g.sourcePath = files[0]
		g.version = version
		g.nTensor = binary.LittleEndian.Uint64(hdr[8:16])
		g.nKV = binary.LittleEndian.Uint64(hdr[16:24])

		return g, nil
	}


	// Multi-file — validate all headers match and combine into multiReaderAt
	shards := make([]*shardHandle, len(files))
	var totalTensors uint64
	var totalKV uint64

	for i, path := range files {
		f, err := os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("gguf: open %q (shard %d): %w", path, i+1, err)
		}

		info, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: stat %q: %w", path, err)
		}

		var hdr [24]byte
		if _, err := io.ReadFull(io.NewSectionReader(f, 0, 24), hdr[:]); err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: read header for %q (shard %d): %w", path, i+1, err)
		}

		if string(hdr[0:4]) != Magic {
			f.Close()
			return nil, fmt.Errorf("gguf: invalid magic in %q (shard %d)", path, i+1)
		}

		version := binary.LittleEndian.Uint32(hdr[4:8])
		if version < Version1 || version > Version3 {
			f.Close()
			return nil, fmt.Errorf("gguf: unsupported version %d in %q (shard %d)", version, path, i+1)
		}

		nTensor := binary.LittleEndian.Uint64(hdr[8:16])
		nKV := binary.LittleEndian.Uint64(hdr[16:24])

		shards[i] = &shardHandle{
			r:       f,
			fileSz:  info.Size(),
			version: version,
			nKV:     nKV,
			nTensor: nTensor,
			path:    path,
			index:   i,
		}

		totalTensors += nTensor
		totalKV += nKV
	}

	g.r = &multiReaderAt{shards: shards}
	g.fileSz = 0 // computed by multiReaderAt
	g.version = shards[0].version
	g.nTensor = totalTensors
	g.nKV = totalKV

	// Validate split metadata consistency across all shards
	if err := g.validateSplitMetadata(shards, totalTensors); err != nil {
		return nil, err
	}

	return g, nil
}

// Open opens a single GGUF file by path and returns a lazy [*GGUF] reader. For multi-shard splits,
// pass all shard paths explicitly to [NewReaderFile] in order — the library validates headers match
// and combines them into a single logical reader. The returned *GGUF must be closed via [GGUF.Close]
// when done.
//
// Example — single file:
//
//	g, err := gguf.NewReaderFile("model.gguf")
//	if err != nil { log.Fatal(err) }
//	defer g.Close()
//
// Example — multi-shard (pass any shard; library finds the rest):
//
//	g, err := gguf.NewReaderFile("shard-00002-of-00003.gguf") // auto-combines all 3 shards
func Open(path string) (*GGUF, error) {
	return NewReaderFile(path)
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
				for range int(count) {
					slenBuf := make([]byte, 8)
					if _, err := readFull(g.r, slenBuf, int64(pos)); err != nil {
						return fmt.Errorf("gguf: kv[%d] array string len: %w", i, err)
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
		for d := range int(shapeLen) {
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
// Tensor streaming read methods
// ---------------------------------------------------------------------------

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
	n := min(len(buf), int(avail))
	data := buf[:n]
	read, err := t.r.ReadAt(data, t.off+off)
	return read, err
}
