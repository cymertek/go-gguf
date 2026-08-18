// Package gguf provides lazy-loaded reading and writing of GGUF (GPT-Generated Unified Format) files.
//
// GGUF is a binary format used to store quantized model weights, developed by llama.cpp.
// It consists of a key-value metadata section and tensor data sections.
//
// The primary entry points are [Open] for reading and [Create] for writing.
// Both operate on io.ReaderAt/io.Writer interfaces, enabling file-backed or network-backed GGUF files.
package gguf

import (
	"errors"
	"fmt"
	"io"
	"sync"
)

//go:generate go run golang.org/x/tools/cmd/stringer@latest -type=BType,GgmlType -output=zz_generated.ggufmetadatavaluetype.stringer.go

// Version constants.
const (
	Version1 = 1
	Version2 = 2
	Version3 = 3
)

// Magic bytes identifying a GGUF file.
const Magic = "GGUF"

// ErrCorrupt is returned when the GGUF file is malformed.
var ErrCorrupt = errors.New("gguf: corrupt file")

// ---------------------------------------------------------------------------
// BType — GGUF metadata value type (wire format enum)
// ---------------------------------------------------------------------------

// BType represents a GGUF metadata value type on the wire format.
// It identifies how a single key-value pair's value is encoded in the file.
//
// Valid values: BTypeUint8, BTypeInt8, BTypeUint16, BTypeInt16, BTypeUint32,
// BTypeInt32, BTypeFloat32, BTypeBool, BTypeString, BTypeArray, BTypeUint64,
// BTypeInt64, BTypeFloat64.
type BType uint32

const (
	BTypeUint8   BType = 0
	BTypeInt8    BType = 1
	BTypeUint16  BType = 2
	BTypeInt16   BType = 3
	BTypeUint32  BType = 4
	BTypeInt32   BType = 5
	BTypeFloat32 BType = 6
	BTypeBool    BType = 7
	BTypeString  BType = 8
	BTypeArray   BType = 9
	BTypeUint64  BType = 10
	BTypeInt64   BType = 11
	BTypeFloat64 BType = 12
)

// Size returns the wire size in bytes of a single scalar value for this type.
// For example, BTypeFloat32.Size() == 4, BTypeBool.Size() == 1.
// Returns 0 for unsupported or array types (BTypeString/BTypeArray have variable sizes).
func (t BType) Size() int {
	switch t {
	case BTypeUint8, BTypeInt8, BTypeBool:
		return 1
	case BTypeUint16, BTypeInt16:
		return 2
	case BTypeUint32, BTypeInt32, BTypeFloat32:
		return 4
	case BTypeUint64, BTypeInt64, BTypeFloat64:
		return 8
	default:
		return 0
	}
}

// ---------------------------------------------------------------------------
// GgmlType — tensor data quantization / layout type
// ---------------------------------------------------------------------------

// GgmlType represents a GGML tensor type, i.e. the quantization scheme or raw floating-point
// format used to encode a tensor's data on disk. Each value selects a specific block layout;
// see [GgmlType.BlockBytes] and [GgmlType.ElementsPerBlock].
type GgmlType uint32

const (
	GgmlF32    GgmlType = 0
	GgmlF16    GgmlType = 1
	GgmlQ4_0   GgmlType = 2
	GgmlQ4_1   GgmlType = 3
	GgmlQ4_1F  GgmlType = 4 // deprecated
	GgmlIQ2XXS GgmlType = 5
	GgmlQ5_0   GgmlType = 6
	GgmlQ5_1   GgmlType = 7
	GgmlQ8_0   GgmlType = 8
	GgmlQ8_1   GgmlType = 9
	GgmlQ2_K   GgmlType = 10
	GgmlQ3_K_S GgmlType = 11
	GgmlQ3_K_L GgmlType = 12
	GgmlIQ3XXS GgmlType = 13
	GgmlIQ3S   GgmlType = 14
	GgmlIQ2S   GgmlType = 15
	GgmlIQ2M   GgmlType = 16
	GgmlIQ4XL  GgmlType = 17
	GgmlIQ4NS  GgmlType = 18
	GgmlQ6_K   GgmlType = 19
	GgmlQ8_K   GgmlType = 20
	// Aliases for tensor types sharing block config with other values
	GgmlQ4_K GgmlType = 13 // maps to IQ3_XXS block config
	GgmlQ5_K GgmlType = 14 // maps to IQ3_S block config
	GgmlIQ4M GgmlType = 21
	GgmlDFP  GgmlType = 22
	GgmlDP   GgmlType = 23
	GgmlI2   GgmlType = 24
	GgmlIM   GgmlType = 25
	GgmlNVFP4 GgmlType = 40 // NVIDIA NVFP4: 64-element block (4x16 sub-blocks), UE4M3 sub-scales
)

// GgmlName returns a short, human-readable name for this quantization type
// (e.g., "Q4_0", "F32", "NVFP4"). Useful for logging and display.
func (t GgmlType) GgmlName() string {
	switch t {
	case GgmlF32:
		return "F32"
	case GgmlF16:
		return "F16"
	case GgmlQ4_0:
		return "Q4_0"
	case GgmlQ4_1:
		return "Q4_1"
	case GgmlIQ2XXS:
		return "IQ2_XXS"
	case GgmlQ5_0:
		return "Q5_0"
	case GgmlQ5_1:
		return "Q5_1"
	case GgmlQ8_0:
		return "Q8_0"
	case GgmlQ8_1:
		return "Q8_1"
	case GgmlQ2_K:
		return "Q2_K"
	case GgmlQ3_K_S, GgmlQ3_K_L:
		return "Q3_K"
	case GgmlIQ3XXS:
		return "IQ3_XXS"
	case GgmlIQ3S:
		return "IQ3_S"
	case GgmlIQ2S:
		return "IQ2_S"
	case GgmlIQ2M:
		return "IQ2_M"
	case GgmlIQ4XL:
		return "IQ4_XL"
	case GgmlIQ4NS:
		return "IQ4_NS"
	case GgmlQ6_K:
		return "Q6_K"
	case GgmlQ8_K:
		return "Q8_K"
	case GgmlIQ4M:
		return "IQ4_M"
	case GgmlDFP:
		return "DFP"
	case GgmlDP:
		return "DP"
	case GgmlI2:
		return "I2"
	case GgmlIM:
		return "IM"
	case GgmlNVFP4:
		return "NVFP4"
	default:
		return fmt.Sprintf("GgmlType(%d)", t)
	}
}

// ElementsPerBlock returns the number of tensor elements represented by one quantization block.
// For raw float types (F32, F16) this is 1. For quantized types it varies: Q4_0/Q5_0/Q8_0 use 32,
// K-quantizations (Q2_K..Q6_K) use 256, NVFP4 uses 64. Returns 0 for unsupported types.
func (t GgmlType) ElementsPerBlock() int {
	switch t {
	case GgmlF32, GgmlF16:
		return 1
	case GgmlQ4_0, GgmlQ5_0, GgmlQ8_0:
		return 32
	case GgmlQ2_K, GgmlQ3_K_S, GgmlQ3_K_L, GgmlQ4_K, GgmlQ5_K, GgmlQ6_K:
		return 256
	case GgmlNVFP4:
		return 64
	default:
		return 0
	}
}

// BlockBytes returns the number of bytes consumed by one quantization block on disk.
// For F32 this is 4, for Q8_0 it is 34 (2-byte scale + 32 signed bytes), etc.
// Returns 0 if the type has no standard block layout (e.g., deprecated or unknown types).
func (t GgmlType) BlockBytes() int {
	switch t {
	case GgmlF32:
		return 4
	case GgmlF16:
		return 2
	case GgmlQ4_0:
		return 18
	case GgmlQ5_0:
		return 22
	case GgmlQ8_0:
		return 34
	case GgmlQ2_K:
		return 64
	case GgmlQ3_K_S:
		return 62
	case GgmlQ3_K_L:
		return 64
	case GgmlQ4_K:
		return 70
	case GgmlQ5_K:
		return 76
	case GgmlQ6_K:
		return 98
	case GgmlNVFP4:
		return 40
	default:
		return 0
	}
}

// IsSupported returns true if this type has a known block layout and is supported
// by [github.com/cymertek/go-quant] for dequantization. A return value of false means the
// type constant exists but no decoder is implemented yet. Use [Tensor.Reader] to stream data
// into go-quant's decoder for dequantization instead of loading full tensors into memory.
func (t GgmlType) IsSupported() bool {
	return t.BlockBytes() != 0
}

// ---------------------------------------------------------------------------
// Value — parsed GGUF metadata value
// ---------------------------------------------------------------------------

// Value holds a single metadata value that has been parsed from the GGUF wire format.
// Use one of the typed accessor methods (AsBool, AsInt, AsUint64, AsFloat, AsString) to
// extract the actual Go value. Each accessor returns a second bool indicating whether the
// conversion succeeded for this BType.
type Value struct {
	BType    BType   // type tag identifying which field is valid
	Int      int64   // integer or bool (true for bool) -- valid for all numeric types
	Float    float64 // floating-point value -- valid when BType is BTypeFloat32 or BTypeFloat64
	Str      string  // string value -- valid when BType is BTypeString
	ElemType BType   // element type for arrays -- valid when BType is BTypeArray
	Raw      []byte  // raw wire bytes for arrays / strings -- valid when BType is BTypeArray or BTypeString
}

// AsBool returns the value as bool and true if the stored BType is BTypeBool,
// otherwise returns false in the second return value. Safe to call on any Value;
// only bool-typed values will succeed.
func (v Value) AsBool() (bool, bool) {
	if v.BType != BTypeBool {
		return false, false
	}
	return v.Int != 0, true
}

// AsInt returns the value as int64 and true if the stored BType is one of the
// signed or unsigned integer types (BTypeUint8 through BTypeInt64). Returns zero
// and false for float, string, bool, or array values.
func (v Value) AsInt() (int64, bool) {
	switch v.BType {
	case BTypeInt8, BTypeInt16, BTypeInt32, BTypeInt64,
		BTypeUint8, BTypeUint16, BTypeUint32, BTypeUint64:
		return v.Int, true
	default:
		return 0, false
	}
}

// AsUint64 returns the value as uint64 and true if the stored BType is one of the
// unsigned integer types (BTypeUint8, BTypeUint16, BTypeUint32, BTypeUint64).
// Signed integer types are rejected. Returns zero and false otherwise.
func (v Value) AsUint64() (uint64, bool) {
	switch v.BType {
	case BTypeUint8, BTypeUint16, BTypeUint32, BTypeUint64:
		return uint64(v.Int), true
	default:
		return 0, false
	}
}

// AsFloat returns the value as float64 and true if the stored BType is one of the
// floating-point types (BTypeFloat32 or BTypeFloat64). Returns zero and false for
// integer, string, bool, or array values. Float32 values are converted to float64.
func (v Value) AsFloat() (float64, bool) {
	switch v.BType {
	case BTypeFloat32, BTypeFloat64:
		return v.Float, true
	default:
		return 0, false
	}
}

// AsString returns the value as string and true if the stored BType is BTypeString.
// Returns empty string and false for all other types.
func (v Value) AsString() (string, bool) {
	if v.BType == BTypeString {
		return v.Str, true
	}
	return "", false
}

// ---------------------------------------------------------------------------
// TensorInfo — tensor metadata from GGUF wire format
// ---------------------------------------------------------------------------

// TensorInfo holds the parsed metadata for a single tensor in a GGUF file.
// It describes the tensor's name, shape (dimensions), quantization type, offset within
// the tensor-data section, and total byte size. The Offset field is relative to dataStart
// (the aligned start of the tensor-data section) and is NOT necessarily 32-byte aligned --
// alignment padding occurs between tensors. NBytes can be zero for unsupported GgmlType values;
// in that case [Tensor.Bytes] will derive it from consecutive offsets.
type TensorInfo struct {
	Name     string    // tensor name (UTF-8)
	Shape    []uint64  // dimension sizes
	GgmlType GgmlType  // quantization/type enum
	Offset   uint64    // relative offset from dataStart (un-aligned, per GGUF spec)
	NBytes   uint64    // total raw tensor data size in bytes
}

// ---------------------------------------------------------------------------
// GGUF — lazy-loaded GGUF file reader
// ---------------------------------------------------------------------------

// GGUF is the primary lazy reader for GGUF files. It accepts any io.ReaderAt at open time,
// reads only the 24-byte header immediately, and walks KV metadata + tensor info on-demand
// via [GGUF.Metadata], [GGUF.Tensors] calls. The returned *GGUF must be closed with [GGUF.Close]
// when no longer needed to release the underlying file handle (if it is an io.Closer).
//
// GGUF is safe for concurrent reads of different tensors or metadata entries, but each method
// call (Metadata, Tensors) takes a mutex and must not be called concurrently with itself.
type GGUF struct {
	r       io.ReaderAt     // any ReaderAt: *os.File, S3 range reader, gzip decompressor, etc.
	fileSz  int64           // total file size (provided by caller at Open time)

	version   uint32        // always Version3 after validation
	nKV       uint64        // from header
	nTensor   uint64        // from header
	alignment uint64        // resolved from KV section or defaults to 32

	mu     sync.Mutex
	kvEnd  uint64          // file offset where KV section ends (set by walkKVSection)
	kvWalk bool            // true once kvEntries are populated

	kvNames  []string        // ordered key names (populated after Metadata() call)
	kvEntries []*kvEntry     // populated after Metadata() call

	tensorInfos  []TensorInfo   // populated after Tensors() call
	dataStart    uint64         // aligned start of tensor data section, set by walkTensorSection

	// Multi-shard support: when file is part of a split GGUF
	splitInfo *splitInfo     // nil for single-file; non-nil for multi-file shards
	sourcePath string        // original file path (for split detection)
}

// splitInfo holds the combined reader state for multi-shard GGUF files.
type splitInfo struct {
	shards       []*shardHandle // all shard readers, in order (0 = metadata shard)
	totalTensors uint64         // sum of nTensor across all shards
}

// shardHandle wraps an io.ReaderAt with its file size and header info.
type shardHandle struct {
	r            io.ReaderAt
	fileSz       int64
	version      uint32
	nKV          uint64
	nTensor      uint64
	path         string // original path for error reporting
	index        int    // 0-based index in split (after sorting by split.no)
	splitNo      int64  // extracted from KV metadata
	splitCount   int64  // extracted from KV metadata
	architecture string // extracted from KV metadata for cross-validation
}

// multiReaderAt combines multiple io.ReaderAt shards into a single logical reader.
type multiReaderAt struct {
	shards []*shardHandle
}

func (mr *multiReaderAt) ReadAt(buf []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("gguf: negative offset")
	}

	totalRead := 0
	pos := off

	for totalRead < len(buf) && pos >= 0 {
		shardIdx := -1
		var remainingInShard int64

		for i := range mr.shards {
			size := mr.shards[i].fileSz
			if pos < size {
				shardIdx = i
				remainingInShard = size - pos
				break
			}
			pos -= size // skip past this shard's entire file
		}

		if shardIdx == -1 || remainingInShard <= 0 {
			break // end of all shards reached
		}

		n := min(int64(len(buf[totalRead:])), remainingInShard)

		read, err := mr.shards[shardIdx].r.ReadAt(buf[totalRead:totalRead+int(n)], pos)
		totalRead += read
		pos += int64(read)

		if err != nil && err != io.EOF {
			return totalRead, err
		}
	}

	return totalRead, nil
}

// kvEntry is a single KV entry with offset info for lazy value loading.
type kvEntry struct {
	key     string      // parsed key name (UTF-8)
	btype   BType       // GGUF value type identifier
	wireSz  int64       // total wire size of value data bytes (excluding key_len, key, btype fields)
	rawOff  uint64      // absolute file offset where value bytes start
	value   Value       // populated for eager/small values or after Value() call
	loaded  bool        // true if value has been parsed/cached in .value
}

// ---------------------------------------------------------------------------
// Tensor — file-backed tensor handle with per-tensor partial-read cache
// ---------------------------------------------------------------------------

// Tensor is a file-backed tensor handle. It wraps [TensorInfo] metadata and provides
// an internal 1 MB read cache to avoid redundant disk seeks when reading overlapping regions.
// A *Tensor should be closed with [Tensor.Close] when no longer needed to release the cached buffer.
type Tensor struct {
	info      TensorInfo  // embedded: Name, Shape, GgmlType, Offset, NBytes
	absOffset uint64      // absolute file offset of aligned tensor data start

	gguf       *GGUF         // parent reader for ReadAt calls
	cacheMu    sync.Mutex    // protects cache fields below
	cache      []byte        // cached raw bytes from previous partial read (relative to absOffset)
	cacheOff   int64         // byte offset within the tensor where cache starts
	cacheEnd   uint64        // absolute file position of cache end (== absOffset + uint64(cacheOff+len(cache)))
	cacheValid bool          // true if cache contains valid data
}

// Info returns a copy of this tensor's metadata. The returned [TensorInfo]
// contains deep-copied slices so modifications to the return value won't affect
// the original tensor's data. Callers should use [Tensor.Bytes], [Tensor.ReadAt],
// or [Tensor.Dequant] for actual data access; Info() is read-only introspection.
func (t *Tensor) Info() TensorInfo {
	info := t.info
	if info.Shape != nil {
		info.Shape = make([]uint64, len(info.Shape))
		copy(info.Shape, t.info.Shape)
	}
	return info
}

// Close releases the per-tensor read cache buffer back to the pool.
func (t *Tensor) Close() {
	t.cacheMu.Lock()
	defer t.cacheMu.Unlock()
	if len(t.cache) > 0 {
		putBuffer(t.cache)
		t.cache = nil
	}
	t.cacheOff = -1
	t.cacheEnd = 0
}

// ---------------------------------------------------------------------------
// MetadataEntry — lazy handle for one KV entry
// ---------------------------------------------------------------------------

// MetadataEntry provides a thin handle to a single metadata key-value pair with
// lazy value loading from the underlying file.
type MetadataEntry struct {
	key    string      // KV name
	btype  BType       // GGUF type tag
	wireSz int64       // total wire size of value data (excluding overhead)
	rawOff uint64      // absolute file offset where value bytes start

	gguf   *GGUF       // parent reader for lazy reads
	mu     sync.Mutex
	val    Value       // populated after first Value() call
	ok     bool        // true if val has been loaded and cached
}

// Name returns the entry's key name as stored in the GGUF file.
func (e *MetadataEntry) Name() string { return e.key }

// BType returns the GGUF value type identifier for this metadata entry,
// indicating how [Value] or one of the typed accessors should be used to read it.
func (e *MetadataEntry) BType() BType { return e.btype }

// Size returns the total wire size of the value data in bytes, excluding the key_len,
// key, and btype fields from the GGUF wire format. Useful for pre-allocating buffers.
func (e *MetadataEntry) Size() int64 { return e.wireSz }

// Value loads and parses the raw KV data from file at the stored offset, caching the
// result so subsequent calls return immediately without disk I/O. For string and array
// values this may involve a seek; for small scalar values (<=64 bytes) the value is
// already eagerly loaded during [GGUF.Metadata] and returns instantly.
func (e *MetadataEntry) Value() (Value, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ok {
		return e.val, nil
	}
	v, err := readLazyValue(e.gguf.r, int64(e.rawOff), e.btype, e.wireSz)
	if err != nil {
		return Value{}, err
	}
	e.val = v
	e.ok = true
	return v, nil
}

// SetValue updates the cached value so it can be written back to a new GGUF file.
func (e *MetadataEntry) SetValue(v any) error {
	switch val := v.(type) {
	case Value:
		e.mu.Lock()
		defer e.mu.Unlock()
		e.val = val
		e.ok = true
	default:
		return fmt.Errorf("gguf: SetValue expects gguf.Value, got %T", v)
	}
	return nil
}

// AsString returns the parsed string value, or an error if the entry's BType is not
// BTypeString. Convenience wrapper around [Value].AsString().
func (e *MetadataEntry) AsString() (string, error) {
	v, err := e.Value()
	if err != nil {
		return "", err
	}
	if s, ok := v.AsString(); ok {
		return s, nil
	}
	return "", fmt.Errorf("gguf: key %q is not a string (got btype %d)", e.key, e.btype)
}

// AsInt64 returns the parsed integer value as int64, or an error if the entry's BType is not
// a supported integer type. Convenience wrapper around [Value].AsInt().
func (e *MetadataEntry) AsInt64() (int64, error) {
	v, err := e.Value()
	if err != nil {
		return 0, err
	}
	i, ok := v.AsInt()
	if !ok {
		return 0, fmt.Errorf("gguf: key %q is not an integer (got btype %d)", e.key, e.btype)
	}
	return i, nil
}

// AsFloat64 returns the parsed float value as float64, or an error if the entry's BType is not
// a supported floating-point type. Convenience wrapper around [Value].AsFloat().
func (e *MetadataEntry) AsFloat64() (float64, error) {
	v, err := e.Value()
	if err != nil {
		return 0, err
	}
	f, ok := v.AsFloat()
	if !ok {
		return 0, fmt.Errorf("gguf: key %q is not a float (got btype %d)", e.key, e.btype)
	}
	return f, nil
}

// ---------------------------------------------------------------------------
// KVEntry — builder-friendly type for writing metadata
// ---------------------------------------------------------------------------

// KVEntry holds a single key-value pair ready to be written into a GGUF file.
// Use [StreamWriter.SetMetadataEntry] or [GGUFWriter.SetKV] with this type.
type KVEntry struct {
	Key   string // metadata key name (UTF-8)
	Value Value  // parsed value; set BType and the appropriate field (Int, Float, Str, Raw)
}

// ---------------------------------------------------------------------------
// Buffer Pool — package-level tiered sync.Pool for reusable byte slices
// ---------------------------------------------------------------------------

var (
	smallPool = &sync.Pool{
		New: func() any { return make([]byte, 0, 64<<10) }, // default 64KB
	}
	largePool = &sync.Pool{
		New: func() any { return make([]byte, 0, 1<<20) }, // default 1MB
	}
)

// getBuffer returns a byte slice of at least minSize bytes from the appropriate pool tier.
func getBuffer(minSize int) []byte {
	if minSize <= 64 {
		buf := make([]byte, minSize)
		return buf
	}
	var p *sync.Pool
	if minSize >= 1<<20 {
		p = largePool
	} else {
		p = smallPool
	}
	b := p.Get().([]byte)
	if cap(b) < minSize {
		buf := make([]byte, minSize)
		p.Put(b[:64<<10]) // put pooled buffer back (truncate to keep within pool size)
		return buf
	}
	return b[:minSize]
}

// putBuffer returns a byte slice to the appropriate pool tier.
func putBuffer(buf []byte) {
	if cap(buf) >= 64<<10 && len(buf) == cap(buf) { // only return full-size buffers that fit in smallPool
		smallPool.Put(buf[:64<<10]) //nolint:staticcheck // minimal allocation, acceptable for pool
	} else if cap(buf) >= 1<<20 && len(buf) == cap(buf) {
		largePool.Put(buf) //nolint:staticcheck // minimal allocation, acceptable for pool
	}
	// otherwise: let GC collect (partial-use or oversized buffers)
}

// ---------------------------------------------------------------------------
// Multi-shard GGUF support — public API methods
// ---------------------------------------------------------------------------

// IsSplit returns true when the underlying file is one shard of a multi-file split GGUF
// (e.g., TestModel-V4). When true, [SplitInfo] will be non-nil and contains per-shard metadata.
func (g *GGUF) IsSplit() bool {
	return g.splitInfo != nil && len(g.splitInfo.shards) > 1
}

// SplitInfo returns metadata about all shards in a multi-shard GGUF split, or nil if this
// is a single-file GGUF. The returned [SplitInfo] contains per-shard handles ([SplitShard])
// that can each be used to read their own metadata and tensors independently.
func (g *GGUF) SplitInfo() *SplitInfo {
	if g.splitInfo == nil {
		return nil
	}
	info := &SplitInfo{
		Count:           len(g.splitInfo.shards),
		TensorsPerShard: make([]uint64, len(g.splitInfo.shards)),
		Shards:          make([]*SplitShard, len(g.splitInfo.shards)),
	}

	for i, s := range g.splitInfo.shards {
		info.TensorsPerShard[i] = s.nTensor

		// Construct GGUF reader directly from shard handle (no header re-read needed)
		shardGGUF := &GGUF{
			r:       s.r,
			fileSz:  s.fileSz,
			version: s.version,
			nTensor: s.nTensor,
			nKV:     s.nKV,
			alignment: defaultAlignment,
		}

		info.Shards[i] = &SplitShard{
			Index:      i,
			Path:       s.path,
			Size:       s.fileSz,
			Version:    s.version,
			NumTensors: s.nTensor,
			NumKV:      s.nKV,
			reader:     shardGGUF,
		}
	}

	return info
}

// SplitInfo contains metadata about a multi-shard GGUF file (e.g., TestModel-V4).
// It reports the total number of shards, per-shard tensor counts from headers, and provides
// [SplitShard] handles for reading each shard's metadata and tensors independently.
type SplitInfo struct {
	Count           int           // total number of shards
	TensorsPerShard []uint64     // tensor count per shard (from headers)
	Shards          []*SplitShard // actual shard handles with readers
}

// SplitShard represents a single shard in a multi-shard GGUF file. It carries file-level
// metadata (path, size, version, tensor/KV counts) and provides access to the shard's own
// [MetadataEntry] list and [*Tensor] handles via its GetMetadata and Tensors methods.
type SplitShard struct {
	Index      int       // 0-based index in the split
	Path       string    // original path to this shard
	Size       int64     // file size in bytes
	Version    uint32    // GGUF version from header
	NumTensors uint64    // tensor count from header
	NumKV      uint64    // KV count from header
	reader     *GGUF     // parent GGUF reader for this shard (nil for non-primary shards)
}

// GetMetadata returns the parsed [MetadataEntry] list for this shard by delegating to
// the underlying [*GGUF]. The returned slice is safe to read concurrently with other
// shards but must not be modified. Returns an error if the shard has no reader attached.
func (s *SplitShard) GetMetadata() ([]*MetadataEntry, error) {
	if s.reader == nil {
		return nil, fmt.Errorf("split shard %d has no reader", s.Index)
	}
	return s.reader.Metadata()
}

// Tensors returns the [*Tensor] handles for this shard by delegating to the underlying
// [*GGUF]. Each handle provides lazy [Tensor.ReadAt], [Tensor.Bytes], and [Tensor.Dequant]
// access. Returns an error if the shard has no reader attached.
func (s *SplitShard) Tensors() ([]*Tensor, error) {
	if s.reader == nil {
		return nil, fmt.Errorf("split shard %d has no reader", s.Index)
	}
	return s.reader.Tensors()
}

// ShardIndex reports the shard index for a [SplitInfo]. This method always returns -1 because
// [SplitInfo] describes a collection of shards, not an individual one. Use [SplitShard.Index]
// instead to get a specific shard's position in the split.
func (s *SplitInfo) ShardIndex() int { return -1 } // Not applicable to SplitInfo itself

