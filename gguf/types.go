// Package gguf provides lazy-loaded reading and writing of GGUF (GPT-Generated Unified Format) files.
//
// GGUF is a binary format used to store quantized model weights, developed by llama.cpp.
// It consists of a key-value metadata section and tensor data sections.
//
// The primary entry points are [Open] for reading and [Create] for writing.
// Both operate on io.ReaderAt/io.Writer interfaces, enabling file-backed or network-backed GGUF files.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
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

// BType represents a GGUF metadata value type.
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

// GgmlType represents a GGML tensor type (quantization scheme or raw float format).
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

// GgmlName returns the human-readable name for this quantization type.
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

// BlockBytes returns the number of bytes consumed by one quantization block.
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

// IsSupported returns true if this type is supported for reading and dequantization.
func (t GgmlType) IsSupported() bool {
	return t.BlockBytes() != 0
}

// ---------------------------------------------------------------------------
// Value — parsed GGUF metadata value
// ---------------------------------------------------------------------------

// Value holds a single metadata value that has been parsed from wire format.
type Value struct {
	BType    BType   // type tag
	Int      int64   // integer or bool (true for bool)
	Float    float64 // floating-point value
	Str      string  // string value
	ElemType BType   // element type for arrays
	Raw      []byte  // raw wire bytes for arrays / strings
}

// AsBool returns the value if it is a bool.
func (v Value) AsBool() (bool, bool) {
	if v.BType != BTypeBool {
		return false, false
	}
	return v.Int != 0, true
}

// AsInt returns the value if it is an integer type.
func (v Value) AsInt() (int64, bool) {
	switch v.BType {
	case BTypeInt8, BTypeInt16, BTypeInt32, BTypeInt64,
		BTypeUint8, BTypeUint16, BTypeUint32, BTypeUint64:
		return v.Int, true
	default:
		return 0, false
	}
}

// AsUint64 returns the value if it is an unsigned integer type.
func (v Value) AsUint64() (uint64, bool) {
	switch v.BType {
	case BTypeUint8, BTypeUint16, BTypeUint32, BTypeUint64:
		return uint64(v.Int), true
	default:
		return 0, false
	}
}

// AsFloat returns the value if it is a float type.
func (v Value) AsFloat() (float64, bool) {
	switch v.BType {
	case BTypeFloat32, BTypeFloat64:
		return v.Float, true
	default:
		return 0, false
	}
}

// AsString returns the value if it is a string type.
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
// reads only the 24-byte header, and walks KV metadata + tensor info on-demand via Seek/ReadAt calls.
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

// readKVSection reads the KV section and extracts split-related fields.
func (s *shardHandle) readKVSection() error {
	pos := uint64(24)

	for i := uint64(0); i < s.nKV; i++ {
		entryStart := pos

		// Read key length (8 bytes)
		keyLenBytes := make([]byte, 8)
		if _, err := readFull(s.r, keyLenBytes, int64(pos)); err != nil {
			return fmt.Errorf("kv[%d] key_len: %w", i, err)
		}
		keyLen := binary.LittleEndian.Uint64(keyLenBytes)

		pos += 8 + keyLen

		// Read btype (4 bytes)
		btypeBytes := make([]byte, 4)
		if _, err := readFull(s.r, btypeBytes, int64(pos)); err != nil {
			return fmt.Errorf("kv[%d] btype: %w", i, err)
		}
		btype := BType(binary.LittleEndian.Uint32(btypeBytes))
		pos += 4

		valueStart := pos // absolute file offset where value bytes start

		var wireSz int64
		var rawLen uint64 // for variable-size types (String, Array)

		switch btype {
		case BTypeBool, BTypeUint8, BTypeInt8:
			wireSz = 1
			pos += uint64(wireSz)
		case BTypeUint16, BTypeInt16:
			wireSz = 2
			pos += uint64(wireSz)
		case BTypeUint32, BTypeInt32, BTypeFloat32:
			wireSz = 4
			pos += uint64(wireSz)
		case BTypeUint64, BTypeInt64, BTypeFloat64:
			wireSz = 8
			pos += uint64(wireSz)
		case BTypeString:
			strLenBytes := make([]byte, 8)
			if _, err := readFull(s.r, strLenBytes, int64(pos)); err != nil {
				return fmt.Errorf("kv[%d] string_len: %w", i, err)
			}
			rawLen = binary.LittleEndian.Uint64(strLenBytes)
			wireSz = int64(rawLen) + 8
			pos += uint64(wireSz)
		case BTypeArray:
			elemTypeBytes := make([]byte, 4)
			countBytes := make([]byte, 8)
			if _, err := readFull(s.r, elemTypeBytes, int64(pos)); err != nil {
				return fmt.Errorf("kv[%d] array_elem_type: %w", i, err)
			}
			if _, err := readFull(s.r, countBytes, int64(pos)+4); err != nil {
				return fmt.Errorf("kv[%d] array_count: %w", i, err)
			}
			elemType := BType(binary.LittleEndian.Uint32(elemTypeBytes))
			count := binary.LittleEndian.Uint64(countBytes)

			pos += 12 // elem_type(4) + count(8) already read
			valueStart = pos

			switch elemType {
			case BTypeString:
				var totalStrData uint64
				for j := uint64(0); j < count; j++ {
					slenBuf := make([]byte, 8)
					if _, err := readFull(s.r, slenBuf, int64(pos)); err != nil {
						return fmt.Errorf("kv[%d] array string len[%d]: %w", i, j, err)
					}
					slen := binary.LittleEndian.Uint64(slenBuf)
					totalStrData += 8 + slen
					pos += uint64(8 + slen)
				}
				wireSz = int64(totalStrData) - 12
			default:
				elemSize := elemType.Size()
				if elemSize == 0 {
					elemSize = 1
				}
				wireSz = int64(count) * int64(elemSize) + 12
				pos += uint64(count) * uint64(elemSize)
			}
		default:
			wireSz = -1 // unsupported
		}

		keyData := make([]byte, keyLen)
		if _, err := readFull(s.r, keyData, int64(entryStart+8)); err != nil {
			return fmt.Errorf("kv[%d] key data: %w", i, err)
		}

		keyStr := string(keyData)

		// Extract split metadata
		if wireSz > 0 && btype == BTypeString {
			strLenBytes := make([]byte, 8)
			if _, err := readFull(s.r, strLenBytes, int64(valueStart)); err != nil {
				return fmt.Errorf("kv[%d] string_len: %w", i, err)
			}
			strLen := binary.LittleEndian.Uint64(strLenBytes)
			strData := make([]byte, strLen)
			if _, err := readFull(s.r, strData, int64(valueStart+8)); err != nil {
				return fmt.Errorf("kv[%d] string data: %w", i, err)
			}
			strVal := string(strData)

			switch keyStr {
			case "general.architecture":
				s.architecture = strVal
			}
		} else if wireSz > 0 && (btype >= BTypeUint8 && btype <= BTypeFloat64) {
			valData := make([]byte, wireSz)
			if _, err := readFull(s.r, valData, int64(valueStart)); err != nil {
				return fmt.Errorf("kv[%d] value: %w", i, err)
			}

			var numVal int64
			switch btype {
			case BTypeUint8:
				numVal = int64(valData[0])
			case BTypeInt8:
				numVal = int64(int8(valData[0]))
			case BTypeUint16:
				numVal = int64(binary.LittleEndian.Uint16(valData))
			case BTypeInt16:
				numVal = int64(int16(binary.LittleEndian.Uint16(valData)))
			case BTypeUint32:
				numVal = int64(binary.LittleEndian.Uint32(valData))
			case BTypeInt32:
				numVal = int64(int32(binary.LittleEndian.Uint32(valData)))
			case BTypeUint64, BTypeInt64:
				numVal = int64(binary.LittleEndian.Uint64(valData))
			}

			switch keyStr {
			case "split.no":
				s.splitNo = numVal
			case "split.count":
				s.splitCount = numVal
			}
		}
	}

	return nil
}

// ---------------------------------------------------------------------------
// Multi-shard GGUF support — detect and combine split files
// ---------------------------------------------------------------------------

// detectSplit checks if this file is part of a multi-shard GGUF split. If so, it locates all shards
// by pattern matching the filename (e.g., "model-00001-of-00003.gguf") and opens them as a unified reader.
func (g *GGUF) detectSplit(sourcePath string) (*splitInfo, error) {
	// First, walk KV section to check for split indicators
	if err := g.walkKVSection(); err != nil {
		return nil, fmt.Errorf("gguf: walk kv section for split detection: %w", err)
	}

	var hasSplit bool
	for _, e := range g.kvEntries {
		if e.key == "split.no" || e.key == "split.count" || e.key == "split.tensors.count" {
			hasSplit = true
			break
		}
	}

	if !hasSplit {
		return nil, nil // not a split file
	}

	// Extract split metadata from KV entries
	splitNo := int64(0)
	splitCount := 1
	var tensorsPerShard []int64

	for _, e := range g.kvEntries {
		if e.key == "split.no" && e.loaded {
			splitNo = e.value.Int
		} else if e.key == "split.count" && e.loaded {
			splitCount = int(e.value.Int)
		} else if e.key == "split.tensors.count" && e.loaded {
			tensorsPerShard = make([]int64, splitCount)
			for i := range tensorsPerShard {
				tensorsPerShard[i] = 0 // will be filled from each shard's header nTensors
			}
		}
	}

	if splitNo < 0 || splitNo >= int64(splitCount) {
		return nil, fmt.Errorf("gguf: invalid split.no=%d for count=%d", splitNo, splitCount)
	}

	// Locate all shard files based on filename pattern
	baseName := extractBaseName(sourcePath) // e.g., "DeepSeek-V4-Flash-0731-UD-IQ1_S" from path

	if baseName == "" {
		return nil, fmt.Errorf("gguf: could not extract base name from %q", sourcePath)
	}

	// Generate glob pattern: baseName-*-of-NNNNN.gguf
	// We need to find all files with the same base name and different shard numbers
	dir := filepath.Dir(sourcePath)
	pattern := filepath.Join(dir, fmt.Sprintf("%s-*-of-%0*d.gguf", baseName, len(strconv.Itoa(splitCount)), splitCount))

	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("gguf: glob pattern %q: %w", pattern, err)
	}

	if len(matches) < int(splitCount) {
		// Try a more flexible pattern without fixed shard count
		pattern = filepath.Join(dir, fmt.Sprintf("%s-*-of-*.gguf", baseName))
		matches, err = filepath.Glob(pattern)
		if err != nil {
			return nil, fmt.Errorf("gguf: glob pattern %q: %w", pattern, err)
		}
		if len(matches) < int(splitCount) {
			return nil, fmt.Errorf("gguf: expected at least %d shards matching %q, found %d", splitCount, pattern, len(matches))
		}
	}

	// Sort matches to ensure correct order (00001, 00002, ...)
	sort.Strings(matches)

	// Open all shards and validate headers + metadata consistency
	shards := make([]*shardHandle, splitCount)

	// First pass: open all shards and read their KV sections to verify they belong together
	type shardMeta struct {
		splitNo    int64
		splitCount int64
		arch       string // for cross-validation
	}
	shardMetas := make([]shardMeta, splitCount)

	for i, path := range matches[:splitCount] {
		f, err := os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("gguf: open shard %d (%s): %w", i+1, path, err)
		}

		info, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: stat shard %d: %w", i+1, err)
		}

		sz := info.Size()
		var hdr [24]byte
		if _, err := io.ReadFull(io.NewSectionReader(f, 0, 24), hdr[:]); err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: read shard %d header: %w", i+1, err)
		}

		version := binary.LittleEndian.Uint32(hdr[4:8])
		if string(hdr[0:4]) != Magic || version != Version3 {
			f.Close()
			return nil, fmt.Errorf("gguf: shard %d has invalid header", i+1)
		}

		nTensors := binary.LittleEndian.Uint64(hdr[8:16])
		nKV := binary.LittleEndian.Uint64(hdr[16:24])

		shards[i] = &shardHandle{
			r:       f,
			fileSz:  sz,
			version: version,
			nKV:     nKV,
			nTensor: nTensors,
			path:    path,
			index:   i,
		}

		// Read KV section to validate split metadata
		if err := shards[i].readKVSection(); err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: shard %d (%s) failed to read KV section: %w", i+1, path, err)
		}

		shardMetas[i] = shardMeta{
			splitNo:    shards[i].splitNo,
			splitCount: shards[i].splitCount,
			arch:       shards[i].architecture,
		}
	}

	// Second pass: validate all shards belong together
	refShard := shardMetas[0] // Use first shard as reference

	for i, meta := range shardMetas {
		if meta.splitNo < 0 || meta.splitNo >= refShard.splitCount {
			return nil, fmt.Errorf("gguf: shard %d has invalid split.no=%d (expected 0 to %d)",
				i+1, meta.splitNo, refShard.splitCount-1)
		}

		if meta.splitCount != refShard.splitCount {
			return nil, fmt.Errorf("gguf: shard %d has split.count=%d, but shard 0 has split.count=%d (shards don't match)",
				i+1, meta.splitCount, refShard.splitCount)
		}

		// Only validate architecture if both shards have it set
		if refShard.arch != "" && meta.arch != "" && meta.arch != refShard.arch {
			return nil, fmt.Errorf("gguf: shard %d has architecture %q, but shard 0 has architecture %q (shards don't match)",
				i+1, meta.arch, refShard.arch)
		}
	}

	// Verify all split.no values are unique and form a complete sequence 0..count-1
	seen := make(map[int64]bool)
	for _, meta := range shardMetas {
		if seen[meta.splitNo] {
			return nil, fmt.Errorf("gguf: duplicate split.no=%d found (shards not in order or corrupted)", meta.splitNo)
		}
		seen[meta.splitNo] = true
	}

	for i := int64(0); i < refShard.splitCount; i++ {
		if !seen[i] {
			return nil, fmt.Errorf("gguf: missing shard with split.no=%d (incomplete split set)", i)
		}
	}

	// Re-sort shards by split.no to ensure correct order
	type indexedShard struct {
		index int
		meta  shardMeta
	}
	indexed := make([]indexedShard, len(shards))
	for i := range shards {
		indexed[i] = indexedShard{i, shardMetas[i]}
	}
	sort.Slice(indexed, func(a, b int) bool {
		return indexed[a].meta.splitNo < indexed[b].meta.splitNo
	})

	// Reorder shards array by split.no
	for i, is := range indexed {
		shards[i] = shards[is.index]
		shards[i].index = i // Update index to match sorted position
	}

	totalTensors := uint64(0)
	for _, s := range shards {
		totalTensors += s.nTensor
	}

	return &splitInfo{
		shards:       shards,
		totalTensors: totalTensors,
	}, nil
}

// extractBaseName extracts the base name from a shard filename, removing the "-NNNNN-of-MMMM" suffix.
func extractBaseName(path string) string {
	base := filepath.Base(path)

	// Match pattern: *-of-*.gguf (e.g., "model-00001-of-00003.gguf")
	parts := strings.Split(base, "-of-")
	if len(parts) != 2 {
		return base // not a split file pattern
	}

	// parts[0] should be the base name (e.g., "DeepSeek-V4-Flash-0731-UD-IQ1_S-00001")
	// parts[1] should be the shard number + extension (e.g., "00003.gguf")

	// Extract just the base name part before the last "-"
	lastDash := strings.LastIndex(parts[0], "-")
	if lastDash == -1 {
		return parts[0] // no dash found, use as-is
	}

	return parts[0][:lastDash] // return everything before the last dash
}

// multiReaderAt wraps multiple io.ReaderAt shards into a single logical reader.
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

		n := int64(len(buf[totalRead:]))
		if n > remainingInShard {
			n = remainingInShard
		}

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

// Tensor is a file-backed tensor handle. It wraps TensorInfo metadata and provides
// a read cache that avoids disk seeks for repeated reads of the same region.
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

// Info returns a copy of this tensor's metadata.
func (t *Tensor) Info() TensorInfo { return t.info }

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

// Name returns the entry's key name.
func (e *MetadataEntry) Name() string { return e.key }

// BType returns the GGUF value type identifier.
func (e *MetadataEntry) BType() BType { return e.btype }

// Size returns the total wire size of the value data in bytes.
func (e *MetadataEntry) Size() int64 { return e.wireSz }

// Value loads and parses the raw KV data from file at stored offset, caching it on first call.
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

// AsString returns the parsed string value.
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

// AsInt64 returns the parsed integer value.
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

// AsFloat64 returns the parsed float value.
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
type KVEntry struct {
	Key   string
	Value Value
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

// ShardIndex returns the index of this GGUF in a split file (0-based).
func (g *GGUF) ShardIndex() int {
	if g.splitInfo == nil || len(g.splitInfo.shards) == 0 {
		return -1
	}
	for i, s := range g.splitInfo.shards {
		if s.r == g.r {
			return i
		}
	}
	return -1
}

// IsSplit returns true if this GGUF is part of a multi-shard file.
func (g *GGUF) IsSplit() bool {
	return g.splitInfo != nil && len(g.splitInfo.shards) > 1
}

// SplitInfo returns information about the split shards, or nil if not a split file.
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
		// Create a proper GGUF reader for each shard by calling OpenFromReader
		shardGGUF, err := OpenFromReader(s.r, s.fileSz)
		if err != nil {
			fmt.Printf("Warning: failed to open shard %d: %v\n", i+1, err)
			continue
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

// SplitInfo contains metadata about a multi-shard GGUF file.
type SplitInfo struct {
	Count           int          // total number of shards
	TensorsPerShard []uint64     // tensor count per shard (from headers)
	Shards          []*SplitShard // actual shard handles with readers
}

// SplitShard represents a single shard in a multi-shard GGUF file.
type SplitShard struct {
	Index      int       // 0-based index in the split
	Path       string    // original path to this shard
	Size       int64     // file size in bytes
	Version    uint32    // GGUF version from header
	NumTensors uint64    // tensor count from header
	NumKV      uint64    // KV count from header
	reader     *GGUF     // parent GGUF reader for this shard (nil for non-primary shards)
}

// GetMetadata returns the metadata entries for this shard.
func (s *SplitShard) GetMetadata() ([]*MetadataEntry, error) {
	if s.reader == nil {
		return nil, fmt.Errorf("split shard %d has no reader", s.Index)
	}
	return s.reader.Metadata()
}

// Tensors returns the tensor handles for this shard.
func (s *SplitShard) Tensors() ([]*Tensor, error) {
	if s.reader == nil {
		return nil, fmt.Errorf("split shard %d has no reader", s.Index)
	}
	return s.reader.Tensors()
}

// ShardIndex returns the index of this GGUF in a split file (0-based).
func (s *SplitInfo) ShardIndex() int { return -1 } // Not applicable to SplitInfo itself

