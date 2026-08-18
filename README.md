# go-gguf

A pure Go library for reading and writing [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) files, the binary format used by llama.cpp for model weights and metadata.

**Features:**
- Lazy loading — only reads file headers initially (~56 bytes per file), walks KV/tensor sections on demand
- Multi-shard support — automatically detects and combines split GGUF files
- Streaming tensor access via `io.ReadSeeker`/`io.ReaderAt` for external library consumption (e.g., go-quant)

```bash
go get github.com/cymertek/go-gguf
```

## Quick Start

### Reading a Single GGUF File

```go
package main

import (
    "fmt"
    "github.com/cymertek/go-gguf"
)

func main() {
    g, err := gguf.NewReaderFile("model.gguf")
    if err != nil {
        panic(err)
    }
    defer g.Close()

    fmt.Printf("Version: %d\n", g.Version())
    fmt.Printf("Tensors: %d\n", g.NumTensors())

    // Read metadata (lazy walk with eager threshold ≤64B)
    meta, _ := g.Metadata()
    for _, entry := range meta {
        v, _ := entry.Value()
        if s, ok := v.AsString(); ok {
            fmt.Printf("%s = %q\n", entry.Name(), s)
        }
    }

    // Read tensors (ONE seek to kvEnd + sequential read of all entries)
    tensors, _ := g.Tensors()
    for _, t := range tensors {
        info := t.Info()
        fmt.Printf("%-40s shape=%v type=%s nBytes=%d\n",
            info.Name, info.Shape, info.GgmlType.GgmlName(), info.NBytes)
    }
}
```

### Reading Multi-Shard GGUF Files

Pass all shard paths explicitly to `NewReaderFile` in order — the library validates headers match and combines into a single logical reader. Shards can be from different directories or mounts:

```go
// Example 1: All shards in same directory
g, err := gguf.NewReaderFile(
    "shard-00001-of-00003.gguf",
    "shard-00002-of-00003.gguf",
    "shard-00003-of-00003.gguf",
)

// Example 2: Shards from different directories/mounts
g, err = gguf.NewReaderFile(
    "/mnt/data1/shard-00001-of-00003.gguf",
    "/mnt/data2/shard-00002-of-00003.gguf",
    "/mnt/data3/shard-00003-of-00003.gguf",
)

if err != nil { panic(err) }
defer g.Close()

if g.IsSplit() {
    info := g.SplitInfo()
    
    fmt.Printf("Multi-shard: %d shards\n", info.Count)
    
    // Read metadata from first shard (typically the metadata-only shard)
    meta, _ := info.Shards[0].GetMetadata()
    for _, entry := range meta[:5] {
        fmt.Printf("  %s = %v\n", entry.Name(), entry.Value())
    }
    
    // Read tensors from each shard independently
    totalTensors := 0
    for i, shard := range info.Shards[1:] {
        tensors, _ := shard.Tensors()
        fmt.Printf("Shard %d: %d tensors\n", i+2, len(tensors))
        totalTensors += len(tensors)
    }
}

// Alternative: pass io.ReaderAt inputs directly (for S3 ranges, network streams, etc.)
f1, _ := os.Open("shard-00001-of-00003.gguf")
defer f1.Close()
f2, _ := os.Open("shard-00002-of-00003.gguf")
defer f2.Close()
f3, _ := os.Open("shard-00003-of-00003.gguf")
defer f3.Close()

g, err := gguf.NewReader(f1, f2, f3)
```

#### Custom Shard Distribution Control

You can control which tensors go into each shard based on size, name patterns, or custom logic:

```go
// Open source file and get all tensors
src, _ := gguf.Open("model.gguf")
tensors, _ := src.Tensors()

// Calculate total size for equal distribution
var totalSize uint64
for _, t := range tensors {
    info := t.Info() // Returns safe copy with copied slices
    totalSize += info.NBytes
}
targetPerShard := totalSize / 2

// Distribute tensors based on cumulative size
shard1Tensors := []*gguf.Tensor{}
shard2Tensors := []*gguf.Tensor{}
cumulativeSize := uint64(0)

for _, t := range tensors {
    info := t.Info()
    
    if cumulativeSize < targetPerShard {
        shard1Tensors = append(shard1Tensors, t)
        cumulativeSize += info.NBytes
    } else {
        shard2Tensors = append(shard2Tensors, t)
    }
}

// Write shards with custom distribution
writeShard("shard-00001-of-00002.gguf", metaEntries, shard1Tensors)
writeShard("shard-00002-of-00002.gguf", nil, shard2Tensors) // No metadata in data shards
```

### Writing a GGUF File

Wrap any `io.Writer` (file, gzip stream, network buffer) with `Create`:

```go
f, _ := os.Create("output.gguf")
defer f.Close()
w := gguf.Create(f) // Accepts io.Writer or io.WriterAt

// Add metadata
w.SetKV("general.architecture", gguf.Value{Str: "llama", BType: gguf.BTypeString})
w.SetKV("llama.context_length", gguf.Value{Int: 2048, BType: gguf.BTypeUint32})

// Add tensors — data is deferred until Close() so RAM usage stays O(1) per tensor
shape := []uint64{512, 32000}
data := makeTestData(512 * 32000 * 4) // F32 = 4 bytes per element

idx := w.AddTensor("tok_embeddings.weight", shape, gguf.GgmlF32)
w.WriteTensorData(idx, bytes.NewReader(data))

// Close flushes header + metadata + tensors to disk (streams deferred readers through pooled buffer)
written, _ := w.Close()
fmt.Printf("Wrote %d bytes\n", written)
```

For direct file paths, use `OpenForWrite` which manages the file handle internally:

```go
w, err := gguf.OpenForWrite("output.gguf")
if err != nil { panic(err) }
defer w.Close() // Also closes underlying file
```

### Streaming Between Files (Zero-Copy)

```go
// Copy tensors from src to dst with filtering — passthrough path uses deferred streaming
src, _ := gguf.Open("input.gguf")
defer src.Close()

f, _ := os.Create("output.gguf")
defer f.Close()
dst := gguf.Create(f)

// Stream tensors (include only weight tensors, exclude bias)
err := gguf.StreamCopy(dst, src, gguf.StreamOptions{
    Include: []string{"*.weight"},
    Exclude: []string{"*bias*"},
})

dst.Close() // Flushes all deferred tensor data to disk
```

### Transforming Tensors (In-Memory)

For transformations that require full tensor data in memory, use the `Transform` callback:

```go
// Custom transform — e.g., apply a scaling function to F32 tensors
err := gguf.StreamCopy(dst, src, gguf.StreamOptions{
    Include: []string{"*.weight"},
    Transform: func(data []byte, info gguf.TensorInfo) ([]byte, error) {
        // Process data in-place (e.g., apply quantization via go-quant library)
        return data, nil
    },
})
```

**Note:** Dequantization and requantization are handled by external libraries like [go-quant](https://github.com/cymertek/go-quant). The Transform callback receives raw tensor bytes — pass them to your quantizer of choice.

## API Reference

### Reading

| Function | Signature | Description |
|----------|-----------|-------------|
| `NewReader` | `func NewReader(ras ...io.ReaderAt) (*GGUF, error)` | Open one or more io.ReaderAt inputs. Single-file: pass one reader (e.g., *os.File). Multi-shard: pass all shard readers in order; library validates headers match and combines into a single logical reader. |
| `NewReaderFile` | `func NewReaderFile(files ...string) (*GGUF, error)` | Open GGUF files from paths. Single-file: pass one path. Multi-shard: pass all shard paths in order — shards can be from different directories/mounts; library validates naming convention and metadata consistency. |
| `Open` | `func Open(path string) (*GGUF, error)` | Convenience alias for `NewReaderFile(path)` — auto-detects multi-shard files by filename pattern (e.g., `model-*-of-NNNNN.gguf`). |
| `(*GGUF).Metadata()` | `func (g *GGUF) Metadata() ([]*MetadataEntry, error)` | Walk the KV section and return all metadata entries. Small values are eagerly parsed; large ones remain file-backed with lazy loading on `.Value()`. |
| `(*GGUF).Tensors()` | `func (g *GGUF) Tensors() ([]*Tensor, error)` | Walk tensor metadata section (ONE seek to kvEnd + N sequential reads) and return all tensors. |
| `(*GGUF).Version()` | `func (g *GGUF) Version() uint32` | Return GGUF file version (always 3 after validation). |
| `(*GGUF).NumTensors()` | `func (g *GGUF) NumTensors() int` | Return number of tensors from header. |
| `(*GGUF).IsSplit()` | `func (g *GGUF) IsSplit() bool` | Return true if this file is part of a multi-shard split. |
| `(*GGUF).SplitInfo()` | `func (g *GGUF) SplitInfo() *SplitInfo` | Return information about the split shards, or nil if not a split file. |

### Tensor Reading

| Function | Signature | Description |
|----------|-----------|-------------|
| `(*Tensor).Info()` | `func (t *Tensor) Info() TensorInfo` | Return tensor metadata as a safe copy with copied slices. Access `.NBytes` for size in bytes, `.Shape` for dimensions, etc. Modifications to the returned struct won't affect the original tensor. |
| `(*Tensor).ReadAt()` | `func (t *Tensor) ReadAt(dst []byte, off int64) (int, error)` | Read bytes from this tensor at the given offset. Uses per-tensor cache to avoid disk I/O for repeated reads. |
| `(*Tensor).Reader()` | `func (t *Tensor) Reader() io.ReadSeeker` | Return a limited io.ReadSeeker over this tensor's raw data. Ideal for streaming to external libraries like [go-quant](https://github.com/cymertek/go-quant) — no full allocation, works with tensors larger than RAM. |
| `(*Tensor).Data()` | `func (t *Tensor) Data() (io.ReaderAt, error)` | Return an io.ReaderAt positioned at this tensor's raw data in the file. Fixed length = NBytes. Useful for random-access reads without cache. |

### Writing

| Function | Signature | Description |
|----------|-----------|-------------|
| `Create` | `func Create(w io.Writer) *GGUFWriter` | Create a new GGUF writer wrapping any io.Writer or io.WriterAt (file, gzip stream, network buffer). If the writer implements io.WriterAt, writes go directly to it; otherwise data is buffered. |
| `OpenForWrite` | `func OpenForWrite(path string) (*GGUFWriter, error)` | Create a writer for the given file path. Equivalent to `Create(os.File)` but manages the file handle internally until Close. Returns an error if the file cannot be opened. |
| `(*GGUFWriter).SetKV()` | `func (w *GGUFWriter) SetKV(key string, v Value) error` | Add or replace a key-value pair in the metadata section. |
| `(*GGUFWriter).GetMetadata()` | `func (w *GGUFWriter) GetMetadata() []KVEntry` | Return copies of all KV entries currently queued for writing. |
| `(*GGUFWriter).AddTensor()` | `func (w *GGUFWriter) AddTensor(name string, shape []uint64, ggmlType GgmlType) uint64` | Queue a tensor definition; returns index for WriteTensorData/NewTensor. The reader is NOT consumed until Close(). |
| `(*GGUFWriter).WriteTensorData()` | `func (w *GGUFWriter) WriteTensorData(idx uint64, r io.Reader) error` | Associate an io.Reader with the queued tensor at idx. Reader is deferred — bytes stream to disk during Close(). Memory usage O(1) regardless of tensor size. |
| `(*GGUFWriter).NewTensor()` | `func (w *GGUFWriter) NewTensor(name string, shape []uint64, ggmlType GgmlType, r io.Reader) error` | Convenience: combines AddTensor + WriteTensorData in one call with deferred consumption. |
| `(*GGUFWriter).NumTensors()` | `func (w *GGUFWriter) NumTensors() int` | Return number of queued tensors. |
| `(*GGUFWriter).Close()` | `func (w *GGUFWriter) Close() (int64, error)` | Flush header + KV section + tensor metadata and stream tensor data. Returns total bytes written. |

### Streaming Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `StreamCopy` | `func StreamCopy(dst *GGUFWriter, src *GGUF, opts StreamOptions) error` | Copy tensors from source GGUF to writer with optional filtering and transformation. Passthrough copies stream directly without loading into memory; transforms require full data in memory. |
| `StreamMerge` | `func StreamMerge(dst *GGUFWriter, sources []*GGUF, opts StreamOptions) error` | Merge tensors from multiple GGUF files into a single writer. KV metadata comes from the first source only. Later sources' metadata entries are skipped to avoid key conflicts. |

### Deferred-Consumption Streaming Writer

The writer uses **deferred consumption** for tensor data — bytes are NOT loaded into memory when `AddTensor`/`WriteTensorData`/`NewTensor` is called. Instead, the underlying `io.Reader` reference is stored and streamed during `Close()`. This enables writing tensors larger than available RAM without allocation pressure.

**How it works:**
1. Call `AddTensor(name, shape, type)` to queue tensor definition — returns index
2. Call `WriteTensorData(idx, reader)` or `NewTensor(name, shape, type, reader)` with any `io.Reader` (network stream, file handle, byte buffer)
3. The reader is NOT consumed at this point — only stored as a reference
4. On `Close()`, the writer streams each tensor's bytes directly to disk

**Memory efficiency:** RAM usage stays O(1) per in-flight tensor regardless of size. A 10 GB tensor writes with minimal peak memory, not 10 GB.

```go
// Example: Stream a large tensor from a network connection without loading it into memory
w := gguf.Create(outFile)
idx := w.AddTensor("huge.weight", []uint64{largeDimension}, gguf.GgmlQ4_0)

// Pass any io.Reader — the library streams bytes at Close() time, not now
err := w.WriteTensorData(idx, networkConn)  // No ReadFull here; reader stored for deferred consumption

// Or use NewTensor convenience method (same deferred behavior)
err = w.NewTensor("another.weight", shape, ggmlType, someReader)

// Close() is where actual streaming happens — only a pooled buffer in RAM at any time
nWritten, err := w.Close()  // Streams each tensor through ~256KB window to disk
```

**Passthrough vs Transform paths:**
- **Passthrough** (no transform): Source reader streams directly to writer via deferred consumption — zero-copy, O(1) memory
- **Transform**: Requires full data in memory for processing, then writes through normal streaming path

### Utilities

| Function | Signature | Description |
|----------|-----------|-------------|
| `ConvertName` | `func ConvertName(name string) string` | Translate tensor names from llama.cpp convention to HuggingFace convention (and vice versa). |
| `MatchPattern` | `func MatchPattern(name string, patterns []string) bool` | Return true if name matches any of the given glob patterns. |

## Types

### `GGUF` — Lazy Reader

The primary reader type. Open with `gguf.Open(path)` or `gguf.NewReader(f)`. Only the 24-byte header is read initially; KV metadata and tensor info are walked lazily on first access.

```go
type GGUF struct {
    // Has unexported fields.
}
```

### `Tensor` — File-Backed Tensor Handle

Wraps `TensorInfo` with a per-tensor read cache that avoids disk seeks for repeated reads of the same region. Provides streaming access methods for reading tensor data without loading it entirely into memory.

```go
type Tensor struct {
    // Has unexported fields.
}
```

**Streaming read methods (no full allocation):**

| Method | Returns | Description |
|--------|---------|-------------|
| `Reader()` | `io.ReadSeeker` | LimitedReader-style sequential reader over tensor bytes. Use for streaming to other libraries or `io.Copy`. |
| `Data()` | `io.ReaderAt` | Seekable reader at aligned file offset. Fixed length = NBytes. Useful for random-access reads without cache. |
| `Read(dst []byte)` | `(int, error)` | Convenience wrapper around `ReadAt(dst, 0)` for `io.Reader`-style usage. |
| `ReadAt(dst []byte, off int64)` | `(int, error)` | Partial read with per-tensor cache overlap optimization. First call reads 1 MB aligned chunk into internal cache; subsequent calls within that range serve from memory without disk I/O. |

**Example — stream tensor to external library:**
```go
tensors, _ := g.Tensors()
for _, t := range tensors {
    r := t.Reader()  // io.ReadSeeker, limited to NBytes
    externalLib.Feed(r)  // Library reads at its own pace
}
```

**Example — partial read with caching:**
```go
t := tensors[0]
buf := make([]byte, 4096)
n, _ := t.ReadAt(buf, 1024)  // First call: disk seek + cache fill
n, _ = t.ReadAt(buf, 2048)   // Second call within same 1MB window: cache hit (no disk I/O)
```

### `TensorInfo` — Parsed Tensor Metadata

```go
type TensorInfo struct {
    Name     string    // tensor name (UTF-8)
    Shape    []uint64  // dimension sizes
    GgmlType GgmlType  // quantization/type enum
    Offset   uint64    // relative offset from dataStart (un-aligned, per GGUF spec)
    NBytes   uint64    // total raw tensor data size in bytes
}
```

### `MetadataEntry` — Lazy KV Entry Handle

Thin handle for a single metadata key-value pair with lazy value loading from the underlying file.

```go
type MetadataEntry struct {
    // Has unexported fields.
}
```

Methods:
- `Name() string` — Return entry's key name
- `BType() BType` — Return GGUF value type identifier
- `Size() int64` — Return total wire size of the value data in bytes
- `Value() (Value, error)` — Load and parse the raw KV data from file at stored offset, caching on first call

### `SplitInfo` — Multi-Shard Metadata

Contains metadata about a multi-shard GGUF file.

```go
type SplitInfo struct {
    Count           int          // total number of shards
    TensorsPerShard []uint64     // tensor count per shard (from headers)
    Shards          []*SplitShard // actual shard handles with readers
}
```

### `SplitShard` — Individual Shard Handle

Represents a single shard in a multi-shard GGUF file. Each shard has its own independent reader for metadata and tensors.

```go
type SplitShard struct {
    Index      int       // 0-based index in the split (after sorting by split.no)
    Path       string    // original path to this shard
    Size       int64     // file size in bytes
    Version    uint32    // GGUF version from header
    NumTensors uint64    // tensor count from header
    NumKV      uint64    // KV count from header
}
```

Methods:
- `GetMetadata() ([]*MetadataEntry, error)` — Return metadata entries for this shard
- `Tensors() ([]*Tensor, error)` — Return tensor handles for this shard

### `StreamOptions` — Streaming Configuration

Configures streaming operations (copy, merge, requantize).

```go
type StreamOptions struct {
    Include []string        // Filter tensors by name pattern. Empty = include all.
    Exclude []string        // Exclude tensors matching these patterns.
    TargetType GgmlType   // Requantize tensors to this type. Empty = passthrough (stream copy).
    Transform func(data []byte, info TensorInfo) ([]byte, error)  // Custom per-tensor hook called before writing. Return nil to write, non-nil to skip. Called with dequantized data.
}
```

### `Value` — Parsed Metadata Value

Holds a single metadata value that has been parsed from wire format.

```go
type Value struct {
    BType    BType   // type tag
    Int      int64   // integer or bool (true for bool)
    Float    float64 // floating-point value
    Str      string  // string value
    ElemType BType   // element type for arrays
    Raw      []byte  // raw wire bytes for arrays / strings
}
```

Methods:
- `AsBool() (bool, bool)` — Return true if value is a bool
- `AsInt() (int64, bool)` — Return true if value is an integer type
- `AsUint64() (uint64, bool)` — Return true if value is an unsigned integer type
- `AsFloat() (float64, bool)` — Return true if value is a float type
- `AsString() (string, bool)` — Return true if value is a string type

### `KVEntry` — Builder for Writing Metadata

Holds a single key-value pair ready to be written into a GGUF file.

```go
type KVEntry struct {
    Key   string
    Value Value
}
```

## Supported Types

### BType (Value Types)

| Constant | Wire Size | Go Type | Description |
|----------|-----------|---------|-------------|
| `BTypeUint8` | 1 byte | int64 | Unsigned 8-bit integer |
| `BTypeInt8` | 1 byte | int64 | Signed 8-bit integer |
| `BTypeBool` | 1 byte | bool | Boolean (0 or 1) |
| `BTypeUint16` | 2 bytes | int64 | Unsigned 16-bit integer |
| `BTypeInt16` | 2 bytes | int64 | Signed 16-bit integer |
| `BTypeUint32` | 4 bytes | int64 | Unsigned 32-bit integer |
| `BTypeInt32` | 4 bytes | int64 | Signed 32-bit integer |
| `BTypeFloat32` | 4 bytes | float64 | 32-bit float |
| `BTypeUint64` | 8 bytes | int64 | Unsigned 64-bit integer |
| `BTypeInt64` | 8 bytes | int64 | Signed 64-bit integer |
| `BTypeFloat64` | 8 bytes | float64 | 64-bit float |
| `BTypeString` | variable | string | UTF-8 string with length prefix |
| `BTypeArray` | variable | multiple | Typed array (elem_type + count + data) |

### GgmlType (Tensor Types)

| Constant | Block Size | Elements/Block | Description |
|----------|-----------|--------------|-------------|
| `GgmlF32` | 4 B | 1 | 32-bit float (raw) |
| `GgmlF16` | 2 B | 1 | 16-bit float |
| `GgmlQ4_0` | 18 B | 32 | 4-bit quantization |
| `GgmlQ5_0` | 22 B | 32 | 5-bit quantization |
| `GgmlQ8_0` | 34 B | 32 | 8-bit quantization |
| `GgmlQ2_K` | 64 B | 256 | K-quants 2-bit |
| `GgmlQ3_K_S` / `GgmlQ3_K_L` | 62/64 B | 256 | K-quants 3-bit (small/large) |
| `GgmlIQ3XXS` | 18 B | 32 | IQ3_XXS quantization |
| `GgmlIQ3S` | 20 B | 32 | IQ3_S quantization |
| `GgmlIQ2S` / `GgmlIQ2M` | variable | variable | IQ2 variants |
| `GgmlIQ4XL` | 56 B | 256 | IQ4_XL quantization |
| `GgmlIQ4NS` | 58 B | 256 | IQ4_NS quantization |
| `GgmlQ4_K` | 70 B | 256 | K-quants 4-bit (alias for IQ3_XXS block config) |
| `GgmlQ5_K` | 76 B | 256 | K-quants 5-bit (alias for IQ3_S block config) |
| `GgmlQ6_K` | 98 B | 256 | K-quants 6-bit |
| `GgmlIQ4M` | variable | variable | IQ4_M quantization |
| `GgmlDFP` / `GgmlDP` | variable | variable | Dynamic format precisions |
| `GgmlI2` / `GgmlIM` | variable | variable | Integer quantizations |
| `GgmlNVFP4` | 40 B | 64 | NVIDIA NVFP4: 64-element block (4x16 sub-blocks), UE4M3 sub-scales |

## Examples

Run the examples against a GGUF file:

```bash
# Read single-file or multi-shard GGUF
go run ./examples/main.go model.gguf           # Demo all access patterns
go run ./examples/verify.go model.gguf         # Cross-reference with pygguf
go run ./examples/split/main.go model-00001-of-00003.gguf  # Multi-shard example

# Inspect split files
go run ./cmd/test_splits model-00001-of-00003.gguf  # Detailed shard inspection
```

## Testing

```bash
# Run all tests (includes multi-shard validation tests)
GOFLAGS="-buildvcs=false" go test ./gguf/... -v

# Skip slow tests (e.g., real model file I/O)
GOFLAGS="-buildvcs=false" go test ./gguf/... -short -v
```

## Architecture Notes

### Lazy Loading Design

The library uses lazy loading to minimize initial memory footprint and startup time:

1. `Open()` reads only the 24-byte header (magic + version + nKV + nTensor)
2. `Metadata()` walks the KV section sequentially, eagerly parsing values ≤64 bytes
3. `Tensors()` performs ONE seek to kvEnd + sequential tensor metadata walk
4. Per-tensor read cache avoids disk seeks for repeated reads of the same region

This design is ideal for:
- **Concurrent file operations**: 100+ GGUF files can be opened with minimal memory (~5.6KB total)
- **Slow media (S3, network)**: Sequential reads minimize seek overhead
- **Large models**: Only needed tensors are loaded into memory

### Multi-Shard Validation

Multi-shard GGUF files undergo rigorous validation before combining:

1. **Header validation**: Each shard must have valid GGUF v3 magic bytes and consistent version
2. **Naming convention enforcement**: All shards must follow `{basename}-{NNNNN}-of-{NNNNN}.gguf` pattern with matching basename (e.g., `TestModel-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf`)
3. **Metadata consistency**: `split.no` and `split.count` values must match across all shards
4. **Index validation**: Each shard's `split.no` must be in range [0, splitCount)
5. **Sequence verification**: Shards are re-sorted by `split.no` to ensure correct order regardless of input order

This prevents silent failures when:
- Shards come from different directories or mounts (validated via metadata, not path)
- Files are missing or corrupted
- Metadata is inconsistent across shards
2. **Metadata extraction**: `split.no`, `split.count`, and `general.architecture` are read from each shard's KV section
3. **Cross-shard consistency**: All shards must agree on `split.count`; architecture must match if present in multiple shards
4. **Sequence verification**: Shards are re-sorted by `split.no` to ensure correct order (0, 1, 2, ...) regardless of disk order

This prevents silent failures when:
- Shards are missing or corrupted
- Files are out of order on disk
- Metadata is inconsistent across shards

## License

Apache 2.0 — see [LICENSE](./LICENSE) for details.
