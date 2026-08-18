// Example: Verify split/join round-trip with SHA256 comparison.
//
// This program demonstrates a complete end-to-end test of the split and join
// operations by:
// 1. Splitting a small GGUF file (test.gguf) into 2 shards
// 2. Joining them back together
// 3. Comparing SHA256 hashes to verify byte-for-byte correctness

package main

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/cymertek/go-gguf"
)

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}

	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

// openAllShards opens a GGUF file and returns all shards from its SplitInfo.
// If the file is not a split file, it returns a single-element slice containing just that file's reader.
func openAllShards(firstShardPath string) ([]*gguf.GGUF, error) {
	src, err := gguf.NewReaderFile(firstShardPath)
	if err != nil {
		return nil, fmt.Errorf("open first shard: %w", err)
	}

	var sources []*gguf.GGUF

	if src.IsSplit() {
		info := src.SplitInfo()
		fmt.Printf("  Multi-shard file detected: %d shards\n", info.Count)

		// Use the split info to get all shard paths, then open each one directly
		// (skipping auto-detection for data shards)
		for _, shard := range info.Shards {
			shardSrc, err := gguf.NewReaderFile(shard.Path)
			if err != nil {
				src.Close()
				return nil, fmt.Errorf("open shard %s: %w", shard.Path, err)
			}
			sources = append(sources, shardSrc)
		}

		// Close the initial reader since we now have individual readers for each shard
		src.Close()
	} else {
		fmt.Println("  Single-file GGUF detected")
		sources = []*gguf.GGUF{src}
	}

	return sources, nil
}

func main() {
	sourceFile := "/tmp/test-model.gguf"
	splitDir := "/tmp/gguf-split-test"
	outputPath := filepath.Join(splitDir, "test-reconstructed.gguf")

	// Create split directory
	if err := os.MkdirAll(splitDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating split directory: %v\n", err)
		os.Exit(1)
	}

	// Step 1: Split the source file into 2 shards
	fmt.Println("=== Step 1: Splitting test.gguf into 2 shards ===")
	if err := splitFile(sourceFile, splitDir, 2); err != nil {
		fmt.Fprintf(os.Stderr, "Split failed: %v\n", err)
		os.Exit(1)
	}

	// Step 2: Join the shards back together
	fmt.Println("\n=== Step 2: Joining shards back into single file ===")

	// Open only the first shard - it will auto-detect and combine all shards
	firstShardPath := filepath.Join(splitDir, "test-model-00001-of-00002.gguf")
	fmt.Printf("  Opening first shard: %s\n", firstShardPath)

	sources, err := openAllShards(firstShardPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening shards: %v\n", err)
		os.Exit(1)
	}

	if err := joinFiles(sources, outputPath); err != nil {
		fmt.Fprintf(os.Stderr, "Join failed: %v\n", err)
		os.Exit(1)
	}

	// Step 3: Compare SHA256 hashes
	fmt.Println("\n=== Step 3: Comparing files ===")
	originalHash, err := sha256File(sourceFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading original file: %v\n", err)
		os.Exit(1)
	}

	reconstructedHash, err := sha256File(outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading reconstructed file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Original SHA256:    %s\n", originalHash)
	fmt.Printf("Reconstructed SHA256: %s\n", reconstructedHash)

	if originalHash == reconstructedHash {
		fmt.Println("\n✓ SUCCESS: Files match byte-for-byte!")
		os.Exit(0)
	} else {
		fmt.Println("\n✗ FAILURE: Files do not match!")
		os.Exit(1)
	}
}

func splitFile(inputPath string, outputDir string, numShards int) error {
	src, err := gguf.NewReaderFile(inputPath)
	if err != nil {
		return fmt.Errorf("open source: %w", err)
	}
	defer src.Close()

	tensors, err := src.Tensors()
	if err != nil {
		return fmt.Errorf("get tensors: %w", err)
	}

	metaEntries, err := src.Metadata()
	if err != nil {
		return fmt.Errorf("get metadata: %w", err)
	}

	baseName := filepath.Base(inputPath)
	baseName = baseName[:len(baseName)-len(filepath.Ext(baseName))]

	// Distribute tensors evenly across all shards
	tensorsPerShard := len(tensors) / numShards
	if tensorsPerShard == 0 {
		tensorsPerShard = 1
	}

	for i := 0; i < numShards; i++ {
		startIdx := i * tensorsPerShard
		endIdx := startIdx + tensorsPerShard
		if endIdx > len(tensors) {
			endIdx = len(tensors)
		}

		shardPath := filepath.Join(outputDir, fmt.Sprintf("%s-%05d-of-%05d.gguf", baseName, i+1, numShards))
		shardTensors := tensors[startIdx:endIdx]

		fmt.Printf("Writing shard %d/%d: tensors [%d, %d)\n", i+1, numShards, startIdx, endIdx)

		if i == 0 {
			// First shard gets metadata + first batch of tensors
			if err := writeMetadataShardWithTensors(shardPath, metaEntries, shardTensors); err != nil {
				return fmt.Errorf("write shard %d: %w", i+1, err)
			}
		} else {
			// Other shards get only their tensors (with split.no metadata)
			if err := writeDataShard(shardPath, shardTensors); err != nil {
				return fmt.Errorf("write shard %d: %w", i+1, err)
			}
		}

		for j, t := range shardTensors {
			info := t.Info()
			fmt.Printf("  Tensor %d: %s, NBytes=%d\n", startIdx+j+1, info.Name, info.NBytes)
		}
	}

	return nil
}

// writeDataShard writes header + tensor metadata + tensor data (no KV section except split.no)
func writeDataShard(path string, tensors []*gguf.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := gguf.Create(f)

	// Add minimal metadata for this shard
	shardIdx := 1 // Will be set by caller
	w.SetKV("split.no", gguf.Value{Int: int64(shardIdx), BType: gguf.BTypeUint16})

	// Add tensors (streaming — data deferred until Close())
	for i, t := range tensors {
		info := t.Info()
		fmt.Printf("    Writing tensor %d: %s (NBytes=%d)\n", i+1, info.Name, info.NBytes)

		idx := w.AddTensor(info.Name, info.Shape, info.GgmlType)

		if err := w.WriteTensorData(idx, t.Reader()); err != nil {
			return fmt.Errorf("stream tensor %s to shard %d: %w", info.Name, i+1, err)
		}
	}

	w.Close()
	return nil
}

// writeMetadataShardWithTensors writes header + KV section + first batch of tensors
func writeMetadataShardWithTensors(path string, metaEntries []*gguf.MetadataEntry, tensors []*gguf.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := gguf.Create(f)

	// Add split metadata
	w.SetKV("split.no", gguf.Value{Int: 0, BType: gguf.BTypeUint16})
	w.SetKV("split.count", gguf.Value{Int: int64(2), BType: gguf.BTypeUint16})

	// Copy all metadata entries
	for _, entry := range metaEntries {
		v, err := entry.Value()
		if err != nil {
			continue
		}
		w.SetKV(entry.Name(), v)
	}

	// Add tensors to this shard (streaming — data deferred until Close())
	for _, t := range tensors {
		info := t.Info()
		idx := w.AddTensor(info.Name, info.Shape, info.GgmlType)

		if err := w.WriteTensorData(idx, t.Reader()); err != nil {
			return fmt.Errorf("stream tensor %s to shard 1: %w", info.Name, err)
		}
	}

	w.Close()
	return nil
}

func joinFiles(sources []*gguf.GGUF, outputPath string) error {
	defer func() {
		for _, s := range sources {
			s.Close()
		}
	}()

	dstWriter, err := gguf.OpenForWrite(outputPath)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	defer dstWriter.Close()

	// Copy metadata excluding split-related entries for single-file output
	firstMeta, metaErr := sources[0].Metadata()
	if metaErr != nil {
		return fmt.Errorf("read metadata: %w", metaErr)
	}

	for _, entry := range firstMeta {
		name := entry.Name()
		// Skip split-related metadata for single-file output
		if name == "split.no" || name == "split.count" || name == "split.tensors.count" {
			continue
		}

		v, err := entry.Value()
		if err != nil {
			fmt.Printf("  Warning: skipping metadata %s: %v\n", name, err)
			continue
		}
		dstWriter.SetKV(name, v)
	}

	for i, src := range sources {
		fmt.Printf("Streaming from shard %d/%d...\n", i+1, len(sources))

		numTensors := src.NumTensors()
		fmt.Printf("  Header says nTensors=%d\n", numTensors)

		// Read metadata first to see what's there
		metaEntries, err := src.Metadata()
		if err != nil {
			return fmt.Errorf("read metadata: %w", err)
		}
		fmt.Printf("  Metadata entries read: %d (header says nKV=%d)\n", len(metaEntries), src.NumTensors())
		for j, entry := range metaEntries {
			v, vErr := entry.Value()
			valStr := ""
			if vErr == nil {
				switch {
				case v.BType == gguf.BTypeString:
					s, _ := v.AsString()
					valStr = fmt.Sprintf("=%q", s)
				default:
					if i, ok := v.AsInt(); ok {
						valStr = fmt.Sprintf("=%d", i)
					} else if f, ok := v.AsFloat(); ok {
						valStr = fmt.Sprintf("=%g", f)
					}
				}
			} else {
				valStr = fmt.Sprintf("(error: %v)", vErr)
			}
			fmt.Printf("    KV[%d]: name=%q btype=%v%s\n", j, entry.Name(), entry.BType(), valStr)
		}

		tensors, err := src.Tensors()
		if err != nil {
			return fmt.Errorf("get tensors: %w", err)
		}

		fmt.Printf("  Found %d tensors via Tensors()\n", len(tensors))
		for j, t := range tensors {
			info := t.Info()
			fmt.Printf("  Tensor %d: name=%q NBytes=%d Offset=%d Type=%v\n",
				j+1, info.Name, info.NBytes, info.Offset, info.GgmlType)
		}

		for _, t := range tensors {
			info := t.Info()

			fmt.Printf("  Tensor: %s, NBytes=%d\n", info.Name, info.NBytes)

			idx := dstWriter.AddTensor(info.Name, info.Shape, info.GgmlType)

			if err := dstWriter.WriteTensorData(idx, t.Reader()); err != nil {
				fmt.Printf("Error streaming tensor %s: %v\n", info.Name, err)
				continue
			}
		}
	}

	writtenBytes, err := dstWriter.Close()
	if err != nil {
		return fmt.Errorf("close output: %w", err)
	}

	fmt.Printf("Joined file written: %.3f KB\n", float64(writtenBytes)/1e3)
	return nil
}
