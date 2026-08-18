// Example: Split a single GGUF file into multiple shards.
//
// Usage: go run ./examples/split <input.gguf> [--shards N]
//
// This demonstrates how to split a large GGUF model file (like Bonsai-8B)
// into multiple smaller shards suitable for distributed loading or storage.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cymertek/go-gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <input.gguf> [--shards N]\n", os.Args[0])
		os.Exit(1)
	}

	inputPath := os.Args[1]
	numShards := 2 // default

	// Parse --shards flag
	for i, arg := range os.Args {
		if arg == "--shards" && i+1 < len(os.Args) {
			n, err := strconv.Atoi(os.Args[i+1])
			if err != nil || n < 2 {
				fmt.Fprintf(os.Stderr, "Error: --shards requires a number >= 2\n")
				os.Exit(1)
			}
			numShards = n
			break
		}
	}

	// Open source file
	src, err := gguf.NewReaderFile(inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening %s: %v\n", inputPath, err)
		os.Exit(1)
	}
	defer src.Close()

	if src.IsSplit() {
		fmt.Fprintf(os.Stderr, "Error: source file is already a multi-shard GGUF\n")
		os.Exit(1)
	}

	// Get all tensors from source
	tensors, err := src.Tensors()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting tensors: %v\n", err)
		os.Exit(1)
	}

	totalTensors := len(tensors)
	tensorsPerShard := totalTensors / numShards
	if tensorsPerShard == 0 {
		tensorsPerShard = 1
	}

	fmt.Printf("Source: %s\n", inputPath)
	fmt.Printf("Total tensors: %d\n", totalTensors)
	fmt.Printf("Splitting into %d shards (approx. %d tensors each)\n\n", numShards, tensorsPerShard)

	// Extract base name for output files
	baseName := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
	dir := filepath.Dir(inputPath)

	// Read all metadata from source
	metaEntries, err := src.Metadata()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading metadata: %v\n", err)
		os.Exit(1)
	}

	// Calculate cumulative tensor sizes to find a good split point
	var cumSizes []uint64
	totalSize := uint64(0)
	for _, t := range tensors {
		totalSize += t.Info().NBytes
		cumSizes = append(cumSizes, totalSize)
	}

	targetSize := totalSize / uint64(numShards)
	splitPoint := 0
	var minDiff uint64 = ^uint64(0)

	for i, size := range cumSizes {
		diff := uint64(0)
		if size > targetSize {
			diff = size - targetSize
		} else {
			diff = targetSize - size
		}
		if diff < minDiff {
			minDiff = diff
			splitPoint = i + 1 // split after this tensor
		}
	}

	// Ensure we don't put all tensors in one shard
	if splitPoint == totalTensors {
		splitPoint = totalTensors / 2
	}
	if splitPoint == 0 {
		splitPoint = 1
	}

	fmt.Printf("Split point: tensor %d (cumulative size: %.3f GB)\n\n", splitPoint, float64(cumSizes[splitPoint-1])/1e9)

	// Create output shards
	shardFiles := make([]*os.File, numShards)
	writers := make([]*gguf.GGUFWriter, numShards)

	for i := 0; i < numShards; i++ {
		outPath := filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", baseName, i+1, numShards))
		f, err := os.Create(outPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating %s: %v\n", outPath, err)
			os.Exit(1)
		}
		shardFiles[i] = f
		writers[i] = gguf.Create(f)
		defer f.Close()
	}

	// Copy metadata to all shards (each shard needs the full KV section)
	for _, entry := range metaEntries {
		v, err := entry.Value()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading metadata %s: %v\n", entry.Name(), err)
			continue
		}
		writers[0].SetKV(entry.Name(), v)
	}

	// Distribute tensors across shards based on size
	// Users can customize this logic to balance by:
	// - Equal number of tensors per shard (default)
	// - Equal total bytes per shard
	// - Grouping related tensors together
	// - Custom heuristics based on tensor names or types

	var tensorIdx int = 0

	for i := 0; i < numShards; i++ {
		startTensor := 0
		endTensor := totalTensors

		if i == 0 {
			startTensor = 0
			endTensor = splitPoint
		} else if i == numShards-1 {
			startTensor = splitPoint + (i-1)*(tensorsPerShard - splitPoint/(numShards-1))
			endTensor = totalTensors
		} else {
			startTensor = splitPoint + (i-1)*tensorsPerShard
			endTensor = startTensor + tensorsPerShard
			if endTensor > totalTensors {
				endTensor = totalTensors
			}
		}

		fmt.Printf("Shard %d/%d: tensors [%d, %d)\n", i+1, numShards, startTensor, endTensor)

		for j := startTensor; j < endTensor && tensorIdx < len(tensors); j++ {
			t := tensors[j]
			info := t.Info()

			// Add tensor to this shard's writer (deferred — data streams at Close())
			idx := writers[i].AddTensor(info.Name, info.Shape, info.GgmlType)

			// Stream tensor data directly from source reader to destination writer
			if err := writers[i].WriteTensorData(idx, t.Reader()); err != nil {
				fmt.Fprintf(os.Stderr, "Error streaming tensor %s: %v\n", info.Name, err)
				continue
			}

			tensorIdx++
		}
	}

	// Close all writers (this finalizes the GGUF files)
	var totalWritten int64
	for i, w := range writers {
		written, err := w.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error closing shard %d: %v\n", i+1, err)
			os.Exit(1)
		}
		totalWritten += written
		shardFiles[i].Close()

		outPath := filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", baseName, i+1, numShards))
		fmt.Printf("  Written: %s (%.3f GB)\n", outPath, float64(written)/1e9)
	}

	fmt.Printf("\nTotal bytes written: %.3f GB\n", float64(totalWritten)/1e9)
	fmt.Println("\nSplit complete!")
}
