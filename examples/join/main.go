// Example: Join multiple GGUF shards into a single file.
//
// Usage: go run ./examples/join <shard1.gguf> [shard2.gguf ...] [--output output.gguf]
//
// This demonstrates how to merge split GGUF files back into a single file,
// which is useful for redistribution or when you need a single-file format.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/cymertek/go-gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <shard1.gguf> [shard2.gguf ...] [--output output.gguf]\n", os.Args[0])
		os.Exit(1)
	}

	var shardPaths []string
	outputPath := ""

	// Parse arguments
	for i, arg := range os.Args {
		if arg == "--output" && i+1 < len(os.Args) {
			outputPath = os.Args[i+1]
			os.Args = append(os.Args[:i], os.Args[i+2:]...) // Remove the flag and value
			i-- // Adjust index since we removed elements
		} else if !strings.HasPrefix(arg, "--") {
			shardPaths = append(shardPaths, arg)
		}
	}

	if outputPath == "" {
		// Default output name based on first shard
		baseName := strings.TrimSuffix(filepath.Base(shardPaths[0]), filepath.Ext(shardPaths[0]))
		outputPath = filepath.Join(filepath.Dir(shardPaths[0]), baseName+".gguf")
	}

	fmt.Printf("Joining %d shards into: %s\n", len(shardPaths), outputPath)

	// Open all source shards
	sources := make([]*gguf.GGUF, 0, len(shardPaths))
	for _, path := range shardPaths {
		g, err := gguf.NewReaderFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening %s: %v\n", path, err)
			os.Exit(1)
		}
		sources = append(sources, g)

		if !g.IsSplit() {
			fmt.Fprintf(os.Stderr, "Warning: %s is not a split file (single shard)\n", path)
		}
	}
	defer func() {
		for _, s := range sources {
			s.Close()
		}
	}()

	// Create output writer
	dstWriter, err := gguf.OpenForWrite(outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output file: %v\n", err)
		os.Exit(1)
	}
	defer dstWriter.Close()

	// Copy metadata from first source (all sources should have identical metadata)
	firstMeta, err := sources[0].Metadata()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading metadata: %v\n", err)
		os.Exit(1)
	}

	for _, entry := range firstMeta {
		v, err := entry.Value()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading metadata %s: %v\n", entry.Name(), err)
			continue
		}
		dstWriter.SetKV(entry.Name(), v)
	}

	// Stream tensors from all sources into the destination
	for i, src := range sources {
		fmt.Printf("Streaming tensors from shard %d/%d...\n", i+1, len(sources))

		// Get tensors from this source
		tensors, err := src.Tensors()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting tensors: %v\n", err)
			os.Exit(1)
		}

		for _, t := range tensors {
			info := t.Info()

			// Add tensor to destination writer (deferred — data streams at Close())
			idx := dstWriter.AddTensor(info.Name, info.Shape, info.GgmlType)

			// Stream tensor data directly from source reader to destination writer
			if err := dstWriter.WriteTensorData(idx, t.Reader()); err != nil {
				fmt.Fprintf(os.Stderr, "Error streaming tensor %s: %v\n", info.Name, err)
				continue
			}
		}
	}

	// Close the destination writer (finalizes the file)
	writtenBytes, err := dstWriter.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error closing output: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nJoin complete!\n")
	fmt.Printf("Output: %s (%.3f GB)\n", outputPath, float64(writtenBytes)/1e9)
}
