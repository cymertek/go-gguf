// Example main.go demonstrating go-gguf lazy reader and zero-copy streaming.
package main

import (
	"fmt"
	"os"
	"path/filepath"

	gguf "github.com/cymertek/go-gguf"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <gguf-file>\n", filepath.Base(os.Args[0]))
		os.Exit(1)
	}

	file := os.Args[1]
	fmt.Printf("Reading GGUF file: %s\n", file)

	rdr, err := gguf.NewReaderFile(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening file: %v\n", err)
		os.Exit(1)
	}
	defer rdr.Close()

	// 1. Read KV metadata (lazy walk with eager threshold ≤64B)
	fmt.Printf("\n=== KV Metadata ===\n")
	metaEntries, err := rdr.Metadata()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading metadata: %v\n", err)
		os.Exit(1)
	}

	for _, entry := range metaEntries {
		v, err := entry.Value()
		if err != nil {
			fmt.Printf("  %s: error loading value: %v\n", entry.Name(), err)
			continue
		}
		fmt.Printf("  %s = %+v (type=%d)\n", entry.Name(), v, entry.BType())
	}

	// 2. Get tensor metadata (ONE seek to kvEnd + sequential read of all tensors)
	fmt.Printf("\n=== Tensor Info ===\n")
	tensors, err := rdr.Tensors()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting tensors: %v\n", err)
		os.Exit(1)
	}

	for i, t := range tensors {
		info := t.Info()
		fmt.Printf("Tensor[%d]: %s (shape=%v, type=%s, nbytes=%d)\n",
			i, info.Name, info.Shape, info.GgmlType.GgmlName(), info.NBytes)

		if i >= 5 { // Show first 6 tensors only for brevity
			fmt.Printf("  ... and %d more\n", len(tensors)-i-1)
			break
		}
	}

	// 3. Read tensor data with per-tensor cache (first read caches, subsequent reads hit cache)
	if len(tensors) > 0 {
		t := tensors[0]
		fmt.Printf("\n=== Tensor Data (with cache) ===\n")
		fmt.Printf("First %d bytes of %s:\n", min(16, int(t.Info().NBytes)), t.Info().Name)

		buf := make([]byte, 16)
		n, err := t.ReadAt(buf, 0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading tensor: %v\n", err)
		} else {
			for i := 0; i < n; i++ {
				fmt.Printf("%02x ", buf[i])
			}
			fmt.Println()
		}

		// Second read from same offset should hit cache (no disk I/O)
		n, err = t.ReadAt(buf, 0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading tensor (cache): %v\n", err)
		} else {
			fmt.Printf("Second read (cached): %d bytes\n", n)
		}

		t.Close() // Release cache buffer back to pool
	}

	// 4. StreamCopy with zero-copy path detection
	fmt.Printf("\n=== Stream Copy Example ===\n")
	if err := streamCopyExample(rdr); err != nil {
		fmt.Fprintf(os.Stderr, "Stream copy error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\nDone!")
}

// streamCopyExample demonstrates zero-copy tensor streaming between files.
func streamCopyExample(src *gguf.GGUF) error {
	dstPath := "/tmp/output.gguf"
	defer os.Remove(dstPath) // Clean up after example

	// Create destination writer (wraps io.Writer for gzip, network, etc.)
	gw, err := gguf.OpenForWrite(dstPath)
	if err != nil {
		return fmt.Errorf("create dst: %w", err)
	}

	// Copy metadata with pattern filter (include only "general.*" keys)
	srcMeta, err := src.Metadata()
	if err != nil {
		gw.Close()
		return fmt.Errorf("get source metadata: %w", err)
	}

	for _, entry := range srcMeta {
		if gguf.MatchPattern(entry.Name(), []string{"general.*"}) {
			v, err := entry.Value()
			if err != nil {
				continue // Skip entries that fail to load
			}
			gw.SetKV(entry.Name(), v)
		}
	}

	// Stream tensors with zero-copy path (no transform, no requantize)
	opts := gguf.StreamOptions{
		Include: []string{"*.weight"}, // Only copy weight tensors
	}

	if err := gguf.StreamCopy(gw, src, opts); err != nil {
		gw.Close()
		return fmt.Errorf("stream copy: %w", err)
	}

	writtenBytes, err := gw.Close()
	if err != nil {
		return fmt.Errorf("close dst writer: %w", err)
	}

	fmt.Printf("Copied to %s (%d bytes written)\n", dstPath, writtenBytes)
	return nil
}
