// Create a small test GGUF file for split/join testing.
package main

import (
	"bytes"
	"fmt"
	"os"

	"github.com/cymertek/go-gguf"
)

func main() {
	outputPath := "/tmp/test-model.gguf"

	w, err := gguf.OpenForWrite(outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating writer: %v\n", err)
		os.Exit(1)
	}

	// Add metadata
	w.SetKV("general.architecture", gguf.Value{Str: "test", BType: gguf.BTypeString})
	w.SetKV("general.file_type", gguf.Value{Int: 2, BType: gguf.BTypeUint32})

	// Add two small tensors (F16 = 2 bytes per element)
	tensor1Shape := []uint64{2, 2}
	tensor1Data := make([]byte, 2*2*2) // F16 = 2 bytes per element
	for i := range tensor1Data {
		tensor1Data[i] = byte(i % 256)
	}

	idx1 := w.AddTensor("tensor1.weight", tensor1Shape, gguf.GgmlF16)
	if err := w.WriteTensorData(idx1, bytes.NewReader(tensor1Data)); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing tensor1: %v\n", err)
		os.Exit(1)
	}

	tensor2Shape := []uint64{3, 3}
	tensor2Data := make([]byte, 3*3*2) // F16 = 2 bytes per element
	for i := range tensor2Data {
		tensor2Data[i] = byte((i + 100) % 256)
	}

	idx2 := w.AddTensor("tensor2.weight", tensor2Shape, gguf.GgmlF16)
	if err := w.WriteTensorData(idx2, bytes.NewReader(tensor2Data)); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing tensor2: %v\n", err)
		os.Exit(1)
	}

	writtenBytes, err := w.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error closing writer: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Created test model: %s (%d bytes)\n", outputPath, writtenBytes)
}
