package main

import (
	"fmt"
	"os"

	gguf "github.com/cymertek/go-gguf/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: inspect_splits <gguf-file>")
		os.Exit(1)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	defer f.Close()

	info, _ := f.Stat()
	rdr, err := gguf.OpenFromReader(f, info.Size())
	if err != nil {
		panic(err)
	}
	defer rdr.Close()

	fmt.Printf("Version: %d\n", rdr.Version())
	fmt.Printf("NumTensors: %d\n", rdr.NumTensors())

	metaEntries, err := rdr.Metadata()
	if err != nil {
		panic(err)
	}

	fmt.Println("\nAll metadata keys:")
	for _, entry := range metaEntries {
		v, err := entry.Value()
		if err != nil {
			fmt.Printf("  %s: ERROR loading value: %v\n", entry.Name(), err)
			continue
		}
		fmt.Printf("  %-40s = %+v (type=%d)\n", entry.Name(), v, entry.BType())
	}

	tensors, err := rdr.Tensors()
	if err != nil {
		panic(err)
	}

	fmt.Printf("\nFirst 10 tensors:\n")
	for i, t := range tensors[:min(10, len(tensors))] {
		info := t.Info()
		fmt.Printf("  [%d] %s (shape=%v type=%s nbytes=%d)\n",
			i, info.Name, info.Shape, info.GgmlType.GgmlName(), info.NBytes)
	}

	totalTensors := uint64(0)
	for _, t := range tensors {
		var n uint64 = 1
		for _, d := range t.Info().Shape {
			n *= d
		}
		totalTensors += n
	}
	fmt.Printf("\nTotal tensor elements: %d\n", totalTensors)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
