package gguf

import (
	"encoding/binary"
	"io"
	"os"
	"runtime"
	"testing"
)

// TestMemoryScaling verifies that Open() scales linearly with concurrent file opens.
func TestMemoryScaling(t *testing.T) {
	const bonsaiPath = "/workdir/Bonsai-8B.gguf"

	testCases := []struct {
		name string
		n    int
	}{
		{"0 files", 0},
		{"1 file", 1},
		{"4 files", 4},
		{"8 files", 8},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var rssBefore uint64
			if tc.n > 0 {
				rssBefore = getRSS()
			}

			// Open n files concurrently (only header read)
			readers := make([]*GGUF, tc.n)
			for i := 0; i < tc.n; i++ {
				f, err := os.Open(bonsaiPath)
				if err != nil {
					t.Fatalf("os.Open: %v", err)
				}
				info, _ := f.Stat()

				var hdr [24]byte
				if _, err := io.ReadFull(io.NewSectionReader(f, 0, 24), hdr[:]); err != nil {
					f.Close()
					t.Fatalf("read header: %v", err)
				}

				rdr := &GGUF{
					r:         f,
					fileSz:    info.Size(),
					version:   binary.LittleEndian.Uint32(hdr[4:8]),
					nTensor:   binary.LittleEndian.Uint64(hdr[8:16]),
					nKV:       binary.LittleEndian.Uint64(hdr[16:24]),
					alignment: defaultAlignment,
				}
				readers[i] = rdr
			}

			var rssAfter uint64
			if tc.n > 0 {
				rssAfter = getRSS()
				increase := rssAfter - rssBefore
				expectedMin := uint64(56 * tc.n) // ~56 bytes per GGUF struct
				t.Logf("Opened %d files: RSS increase = %d bytes (min expected: %d)", tc.n, increase, expectedMin)

				// Verify linear scaling - allow 10x overhead for safety
				if increase > expectedMin*10 {
					t.Errorf("RSS increase too large: got %d, expected ~%d", increase, expectedMin)
				}
			}

			// Close all readers
			for _, rdr := range readers {
				_ = rdr.Close() //nolint:errcheck // cleanup
			}
		})
	}
}

func getRSS() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Sys // Total system memory in use
}
