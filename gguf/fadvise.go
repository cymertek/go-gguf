package gguf

import (
	"fmt"
	"os"
	"golang.org/x/sys/unix"
)

// HintSequential marks file region for sequential access.
func HintSequential(f *os.File, offset int64, count int64) error {
	err := unix.Fadvise(int(f.Fd()), offset, count, 2)
	if err != nil {
		return fmt.Errorf("gguf: fadvise sequential failed: %w", err)
	}
	return nil
}

// HintRandom marks file region for random access.
func HintRandom(f *os.File, offset int64, count int64) error {
	err := unix.Fadvise(int(f.Fd()), offset, count, 1)
	if err != nil {
		return fmt.Errorf("gguf: fadvise random failed: %w", err)
	}
	return nil
}

// HintDiscard releases page cache for a file region (FADV_DONTNEED).
func HintDiscard(f *os.File, offset int64, length int64) error {
	err := unix.Fadvise(int(f.Fd()), offset, length, 4)
	if err != nil {
		return fmt.Errorf("gguf: fadvise discard failed: %w", err)
	}
	return nil
}

// HintNoReuse marks a region as unlikely to be reused soon (FADV_NOREUSE).
func HintNoReuse(f *os.File, offset int64, count int64) error {
	err := unix.Fadvise(int(f.Fd()), offset, count, 3)
	if err != nil {
		return fmt.Errorf("gguf: fadvise no-reuse failed: %w", err)
	}
	return nil
}