package gguf

import (
	"fmt"
	"os"
	"golang.org/x/sys/unix"
)

// HintSequential advises the OS kernel to expect sequential reads from [f] starting at [offset]
// for approximately [count] bytes. This enables read-ahead optimizations that improve throughput
// when scanning tensor data linearly. Only effective on Linux; silently ignored on other platforms.
func HintSequential(f *os.File, offset int64, count int64) error {
	err := unix.Fadvise(int(f.Fd()), offset, count, 2)
	if err != nil {
		return fmt.Errorf("gguf: fadvise sequential failed: %w", err)
	}
	return nil
}

// HintRandom advises the OS kernel that reads from [f] in the range [offset, offset+count] will be
// random (non-sequential). This disables read-ahead optimizations and may improve performance when
// seeking to scattered tensor offsets. Only effective on Linux; silently ignored on other platforms.
func HintRandom(f *os.File, offset int64, count int64) error {
	err := unix.Fadvise(int(f.Fd()), offset, count, 1)
	if err != nil {
		return fmt.Errorf("gguf: fadvise random failed: %w", err)
	}
	return nil
}

// HintDiscard tells the OS kernel that pages in the range [offset, offset+length] of [f] are no
// longer needed and can be released from the page cache. This is equivalent to POSIX fadvise(FADV_DONTNEED)
// and frees memory when you know a region will not be re-read soon (e.g., after fully loading a shard).
// Only effective on Linux; silently ignored on other platforms.
func HintDiscard(f *os.File, offset int64, length int64) error {
	err := unix.Fadvise(int(f.Fd()), offset, length, 4)
	if err != nil {
		return fmt.Errorf("gguf: fadvise discard failed: %w", err)
	}
	return nil
}

// HintNoReuse advises the OS kernel that data in the range [offset, offset+count] of [f] is unlikely
// to be read again. This allows the kernel to free pages aggressively without waiting for them to be
// evicted by normal LRU pressure. Useful after processing a metadata section you will not revisit.
// Only effective on Linux; silently ignored on other platforms.
func HintNoReuse(f *os.File, offset int64, count int64) error {
	err := unix.Fadvise(int(f.Fd()), offset, count, 3)
	if err != nil {
		return fmt.Errorf("gguf: fadvise no-reuse failed: %w", err)
	}
	return nil
}