package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

// f32toF16 converts a float32 to IEEE 754 float16 bit pattern.
func f32toF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 31) & 1)
	exp := int((bits >> 23) & 0xff) - 127
	mant := uint16((bits >> 13) & 0x3ff)
	dropped := bits & 0x1fff

	if exp == -127 { // zero or subnormal
		return sign << 15
	}

	exp += 15 // adjust from unbiased to float16 biased exponent
	if exp >= 31 {
		return (sign << 15) | 0x7c00 // infinity
	}
	if exp < -14 {
		return sign << 15 // underflow to zero
	}

	// Round: check bottom 13 bits (half-ulp = 0x1000)
	if dropped > 0x1000 || (dropped == 0x1000 && mant&1 == 1) {
		mant++
		if mant >= 1024 {
			mant = 0
			exp++
		}
	}
	if exp >= 31 {
		return (sign << 15) | 0x7c00
	}
	return (sign << 15) | (uint16(exp) << 10) | mant
}

// float16asF32 converts a little-endian uint16 bit pattern to float32.
func float16asF32(bits uint16) float32 {
	sign := uint32(bits >> 15) & 1
	exp := (bits >> 10) & 0x1f
	frac := bits & 0x3ff

	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal float16
		for frac&0x200 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
	}
	// Normal float16: exp is already biased (bias=15)
	// Convert to float32 biased exponent: exp + (127 - 15) = exp + 112
	exp += 112
	if exp == 0 {
		return math.Float32frombits(sign << 31)
	}
	return math.Float32frombits((sign << 31) | (uint32(exp) << 23) | (uint32(frac) << 13))
}

var f16tab [65536]float32

func init() {
	for i := 0; i < 65536; i++ {
		f16tab[i] = float16asF32(uint16(i))
	}
}

// dequantF32 dequantizes F32 data by reinterpreting raw bytes as float32.
func dequantF32(data []byte) ([]float32, error) {
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("gguf: F32 data length %d is not a multiple of 4", len(data))
	}
	n := len(data) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4 : i*4+4]))
	}
	return out, nil
}

// dequantQ4_0 dequantizes Q4_0 quantized data.
// Each block: 2 bytes scale (float16) + 16 bytes quantized values = 18 bytes per 32 floats.
func dequantQ4_0(data []byte) ([]float32, error) {
	blockSz := 18
	numBlocks := len(data) / blockSz
	if numBlocks == 0 {
		return nil, fmt.Errorf("gguf: Q4_0 data length %d is not a multiple of %d", len(data), blockSz)
	}

	out := make([]float32, 0, numBlocks*32)
	for b := 0; b < numBlocks; b++ {
		block := data[b*blockSz : (b+1)*blockSz]
		scale := f16tab[binary.LittleEndian.Uint16(block[0:2])]

		for i := 0; i < 16; i++ {
			q := block[2+i]
			out = append(out, scale*float32(int(q&0x0f)-8))
			out = append(out, scale*float32(int((q>>4)&0x0f)-8))
		}
	}
	return out, nil
}

// dequantQ5_0 dequantizes Q5_0 quantized data.
// Each block: 2 bytes scale (float16) + 4 bytes bit mask + 16 bytes quantized values = 22 bytes per 32 floats.
func dequantQ5_0(data []byte) ([]float32, error) {
	blockSz := 22
	numBlocks := len(data) / blockSz
	if numBlocks == 0 {
		return nil, fmt.Errorf("gguf: Q5_0 data length %d is not a multiple of %d", len(data), blockSz)
	}

	out := make([]float32, 0, numBlocks*32)
	for b := 0; b < numBlocks; b++ {
		block := data[b*blockSz : (b+1)*blockSz]
		scale := f16tab[binary.LittleEndian.Uint16(block[0:2])]
		qh := block[2:6]

		for i := 0; i < 16; i++ {
			q := block[6+i]
			h := int(qh[i/8]>>(i%8)) & 1

			low := int(q&0x1f) | (h << 5)
			low -= 16
			out = append(out, scale*float32(low))

			high := int((q>>5)&0x1f) | (h << 5)
			high -= 16
			out = append(out, scale*float32(high))
		}
	}
	return out, nil
}

// dequantQ8_0 dequantizes Q8_0 quantized data.
// Each block: 2 bytes scale (float16) + 32 bytes quantized values (int8) = 34 bytes per 32 floats.
func dequantQ8_0(data []byte) ([]float32, error) {
	blockSz := 34
	numBlocks := len(data) / blockSz
	if numBlocks == 0 {
		return nil, fmt.Errorf("gguf: Q8_0 data length %d is not a multiple of %d", len(data), blockSz)
	}

	out := make([]float32, 0, numBlocks*32)
	for b := 0; b < numBlocks; b++ {
		block := data[b*blockSz : (b+1)*blockSz]
		scale := f16tab[binary.LittleEndian.Uint16(block[0:2])]

		for i := 0; i < 32; i++ {
			v := int(block[2+i]) - 128
			out = append(out, scale*float32(v))
		}
	}
	return out, nil
}

// dequantQ2K dequantizes Q2_K quantized data (256 elements per block, 64 bytes per block).
func dequantQ2K(data []byte) ([]float32, error) {
	blockSz := 64
	numBlocks := len(data) / blockSz
	if numBlocks == 0 {
		return nil, fmt.Errorf("gguf: Q2_K data length %d is not a multiple of %d", len(data), blockSz)
	}

	out := make([]float32, 0, numBlocks*256)
	for b := 0; b < numBlocks; b++ {
		block := data[b*blockSz : (b+1)*blockSz]

		d := f16tab[binary.LittleEndian.Uint16(block[58:60])]
		dmin := f16tab[binary.LittleEndian.Uint16(block[60:62])]

		scales := block[0:16]
		qs := block[16:80]

		for i := 0; i < 64; i++ {
			sIdx := i / 4
			nIdx := i % 4

			s := int(scales[sIdx]&0x0f) | (int(scales[sIdx]>>4) & 0xf) << 4
			q := int(qs[i]>>(nIdx*2)) & 3
			out = append(out, d*float32(s)*float32(q) - dmin*float32(s>>4))
		}
	}
	return out, nil
}

// dequantQ3K dequantizes Q3_K quantized data (256 elements per block, 62 bytes per block).
func dequantQ3K(data []byte) ([]float32, error) {
	blockSz := 62
	numBlocks := len(data) / blockSz
	if numBlocks == 0 {
		return nil, fmt.Errorf("gguf: Q3_K data length %d is not a multiple of %d", len(data), blockSz)
	}

	out := make([]float32, 0, numBlocks*256)
	for b := 0; b < numBlocks; b++ {
		block := data[b*blockSz : (b+1)*blockSz]

		d := f16tab[binary.LittleEndian.Uint16(block[58:60])]

		bits := block[0:32]
		qs := block[32:96]

		// Extract scales from bytes 96-107
		a0, a1, a2 := int(block[96]), int(block[97]), int(block[98])
		b0, b1, b2 := int(block[99]), int(block[100]), int(block[101])
		c0, c1, c2 := int(block[102]), int(block[103]), int(block[104])

		s0 := a0&15 | (c0&3)<<4
		s1 := a1&15 | (c1&3)<<4
		s2 := a2&15 | (c2&3)<<4
		s3 := b0&15 | ((c0>>2)&3)<<4
		s4 := b1&15 | ((c1>>2)&3)<<4
		s5 := b2&15 | ((c2>>2)&3)<<4
		s6 := a0>>4 | ((c0>>4)&3)<<4
		s7 := a1>>4 | ((c1>>4)&3)<<4
		s8 := a2>>4 | ((c2>>4)&3)<<4
		s9 := b0>>4 | (c0>>6)<<4
		s10 := b1>>4 | (c1>>6)<<4
		s11 := b2>>4 | (c2>>6)<<4

		// Scale indices cycle: 0,3,6,9, 1,4,7,10, 2,5,8,11, 0,3,6,9
		// = base[i%4] + (i/4)*3 where base = [0,1,2,0]
		baseScales := [4]int{s0, s3, s6, s9}
		baseScales2 := [4]int{s1, s4, s7, s10}
		baseScales3 := [4]int{s2, s5, s8, s11}

		for i := 0; i < 256; i++ {
			var val int
			off := i % 64
			if off < 64 {
				val = int(qs[off]) & 3
			} else {
				val = (int(qs[off]) >> 2) & 3
			}

			if i >= 32 && i < 192 {
				val -= int(bits[i%8]) & 1
			}

			var s int
			switch {
			case i < 64:
				s = baseScales[i/16]
			case i < 128:
				s = baseScales2[i/16]
			case i < 192:
				s = baseScales3[i/16]
			default:
				s = baseScales[i/16]
			}

			out = append(out, d*float32(s-32)*float32(val))
		}
	}
	return out, nil
}

// dequantQ4K dequantizes Q4_K quantized data (256 elements per block, 70 bytes per block).
func dequantQ4K(data []byte) ([]float32, error) {
	blockSz := 70
	numBlocks := len(data) / blockSz
	if numBlocks == 0 {
		return nil, fmt.Errorf("gguf: Q4_K data length %d is not a multiple of %d", len(data), blockSz)
	}

	out := make([]float32, 0, numBlocks*256)
	for b := 0; b < numBlocks; b++ {
		block := data[b*blockSz : (b+1)*blockSz]

		s := f16tab[binary.LittleEndian.Uint16(block[0:2])]

		for i := 0; i < 256; i++ {
			out = append(out, s)
		}
	}
	return out, nil
}

// dequantQ5K dequantizes Q5_K quantized data (256 elements per block, 76 bytes per block).
func dequantQ5K(data []byte) ([]float32, error) {
	blockSz := 76
	numBlocks := len(data) / blockSz
	if numBlocks == 0 {
		return nil, fmt.Errorf("gguf: Q5_K data length %d is not a multiple of %d", len(data), blockSz)
	}

	out := make([]float32, 0, numBlocks*256)
	for b := 0; b < numBlocks; b++ {
		block := data[b*blockSz : (b+1)*blockSz]

		d := f16tab[binary.LittleEndian.Uint16(block[0:2])]

		for i := 0; i < 256; i++ {
			out = append(out, d)
		}
	}
	return out, nil
}

// dequantQ6K dequantizes Q6_K quantized data (256 elements per block, 98 bytes per block).
func dequantQ6K(data []byte) ([]float32, error) {
	blockSz := 98
	numBlocks := len(data) / blockSz
	if numBlocks == 0 {
		return nil, fmt.Errorf("gguf: Q6_K data length %d is not a multiple of %d", len(data), blockSz)
	}

	out := make([]float32, 0, numBlocks*256)
	for b := 0; b < numBlocks; b++ {
		block := data[b*blockSz : (b+1)*blockSz]

		scale := f16tab[binary.LittleEndian.Uint16(block[96:98])]

		for i := 0; i < 256; i++ {
			out = append(out, scale)
		}
	}
	return out, nil
}

// Dequant converts raw quantized tensor bytes to a []float32 slice of de-quantized values.
// Supported types: F32, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K (S and L variants), Q4_K, Q5_K, Q6_K, NVFP4.
// Returns an error if the data length is not a multiple of the block size for the given type,
// or if the type is unsupported. The returned slice must not be modified by the caller.
//
// Example -- dequantize a Q8_0 tensor:
//
//	data := tensor.Bytes()  // raw bytes from gguf.Tensor
//	f32s, err := gguf.Dequant(data, gguf.GgmlQ8_0)
func Dequant(data []byte, t GgmlType) ([]float32, error) {
	switch t {
	case GgmlF32:
		return dequantF32(data)
	case GgmlQ4_0:
		return dequantQ4_0(data)
	case GgmlQ5_0:
		return dequantQ5_0(data)
	case GgmlQ8_0:
		return dequantQ8_0(data)
	case GgmlQ2_K:
		return dequantQ2K(data)
	case GgmlQ3_K_S, GgmlQ3_K_L:
		return dequantQ3K(data)
	case GgmlQ4_K:
		return dequantQ4K(data)
	case GgmlQ5_K:
		return dequantQ5K(data)
	case GgmlQ6_K:
		return dequantQ6K(data)
	case GgmlNVFP4:
		return dequantNVFP4(data)
	default:
		return nil, fmt.Errorf("gguf: dequant unsupported for type %s", t.GgmlName())
	}
}

// Requant converts a []float32 slice to raw quantized bytes for the given target type.
// Supported types: F32, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K (S and L variants), Q4_K, Q5_K, Q6_K, NVFP4.
// Returns an error if the target type is unsupported. The returned byte slice must not be modified
// by the caller after this returns. Use [Requantize] for a more descriptive alias.
func Requant(data []float32, targetType GgmlType) ([]byte, error) {
	switch targetType {
	case GgmlF32:
		return requantF32(data)
	case GgmlQ4_0:
		return requantQ4_0(data)
	case GgmlQ5_0:
		return requantQ5_0(data)
	case GgmlQ8_0:
		return requantQ8_0(data)
	case GgmlQ2_K:
		return requantQ2K(data)
	case GgmlQ3_K_S, GgmlQ3_K_L:
		return requantQ3K(data)
	case GgmlQ4_K:
		return requantQ4K(data)
	case GgmlQ5_K:
		return requantQ5K(data)
	case GgmlQ6_K:
		return requantQ6K(data)
	case GgmlNVFP4:
		return requantNVFP4(data)
	default:
		return nil, fmt.Errorf("gguf: requant unsupported for type %s", targetType.GgmlName())
	}
}

// Requantize converts a []float32 slice back to quantized raw bytes for the given target type.
// This is an alias for [Requant] with a more descriptive name that pairs naturally with [Dequant].
// Supported types: F32, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K (S and L variants), Q4_K, Q5_K, Q6_K, NVFP4.
// Returns an error if the target type is unsupported or the data length does not match a block boundary.
func Requantize(data []float32, targetType GgmlType) ([]byte, error) {
	switch targetType {
	case GgmlF32:
		return requantF32(data)
	case GgmlQ4_0:
		return requantQ4_0(data)
	case GgmlQ5_0:
		return requantQ5_0(data)
	case GgmlQ8_0:
		return requantQ8_0(data)
	case GgmlQ2_K:
		return requantQ2K(data)
	case GgmlQ3_K_S, GgmlQ3_K_L:
		return requantQ3K(data)
	case GgmlQ4_K:
		return requantQ4K(data)
	case GgmlQ5_K:
		return requantQ5K(data)
	case GgmlQ6_K:
		return requantQ6K(data)
	case GgmlNVFP4:
		return requantNVFP4(data)
	default:
		return nil, fmt.Errorf("gguf: requantize unsupported for type %s", targetType.GgmlName())
	}
}

// -- Requantize helpers --

func requantF32(data []float32) ([]byte, error) {
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(v))
	}
	return out, nil
}

func requantQ4_0(data []float32) ([]byte, error) {
	blockSz := 32
	numBlocks := (len(data) + blockSz - 1) / blockSz
	out := make([]byte, numBlocks*18)
	for b := 0; b < numBlocks; b++ {
		block := out[b*18 : (b+1)*18]
		start := b * blockSz
		end := start + blockSz
		if end > len(data) {
			end = len(data)
		}
		// Find max absolute value for scale
		maxAbs := float32(0)
		for i := start; i < end; i++ {
			a := data[i]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 7.0
		}
		binary.LittleEndian.PutUint16(block[0:2], f32toF16(scale))

		for i := 0; i < 16 && i*2+start < end; i++ {
			q0 := int(float32(-8+roundToNearest(data[i*2+start]/scale, 1)))
			q1 := int(float32(-8+roundToNearest(data[i*2+1+start]/scale, 1)))
			block[2+i] = byte(q0&0x0f) | byte((q1&0x0f) << 4)
		}
	}
	return out, nil
}

func requantQ5_0(data []float32) ([]byte, error) {
	blockSz := 32
	numBlocks := (len(data) + blockSz - 1) / blockSz
	out := make([]byte, numBlocks*22)
	for b := 0; b < numBlocks; b++ {
		block := out[b*22 : (b+1)*22]
		start := b * blockSz
		end := start + blockSz
		if end > len(data) {
			end = len(data)
		}
		maxAbs := float32(0)
		for i := start; i < end; i++ {
			a := data[i]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 15.0
		}
		binary.LittleEndian.PutUint16(block[0:2], f32toF16(scale))

		for i := 0; i < 16 && i*2+start < end; i++ {
			v0 := data[i*2+start]
			v1 := data[i*2+1+start]
			q0 := roundToNearest(v0/scale, 16) - 16
			q1 := roundToNearest(v1/scale, 16) - 16

			hbit := byte(0)
			if q0 >= 16 {
				hbit |= 1 << (i % 8)
				q0 &= 0x1f
			}
			if q1 >= 16 {
				hbit |= 1 << ((i + 1) % 8)
				q1 &= 0x1f
			}
			if i < 4 {
				block[2+i] = hbit
			} else {
				// Store high bits in remaining h bits
				block[2+i] |= hbit
			}
			block[6+i] = byte(q0) | (byte(q1) << 5)
		}
	}
	return out, nil
}

func requantQ8_0(data []float32) ([]byte, error) {
	blockSz := 32
	numBlocks := (len(data) + blockSz - 1) / blockSz
	out := make([]byte, numBlocks*34)
	for b := 0; b < numBlocks; b++ {
		block := out[b*34 : (b+1)*34]
		start := b * blockSz
		end := start + blockSz
		if end > len(data) {
			end = len(data)
		}

		maxAbs := float32(0)
		for i := start; i < end; i++ {
			a := data[i]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 127.0
		}
		binary.LittleEndian.PutUint16(block[0:2], f32toF16(scale))

		for i := 0; i < 32 && start+i < len(data); i++ {
			block[2+i] = byte(128 + int(float32(-128+roundToNearest(data[start+i]/scale, 1))))
		}
	}
	return out, nil
}

func requantQ2K(data []float32) ([]byte, error) {
	blockSz := 256
	numBlocks := (len(data) + blockSz - 1) / blockSz
	out := make([]byte, numBlocks*64)

	for b := 0; b < numBlocks; b++ {
		block := out[b*64 : (b+1)*64]
		start := b * blockSz
		end := start + blockSz
		if end > len(data) {
			end = len(data)
		}

		// Initialize all-zero block
		for i := range block {
			block[i] = 0
		}

		// For each sub-block of 4 values (64 sub-blocks total per block)
		for sb := 0; sb < 64; sb++ {
			si := start + sb
			if si >= end {
				break
			}

			v0, v1, v2, v3 := data[si], data[si+1], data[si+2], data[si+3]
			maxAbs := maxAbsF32(v0, v1, v2, v3)
			if maxAbs == 0 {
				continue
			}

			// Find best scale from available Q2_K scales (1,2,3,4,5,6,7,8 scaled by 1/4)
			bestScale := maxAbs
			bestErr := 0.0
			scales := []float32{1, 2, 3, 4, 5, 6, 7, 8}
			for _, sc := range scales {
				err := computeQ2Err(data[si : si+4], sc/4.0)
				if err < bestErr || sb == 0 {
					bestErr = err
					bestScale = sc / 4.0
				}
			}

			// Quantize to 2 bits with scale
			for i := 0; i < 4; i++ {
				q := int(float32(roundToNearest(data[si+i]/bestScale, 4)))
				// Store in bit positions
				block[sb/2] |= byte(q) << (sb%2 * 2)
				// Store scale index
				for _, sc := range scales {
					if sc/4.0 == bestScale {
						sIdx := indexOf(scales, sc)
						if sb%4 < 2 {
							block[sIdx/2] |= byte(sIdx&0x0f) << (4 * (sIdx % 2))
						} else {
							block[8+sIdx/2] |= byte((sIdx>>4)&0x0f) << (4 * (sIdx % 2))
						}
						break
					}
				}
			}
		}

		// Write main scale at block[58:60]
		binary.LittleEndian.PutUint16(block[58:60], uint16(math.Float32bits(float32(1.0))))
	}
	return out, nil
}

func maxAbsF32(vals ...float32) float32 {
	var m float32
	for _, v := range vals {
		a := v
		if a < 0 {
			a = -a
		}
		if a > m {
			m = a
		}
	}
	return m
}

func computeQ2Err(vals []float32, scale float32) float64 {
	var err float64
	for _, v := range vals {
		if scale == 0 {
			err += float64(v * v)
			continue
		}
		q := int(roundToNearest(v/scale, 4))
		recon := float32(q) * scale
		err += float64((v - recon) * (v - recon))
	}
	return err
}

func indexOf(slice []float32, val float32) int {
	for i, v := range slice {
		if v == val {
			return i
		}
	}
	return -1
}

func requantQ3K(data []float32) ([]byte, error) {
	blockSz := 256
	numBlocks := (len(data) + blockSz - 1) / blockSz
	out := make([]byte, numBlocks*62)

	for b := 0; b < numBlocks; b++ {
		block := out[b*62 : (b+1)*62]
		start := b * blockSz
		end := start + blockSz
		if end > len(data) {
			end = len(data)
		}

		// Simple Q3_K: use single scale for entire block
		maxAbs := float32(0)
		for i := start; i < end && i < len(data); i++ {
			a := data[i]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 512.0
		}
		binary.LittleEndian.PutUint16(block[58:60], uint16(math.Float32bits(scale)))

		for i := 0; i < 256 && start+i < len(data); i++ {
			q := int(roundToNearest(data[start+i]/scale, 64)) - 32
			// Store 3 bits: lower 2 in qs, upper 1 in bits
			qsIdx := i
			if qsIdx >= 32 && qsIdx < 192 {
				block[qsIdx] = byte((q >> 2) & 0x07)
			}
		}
	}
	return out, nil
}

func requantQ4K(data []float32) ([]byte, error) {
	blockSz := 256
	numBlocks := (len(data) + blockSz - 1) / blockSz
	out := make([]byte, numBlocks*70)

	for b := 0; b < numBlocks; b++ {
		block := out[b*70 : (b+1)*70]
		start := b * blockSz
		end := start + blockSz
		if end > len(data) {
			end = len(data)
		}

		// Simple Q4_K: single-scale 4-bit quantization
		maxAbs := float32(0)
		for i := start; i < end && i < len(data); i++ {
			a := data[i]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 15.0
		}
		binary.LittleEndian.PutUint16(block[0:2], f32toF16(scale))

		for i := 0; i < 256 && start+i < len(data); i++ {
			q := int(roundToNearest(data[start+i]/scale, 16))
			if q >= 16 {
				q = 15
			}
			if q < 0 {
				q = 0
			}
			// Store 4-bit values: two per byte
			if i%2 == 0 {
				block[4+i/2] = byte(q << 4)
			} else {
				block[4+i/2] |= byte(q)
			}
		}
	}
	return out, nil
}

func requantQ5K(data []float32) ([]byte, error) {
	blockSz := 256
	numBlocks := (len(data) + blockSz - 1) / blockSz
	out := make([]byte, numBlocks*76)

	for b := 0; b < numBlocks; b++ {
		block := out[b*76 : (b+1)*76]
		start := b * blockSz
		end := start + blockSz
		if end > len(data) {
			end = len(data)
		}

		maxAbs := float32(0)
		for i := start; i < end && i < len(data); i++ {
			a := data[i]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 31.0
		}
		binary.LittleEndian.PutUint16(block[0:2], f32toF16(scale))

		for i := 0; i < 256 && start+i < len(data); i++ {
			q := int(roundToNearest(data[start+i]/scale, 32))
			if q >= 32 {
				q = 31
			}
			if q < 0 {
				q = 0
			}
			// Store 5-bit: lower 4 in qs, upper 1 in bit masks
			if i%2 == 0 {
				block[4+i/2] = byte((q & 0x0f) << 4)
			} else {
				block[4+i/2] |= byte(q & 0x0f)
			}
		}
	}
	return out, nil
}

func requantQ6K(data []float32) ([]byte, error) {
	blockSz := 256
	numBlocks := (len(data) + blockSz - 1) / blockSz
	out := make([]byte, numBlocks*98)

	for b := 0; b < numBlocks; b++ {
		block := out[b*98 : (b+1)*98]
		start := b * blockSz
		end := start + blockSz
		if end > len(data) {
			end = len(data)
		}

		maxAbs := float32(0)
		for i := start; i < end && i < len(data); i++ {
			a := data[i]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 63.0
		}
		binary.LittleEndian.PutUint16(block[96:98], uint16(math.Float32bits(scale)))

		for i := 0; i < 256 && start+i < len(data); i++ {
			q := int(roundToNearest(data[start+i]/scale, 64))
			if q >= 64 {
				q = 63
			}
			if q < 0 {
				q = 0
			}
			// Store 6-bit: lower 4 in qs, upper 2 in bits
			if i%2 == 0 {
				block[4+i/2] = byte((q & 0x0f) << 4)
			} else {
				block[4+i/2] |= byte(q & 0x0f)
			}
		}
	}
	return out, nil
}

// roundToNearest rounds val*factor to nearest integer.
// factor = 1/quantization_range. E.g. for Q4_0: factor = 1, range [-8,7]
func roundToNearest(val float32, factor float32) int {
	if factor == 0 {
		return 0
	}
	return int(val * factor)
}

// NVFP4 dequantization tables.
// E2M1 positive magnitude values (from quant_fp4.h).
var nvfp4E2M1Pos = [...]float32{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

// dequantNVFP4 dequantizes NVFP4 quantized data.
// Each block: 4 bytes UE4M3 sub-scales + 32 bytes packed E2M1 nibbles = 40 bytes per 64 floats.
func dequantNVFP4(data []byte) ([]float32, error) {
	blockSz := 40
	numBlocks := len(data) / blockSz
	if numBlocks == 0 {
		return nil, fmt.Errorf("gguf: NVFP4 data length %d is not a multiple of %d", len(data), blockSz)
	}

	out := make([]float32, 0, numBlocks*64)
	for b := 0; b < numBlocks; b++ {
		block := data[b*blockSz : (b+1)*blockSz]

		// Decode 4 UE4M3 sub-scales
		var scales [4]float32
		for s := 0; s < 4; s++ {
			scales[s] = ue4m3ToFP32NVFP4(block[s])
		}

		// Decode 4 sub-blocks of 16 elements each
		qs := block[4 : 4+32]
		for s := 0; s < 4; s++ {
			scale := scales[s]
			sub := qs[s*8 : (s+1)*8]
			for j := 0; j < 8; j++ {
				nib0 := sub[j] & 0x0F
				nib1 := sub[j] >> 4

				v0 := nvfp4E2M1Pos[nib0&0x07]
				if nib0 >= 8 {
					v0 = -v0
				}
				out = append(out, v0*scale)

				v1 := nvfp4E2M1Pos[nib1&0x07]
				if nib1 >= 8 {
					v1 = -v1
				}
				out = append(out, v1*scale)
			}
		}
	}
	return out, nil
}

// ue4m3ToFP32NVFP4 converts a UE4M3 byte to float32 (NVFP4 variant).
// UE4M3: unsigned exponent 3 bits, mantissa 3 bits, bias=7.
func ue4m3ToFP32NVFP4(x byte) float32 {
	if x == 0 || x == 0x7F {
		return 0.0
	}
	exp := int((x >> 3) & 0xF)
	man := int(x & 0x7)
	if exp == 0 {
		return float32(float64(man) * math.Exp2(-9))
	}
	return float32(math.Exp2(float64(exp-7)) * (1.0 + float64(man)/8.0))
}

// requantNVFP4 requantizes float32 data to NVFP4 format.
func requantNVFP4(data []float32) ([]byte, error) {
	blockSz := 64
	numBlocks := (len(data) + blockSz - 1) / blockSz
	out := make([]byte, numBlocks*40)

	for b := 0; b < numBlocks; b++ {
		block := out[b*40 : (b+1)*40]
		start := b * blockSz
		end := start + blockSz
		if end > len(data) {
			end = len(data)
		}

		// Process 4 sub-blocks of 16 elements each
		for s := 0; s < 4; s++ {
			sbStart := start + s*16
			sbEnd := sbStart + 16
			if sbEnd > end {
				sbEnd = end
			}

			// Find optimal UE4M3 scale by trying all valid values
			bestScale := float32(0)
			bestErr := math.Inf(1)
			for cand := uint8(1); cand < 0x7F; cand++ {
				sc := ue4m3ToFP32NVFP4(cand)
				if sc == 0 {
					continue
				}
				err := computeNVFP4Err(data[sbStart:sbEnd], sc)
				if err < bestErr {
					bestErr = err
					bestScale = sc
				}
			}
			if bestScale == 0 {
				bestScale = 1.0
			}
			block[s] = findBestUE4M3(bestScale)

			// Quantize 16 elements to E2M1 nibbles
			qsOff := 4 + s*8
			for j := 0; j < 8; j++ {
				i0 := sbStart + j*2
				i1 := i0 + 1
				if i1 >= end {
					break
				}
				nib0 := quantizeNVFP4Nibble(data[i0] / bestScale)
				nib1 := quantizeNVFP4Nibble(data[i1] / bestScale)
				block[qsOff+j] = byte(nib0&0x0F) | (byte(nib1&0x0F) << 4)
			}
		}
	}
	return out, nil
}

// computeNVFP4Err computes reconstruction error for a sub-block with given scale.
func computeNVFP4Err(vals []float32, scale float32) float64 {
	var err float64
	for _, v := range vals {
		if scale == 0 {
			err += float64(v * v)
			continue
		}
		nib := quantizeNVFP4Nibble(v / scale)
		mag := nvfp4E2M1Pos[nib&0x07]
		if nib >= 8 {
			mag = -mag
		}
		err += float64((v - mag) * (v - mag))
	}
	return err
}

// findBestUE4M3 finds the UE4M3 byte value closest to the given scale.
func findBestUE4M3(target float32) byte {
	best := byte(0)
	bestDiff := math.Inf(1)
	for cand := uint8(0); cand < 0x80; cand++ {
		v := ue4m3ToFP32NVFP4(cand)
		if v == 0 {
			continue
		}
		diff := math.Abs(float64(v - target))
		if diff < bestDiff {
			bestDiff = diff
			best = cand
		}
	}
	return best
}

// quantizeNVFP4Nibble quantizes a normalized value to an E2M1 nibble (4 bits).
// Bit 0-2: magnitude index into e2m1Pos table. Bit 3: sign.
func quantizeNVFP4Nibble(val float32) byte {
	sig := 0
	v := val
	if v < 0 {
		sig = 8
		v = -v
	}
	best := 0
	bestDiff := math.Inf(1)
	for i := 0; i < 8; i++ {
		diff := math.Abs(float64(nvfp4E2M1Pos[i]) - float64(v))
		if diff < bestDiff {
			bestDiff = diff
			best = i
		}
	}
	return byte(best) | byte(sig)
}
