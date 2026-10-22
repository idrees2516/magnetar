//! Bitslicing: 32 field elements share one `u32` per bit-plane.
//!
//! A *group* of 32 elements is stored as 128 words: word `j` holds bit `j`
//! (tower coordinate `j`) of each of the 32 elements, with element `e`'s bit
//! at lane `e`. Field addition of two groups becomes 128 word-XORs; the
//! tower multiplication tree ([`crate::tree`]) becomes a sequence of
//! plane-level XOR/AND ops, which is exactly what the GPU kernel executes.
//!
//! Layout: **group-major rows**, matching the GPU tensor `[groups, 128]`:
//! `words[g * 128 + j]` is plane `j` of group `g`.

#![forbid(unsafe_code)]

/// Elements per bitsliced group.
pub const LANES: usize = 32;
/// Tower coordinates per element (== planes per group).
pub const PLANES: usize = 128;
/// u32 words per group.
pub const WORDS_PER_GROUP: usize = LANES * 4; // == PLANES

/// Bitsliced storage in group-major layout, identical to the GPU tensor
/// layout `[groups, 128]` (row `g` = plane words of group `g`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitPlanes {
    /// Group-major plane words: `words[g * 128 + j]` is plane `j` of group `g`.
    pub words: Vec<u32>,
    /// Number of 32-element groups.
    pub groups: usize,
}

/// Packing failure modes.
#[derive(Debug)]
pub enum PackError {
    /// Element count must be a positive multiple of 32.
    NotMultipleOfLanes(usize),
}

impl core::fmt::Display for PackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PackError::NotMultipleOfLanes(n) => {
                write!(f, "element count {n} is not a positive multiple of {LANES}")
            }
        }
    }
}

impl BitPlanes {
    /// Allocates zeroed storage for `groups` groups.
    #[must_use]
    pub fn zeroed(groups: usize) -> Self {
        Self {
            words: vec![0u32; groups * WORDS_PER_GROUP],
            groups,
        }
    }

    /// Packs 32n elements (little-endian coordinate vectors) into planes.
    ///
    /// One-time host-side cost: ~128 bit extractions per element. For large
    /// witness sizes use the GPU pack kernel (the `cutile` feature,
    /// [`crate::gpu`]).
    pub fn pack(evals: &[crate::F128]) -> Result<Self, PackError> {
        if evals.is_empty() || evals.len() % LANES != 0 {
            return Err(PackError::NotMultipleOfLanes(evals.len()));
        }
        let groups = evals.len() / LANES;
        let mut words = vec![0u32; groups * WORDS_PER_GROUP];
        for (g, chunk) in evals.chunks_exact(LANES).enumerate() {
            for (j, w) in words[g * WORDS_PER_GROUP..g * WORDS_PER_GROUP + PLANES]
                .iter_mut()
                .enumerate()
            {
                let mut acc = 0u32;
                for (e, elem) in chunk.iter().enumerate() {
                    if (elem.0 >> j) & 1 == 1 {
                        acc |= 1u32 << e;
                    }
                }
                *w = acc;
            }
        }
        Ok(Self { words, groups })
    }

    /// Unpacks back to 32n elements.
    #[must_use]
    pub fn unpack(&self) -> Vec<crate::F128> {
        let mut out = vec![crate::F128::ZERO; self.groups * LANES];
        for g in 0..self.groups {
            for j in 0..PLANES {
                let w = self.words[g * WORDS_PER_GROUP + j];
                if w == 0 {
                    continue;
                }
                for e in 0..LANES {
                    if (w >> e) & 1 == 1 {
                        out[g * LANES + e].0 |= 1u128 << j;
                    }
                }
            }
        }
        out
    }

    /// Plane word `j` of group `g`.
    #[must_use]
    pub fn plane(&self, g: usize, j: usize) -> u32 {
        self.words[g * WORDS_PER_GROUP + j]
    }

    /// Interprets a raw group-major word slice (GPU tensor download) as planes.
    #[must_use]
    pub fn from_group_major(words: Vec<u32>) -> Self {
        debug_assert!(words.len() % WORDS_PER_GROUP == 0);
        let groups = words.len() / WORDS_PER_GROUP;
        Self { words, groups }
    }

    /// XOR of all 32 elements of group `g` (field sum).
    #[must_use]
    pub fn group_sum(&self, g: usize) -> crate::F128 {
        let mut acc = 0u128;
        for j in 0..PLANES {
            if parity32(self.words[g * WORDS_PER_GROUP + j]) == 1 {
                acc |= 1u128 << j;
            }
        }
        crate::F128(acc)
    }
}

/// Parity of the 32 bits of a word (XOR of all lanes).
#[inline]
pub fn parity32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    x & 1
}

/// The 5-stage 32x32 bit-matrix transpose used by the GPU bit packing; provided
/// here as a lane-sliced simulator so the CUDA shuffle sequence can be
/// validated on the host.
///
/// `words[e]` is the 32-bit word of element `e`; after the call `words[b]`
/// is the plane word holding bit `b` of each element at lane `e`.
pub fn transpose32_lane_sim(words: &mut [u32; LANES]) {
    // Classical 5-stage 32x32 bit-matrix transpose. Stage masks are the
    // interleaved patterns (d ones, d zeros alternating), and each stage
    // exchanges the low-d sub-words of the low lane with the high-d
    // sub-words of the partner lane, within every 2d-sized block:
    //   d=16: 0x0000FFFF, d=8: 0x00FF00FF, d=4: 0x0F0F0F0F,
    //   d=2:  0x33333333, d=1: 0x55555555
    const MASKS: [u32; 5] = [
        0x0000_FFFF,
        0x00FF_00FF,
        0x0F0F_0F0F,
        0x3333_3333,
        0x5555_5555,
    ];
    for (stage, &mask) in MASKS.iter().enumerate() {
        let d: usize = 1 << (4 - stage);
        for lane in 0..LANES {
            if lane & d == 0 {
                let partner = lane ^ d;
                let a = words[lane];
                let b = words[partner];
                words[lane] = (a & mask) | ((b & mask) << d);
                words[partner] = (b & !mask) | ((a & !mask) >> d);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;
    use crate::F128;

    struct SplitMix(u64);
    impl crate::Rng for SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x09E3_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let mut r = SplitMix(7);
        for n in [32usize, 64, 128, 1024] {
            let evals: Vec<F128> = (0..n).map(|_| F128::random(&mut r)).collect();
            let planes = BitPlanes::pack(&evals).expect("pack");
            assert_eq!(planes.groups, n / 32);
            assert_eq!(planes.unpack(), evals);
        }
    }

    #[test]
    fn pack_rejects_bad_sizes() {
        assert!(BitPlanes::pack(&[]).is_err());
        assert!(BitPlanes::pack(&[F128::ZERO; 33]).is_err());
    }

    #[test]
    fn group_sum_matches_scalar() {
        let mut r = SplitMix(11);
        let evals: Vec<F128> = (0..64).map(|_| F128::random(&mut r)).collect();
        let planes = BitPlanes::pack(&evals).expect("pack");
        for g in 0..planes.groups {
            let mut scalar = F128::ZERO;
            for e in 0..32 {
                scalar += evals[g * 32 + e];
            }
            assert_eq!(planes.group_sum(g), scalar, "group {g}");
        }
    }

    #[test]
    fn transpose_sim_matches_naive() {
        let mut r = SplitMix(0x7133);
        for _ in 0..50 {
            let mut words = [0u32; LANES];
            for w in words.iter_mut() {
                *w = r.next_u64() as u32;
            }
            let reference = words;
            transpose32_lane_sim(&mut words);
            // naive: out[b] = sum_e bit_b(reference[e]) << e
            for (b, got) in words.iter().enumerate() {
                let mut expect = 0u32;
                for (e, w) in reference.iter().enumerate() {
                    if (w >> b) & 1 == 1 {
                        expect |= 1 << e;
                    }
                }
                assert_eq!(*got, expect, "plane {b}");
            }
        }
    }

    #[test]
    fn parity32_basics() {
        assert_eq!(parity32(0), 0);
        assert_eq!(parity32(1), 1);
        assert_eq!(parity32(0xFFFF_FFFF), 0); // 32 ones
        assert_eq!(parity32(0x8000_0001), 0); // two ones
        assert_eq!(parity32(0x8000_0000), 1); // one one
    }
}
