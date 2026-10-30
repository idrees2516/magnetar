//! Bit-plane layouts for the packed (bitsliced) witness representation.
//!
//! A packed instance `A ∈ F^{m×κ}` is stored as `κ` *column planes* of `m`
//! bits each (the paper's `a^(j)` columns, Eq. (5)): plane `j` holds the
//! witness bits `a[j·m + i]`, bit `i` at word `i/64`, bit `i%64`.
//!
//! Heavy kernels are word-wise over `u64` lanes (auto-vectorizing Rust
//! slices); the bit→byte spread uses BMI2 `PDEP` when available (the
//! 3-op/row path) with a portable fallback, mirroring the repo's measured
//! SIMD-dispatch discipline.

use magnetar_field::mul;
use magnetar_field::Gf128;
use std::sync::OnceLock;

/// A set of `kappa` bit-planes over `m` rows (m must be a multiple of 64).
#[derive(Clone)]
pub struct PlaneSet {
    pub m: usize,
    pub kappa: usize,
    /// `planes[j * words + w]` = word `w` of column-plane `j`.
    pub planes: Vec<u64>,
    pub words: usize,
}

impl PlaneSet {
    pub fn zeros(m: usize, kappa: usize) -> Self {
        assert!(m.is_multiple_of(64), "m must be padded to a multiple of 64");
        let words = m / 64;
        PlaneSet {
            m,
            kappa,
            planes: vec![0u64; kappa * words],
            words,
        }
    }

    #[inline(always)]
    pub fn plane(&self, j: usize) -> &[u64] {
        &self.planes[j * self.words..(j + 1) * self.words]
    }

    #[inline(always)]
    pub fn plane_mut(&mut self, j: usize) -> &mut [u64] {
        let w = self.words;
        &mut self.planes[j * w..(j + 1) * w]
    }

    #[inline(always)]
    pub fn bit(&self, j: usize, i: usize) -> bool {
        let w = self.planes[j * self.words + i / 64];
        (w >> (i % 64)) & 1 == 1
    }

    #[inline(always)]
    pub fn set_bit(&mut self, j: usize, i: usize, v: bool) {
        let w = &mut self.planes[j * self.words + i / 64];
        if v {
            *w |= 1u64 << (i % 64);
        } else {
            *w &= !(1u64 << (i % 64));
        }
    }

    /// `self[j] ^= a[j] & b[j]` — the Hadamard products `a^(j) ∘ b^(j)`.
    pub fn and_into(&mut self, j: usize, a: &PlaneSet, b: &PlaneSet) {
        let w = self.words;
        let sa = &a.planes[j * w..(j + 1) * w];
        let sb = &b.planes[j * w..(j + 1) * w];
        let sc = &mut self.planes[j * w..(j + 1) * w];
        for k in 0..w {
            sc[k] ^= sa[k] & sb[k];
        }
    }

    /// Parity inner product of two planes: `⊕_i a[i]·b[i]` (AND + popcount
    /// parity — the GF(2) dot of the small-value sumcheck papers).
    /// Dispatches to the AVX-512 VPOPCNTQ kernel ([`crate::simd`]).
    pub fn parity_dot(a: &[u64], b: &[u64]) -> u64 {
        crate::simd::parity_dot_v(a, b)
    }

    /// The encoding primitive of the packed Hadamard prover:
    /// `out_c ← XOR of planes { j : selector bit j set }` for `c in 0..8`.
    ///
    /// This is the bitsliced evaluation `gA(γ') = Σ_j a^(j)·ψ_j(γ')` — the
    /// selector `σ_c` is the c-th bit of each code `ψ_j(γ')` ("precomputed
    /// base-polynomial evaluations", §4.2).
    ///
    /// **The j-outer pass**: each source plane streams once and XORs into
    /// every selected output plane through the AVX-512 lanes
    /// ([`crate::simd::xor_into`]) — the callers that re-encode many nodes
    /// over the same source batch them ([`Self::encode_k4`]) so the
    /// 8·m-byte source is re-read once per batch, not once per node.
    ///
    /// The batched form of [`Self::encode_k`] for the η pipeline: four
    /// selector sets against the same source planes in one pass — the
    /// `8·m`-byte source streams once per batch instead of once per node
    /// (a 4× traffic reduction across the 127 off-domain nodes; the batch's
    /// 4·8 output planes stay L2-resident).
    pub fn encode_k4(&self, sels: &[[[u64; 2]; 8]; 4], out: &mut [[Vec<u64>; 8]; 4]) {
        debug_assert_eq!(self.words, out[0][0].len());
        for b in 0..4 {
            for oc in out[b].iter_mut() {
                oc.iter_mut().for_each(|x| *x = 0);
            }
        }
        // Each source plane streams once per batch and XORs into every
        // selected output plane (the batch's 32 outputs are read/written
        // L2-resident; the single-node form re-streams the source per node).
        for j in 0..self.kappa {
            let sp = self.plane(j);
            let (word, bit) = (j / 64, j % 64);
            for b in 0..4 {
                for c in 0..8 {
                    if (sels[b][c][word] >> bit) & 1 == 1 {
                        crate::simd::xor_into(&mut out[b][c], sp);
                    }
                }
            }
        }
    }
    pub fn encode_k(&self, selectors: &[[u64; 2]; 8], out: &mut [Vec<u64>; 8]) {
        let w = self.words;
        for oc in out.iter_mut() {
            debug_assert_eq!(oc.len(), w);
            oc.iter_mut().for_each(|x| *x = 0);
        }
        for j in 0..self.kappa {
            let sp = self.plane(j);
            let (word, bit) = (j / 64, j % 64);
            for c in 0..8 {
                if (selectors[c][word] >> bit) & 1 == 1 {
                    crate::simd::xor_into(&mut out[c], sp);
                }
            }
        }
    }
}

fn bmi2_ok() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("bmi2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Spread 8 bit-planes of 64 rows into 64 bytes:
/// `out[r] = Σ_c bit_r(words[c]) << c` — the bit→byte transpose that feeds
/// the GF(2^8) product pipeline. BMI2 route: 8 `PDEP`s per 8 rows.
pub fn spread8(words: &[u64; 8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    if bmi2_ok() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            spread8_bmi2(words, &mut out)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            spread8_naive(words, &mut out)
        }
    } else {
        spread8_naive(words, &mut out);
    }
    out
}

#[cfg(target_arch = "x86_64")]
unsafe fn spread8_bmi2(words: &[u64; 8], out: &mut [u8; 64]) {
    use std::arch::x86_64::_pdep_u64;
    // SAFETY: bmi2_ok() verified the feature; _pdep_u64 is safe for any u64.
    unsafe {
        const BASE: u64 = 0x0101010101010101;
        for g in 0..8 {
            let mut acc = 0u64;
            for c in 0..8 {
                let src = words[c] >> (8 * g);
                acc |= _pdep_u64(src, BASE << c);
            }
            let bytes = acc.to_le_bytes();
            out[g * 8..g * 8 + 8].copy_from_slice(&bytes);
        }
    }
}

fn spread8_naive(words: &[u64; 8], out: &mut [u8; 64]) {
    for r in 0..64 {
        let mut b = 0u8;
        for c in 0..8 {
            b |= (((words[c] >> r) & 1) as u8) << c;
        }
        out[r] = b;
    }
}

/// In-place 64×64 bit-matrix transpose, by the block-exchange recursion:
/// `M = [A B; C D]` → exchange the anti-diagonal blocks `B ↔ C`, then
/// recurse on the four quadrants. (`a[i]` = row `i`; column `j` = bit `j`.)
pub fn flip64(a: &mut [u64; 64]) {
    transpose_rec(a, 0, 0, 64);
}

fn transpose_rec(a: &mut [u64], row0: usize, bit0: usize, n: usize) {
    if n <= 1 {
        return;
    }
    let half = n / 2;
    // exchange (rows [row0, row0+half) × bits [bit0+half, bit0+n))
    // with     (rows [row0+half, row0+n) × bits [bit0, bit0+half))
    for r in 0..half {
        let i = row0 + r;
        let j = row0 + half + r;
        let mask_lo = ((1u64 << half) - 1) << bit0;
        let mask_hi = mask_lo << half;
        let x = a[i] & mask_hi; // B-part of row i
        let y = a[j] & mask_lo; // C-part of row j
        a[i] = (a[i] & !mask_hi) | (y << half);
        a[j] = (a[j] & !mask_lo) | (x >> half);
    }
    transpose_rec(a, row0, bit0, half);
    transpose_rec(a, row0, bit0 + half, half);
    transpose_rec(a, row0 + half, bit0, half);
    transpose_rec(a, row0 + half, bit0 + half, half);
}

/// Transpose `m` E-words into 128 coordinate planes of `m` bits.
/// Cache-blocked over 8 chunks (one cache line per plane per block) with
/// the AVX-512 64×64 transpose kernel.
pub fn words_to_planes(words: &[Gf128], m: usize) -> Vec<u64> {
    assert!(m.is_multiple_of(64));
    let wp = m / 64;
    let mut planes = vec![0u64; 128 * wp];
    let mut tiles = [[0u64; 8]; 128];
    for cb in (0..wp).step_by(8) {
        let ncb = (wp - cb).min(8);
        for k in 0..ncb {
            let base = (cb + k) * 64;
            let mut lo = [0u64; 64];
            let mut hi = [0u64; 64];
            for r in 0..64 {
                let w = words[base + r].0;
                lo[r] = w as u64;
                hi[r] = (w >> 64) as u64;
            }
            crate::simd::flip64_v(&mut lo);
            crate::simd::flip64_v(&mut hi);
            for e in 0..64 {
                tiles[e][k] = lo[e];
                tiles[64 + e][k] = hi[e];
            }
        }
        for e in 0..128 {
            planes[e * wp + cb..][..ncb].copy_from_slice(&tiles[e][..ncb]);
        }
    }
    planes
}

/// The inverse transpose: 128 coordinate planes of `m` bits → `m` E-words.
/// Cache-blocked (one line per plane per block) with the AVX-512 kernel.
pub fn planes_to_words(planes: &[u64], m: usize) -> Vec<Gf128> {
    assert!(m.is_multiple_of(64));
    let wp = m / 64;
    let mut words = vec![Gf128(0); m];
    let mut tiles = [[0u64; 8]; 128];
    for cb in (0..wp).step_by(8) {
        let ncb = (wp - cb).min(8);
        for e in 0..128 {
            tiles[e][..ncb].copy_from_slice(&planes[e * wp + cb..][..ncb]);
        }
        for k in 0..ncb {
            let mut lo = [0u64; 64];
            let mut hi = [0u64; 64];
            for e in 0..64 {
                lo[e] = tiles[e][k];
                hi[e] = tiles[64 + e][k];
            }
            crate::simd::flip64_v(&mut lo);
            crate::simd::flip64_v(&mut hi);
            let base = (cb + k) * 64;
            for r in 0..64 {
                words[base + r] = Gf128((lo[r] as u128) | ((hi[r] as u128) << 64));
            }
        }
    }
    words
}

/// The row-weight tensor `l = ⊗_r (l_{r,0}, l_{r,1})`, entries in `E`
/// (Algorithm 2's row weight; the paper's two-entry-factor product,
/// Eq. (4)).
pub struct RowWeightTensor {
    pub log_m: usize,
    pub m: usize,
    /// `(l_{r,0}, l_{r,1})` per round `r` — factor `r` ↔ row bit `r`.
    pub factors: Vec<(Gf128, Gf128)>,
    /// E-word materialisation: `words[i] = Π_r l_{r, bit_r(i)}`.
    pub words: Vec<Gf128>,
    /// Coordinate planes: `planes[e * wp + w]`, bit `i%64` = coordinate bit
    /// `e` of `words[i]`.
    pub planes: Vec<u64>,
    pub words_per_plane: usize,
}

impl RowWeightTensor {
    pub fn new(factors: Vec<(Gf128, Gf128)>, with_words: bool) -> Self {
        let log_m = factors.len();
        let m = 1usize << log_m;
        // E-words by the doubling recursion (block-major; factor r ↔ bit r).
        let mut words = Vec::new();
        if with_words {
            words = vec![Gf128(1)];
            for (r, &(f0, f1)) in factors.iter().enumerate() {
                let half = 1usize << r;
                words.resize(half * 2, Gf128(0));
                for i in 0..half {
                    let v = words[i];
                    words[i] = mul(v, f0);
                    words[half + i] = mul(v, f1);
                }
            }
        }
        let planes = words_to_planes(&words, m);
        RowWeightTensor {
            log_m,
            m,
            factors,
            words,
            planes,
            words_per_plane: m / 64,
        }
    }

    #[inline(always)]
    pub fn plane(&self, e: usize) -> &[u64] {
        &self.planes[e * self.words_per_plane..(e + 1) * self.words_per_plane]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread8_matches_naive() {
        let mut words = [0u64; 8];
        let mut x = 0x243F6A8885A308D3u64;
        for w in words.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *w = x;
        }
        let mut naive = [0u8; 64];
        spread8_naive(&words, &mut naive);
        assert_eq!(spread8(&words), naive);
    }

    #[test]
    fn flip64_is_transpose() {
        // transpose twice = identity, and the naive check on a small pattern.
        let mut x = [0u64; 64];
        let mut expected = [[false; 64]; 64];
        let mut r = 0x9E3779B97F4A7C15u64;
        for i in 0..64 {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            x[i] = r;
            for b in 0..64 {
                expected[i][b] = (r >> b) & 1 == 1;
            }
        }
        flip64(&mut x);
        for i in 0..64 {
            for b in 0..64 {
                assert_eq!(
                    (x[i] >> b) & 1 == 1,
                    expected[b][i],
                    "flip must map (i,b) → (b,i)"
                );
            }
        }
        flip64(&mut x);
        let mut r2 = 0x9E3779B97F4A7C15u64;
        for i in 0..64 {
            r2 ^= r2 << 13;
            r2 ^= r2 >> 7;
            r2 ^= r2 << 17;
            assert_eq!(x[i], r2, "double flip = identity");
        }
    }

    #[test]
    fn tensor_words_and_planes_agree() {
        use magnetar_field::mul;
        // small log_m with random factors
        let mut r = 0xabcdefu128;
        let mut next = move || {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
            Gf128(r)
        };
        let factors: Vec<(Gf128, Gf128)> = (0..7).map(|_| (next(), next())).collect();
        let t = RowWeightTensor::new(factors.clone(), true);
        // words[i] = Π_r f_{r, bit_r(i)} — check against direct products.
        for i in 0..t.m {
            let mut v = Gf128(1);
            for (r, &(f0, f1)) in factors.iter().enumerate() {
                v = mul(v, if (i >> r) & 1 == 1 { f1 } else { f0 });
            }
            assert_eq!(t.words[i], v, "word {i}");
        }
        // planes agree with the words' coordinate bits
        for i in 0..t.m {
            for e in 0..128 {
                let pw = t.planes[e * t.words_per_plane + i / 64];
                let bit = (pw >> (i % 64)) & 1;
                assert_eq!(
                    bit as u16,
                    ((t.words[i].0 >> e) & 1) as u16,
                    "plane {e} row {i}"
                );
            }
        }
    }

    #[test]
    fn planes_bit_rw() {
        let mut ps = PlaneSet::zeros(128, 4);
        ps.set_bit(2, 65, true);
        ps.set_bit(2, 3, true);
        assert!(ps.bit(2, 65));
        assert!(ps.bit(2, 3));
        assert!(!ps.bit(2, 64));
        assert!(!ps.bit(1, 65));
    }
}
