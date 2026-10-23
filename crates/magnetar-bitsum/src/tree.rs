//! The bitsliced tower multiplier — the *exact* algorithm the GPU kernel runs,
//! mirrored 1:1 on the host.
//!
//! Multiplication by a public constant (a Fiat-Shamir challenge, identical
//! for every element of a sumcheck round) is a GF(2)-linear map on coordinate
//! vectors, so it applies to whole bitsliced groups with plain word ops:
//!
//! ```text
//! tree(a, masks)[j] = XOR over i of  M[j][i] * a[i]        (M = mul-by-alpha)
//! ```
//!
//! computed by the Karatsuba tower recursion
//!
//! ```text
//! split a and masks into coordinate halves (lo, hi):
//!     r0  = tree(a_lo, m_lo)              r1  = tree(a_hi, m_hi)
//!     r2  = tree(a_lo ^ a_hi, m_lo ^ m_hi)
//!     nu  = tree(r1, NUS[k-1])            // constant masks: single coordinate
//!     out = (r0 ^ nu) ++ (r2 ^ r0)
//! leaf (width 1): out = a & mask
//! ```
//!
//! The GPU kernel (the `cutile` feature, `src/gpu/kernels.rs`) transliterates
//! this recursion into seven device functions `tower_w1..tower_w128` over
//! `Tile<u32, [TG, width]>` slices; the tests here assert bit-exact
//! equivalence with scalar field multiplication, so GPU parity reduces to
//! "the kernel compiles to the same op sequence".
//!
//! # Cost model (measured, not guessed)
//!
//! The pruned tree (skip subtrees whose constant masks are all zero — a pure
//! optimization, semantics identical) executes ~28k word-ops per 128-plane
//! group of 32 elements on the host reference (~875 per element). On the GPU
//! the same op sequence over `[TG, 128]` tiles is ~28k tile-ops per TG-group
//! batch. Phase-2 replaces this with a host-compiled CSE XOR-circuit
//! (~270 word-ops/element); see `docs/architecture/bitsum.md`.

#![forbid(unsafe_code)]

use crate::bitslice::{BitPlanes, PLANES, WORDS_PER_GROUP};
use crate::field::F128;

/// Coordinate-mask words for a constant: `masks[j] = all-ones` iff coordinate
/// `j` of the constant is 1.
#[must_use]
pub fn alpha_masks(alpha: F128) -> [u32; PLANES] {
    let mut masks = [0u32; PLANES];
    for (j, m) in masks.iter_mut().enumerate() {
        *m = if (alpha.0 >> j) & 1 == 1 {
            0xFFFF_FFFF
        } else {
            0
        };
    }
    masks
}

/// Pruned tree multiplication of one group's planes by a constant.
///
/// This is the reference the GPU kernel mirrors; pruning skips zero-mask
/// subtrees (identical results, fewer ops).
#[must_use]
pub fn tree_mul_const_group(a: &[u32; PLANES], masks: &[u32; PLANES]) -> [u32; PLANES] {
    let mut out = [0u32; PLANES];
    tree_rec(a, masks, PLANES, &mut out);
    out
}

/// Unpruned variant: the exact op sequence the GPU executes if its JIT does
/// not fold constant masks. Bit-identical to [`tree_mul_const_group`].
#[must_use]
pub fn tree_mul_const_group_unpruned(a: &[u32; PLANES], masks: &[u32; PLANES]) -> [u32; PLANES] {
    let mut out = [0u32; PLANES];
    tree_rec_unpruned(a, masks, PLANES, &mut out);
    out
}

fn nu_mask_slice(width: usize) -> Vec<u32> {
    // NUS[k] for width = 2^k is the single coordinate (width - 1).
    let mut m = vec![0u32; width];
    m[width - 1] = 0xFFFF_FFFF;
    m
}

fn tree_rec(a: &[u32], masks: &[u32], width: usize, out: &mut [u32]) {
    if width == 1 {
        out[0] = a[0] & masks[0];
        return;
    }
    let half = width / 2;
    if masks.iter().all(|&m| m == 0) {
        // constant is zero in this coordinate range: subtree prunes
        for v in out.iter_mut() {
            *v = 0;
        }
        return;
    }
    let a_lo = &a[..half];
    let a_hi = &a[half..];
    let m_lo = &masks[..half];
    let m_hi = &masks[half..];

    let mut r0 = vec![0u32; half];
    let mut r1 = vec![0u32; half];
    let mut r2 = vec![0u32; half];
    tree_rec(a_lo, m_lo, half, &mut r0);
    tree_rec(a_hi, m_hi, half, &mut r1);
    // r2 needs a_lo ^ a_hi
    let mut ax = vec![0u32; half];
    for i in 0..half {
        ax[i] = a_lo[i] ^ a_hi[i];
    }
    let mut mx = vec![0u32; half];
    for i in 0..half {
        mx[i] = m_lo[i] ^ m_hi[i];
    }
    tree_rec(&ax, &mx, half, &mut r2);

    // nu multiplication: constant masks = NUS[k-1] restricted to `half` coords
    let nu = nu_mask_slice(half);
    let mut nu_r1 = vec![0u32; half];
    tree_rec(&r1, &nu, half, &mut nu_r1);

    for i in 0..half {
        out[i] = r0[i] ^ nu_r1[i];
        out[half + i] = r2[i] ^ r0[i];
    }
}

fn tree_rec_unpruned(a: &[u32], masks: &[u32], width: usize, out: &mut [u32]) {
    if width == 1 {
        out[0] = a[0] & masks[0];
        return;
    }
    let half = width / 2;
    let a_lo = &a[..half];
    let a_hi = &a[half..];
    let m_lo = &masks[..half];
    let m_hi = &masks[half..];

    let mut r0 = vec![0u32; half];
    let mut r1 = vec![0u32; half];
    let mut r2 = vec![0u32; half];
    tree_rec_unpruned(a_lo, m_lo, half, &mut r0);
    tree_rec_unpruned(a_hi, m_hi, half, &mut r1);
    let mut ax = vec![0u32; half];
    let mut mx = vec![0u32; half];
    for i in 0..half {
        ax[i] = a_lo[i] ^ a_hi[i];
        mx[i] = m_lo[i] ^ m_hi[i];
    }
    tree_rec_unpruned(&ax, &mx, half, &mut r2);

    let nu = nu_mask_slice(half);
    let mut nu_r1 = vec![0u32; half];
    tree_rec_unpruned(&r1, &nu, half, &mut nu_r1);

    for i in 0..half {
        out[i] = r0[i] ^ nu_r1[i];
        out[half + i] = r2[i] ^ r0[i];
    }
}

/// Binds one sumcheck variable over a full plane tensor:
/// `out[g] = lo[g] + alpha * (lo[g] + hi[g])` for every group.
///
/// This is the whole per-round GPU workload (one `bind_round` launch plus a
/// reduction); provided here for host-side parity testing.
#[must_use]
pub fn bind_planes(planes: &BitPlanes, alpha: F128) -> BitPlanes {
    assert!(planes.groups % 2 == 0, "group count must be even");
    let half_groups = planes.groups / 2;
    let masks = alpha_masks(alpha);
    let mut out = BitPlanes::zeroed(half_groups);
    for g in 0..half_groups {
        let lo: &[u32] = &planes.words[g * WORDS_PER_GROUP..(g + 1) * WORDS_PER_GROUP];
        let hi: &[u32] = &planes.words
            [(g + half_groups) * WORDS_PER_GROUP..(g + half_groups + 1) * WORDS_PER_GROUP];
        let mut d = [0u32; PLANES];
        for j in 0..PLANES {
            d[j] = lo[j] ^ hi[j];
        }
        let e = tree_mul_const_group(&d, &masks);
        let out_row = &mut out.words[g * WORDS_PER_GROUP..(g + 1) * WORDS_PER_GROUP];
        for j in 0..PLANES {
            out_row[j] = lo[j] ^ e[j];
        }
    }
    out
}

/// Round reduction: XOR of all elements in the first `count` groups,
/// i.e. the `S0` message of a sumcheck round. Mirrors the GPU
/// `reduce_partial` hierarchy plus the host parity fold.
#[must_use]
pub fn reduce_planes(planes: &BitPlanes, count: usize) -> F128 {
    let mut acc_words = [0u32; PLANES];
    for g in 0..count {
        let row = &planes.words[g * WORDS_PER_GROUP..(g + 1) * WORDS_PER_GROUP];
        for (w, a) in row.iter().zip(acc_words.iter_mut()) {
            *a ^= w;
        }
    }
    let mut acc = 0u128;
    for (j, w) in acc_words.iter().enumerate() {
        if crate::bitslice::parity32(*w) == 1 {
            acc |= 1u128 << j;
        }
    }
    F128(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitslice::LANES;
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

    fn random_group(r: &mut SplitMix) -> [u32; PLANES] {
        let evals: Vec<F128> = (0..LANES).map(|_| F128::random(r)).collect();
        let planes = BitPlanes::pack(&evals).expect("pack");
        let mut group = [0u32; PLANES];
        group.copy_from_slice(&planes.words[..WORDS_PER_GROUP]);
        group
    }

    #[test]
    fn tree_matches_scalar_mul() {
        let mut r = SplitMix(0x5eed);
        for _ in 0..25 {
            let group = random_group(&mut r);
            let alpha = F128::random(&mut r);
            let masks = alpha_masks(alpha);

            let got = tree_mul_const_group(&group, &masks);
            let got_unpruned = tree_mul_const_group_unpruned(&group, &masks);
            assert_eq!(got, got_unpruned, "pruned == unpruned");

            // scalar expectation
            let elements: Vec<F128> = {
                let planes = BitPlanes::from_group_major({
                    let mut w = Vec::new();
                    w.extend_from_slice(&group);
                    w
                });
                planes.unpack()
            };
            for (e, elem) in elements.iter().enumerate() {
                let expected = *elem * alpha;
                let mut got_elem = 0u128;
                for (j, w) in got.iter().enumerate() {
                    if (w >> e) & 1 == 1 {
                        got_elem |= 1u128 << j;
                    }
                }
                assert_eq!(F128(got_elem), expected, "element {e}");
            }
        }
    }

    #[test]
    fn tree_handles_zero_and_special_constants() {
        let mut r = SplitMix(3);
        let group = random_group(&mut r);
        for alpha in [
            F128::ZERO,
            F128::ONE,
            F128(1u128 << 127),
            F128(u128::MAX),
            F128::random(&mut r),
        ] {
            let masks = alpha_masks(alpha);
            let got = tree_mul_const_group(&group, &masks);
            let got2 = tree_mul_const_group_unpruned(&group, &masks);
            assert_eq!(got, got2);
            // scalar expectation via unpack
            let mut w = Vec::new();
            w.extend_from_slice(&group);
            let elements = BitPlanes::from_group_major(w).unpack();
            for (e, elem) in elements.iter().enumerate() {
                let mut got_elem = 0u128;
                for (j, w) in got.iter().enumerate() {
                    if (w >> e) & 1 == 1 {
                        got_elem |= 1u128 << j;
                    }
                }
                assert_eq!(F128(got_elem), *elem * alpha, "alpha {alpha}");
            }
        }
    }

    #[test]
    fn bind_planes_matches_scalar_binding() {
        let mut r = SplitMix(0xb1d);
        let evals: Vec<F128> = (0..64).map(|_| F128::random(&mut r)).collect();
        let planes = BitPlanes::pack(&evals).expect("pack");
        let alpha = F128::random(&mut r);
        let bound = bind_planes(&planes, alpha);
        let bound_elems = bound.unpack();
        for i in 0..32 {
            let lo = evals[i];
            let hi = evals[i + 32];
            let expected = lo + alpha * (lo + hi);
            assert_eq!(bound_elems[i], expected, "element {i}");
        }
    }

    #[test]
    fn reduce_planes_matches_scalar() {
        let mut r = SplitMix(0xed0ce);
        let evals: Vec<F128> = (0..128).map(|_| F128::random(&mut r)).collect();
        let planes = BitPlanes::pack(&evals).expect("pack");
        let s0 = reduce_planes(&planes, 2); // first 2 groups == first 64 elements
        let mut scalar = F128::ZERO;
        for e in &evals[..64] {
            scalar += *e;
        }
        assert_eq!(s0, scalar);
    }

    #[test]
    fn op_count_pruned_tree() {
        // Sanity check on the documented cost model (~28k ops / group).
        use std::cell::Cell;
        thread_local! {
            static OPS: Cell<u64> = const { Cell::new(0) };
        }
        fn rec(a: &[u32], masks: &[u32], width: usize) -> Vec<u32> {
            if width == 1 {
                OPS.with(|o| o.set(o.get() + 1));
                return vec![a[0] & masks[0]];
            }
            let half = width / 2;
            if masks.iter().all(|&m| m == 0) {
                return vec![0u32; width];
            }
            OPS.with(|o| o.set(o.get() + 3));
            let r0 = rec(&a[..half], &masks[..half], half);
            let r1 = rec(&a[half..], &masks[half..], half);
            OPS.with(|o| o.set(o.get() + 2));
            let ax: Vec<u32> = (0..half).map(|i| a[i] ^ a[half + i]).collect();
            let mx: Vec<u32> = (0..half).map(|i| masks[i] ^ masks[half + i]).collect();
            let r2 = rec(&ax, &mx, half);
            let nu = nu_mask_slice(half);
            let nu_r1 = rec(&r1, &nu, half);
            OPS.with(|o| o.set(o.get() + 2));
            let mut out = vec![0u32; width];
            for i in 0..half {
                out[i] = r0[i] ^ nu_r1[i];
                out[half + i] = r2[i] ^ r0[i];
            }
            out
        }
        let mut r = SplitMix(0xc057);
        let group = random_group(&mut r);
        let alpha = F128::random(&mut r);
        OPS.with(|o| o.set(0));
        let _ = rec(&group, &alpha_masks(alpha), PLANES);
        let ops = OPS.with(|o| o.get());
        assert!(ops > 10_000 && ops < 60_000, "unexpected op count {ops}");
    }
}
