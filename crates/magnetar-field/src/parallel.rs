//! Parallel primitives (docs/PERFORMANCE.md rules R1–R4, R8).
//!
//! All `K = GF(2^128)` arithmetic (`+` is XOR, `×` is commutative and
//! associative) is **order-independent under reduction**, so every
//! parallel reduction here is bit-identical to the sequential result —
//! Fiat–Shamir transcripts stay deterministic regardless of thread
//! scheduling. What must NOT be parallelized is the *sequencing* of
//! transcript absorbs, which stays strictly sequential in the protocol
//! layer.
//!
//! The fold uses a fresh output buffer (R8: disjoint writes, no in-place
//! aliasing — the sequential `bind_var` is order-dependent in place, so an
//! in-place parallel map would race). Sizing rule (R2): tasks are
//! dispatched only above a measured work threshold; below it the
//! sequential kernel runs.

use crate::gf128::{mul, Gf128};
use rayon::prelude::*;

/// Minimum number of field elements before work is dispatched to rayon
/// (R2). 4096 entries ≈ 64 KiB working set — thousands of CLMULs, enough
/// to amortize task dispatch.
pub const PAR_THRESHOLD: usize = 4096;

/// Target entries per rayon task once a table is large enough (multiple
/// tasks per core keep the tail balanced).
pub const PAR_CHUNK: usize = 1 << 15;

/// Parallel sum-check fold (R1): bind variable `var` of `table` to the
/// challenge `r`, producing `T'(u) = T(0,u) + r·(T(0,u) + T(1,u))`.
/// Bit-identical to [`crate::bind_var`] for every input.
pub fn par_bind_var(table: &mut Vec<Gf128>, var: usize, r: Gf128) {
    debug_assert!(table.len().is_power_of_two());
    if table.len() == 1 {
        return;
    }
    let len = table.len();
    let out_len = len / 2;
    if out_len < PAR_THRESHOLD {
        crate::bind_var(table, var, r);
        return;
    }
    let block = 1usize << var;
    let mut out = vec![Gf128(0); out_len];
    let src = table.as_slice();
    let step = PAR_CHUNK.min(out_len);
    out.par_chunks_mut(step)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let start = ci * step;
            for (rel, o) in chunk.iter_mut().enumerate() {
                let k = start + rel;
                // Source pair: k with a hole at bit `var` (bit=0 / bit=1).
                let lo = insert_bit(k, var, false);
                let a = src[lo];
                let b = src[lo | block];
                *o = a + mul(r, a + b);
            }
        });
    *table = out;
}

/// Insert a bit at position `var` of `k` (the inverse of deleting it) —
/// the source index of folded entry `k`.
#[inline]
fn insert_bit(k: usize, var: usize, bit: bool) -> usize {
    let low = k & ((1usize << var) - 1);
    let high = k >> var;
    (high << (var + 1)) | (usize::from(bit) << var) | low
}

/// Parallel element-wise product, XOR-reduced (R1): `Σ_i mul(a[i], b[i])`.
/// Order-independent — bit-identical to the sequential dot product.
pub fn par_dot_xor(a: &[Gf128], b: &[Gf128]) -> Gf128 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = Gf128(0);
    if a.len() < PAR_THRESHOLD {
        for i in 0..a.len() {
            acc += mul(a[i], b[i]);
        }
        return acc;
    }
    a.par_chunks(PAR_CHUNK)
        .zip(b.par_chunks(PAR_CHUNK))
        .map(|(x, y)| {
            let mut s = Gf128(0);
            for i in 0..x.len() {
                s += mul(x[i], y[i]);
            }
            s
        })
        .sum::<Gf128>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bind_var;
    use rand::RngExt;

    #[test]
    fn par_bind_matches_seq() {
        let mut rng = rand::rng();
        // 2^13 outputs cross the parallel threshold.
        for vars in [13usize, 15] {
            for var in 0..vars {
                let base: Vec<Gf128> = (0..(1usize << vars))
                    .map(|_| Gf128(rng.random::<u128>()))
                    .collect();
                let r = Gf128(rng.random::<u128>());
                let mut a = base.clone();
                let mut b = base;
                bind_var(&mut a, var, r);
                par_bind_var(&mut b, var, r);
                assert_eq!(a, b, "vars={vars} var={var}");
            }
        }
    }

    #[test]
    fn dot_matches_seq() {
        let mut rng = rand::rng();
        let n = 1usize << 17;
        let a: Vec<Gf128> = (0..n).map(|_| Gf128(rng.random::<u128>())).collect();
        let b: Vec<Gf128> = (0..n).map(|_| Gf128(rng.random::<u128>())).collect();
        let mut seq = Gf128(0);
        for i in 0..n {
            seq += mul(a[i], b[i]);
        }
        assert_eq!(seq, par_dot_xor(&a, &b));
    }
}
