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
///
/// The multiply is the dominant cost: when AVX-512 VPCLMULQDQ is available
/// the `r·(lo+hi)` products go through the 4-lane batched pipeline
/// ([`crate::avx::mul4_broadcast`]).
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
    // Contiguous-run fast paths: binding the LSB (var = 0: pairs are
    // (k, k+block), two straight runs) or the top bit (var = log2(len)-1:
    // the low run is the whole first half) feeds the fused VPCLMULQDQ
    // fold kernel directly. Middle variables fall back to the strided
    // per-element form (they only occur for a handful of rounds).
    let top_var = len.trailing_zeros() as usize - 1;
    if var == 0 || var == top_var {
        let step = PAR_CHUNK.min(out_len);
        out.par_chunks_mut(step)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let start = ci * step;
                let done = if var == 0 {
                    // Interleaved pairs: out[k] folds (src[2k], src[2k+1]);
                    // the chunk's outputs start at global `start`, so the
                    // pairs start at 2·start.
                    let s = &src[2 * start..2 * (start + chunk.len())];
                    let done = crate::avx::fold_pairs(chunk, s, r, chunk.len());
                    for k in done..chunk.len() {
                        let a = s[2 * k];
                        let b = s[2 * k + 1];
                        chunk[k] = a + mul(r, a + b);
                    }
                    done
                } else {
                    // Top bit: out[k] folds (src[k], src[k + block]).
                    let s = &src[start..start + chunk.len() + block];
                    let done = crate::avx::fold_run(chunk, s, r, block, chunk.len());
                    for k in done..chunk.len() {
                        let a = s[k];
                        let b = s[k + block];
                        chunk[k] = a + mul(r, a + b);
                    }
                    done
                };
                let _ = done;
            });
        *table = out;
        return;
    }
    let step = PAR_CHUNK.min(out_len);
    out.par_chunks_mut(step)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let start = ci * step;
            let mut k = start;
            let end = start + chunk.len();
            // 2-lane batched body (strided sources).
            while k + 2 <= end {
                let lo0 = insert_bit(k, var, false);
                let lo1 = insert_bit(k + 1, var, false);
                let t = [
                    Gf128(src[lo0].0 ^ src[lo0 | block].0),
                    Gf128(src[lo1].0 ^ src[lo1 | block].0),
                ];
                let got = crate::avx::mul2_inline(r, t);
                chunk[k - start] = Gf128(src[lo0].0 ^ got[0].0);
                chunk[k + 1 - start] = Gf128(src[lo1].0 ^ got[1].0);
                k += 2;
            }
            while k < end {
                let lo = insert_bit(k, var, false);
                let a = src[lo];
                let b = src[lo | block];
                chunk[k - start] = a + mul(r, a + b);
                k += 1;
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

/// Parallel fold of a **bit-valued** table (the small-value setting):
/// `T'(u) = T(0,u) + r·(T(0,u) + T(1,u))` where `(T(0,u) + T(1,u)) ∈ {0, 1}`
/// — the multiply degenerates to a branchless select, no CLMUL.
/// Bit-identical to [`par_bind_var`] for bit-valued inputs.
pub fn par_bind_var_bit(table: &mut Vec<Gf128>, var: usize, r: Gf128) {
    debug_assert!(table.len().is_power_of_two());
    if table.len() == 1 {
        return;
    }
    let len = table.len();
    let out_len = len / 2;
    if out_len < PAR_THRESHOLD {
        let block = 1usize << var;
        let mut out = Vec::with_capacity(out_len);
        let src = table.as_slice();
        let mut i = 0usize;
        while i < len {
            for j in 0..block {
                let idx0 = i + j;
                let t0 = src[idx0];
                let t1 = src[idx0 | block];
                let mask = ((t0.0 ^ t1.0) & 1).wrapping_neg();
                out.push(Gf128(t0.0 ^ (r.0 & mask)));
            }
            i += 2 * block;
        }
        *table = out;
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
                let lo = insert_bit(k, var, false);
                let t0 = src[lo];
                let t1 = src[lo | block];
                let mask = ((t0.0 ^ t1.0) & 1).wrapping_neg();
                *o = Gf128(t0.0 ^ (r.0 & mask));
            }
        });
    *table = out;
}

/// Parallel **gather** (delete-bit copy, slice selectable):
/// `out[k] = table[insert_bit(k, var, slice)]` — the slab update for
/// equality public factors (the never-fold discipline: the bound
/// variable's eq factor is a *constant* scale, so the table only needs
/// one bit-slice; the accumulated scaling is absorbed by the term
/// coefficients). No field multiplications at all.
pub fn par_gather_bit_at(table: &mut Vec<Gf128>, var: usize, one: bool) {
    debug_assert!(table.len().is_power_of_two());
    if table.len() == 1 {
        return;
    }
    let len = table.len();
    let out_len = len / 2;
    let block = 1usize << var;
    let off = usize::from(one) * block;
    if out_len < PAR_THRESHOLD {
        let mut out = Vec::with_capacity(out_len);
        let src = table.as_slice();
        let mut i = 0usize;
        while i < len {
            out.extend_from_slice(&src[i + off..i + off + block]);
            i += 2 * block;
        }
        *table = out;
        return;
    }
    let mut out = vec![Gf128(0); out_len];
    let src = table.as_slice();
    let step = PAR_CHUNK.min(out_len);
    out.par_chunks_mut(step)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let start = ci * step;
            for (rel, o) in chunk.iter_mut().enumerate() {
                let k = start + rel;
                *o = src[insert_bit(k, var, one)];
            }
        });
    *table = out;
}

/// The bit-zero gather (see [`par_gather_bit_at`]).
pub fn par_gather_bit(table: &mut Vec<Gf128>, var: usize) {
    par_gather_bit_at(table, var, false);
}

/// Parallel dot of eq weights with a **bit-valued** column (the
/// instance-fold of the bit witness `z`): `Σ_u eq[u] · z_bit[u]` — every
/// product is a select, no CLMUL.
pub fn par_dot_bit_xor(w: &[Gf128], bits: &[Gf128]) -> Gf128 {
    debug_assert_eq!(w.len(), bits.len());
    let mut acc = Gf128(0);
    if w.len() < PAR_THRESHOLD {
        for i in 0..w.len() {
            acc = Gf128(acc.0 ^ (w[i].0 & bits[i].0.wrapping_neg()));
        }
        return acc;
    }
    w.par_chunks(PAR_CHUNK)
        .zip(bits.par_chunks(PAR_CHUNK))
        .map(|(x, y)| {
            let mut s = Gf128(0);
            for i in 0..x.len() {
                s = Gf128(s.0 ^ (x[i].0 & y[i].0.wrapping_neg()));
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
                if a != b {
                    let idx = a.iter().zip(b.iter()).position(|(x, y)| x != y).unwrap();
                    panic!(
                        "MISMATCH vars={vars} var={var} idx={idx} seq={:x} par={:x}",
                        a[idx].0, b[idx].0
                    );
                }
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

    #[test]
    fn bit_fold_matches_generic() {
        let mut rng = rand::rng();
        for vars in [8usize, 14] {
            for var in 0..vars {
                let base: Vec<Gf128> = (0..(1usize << vars))
                    .map(|_| Gf128((rng.random::<u128>() >> 127) as u128)) // {0,1}
                    .collect();
                let r = Gf128(rng.random::<u128>());
                let mut a = base.clone();
                let mut b = base;
                par_bind_var(&mut a, var, r);
                par_bind_var_bit(&mut b, var, r);
                assert_eq!(a, b, "vars={vars} var={var}");
            }
        }
    }

    #[test]
    fn gather_is_bit_zero_slice() {
        let mut rng = rand::rng();
        let vars = 7;
        let mut table: Vec<Gf128> = (0..(1usize << vars))
            .map(|i| Gf128(i as u128 * 3 + 1))
            .collect();
        for var in 0..vars {
            let mut g = table.clone();
            par_gather_bit(&mut g, var);
            let block = 1usize << var;
            let mut k = 0usize;
            let mut i = 0usize;
            while i < table.len() {
                for j in 0..block {
                    assert_eq!(g[k], table[i + j]);
                    k += 1;
                }
                i += 2 * block;
            }
            let _ = &mut rng;
        }
    }

    #[test]
    fn bit_dot_matches_generic() {
        let mut rng = rand::rng();
        let n = 1usize << 13;
        let w: Vec<Gf128> = (0..n).map(|_| Gf128(rng.random::<u128>())).collect();
        let bits: Vec<Gf128> = (0..n)
            .map(|_| Gf128((rng.random::<u128>() >> 127) as u128))
            .collect();
        let mut seq = Gf128(0);
        for i in 0..n {
            seq += mul(w[i], bits[i]);
        }
        assert_eq!(seq, par_dot_bit_xor(&w, &bits));
    }
}
