//! Dense multilinear tables over `K` (paper §3.1).
//!
//! Index convention: little-endian; **variable 0 is the least-significant
//! index bit**. The multilinear extension is
//! `v̂(x) = Σ_b v[b]·eq(x, b)`.
//!
//! Two workhorse operations:
//! * `eval_mle` — evaluate the extension at an arbitrary point (successive
//!   folding, `O(2^ℓ)`).
//! * `bind_var` / `fold_pairs` — the partial evaluation used by sum-check:
//!   binding variable `i` to challenge `r` replaces the table `T` by
//!   `T'(u') = (1 + r)·T(0, u') + r·T(1, u')` (identical to `1−r` in
//!   characteristic two).

use crate::gf128::{mul, Gf128};

/// A dense multilinear: `2^ℓ` values, variables in little-endian order.
#[derive(Clone, Debug, PartialEq)]
pub struct Multilinear {
    pub values: Vec<Gf128>,
}

impl Multilinear {
    pub fn new(values: Vec<Gf128>) -> Self {
        debug_assert!(values.len().is_power_of_two());
        Multilinear { values }
    }

    #[inline]
    pub fn num_vars(&self) -> usize {
        self.values.len().trailing_zeros() as usize
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Random access by Boolean index.
    #[inline]
    pub fn get(&self, index: usize) -> Gf128 {
        self.values[index]
    }
}

/// Fold variable `var` of the table layout (variable = index bit position).
/// After the call the table has half the entries and variable `var` is gone
/// (higher variables shift down by one).
///
/// This is the standard sum-check fold: the resulting table equals the
/// partial evaluation of the MLE at `x_var = r`, and each entry is
/// `T(0,·) + r·(T(0,·) + T(1,·)) = Σ_prefix eq(r, prefix)·T(prefix,·)` —
/// the identity the whole stack relies on. Constant time.
pub fn bind_var(table: &mut Vec<Gf128>, var: usize, r: Gf128) {
    debug_assert!(table.len().is_power_of_two());
    debug_assert!(var < 64);
    debug_assert!(table.len() > (1usize << var) >> 1);
    let block = 1usize << var;
    let mut out = 0usize;
    let mut i = 0usize;
    while i < table.len() {
        for j in 0..block {
            let idx0 = i + j;
            let idx1 = idx0 | block;
            let t0 = table[idx0];
            let t1 = table[idx1];
            table[out] = t0 + mul(r, t0 + t1);
            out += 1;
        }
        i += 2 * block;
    }
    table.truncate(out);
}

/// Alias of [`bind_var`] used by sum-check code (paper's "folding").
pub fn fold_pairs(table: &mut Vec<Gf128>, var: usize, r: Gf128) {
    bind_var(table, var, r)
}

/// Evaluate the multilinear extension at `point` (little-endian variable
/// order: `point[0]` is variable 0 = LSB).
///
/// Uses the fold: binding every variable to its challenge in reverse order
/// leaves a single value; equivalently `v̂(r) = Σ_b eq(r, b) v[b]`.
pub fn eval_mle(values: &[Gf128], point: &[Gf128]) -> Gf128 {
    debug_assert_eq!(values.len(), 1 << point.len());
    let mut t = values.to_vec();
    // Bind from the most significant variable down so the fold keeps the
    // little-endian layout consistent.
    for (k, &r) in point.iter().enumerate().rev() {
        fold_pairs(&mut t, k, r);
    }
    debug_assert_eq!(t.len(), 1);
    t[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq::EqBits;
    use rand::RngExt;

    #[test]
    fn eval_matches_eq_sum() {
        let mut rng = rand::rng();
        for nv in [1usize, 2, 3, 5, 8] {
            let values: Vec<Gf128> = (0..(1 << nv)).map(|_| rng.random()).collect();
            let point: Vec<Gf128> = (0..nv).map(|_| rng.random()).collect();
            let got = eval_mle(&values, &point);
            let eq = EqBits::build(&point);
            let mut want = Gf128(0);
            for i in 0..(1 << nv) {
                want += mul(values[i], eq.values[i]);
            }
            assert_eq!(got, want, "nv = {nv}");
        }
    }

    #[test]
    fn fold_is_partial_evaluation() {
        let mut rng = rand::rng();
        for nv in [2usize, 4, 6] {
            let values: Vec<Gf128> = (0..(1 << nv)).map(|_| rng.random()).collect();
            let r: Gf128 = rng.random();
            // Fold variable 0.
            let mut t = values.clone();
            fold_pairs(&mut t, 0, r);
            // The folded table must equal the MLE with x0 = r at every
            // remaining Boolean point.
            let point_partial = vec![r];
            for idx in 0..(1 << (nv - 1)) {
                // remaining point: (idx with variables 1..) — evaluate the
                // original MLE at (r, bits of idx shifted).
                let mut point = vec![r];
                for k in 1..nv {
                    point.push(Gf128(((idx >> (k - 1)) & 1) as u128));
                }
                let want = eval_mle(&values, &point);
                assert_eq!(t[idx], want, "nv={nv} idx={idx}");
                let _ = point_partial;
            }
        }
    }

    #[test]
    fn fold_matches_eq_fold_identity() {
        // fold^n with challenges r equals eq-weighted sum: T_n = Σ eq(r,b)T(b).
        let mut rng = rand::rng();
        let nv = 5;
        let values: Vec<Gf128> = (0..(1 << nv)).map(|_| rng.random()).collect();
        let point: Vec<Gf128> = (0..nv).map(|_| rng.random()).collect();
        let mut t = values.clone();
        for (k, &r) in point.iter().enumerate().rev() {
            fold_pairs(&mut t, k, r);
        }
        assert_eq!(t.len(), 1);
        assert_eq!(t[0], eval_mle(&values, &point));
    }
}
