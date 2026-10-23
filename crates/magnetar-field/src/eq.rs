//! Equality-function tables over characteristic two (paper §3.1).
//!
//! For `b ∈ {0,1}` and `x ∈ K`: in characteristic two,
//! `eq(x, b) = (1+x)(1+b) + x·b = 1 + x + b` — the cross terms cancel.
//! Multilinearly: `eq(r, b) = Π_i (1 + r_i + b_i)`, and
//! `Σ_{b∈{0,1}^ℓ} eq(r, b) = 1` for every point `r` (each factor sums to one
//! over the Boolean pair). These identities are used by every sum-check in
//! the stack.

use crate::gf128::{mul, Gf128, ONE};

/// The characteristic-two univariate equality factor: `eq(x, b) = 1 + x + b`.
#[inline(always)]
pub fn eq_char2(x: Gf128, b: bool) -> Gf128 {
    if b {
        x
    } else {
        ONE + x
    }
}

/// A precomputed `eq` table over the Boolean cube: entry `i` (little-endian
/// bit convention, variable 0 = LSB) is `eq(r, i)`.
#[derive(Clone, Debug)]
pub struct EqBits {
    /// 2^ℓ values: `eq(r, b)` for every Boolean `b`.
    pub values: Vec<Gf128>,
    /// The point `r` the table is built from (kept for auditing / tests).
    pub point: Vec<Gf128>,
}

impl EqBits {
    /// Build `eq(r, ·)` over the whole cube: the tensor product of the
    /// per-variable factors `[1 + r_i, r_i]`. Cost `O(2^ℓ)` multiplications.
    ///
    /// Index convention (little-endian): entry `i` has bit `k` = variable `k`,
    /// i.e. `values[i] = eq(r, bits_of(i))` with `point[0]` = LSB. The tensor
    /// product is therefore processed in reverse variable order so the last
    /// factor applied lands in the least-significant bit.
    pub fn build(point: &[Gf128]) -> Self {
        let mut values = vec![ONE];
        for &r in point.iter().rev() {
            let f0 = ONE + r; // eq(r, 0)
            let f1 = r; //     eq(r, 1)
            let mut next = Vec::with_capacity(values.len() * 2);
            for &v in &values {
                next.push(mul(v, f0));
                next.push(mul(v, f1));
            }
            values = next;
        }
        EqBits {
            values,
            point: point.to_vec(),
        }
    }

    /// Number of variables.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.point.len()
    }
}

/// Evaluate `eq(r, b)` at a single Boolean point `b` (little-endian bits).
pub fn eq_table(point: &[Gf128], b_bits: &[bool]) -> Gf128 {
    debug_assert_eq!(point.len(), b_bits.len());
    let mut acc = ONE;
    for (&r, &b) in point.iter().zip(b_bits.iter()) {
        acc = mul(acc, eq_char2(r, b));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf128::ONE;
    use rand::RngExt;

    #[test]
    fn eq_identity_holds() {
        let mut rng = rand::rng();
        for _ in 0..8 {
            let point: Vec<Gf128> = (0..10).map(|_| rng.random()).collect();
            let t = EqBits::build(&point);
            // Sum over the cube is 1.
            let sum = t.values.iter().fold(Gf128(0), |a, b| a + *b);
            assert_eq!(sum, ONE);
            // Each entry matches direct evaluation.
            for (i, &v) in t.values.iter().enumerate() {
                let bits: Vec<bool> = (0..point.len()).map(|k| (i >> k) & 1 == 1).collect();
                assert_eq!(v, eq_table(&point, &bits), "entry {i}");
            }
        }
    }

    #[test]
    fn char2_identity() {
        let mut rng = rand::rng();
        for _ in 0..64 {
            let x: Gf128 = rng.random();
            assert_eq!(eq_char2(x, true), x);
            assert_eq!(eq_char2(x, false), ONE + x);
            // eq(x,0) + eq(x,1) = 1.
            assert_eq!(eq_char2(x, false) + eq_char2(x, true), ONE);
        }
    }
}
