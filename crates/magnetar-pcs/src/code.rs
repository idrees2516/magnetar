//! The affine-substitution codeword (the "subspace extension" role).
//!
//! `C(x, ε) = T̂(x + L(ε))` over `{0,1}^n × {0,1}^k`, with `L(ε) = Σ ε_j γ_j`
//! and `γ_j ∈ K^n` derived from a public seed. Encoding via the
//! δ-shifted Möbius transform in `O(n·2^{n+k})`.

use magnetar_field::{mul, Gf128, ZERO};
use magnetar_hash::blake3_domain;
use rayon::prelude::*;

/// Public parameters of the code: `n` table variables, `k` extension
/// variables, and the seed-derived substitution vectors `γ_1..γ_k`.
/// Minimum block length before codeword blocks encode in parallel (R2).
const PAR_ENCODE_THRESHOLD: usize = 1 << 13;

#[derive(Clone, Debug)]
pub struct CodeParams {
    pub n: usize,
    pub k: usize,
    /// `gammas[j]` is `γ_{j+1} ∈ K^n`.
    pub gammas: Vec<Vec<Gf128>>,
}

impl CodeParams {
    /// Derive parameters deterministically from a domain label (transparent
    /// setup, no trapdoor). The `γ`'s are hash-derived and treated as
    /// generic points (see the crate-level distance discussion).
    pub fn derive(n: usize, k: usize, domain: &str) -> Self {
        let mut gammas = Vec::with_capacity(k);
        for j in 0..k {
            let mut g = Vec::with_capacity(n);
            for i in 0..n {
                // 16 bytes per field element from BLAKE3 with position-bound
                // labels.
                let d = blake3_domain(&[
                    b"magnetar/pcs/gamma/v1",
                    domain.as_bytes(),
                    &(n as u64).to_le_bytes(),
                    &(k as u64).to_le_bytes(),
                    &(j as u64).to_le_bytes(),
                    &(i as u64).to_le_bytes(),
                ]);
                let mut lo = [0u8; 16];
                lo.copy_from_slice(&d[0..16]);
                g.push(Gf128(u128::from_le_bytes(lo)));
            }
            gammas.push(g);
        }
        CodeParams { n, k, gammas }
    }

    /// `L(ε) ∈ K^n` for a Boolean `ε` bit-vector.
    pub fn l_of(&self, eps_bits: &[bool]) -> Vec<Gf128> {
        debug_assert_eq!(eps_bits.len(), self.k);
        let mut out = vec![ZERO; self.n];
        for (j, &e) in eps_bits.iter().enumerate() {
            if e {
                for i in 0..self.n {
                    out[i] += self.gammas[j][i];
                }
            }
        }
        out
    }

    /// Codeword length `2^{n+k}`.
    pub fn codeword_len(&self) -> usize {
        1 << (self.n + self.k)
    }
}

/// A codeword: values over `{0,1}^n × {0,1}^k` laid out with the `x` part in
/// the low `n` bits (little-endian) and the `ε` part in the high `k` bits.
#[derive(Clone, Debug)]
pub struct Codeword {
    pub params: CodeParams,
    pub values: Vec<Gf128>,
}

impl Codeword {
    /// Encode a table `T: {0,1}^n → K`:
    /// for each `ε`, evaluate `T̂` at the shifted cube `{x + L(ε)}`.
    ///
    /// Uses the δ-shifted monomial-coefficient transform: writing
    /// `T̂ = Σ_S c_S Π_{i∈S} y_i`, the shifted evaluation at `y = x + δ` is
    /// `Σ_S d_S(δ) Π_{i∈S} x_i` with
    /// `d_S(δ) = Σ_{S' ⊇ S} c_{S'} Π_{i∈S'∖S} δ_i`, computed by a downward
    /// weighted Möbius transform in `O(n·2^n)` per `ε`.
    pub fn encode(params: &CodeParams, table: &[Gf128]) -> Self {
        let n = params.n;
        debug_assert_eq!(table.len(), 1 << n);
        // Step 1: values -> monomial coefficients (standard Möbius).
        let coeffs = values_to_monomials(table, n);
        let mut values = vec![ZERO; params.codeword_len()];
        // Each ε-block writes a disjoint output slice (docs/PERFORMANCE.md
        // R3/R8): encode the blocks with chunked parallelism — bit-identical
        // to the sequential encode.
        let block_len = 1usize << n;
        let n_blocks = 1usize << params.k;
        if block_len >= PAR_ENCODE_THRESHOLD {
            values
                .par_chunks_mut(block_len)
                .enumerate()
                .for_each(|(eps, out)| {
                    let eps_bits: Vec<bool> = (0..params.k).map(|j| (eps >> j) & 1 == 1).collect();
                    let delta = params.l_of(&eps_bits);
                    let shifted = shift_monomials(&coeffs, n, &delta);
                    write_cube_values(&shifted, n, out);
                });
        } else {
            for eps in 0..n_blocks {
                let eps_bits: Vec<bool> = (0..params.k).map(|j| (eps >> j) & 1 == 1).collect();
                let delta = params.l_of(&eps_bits);
                let shifted = shift_monomials(&coeffs, n, &delta);
                let block = eps << n;
                write_cube_values(&shifted, n, &mut values[block..block + block_len]);
            }
        }
        Codeword {
            params: params.clone(),
            values,
        }
    }

    /// Fold variable `var` (an `x`-bit, little-endian) at challenge `r`:
    /// `D'(x') = (1+r)·D[bit=0] + r·D[bit=1]`. This equals the substitution
    /// `y_var → y_var + r` in the affine argument, so the family is closed.
    pub fn fold(&self, var: usize, r: Gf128) -> Codeword {
        debug_assert!(var < self.params.n);
        let n = self.params.n;
        let block = 1usize << var;
        let m = self.values.len();
        let mut out = Vec::with_capacity(m / 2);
        let mut i = 0usize;
        while i < m {
            for j in 0..block {
                let a = self.values[i + j];
                let b = self.values[(i + j) | block];
                out.push(a + mul(r, a + b));
            }
            i += 2 * block;
        }
        let mut params = self.params.clone();
        params.n = n - 1;
        // The gammas keep their coordinates; the folded variable's γ
        // coordinate now plays the role of the ε-shift for that direction —
        // the affine map (x', ε) -> x' + L'(ε) with L' = L restricted.
        // (The removed coordinate's shift is absorbed into the folded
        // multilinear's constant structure; see `fold`'s doc identity.)
        Codeword {
            params,
            values: out,
        }
    }

    /// After folding all `n` x-variables: `D_n(ε) = T̂(L(ε) + r*)`.
    /// The value at `ε = 0` is `T̂(r*)` — the claim.
    #[inline]
    pub fn tail_entry(&self, eps: usize) -> Gf128 {
        debug_assert_eq!(self.params.n, 0);
        self.values[eps]
    }
}

/// Values → monomial coefficients of the multilinear (little-endian bits).
/// With `v[x] = Σ_{S ⊆ x} c_S`, the coefficients are `c[S] = Σ_{S' ⊆ S}
/// v[S']` in characteristic two (Möbius signs vanish), i.e. the **same**
/// upward zeta transform as the forward direction — the zeta matrix is an
/// involution mod 2 (`M² = I`).
fn values_to_monomials(values: &[Gf128], n: usize) -> Vec<Gf128> {
    let mut c = values.to_vec();
    for i in 0..n {
        let block = 1usize << i;
        let mut s = 0usize;
        while s < c.len() {
            for j in 0..block {
                let with = s + j + block;
                let without = s + j;
                c[with] = c[with] + c[without];
            }
            s += 2 * block;
        }
    }
    c
}

/// Shift monomial coefficients by δ: `d_S = Σ_{S' ⊇ S} c_{S'} Π_{i∈S'∖S} δ_i`.
fn shift_monomials(coeffs: &[Gf128], n: usize, delta: &[Gf128]) -> Vec<Gf128> {
    let mut d = coeffs.to_vec();
    // Downwards pass over variables: for each i, for each S containing i:
    // d[S∖{i}] += δ_i · d[S]. The pair loop is the strided-axpy shape —
    // the VPCLMULQDQ lane kernel (bit-identical to the scalar fold).
    for i in 0..n {
        let r = delta[i];
        let block = 1usize << i;
        let mut s = 0usize;
        while s < d.len() {
            let (head, _tail) = d.split_at_mut(s + 2 * block);
            let (pre, pair) = head.split_at_mut(s);
            debug_assert_eq!(pre.len(), s);
            let (lo, hi) = pair.split_at_mut(block);
            let done = magnetar_field::avx::axpy_run(lo, hi, r, block);
            for j in done..block {
                lo[j] += mul(r, hi[j]);
            }
            s += 2 * block;
        }
    }
    d
}

/// Evaluate a multilinear given in monomial-coefficient form on the whole
/// cube: `v[x] = Σ_{S ⊆ x} c_S` — the (upward) zeta transform, `O(n·2^n)`.
/// In characteristic two the transform is an involution (M² = I), which is
/// why the same pass shape inverts `values_to_monomials`.
fn write_cube_values(coeffs: &[Gf128], n: usize, out: &mut [Gf128]) {
    let mut v = coeffs.to_vec();
    for i in 0..n {
        let block = 1usize << i;
        let mut s = 0usize;
        while s < v.len() {
            for j in 0..block {
                let with = s + j + block;
                let without = s + j;
                v[with] = v[with] + v[without];
            }
            s += 2 * block;
        }
    }
    out.copy_from_slice(&v);
}

/// Inverse of `values_to_monomials` (for tests): monomials → values.
#[cfg(test)]
fn monomials_to_values(coeffs: &[Gf128], n: usize) -> Vec<Gf128> {
    let mut v = coeffs.to_vec();
    for i in 0..n {
        let block = 1usize << i;
        let mut s = 0usize;
        while s < v.len() {
            for j in 0..block {
                let with = s + j + block;
                let without = s + j;
                v[with] = v[with] + v[without];
            }
            s += 2 * block;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rand::RngExt::random::<u128>(rng))
    }

    #[test]
    fn monomial_transforms_roundtrip() {
        let mut rng = rand::rng();
        for n in [1usize, 2, 3, 5] {
            let values: Vec<Gf128> = (0..(1 << n)).map(|_| rg(&mut rng)).collect();
            let c = values_to_monomials(&values, n);
            let back = monomials_to_values(&c, n);
            assert_eq!(back, values, "n = {n}");
        }
    }

    #[test]
    fn shifted_evaluation_identity() {
        let mut rng = rand::rng();
        // C(x, eps) must equal eval_mle(T, x + L(eps)) for Boolean x, eps.
        let n = 3;
        let k = 2;
        let params = CodeParams::derive(n, k, "test-identity");
        let table: Vec<Gf128> = (0..(1 << n)).map(|_| rg(&mut rng)).collect();
        let cw = Codeword::encode(&params, &table);
        for eps in 0..(1usize << k) {
            let eps_bits: Vec<bool> = (0..k).map(|j| (eps >> j) & 1 == 1).collect();
            let delta = params.l_of(&eps_bits);
            for x in 0..(1usize << n) {
                let x_bits: Vec<bool> = (0..n).map(|i| (x >> i) & 1 == 1).collect();
                // point = x + delta, coordinate-wise in K.
                let point: Vec<Gf128> = (0..n)
                    .map(|i| Gf128(x_bits[i] as u128) + delta[i])
                    .collect();
                let want = magnetar_field::eval_mle(&table, &point);
                let idx = x | (eps << n);
                assert_eq!(cw.values[idx], want, "x={x} eps={eps}");
            }
        }
    }

    #[test]
    fn fold_identity_substitution() {
        let mut rng = rand::rng();
        let n = 3;
        let k = 2;
        let params = CodeParams::derive(n, k, "test-fold");
        let table: Vec<Gf128> = (0..(1 << n)).map(|_| rg(&mut rng)).collect();
        let cw = Codeword::encode(&params, &table);
        let r = rg(&mut rng);
        let folded = cw.fold(0, r);
        // D'(x', eps) = T_hat(x' + r*e_0 + L(eps))  [substitution y_0 -> y_0 + r]
        for eps in 0..(1usize << k) {
            let eps_bits: Vec<bool> = (0..k).map(|j| (eps >> j) & 1 == 1).collect();
            let delta = params.l_of(&eps_bits);
            for x in 0..(1usize << (n - 1)) {
                let x_bits: Vec<bool> = (0..(n - 1)).map(|i| (x >> i) & 1 == 1).collect();
                // Fold at var 0 substitutes y_0 -> y_0 + r, so coordinate 0
                // is L_0(eps) + r and coordinate j is x_{j-1} + L_j(eps).
                let mut point = vec![r + delta[0]];
                for i in 0..(n - 1) {
                    point.push(Gf128(x_bits[i] as u128) + delta[i + 1]);
                }
                let want = magnetar_field::eval_mle(&table, &point);
                let idx = x | (eps << (n - 1));
                assert_eq!(folded.values[idx], want, "x={x} eps={eps}");
            }
        }
    }

    #[test]
    fn full_fold_tail_equals_eval() {
        let mut rng = rand::rng();
        let n = 4;
        let k = 2;
        let params = CodeParams::derive(n, k, "test-tail");
        let table: Vec<Gf128> = (0..(1 << n)).map(|_| rg(&mut rng)).collect();
        let mut cw = Codeword::encode(&params, &table);
        // Fold all n variables at random challenges (variable 0 repeatedly:
        // after each fold the next original variable sits at bit 0).
        let mut challenges = Vec::new();
        let mut gammas_remaining = params.gammas.clone();
        let _ = &mut gammas_remaining;
        for _ in 0..n {
            let r = rg(&mut rng);
            challenges.push(r);
            let next = cw.fold(0, r);
            cw = next;
        }
        // D_n(eps) = T_hat(L(eps) + r*) where r* = (challenges in order).
        for eps in 0..(1usize << k) {
            let eps_bits: Vec<bool> = (0..k).map(|j| (eps >> j) & 1 == 1).collect();
            let delta = params.l_of(&eps_bits);
            let point: Vec<Gf128> = (0..n).map(|i| challenges[i] + delta[i]).collect();
            let want = magnetar_field::eval_mle(&table, &point);
            assert_eq!(cw.tail_entry(eps), want, "eps = {eps}");
        }
        // eps = 0: T_hat(r*).
        let rstar: Vec<Gf128> = challenges.clone();
        assert_eq!(cw.tail_entry(0), magnetar_field::eval_mle(&table, &rstar));
    }
}
