//! # magnetar-sumcheck
//!
//! The sum-check protocol over `K = F_{2^128}` (paper §3.2), as the workhorse
//! of every reduction in the stack: Spartan's outer/inner arguments (§4.1),
//! BinaryTwist / BinaryShout (§4.4), BinaryRouter (§5.4), and MosaicPCS's
//! batching (§5.2).
//!
//! ## Engine
//!
//! The summand is a set of **terms**: `g(x) = Σ_t coeff_t · Π_j f_{t,j}(x)`
//! where each factor is either
//! * *dense* — a prover-held multilinear table, folded round by round; or
//! * *public* — a verifier-evaluable multilinear ([`PublicFactor`]).
//!
//! In characteristic two the round check collapses to
//! `Σ_{j≥1} c_j = running` (the constant coefficient cancels), and a
//! multilinear `eq` factor is the linear `1 + r + b`.
//!
//! ## Binding orders
//!
//! The engine binds variables one per round. Two orders cover the stack:
//! [`BindOrder::Increasing`] (variables 0, 1, 2, …) and
//! [`BindOrder::HighBitsFirst`] (the top `k` variables in decreasing order,
//! then the rest increasing — used by MosaicPCS to bind the 7 packing
//! variables first, which turns the bit witness into the c*-folded table
//! that ring switching discharges).
//!
//! The round-by-round API (`message` / `fold`) lets Virtual Spartan run two
//! outer arguments in lockstep, sharing instance challenges (§5.3).

#![forbid(unsafe_code)]
// Proof-system style: loops indexed over Boolean-cube coordinates and
// multi-input reductions are the domain's normal shape.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod engine;
pub mod public;
pub mod sparse;

pub use engine::{SumcheckProver, SumcheckVerifier};
pub use public::{PublicFactor, SumEqFactor, TableFactor};
pub use sparse::{SparseFactor, SparseProver, SparseSpec};

use magnetar_field::{mul, Gf128, ONE, ZERO};

/// One additive term of the summand: `coeff · Π(dense) · Π(public)`.
#[derive(Clone, Debug)]
pub struct Term {
    pub coeff: Gf128,
    /// Indices into the dense-factor list.
    pub dense: Vec<usize>,
    /// Indices into the public-factor list.
    pub public: Vec<usize>,
}

/// Which original variable each round binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindOrder {
    /// Round `i` binds original variable `i`.
    Increasing,
    /// Rounds `0..k` bind original variables `m-1, m-2, …, m-k` (the top `k`
    /// bits first, in decreasing order); later rounds bind the remaining
    /// variables in increasing order.
    HighBitsFirst { high: usize },
}

impl BindOrder {
    /// The original variable bound in round `i` of an `m`-variable sumcheck.
    pub fn round_var(&self, m: usize, i: usize) -> usize {
        match *self {
            BindOrder::Increasing => i,
            BindOrder::HighBitsFirst { high } => {
                if i < high {
                    m - 1 - i
                } else {
                    i - high
                }
            }
        }
    }

    /// The current bit position to fold at round `i` (after previous rounds'
    /// removals). For `HighBitsFirst` the first `high` rounds fold the
    /// current top bit; afterwards each round folds bit 0.
    pub fn current_bit(&self, m: usize, i: usize) -> usize {
        match *self {
            BindOrder::Increasing => 0,
            BindOrder::HighBitsFirst { high } => {
                if i < high {
                    m - 1 - i
                } else {
                    0
                }
            }
        }
    }

    /// Map per-round challenges to the point in *original* coordinates.
    pub fn point(&self, m: usize, challenges: &[Gf128]) -> Vec<Gf128> {
        debug_assert_eq!(challenges.len(), m);
        let mut p = vec![ZERO; m];
        for (i, &c) in challenges.iter().enumerate() {
            p[self.round_var(m, i)] = c;
        }
        p
    }
}

/// A fully-specified sumcheck: `Σ_{x∈{0,1}^m} g(x) = claim`.
#[derive(Debug)]
pub struct SumcheckSpec {
    /// Number of variables `m`.
    pub num_vars: usize,
    /// Dense factor tables (each of length `2^m`).
    pub dense: Vec<Vec<Gf128>>,
    /// Public factors.
    pub public: Vec<Box<dyn PublicFactor + Send + Sync>>,
    /// Additive terms.
    pub terms: Vec<Term>,
    /// The claimed sum.
    pub claim: Gf128,
}

impl SumcheckSpec {
    /// Maximum individual degree of the summand (for message length).
    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|t| t.dense.len() + t.public.len())
            .max()
            .unwrap_or(0)
    }

    /// Sanity: lengths and indices are coherent.
    pub fn validate(&self) -> Result<(), &'static str> {
        let n = 1usize << self.num_vars;
        if self.dense.iter().any(|d| d.len() != n) {
            return Err("dense factor length mismatch");
        }
        for f in &self.public {
            if f.num_vars() != self.num_vars {
                return Err("public factor variable count mismatch");
            }
        }
        for t in &self.terms {
            for &d in &t.dense {
                if d >= self.dense.len() {
                    return Err("dense index out of range");
                }
            }
            for &p in &t.public {
                if p >= self.public.len() {
                    return Err("public index out of range");
                }
            }
        }
        Ok(())
    }
}

/// Per-round message: the univariate coefficients `[c_0, …, c_d]` of the
/// round polynomial in the round variable.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundMsg(pub Vec<Gf128>);

/// Lagrange interpolation on the fixed node set `{0, 1, 2, …, d}` (distinct
/// constants of `K`). Given values at these points, returns the polynomial
/// coefficients, degree `d`.
fn interpolate(values: &[Gf128]) -> Vec<Gf128> {
    let d = values.len() - 1;
    // The engine counts the total factor count per term (conservative upper
    // bound on the per-variable degree). The largest term in the stack is
    // the store routing: eq_o·eq_j·X·Q_st·(a-links) = 8 factors.
    debug_assert!(d <= 8, "sumcheck degree beyond 8 not used in Magnetar");
    // Solve via Lagrange basis polynomials evaluated numerically.
    // nodes x_i = i (0..=d)
    let nodes: Vec<Gf128> = (0..=d).map(|i| Gf128(i as u128)).collect();
    // Build Lagrange polynomials L_i(X) and accumulate value_i * L_i.
    // Represent polynomials as coefficient vectors.
    let mut acc = vec![ZERO; d + 1];
    for (i, &val) in values.iter().enumerate() {
        // L_i(X) = Π_{j≠i} (X - x_j) / (x_i - x_j); char 2: minus = plus.
        let mut poly = vec![ONE]; // constant 1
        let mut denom = ONE;
        for (j, &xj) in nodes.iter().enumerate() {
            if j == i {
                continue;
            }
            // poly *= (X + xj)  →  shift coefficients up.
            let mut next = vec![ZERO; poly.len() + 1];
            for (k, &c) in poly.iter().enumerate() {
                next[k + 1] += c; // X·c
                next[k] += mul(c, xj); // xj·c
            }
            poly = next;
            denom = mul(denom, nodes[i] + xj);
        }
        let scale = mul(val, magnetar_field::inv(denom));
        for (k, &c) in poly.iter().enumerate() {
            acc[k] += mul(scale, c);
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_is_exact() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for d in 1..=3usize {
            let coeffs: Vec<Gf128> = (0..=d).map(|_| Gf128(rng.random::<u128>())).collect();
            let values: Vec<Gf128> = (0..=d)
                .map(|i| {
                    // evaluate at X = i
                    let mut acc = ZERO;
                    let mut xp = ONE;
                    for &c in &coeffs {
                        acc += mul(c, xp);
                        xp = mul(xp, Gf128(i as u128));
                    }
                    acc
                })
                .collect();
            assert_eq!(interpolate(&values), coeffs, "degree {d}");
        }
    }
}
#[cfg(test)]
mod extra_tests {
    #[test]
    fn interpolation_high_degree() {
        use super::*;
        use rand::RngExt;
        let mut rng = rand::rng();
        for d in 4..=6usize {
            let coeffs: Vec<Gf128> = (0..=d).map(|_| Gf128(rng.random::<u128>())).collect();
            let values: Vec<Gf128> = (0..=d)
                .map(|i| {
                    let mut acc = ZERO;
                    let mut xp = ONE;
                    for &c in &coeffs {
                        acc += mul(c, xp);
                        xp = mul(xp, Gf128(i as u128));
                    }
                    acc
                })
                .collect();
            assert_eq!(interpolate(&values), coeffs, "degree {d}");
        }
    }
}
