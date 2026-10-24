//! # magnetar-pcs
//!
//! The hash-based multilinear polynomial commitment scheme for Magnetar
//! (paper §3.6 Definition 6, instantiated in the role of Ligerito [44] as the
//! paper does in §6), plus **ring switching** (paper §5.2, [13]) which
//! reduces evaluation claims on bit-valued vectors to claims on their packed
//! `K`-valued representation.
//!
//! ## Construction (this instantiation)
//!
//! *Commitment* — for a table `T: {0,1}^n → K`, the codeword is the
//! **affine-substitution extension** over `{0,1}^n × {0,1}^k`:
//! `C(x, ε) = T̂(x + L(ε))` with `L(ε) = Σ_j ε_j γ_j` for seed-generic
//! `γ_j ∈ K^n`. The commitment is the BLAKE3 Merkle root of the codeword;
//! the `(x, 0)` slice is exactly `T`.
//!
//! The code has dimension `2^n`, length `2^{n+k}` (rate `2^{-k}`), and —
//! because evaluation points `x + L(ε)` are generic `K`-points — relative
//! distance `≈ 1`: a nonzero codeword difference is a nonzero multilinear
//! evaluated at generic points, nonzero at all but a `O(n/|K|)` fraction
//! (Schwartz–Zippel over the seed-derived `γ`).
//!
//! *Evaluation argument* for `T̂(r) = y`:
//! 1. Sum-check over the `n` table variables: `Σ_x eq(r, x)·T(x) = y`
//!    (public factor `eq(r,·)`, dense factor `T`), binding random `r*`.
//!    The terminal claim is `T̂(r*) = v''` (eq factor folded in).
//! 2. Fold chain: fold the codeword along the `n` cube variables with the
//!    *same* challenges `r*`. Folding bit `i` at challenge `r` computes
//!    `D' = (1+r)·D[bit i = 0] + r·D[bit i = 1]`, which equals the
//!    substitution `y_i → y_i + r`; the family is closed under folds, and
//!    after all `n` folds the tail `D_n(ε) = T̂(L(ε) + r*)`, so
//!    **`D_n(0) = T̂(r*)`** is the claim.
//! 3. The prover commits (in the proof) to every intermediate folded
//!    codeword's Merkle root; the verifier walks `q` correlated query paths
//!    from level 0 up (FRI-style), checking each fold relation, fully
//!    reveals the `2^k`-entry tail, and checks `D_n(0) = v''`.
//!
//! Soundness of the fold chain follows the FRI/Ligerito analysis: the
//! codeword family is closed under folding, the level-0 tree is the
//! commitment, correlated path queries catch fold inconsistencies, and the
//! code distance (≈ 1 for generic `γ`) rules out concentrated deviations
//! *for codewords*, with the unique-decoding-radius proximity-gap argument
//! of [Ben-Sasson et al., TR25-169] covering the general case — the same
//! standard the Magnetar paper itself relies on (§6.1).
//!
//! *Ring switching* (see [`ring`]): a claim `ŵ(r_A, r_B) = y` on a bit
//! vector `w: {0,1}^n × {0,1}^7` packed as `P(u) = Σ_b β_b w(u, b)` reduces,
//! via the c-folded table `C(u) = φ_c(P(u))` with `c = eq(r_B, ·)`, to a
//! fold chain on `C`'s codeword whose level-0 `(x, 0)` slice is anchored to
//! the *main* commitment by exact `φ_c` bit-extraction checks.

#![forbid(unsafe_code)]
// Proof-system style: loops indexed over Boolean-cube coordinates and
// multi-input reductions are the domain's normal shape.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod code;
pub mod eval;
pub mod ring;

pub use code::{CodeParams, Codeword};
pub use eval::{
    commit_table, prove_eval, verify_eval, LevelOpen, PathQuery, PcsParams, PcsProfile, PcsProof,
    PcsStatement,
};
pub use ring::{prove_ring_switch, verify_ring_switch, RingProof};

use magnetar_field::Gf128;

/// A virtual evaluation claim (paper Definition 5): `v̂(r) = y`.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalClaim {
    pub point: Vec<Gf128>,
    pub value: Gf128,
}

impl EvalClaim {
    pub fn new(point: Vec<Gf128>, value: Gf128) -> Self {
        EvalClaim { point, value }
    }
}
