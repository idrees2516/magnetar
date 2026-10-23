//! # magnetar-bitsum
//!
//! Bitsliced multilinear **sum-check over GF(2^128)** in the tower basis —
//! Phase 1 of the 150M-cycles/s GPU proving program (see the
//! [roadmap](https://github.com/idrees2516/magnetar/blob/main/ROADMAP.md)
//! and `docs/architecture/bitsum.md`): a fully tested host oracle plus, behind
//! the `cutile` feature, the NVlabs **cuTile** GPU kernels that mirror the
//! oracle one-to-one.
//!
//! ## Module map
//!
//! | Module | Contents |
//! |---|---|
//! | [`field`] | GF(2^128) tower field, verified constants `NUS[k] = 1 << (2^k - 1)` |
//! | [`bitslice`] | 32-element bitsliced groups, pack/unpack, parity, transpose |
//! | [`tree`] | the tower multiplication tree — the exact GPU kernel algorithm |
//! | [`transcript`] | Fiat-Shamir over the repo-standard SHA3-256 duplex (`magnetar-hash`) |
//! | [`interop`] | tower ↔ `K` (GCM polynomial basis) conversion and the `Gf128` bridge |
//! | [`sumcheck`] | protocol: prove / prove_planes / verify / eval_mle |
//! | [`gpu`] | cuTile kernels + host orchestration (feature `cutile`) |
//!
//! ## Quick start
//!
//! ```
//! use magnetar_bitsum::{F128, Transcript, sumcheck};
//!
//! // witness table of 2^6 entries
//! let evals: Vec<F128> = (0..64)
//!     .map(|i| F128::from_u64(i as u64 * 0x9E37_9B97))
//!     .collect();
//!
//! // prove
//! let mut t = Transcript::new("my-instance");
//! let out = sumcheck::prove(&evals, &mut t).expect("prover");
//!
//! // verify
//! let mut tv = Transcript::new("my-instance");
//! let oc =
//!     sumcheck::verify_with_final_eval(&out.proof, out.claimed_sum, out.final_eval, &mut tv)
//!         .expect("verifier");
//! assert_eq!(oc.final_claim, out.final_eval);
//! ```
//!
//! Tables in the stack's own `K = F_{2^128}` (GCM polynomial basis,
//! `magnetar-field`) enter through [`interop::prove_gcm`] — the conversion is
//! a verified field isomorphism, so claims made on `K` tables carry over
//! exactly.
//!
//! ## The GPU parity guarantee
//!
//! The GPU prover runs, per round: a plane-XOR reduction (`S0`), then
//! `out = lo ^ tree(lo ^ hi, alpha)` — literally [`tree::reduce_planes`]
//! and [`tree::bind_planes`] with the tree of [`tree::tree_mul_const_group`].
//! The crate's tests prove those functions equal scalar field arithmetic,
//! so a GPU kernel that compiles to this op sequence produces bit-identical
//! proofs to the CPU prover (checked end-to-end by the `gpu_parity`
//! acceptance test on CUDA hardware).

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Proof-system style: loops indexed over Boolean-cube coordinates and
// multi-input reductions are the domain's normal shape.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod bitslice;
pub mod field;
pub mod interop;
pub mod sumcheck;
pub mod transcript;
pub mod tree;

#[cfg(feature = "cutile")]
pub mod gpu;

pub use bitslice::BitPlanes;
pub use field::F128;
pub use sumcheck::{
    absorb_round, eval_mle, prove, prove_from_tail, prove_planes, verify, verify_with_final_eval,
    ProveError, ProveOutput, ProverState, RoundMessage, SumcheckProof, VerifyError, VerifyOutcome,
};
pub use transcript::Transcript;

/// Tiny deterministic RNG abstraction used by tests and benches
/// (the prover itself never needs randomness).
pub trait Rng {
    /// Advances the generator and returns the next 64 bits.
    fn next_u64(&mut self) -> u64;

    /// Convenience: two 64-bit draws glued into one 128-bit value.
    fn next_u128(&mut self) -> u128 {
        let lo = self.next_u64() as u128;
        let hi = self.next_u64() as u128;
        (hi << 64) | lo
    }
}
