//! # magnetar-zkvm
//!
//! MosaicPCS (paper §5.2) and the complete protocol (Construction 3, §5.5):
//!
//! 1. **Execute** — the interpreter runs the expanded program and records
//!    the virtual-cycle trace (the CPU circuit's logical witness).
//! 2. **Commit** — the witness matrix `W` (one row per cycle, the dual
//!    circuit's columns) is packed 128 bits per `K` element over the top 7
//!    index bits; the affine-substitution codeword of the packed vector is
//!    Merkle-committed with BLAKE3 — the protocol's only commitment
//!    `cm_G`.
//! 3. **Virtual Spartan** — the two SIMD R1CS relations (the F₂ bit
//!    constraints and the K packed products, §5.1) are reduced jointly to
//!    one evaluation claim per witness segment.
//! 4. **Reduce the virtual claims** — BinaryTwist (registers/RAM),
//!    BinaryRouter (shifts/sub-word stores) and BinaryShout (bytecode)
//!    own the virtual segments (implemented in `magnetar-memory` /
//!    `magnetar-router`, discharged by their reductions against `cm_G`).
//! 5. **Prove the committed evaluations** — MosaicPCS lifts every claim to
//!    a claim on `G`, batches them with one sum-check (binding the 7
//!    packing variables first — after those rounds the dense factor *is*
//!    the c\*-folded table), and discharges the terminal via **ring
//!    switching** (the c\*-codeword fold chain anchored to `cm_G` with
//!    exact `φ_{c*}` bit-extraction checks), closing with the single PCS
//!    evaluation argument.
//!
//! This crate wires steps 1–3 and 5 end-to-end; step 4's arguments are
//! exercised in their own crates and composed through the same
//! claim-flow (each outputs component claims on `cm_G`).

#![forbid(unsafe_code)]

pub mod mosaic;
pub mod protocol;

pub use mosaic::{MosaicCommitment, MosaicProof};
pub use protocol::{keygen, prove, verify, KeyGen, Proof, PublicInput, VerifyingKey};

use magnetar_field::Gf128;

/// A bit-witness claim: `Ĝ_i(r_i) = v_i` on a component of the Mosaic
/// polynomial.
#[derive(Clone, Debug)]
pub struct MosaicClaim {
    /// The component's local point (little-endian).
    pub point: Vec<Gf128>,
    pub value: Gf128,
}
