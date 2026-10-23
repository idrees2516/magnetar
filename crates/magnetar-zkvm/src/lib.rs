//! # magnetar-zkvm
//!
//! MosaicPCS (paper §5.2) and the **complete** protocol (Construction 3,
//! §5.5) — all seven steps, with the virtual-claim discharge (step 4)
//! fully integrated:
//!
//! 1. **Execute** — the interpreter runs the expanded program and records
//!    the virtual-cycle trace plus the memory-access log.
//! 2. **Commit** — the direct witness segments and the auxiliary
//!    polynomials (the one-hot address components and the increments) form
//!    one MosaicPCS polynomial over aligned subcubes; its packed codeword
//!    is Merkle-committed with BLAKE3 — the protocol's only commitment
//!    `cm_G`.
//! 3. **Virtual Spartan** — the two SIMD R1CS relations (the F₂ bit
//!    constraints and the K packed products, §5.1) are reduced jointly to
//!    one evaluation claim per witness segment: the direct claims are held
//!    for the PCS, the virtual claims forwarded to step 4.
//! 4. **Reduce the virtual claims** — BinaryTwist for registers (with the
//!    Ψ-sum-checks binding the register-index claims), BinaryTwist for RAM
//!    (with the address claims and the verifier-sampled increment claim),
//!    BinaryRouter for shifts and sub-word stores, and BinaryShout for the
//!    bytecode (the joint-space bus Φ, the micro-PC Ψ, and the routing
//!    selector checks) turn every virtual claim into evaluation claims on
//!    the components of `cm_G`.
//! 5. **Prove the committed evaluations** — MosaicPCS lifts every claim
//!    for free (aligned subcubes), batches them with one sum-check
//!    (binding the 7 packing variables first), and discharges the terminal
//!    via **ring switching** (the c\*-codeword fold chain anchored to
//!    `cm_G` with exact `φ_{c*}` bit-extraction checks), closing with the
//!    single PCS evaluation argument.

#![forbid(unsafe_code)]
// Proof-system style: loops indexed over Boolean-cube coordinates and
// multi-input reductions are the domain's normal shape.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod mosaic;
pub mod protocol;
pub mod rom;
pub mod stats;

pub use mosaic::{
    pack_tables, pack_tables_slice, MosaicClaim, MosaicCommitment, MosaicLayout, MosaicProof,
};
pub use protocol::{
    boundary_point_claims, build_segment, keygen, keygen_with, prove, prove_inner, verify,
    verify_inner, Boundary, InnerProof, KeyGen, Proof, PublicInput, SegmentStatement, VerifyingKey,
    WitnessData, AMOUNT_BITS,
};
pub use rom::RomTables;
pub use stats::{proof_stats, ProofStats};
