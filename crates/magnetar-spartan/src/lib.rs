//! # magnetar-spartan
//!
//! Spartan over binary fields (paper §4.1–§4.3): the R1CS-to-evaluation
//! reduction (Construction 1), its SIMD extension (§4.2, Phalanx), and
//! **Virtual Spartan** (Construction 2, §5.3) which reduces *two* SIMD R1CS
//! relations over one shared bit-valued witness to one evaluation claim per
//! witness segment.
//!
//! ## Layout conventions
//!
//! * The joint SIMD space is row-major: the column index `y` occupies the
//!   **low** `log m` bits and the instance index `u` the **high** `log q`
//!   bits, so the outer sum-check binds the instance variables first
//!   (`HighBitsFirst`) — exactly the lockstep order of Construction 2.
//! * `z = (w, 1, x)` with `w` in the low half of the columns: Equation (1)
//!   therefore splits at the **most significant** column variable,
//!   `ẑ(y) = (1 + y_msb)·ŵ(y_low) + y_msb·(1,x)^(y_low)`.
//! * Witness segments occupy aligned column subcubes: a segment of `2^s`
//!   columns starting at a multiple of `2^s` — the suffix `b_i` is the high
//!   `(log m − s)` bits of its offset, and
//!   `Ŵ(r_inst, r_col) = Σ_i eq(q_i, b_i)·Ŵ_i(r_inst, r_i)`.

#![forbid(unsafe_code)]

pub mod r1cs;
pub mod virtual_spartan;

pub use r1cs::{SparseMatRef, SparseMatrix};
pub use virtual_spartan::{
    prove_virtual_spartan, verify_virtual_spartan, DualRelation, SegmentLayout, SegmentSpec,
    VirtualSpartanProof,
};

use magnetar_field::Gf128;

/// An evaluation claim on a witness segment: `Ŵ_i(r_inst ‖ r_i) = v_i`
/// (paper Definition 5, output of Construction 2 step 8).
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentClaim {
    pub segment: usize,
    /// `r_inst` (SIMD instance coordinates).
    pub r_inst: Vec<Gf128>,
    /// Segment-local column coordinates.
    pub r_local: Vec<Gf128>,
    pub value: Gf128,
}
