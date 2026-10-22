//! # magnetar-cpu
//!
//! The RV64I machine of Magnetar §5.1: decoder, microcode expansion,
//! interpreter, and the **dual SIMD R1CS circuit** — one relation over
//! `B = F_2` for bit constraints and one over `K = F_{2^128}` for packed
//! products and packed linear checks, sharing one bit-valued witness
//! matrix `W` (paper §5.1).
//!
//! ## Micro-PC and expansion
//!
//! Preprocessing deterministically expands each RV64I instruction into one
//! or more **virtual instructions** (a public bytecode table `ROM` indexed
//! by the micro-PC). Register-shift instructions expand into a fixed
//! sequence of static-routing stages (paper §5.4); everything else is one
//! virtual instruction. A trace of `T = 2^n` virtual cycles is the
//! execution (§2.1 step 1).

#![forbid(unsafe_code)]
// Proof-system style: loops indexed over Boolean-cube coordinates and
// multi-input reductions are the domain's normal shape.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod circuit;
pub mod decode;
pub mod expand;
pub mod interp;

pub use circuit::{build_circuit, CircuitSpec, WITNESS_LAYOUT};
pub use decode::{decode, Inst, Opcode};
pub use expand::{expand_program, RomRow, VirtualInst};
pub use interp::{execute, execute_chunk, execute_padded, ExecState, Trace};

/// A 64-bit word.
pub type Word = u64;

/// Register file size: 32 architectural registers padded to 64 cells
/// (writes to `x0` are redirected to the padding cells — paper's §2.1
/// register-file note).
pub const REGS: usize = 64;

/// RAM cells (words), log2 size in the test configuration.
pub const RAM_WORDS_LOG2: usize = 8;
