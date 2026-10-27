//! # magnetar-mikros
//!
//! Bitwidth-aware SNARK components over extensions of small fields —
//! the Mikros framework (Cong, *Mikros: Bitwidth-Aware SNARKs over
//! Extensions of Small Fields*, IACR ePrint 2026/2646) instantiated on this
//! stack's fields: `F = GF(2) ⊂ K = GF(2^8) ⊂ E = GF(2^128)` with packing
//! width `κ = 128`.
//!
//! The crate maps the paper's building blocks as:
//!
//! | Paper object | Module | Notes |
//! |---|---|---|
//! | GF(2^8) intermediate field `K` (§4.2 small-field variant) | [`gf8`] | the unique order-256 subfield of `E`, AES-polynomial code basis (GFNI-compatible) |
//! | Shared interpolation data (App. B.1) | [`interp`] | `H` = a 7-dim subspace, `Ψ`, Lagrange `ψ_j`, off-domain nodes, `V⁻¹` |
//! | Algebraic vector packing (§4.1) | [`planes`] | column-plane layouts, selector-XOR encodings, bit↔byte spreads |
//! | Fact 1 inner-product reduction (§3.2) | [`fact1`] | `[p0,p1,s]` degree-2 rounds |
//! | Packed Hadamard proof, Algorithm 2 (§4.2) | [`hadamard`] | the Table-3 benchmark core |
//! | Packed inner-product proof, Algorithm 1 (§4.1) | [`innerprod`] | monomial/reciprocal packing |
//! | Packed PCS opening, Algorithm 4 (§4.4) | [`pcs_open`] | `U = Ξ^T A` column claims + the char-2 `W` recursion (Eq. 25) |
//! | Packed permutation proof, Algorithm 3 (§4.3) | [`perm`] | grand products from Hadamard products |
//! | Batched-instance R1CS, Algorithm 6 (§5.3) | [`batch`] | the Table-4 hash benchmarks (public wiring) |
//! | Binary hash circuits | [`circuits`] | SHA-256 compression, Keccak-f[1600], BLAKE3 compression |
//!
//! Benchmarks live in `src/bin/mikros-bench.rs` mirroring the paper's §6
//! evaluation (the 2^30 binary Hadamard PIOP, and end-to-end 256-instance
//! hash batches).

#![cfg_attr(target_arch = "x86_64", allow(unsafe_code))]
#![cfg_attr(not(target_arch = "x86_64"), deny(unsafe_code))]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod batch;
pub mod circuits;
pub mod fact1;
pub mod gf8;
pub mod hadamard;
pub mod innerprod;
pub mod interp;
pub mod pcs_open;
pub mod perm;
pub mod planes;
pub mod r1cs;
