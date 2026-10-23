//! cuTile GPU prover plane for the bitsliced sum-check (feature `cutile`).
//!
//! NVlabs **cuTile** (`cutile-rs`) kernels and host prover for the bitsliced
//! GF(2^128) multilinear sum-check — the Phase-1 kernel spec of the GPU
//! proving program.
//!
//! * [`kernels`] — the `#[cutile::module]` kernels: `pack_planes`,
//!   `bind_round` (tower multiplication tree over `[TG, 128]` tiles),
//!   `reduce_partial` (XOR row reduction).
//! * [`prover`] — round orchestration, hierarchical reduction, CPU tail via
//!   the shared protocol code (so GPU and CPU proofs are bit-identical on
//!   the same transcript).
//!
//! ## Requirements
//!
//! * Build: CUDA 13.2+ toolkit (`CUDA_TOOLKIT_PATH`), Rust stable 1.89+.
//! * Run: NVIDIA GPU with compute capability sm_80+ (H100 = sm_90).
//! * Kernels JIT-compile through CUDA Tile IR on first launch.
//!
//! ## Correctness model
//!
//! The GPU kernels are a 1:1 transliteration of algorithms that are
//! bit-exact tested in the host oracle ([`crate::tree::bind_planes`],
//! [`crate::tree::reduce_planes`], [`crate::bitslice::BitPlanes::pack`]).
//! The acceptance gate on hardware is the `gpu_parity` integration test
//! (`BITSUM_GPU_TESTS=1`), which additionally checks the full prover
//! against the CPU reference on identical transcripts. The kernels are
//! written against the published cuTile DSL/host API and are compiled and
//! gated by the manual `bitsum-cuda` CI workflow; see
//! `docs/architecture/bitsum.md` for the bring-up checklist.

pub mod kernels;
pub mod prover;

pub use prover::{
    bind_round_on_device, download_words, pack_on_device, prove, reduce_to_element, upload_words,
    Error, Result, BIND_TG, GPU_MIN_GROUPS, PACK_TG, REDUCE_TGR,
};
