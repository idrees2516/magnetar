//! Segmented proving — the roadmap's Horizon-2 "segments share the
//! commitment" mechanism (paper §6.2: prove long executions in 2^20–2^24
//! cycle chunks and compose).
//!
//! **What this module implements (v1).** An execution split into `S`
//! chunks of `T` cycles, executed with explicit state handoff
//! ([`magnetar_cpu::interp::execute_chunks`]), proven under **one**
//! MosaicPCS commitment: the chunk boundaries are pure bookkeeping —
//! the concatenated records are exactly the monolithic trace, the
//! register/RAM Twist instances span the whole row space (so the
//! cross-cycle continuity carries across chunk boundaries for free),
//! and the per-segment claims discharge through the single aggregated
//! Mosaic batching sum-check (`S` sets of component claims, one
//! α-random-linear fold, **one** ring-switch evaluation argument).
//!
//! **What this buys today.** The measured claim-aggregation saving
//! ([`aggregation_report`]): the shared-commitment proof of `S×T` cycles
//! versus `S` independent `T`-cycle proofs — one evaluation argument, one
//! commitment, one transcript chain instead of `S` of each. The
//! measured table lives in `docs/BENCHMARKS.md`.
//!
//! **What it composes with (the full Horizon-2 design).** The segments
//! feed the tree-IVC schedule: pairs of chunk proofs verify inside a
//! binary tree whose internal nodes batch the children's claims, and
//! the small **wrap** instance (an implementation of this verifier,
//! proved once at a low PCS rate) is the only public artifact. The
//! verifier-guest circuit, the tree schedule, and the dual-rate wrap
//! are designed in `docs/RECURSION.md` — their absence here is tracked
//! in `docs/STATUS.md`.

use crate::protocol::{
    keygen, prove_padded_trace, verify, KeyGen, Proof, PublicInput, VerifyingKey,
};
use magnetar_cpu::interp::{execute_chunks, Trace};

/// Keygen for a segmented execution: `chunks` segments of `chunk_len`
/// cycles (the trace length is their product, a power of two).
pub fn keygen_segments(
    program: Vec<u32>,
    chunk_len: usize,
    chunks: usize,
) -> (KeyGen, VerifyingKey) {
    assert!(
        chunk_len.is_power_of_two() && chunks.is_power_of_two(),
        "the chunk length and count must be powers of two"
    );
    keygen(program, chunk_len * chunks)
}

/// Execute the program in `chunks` segments with explicit state handoff
/// and prove the whole execution under one shared commitment. The
/// records are the monolithic trace by construction (tested); the
/// proof is the standard Construction-3 artifact.
pub fn prove_segments(pk: &KeyGen, pub_in: &PublicInput) -> Proof {
    let chunk_len = pk.trace_len / segment_count(pk);
    let (chunks, final_state) =
        execute_chunks(&pk.program, &pub_in.input, chunk_len, segment_count(pk));
    assert_eq!(
        final_state.output, pub_in.output,
        "segmented execution output mismatch"
    );
    let mut records = Vec::with_capacity(pk.trace_len);
    for c in chunks {
        records.extend(c);
    }
    assert_eq!(records.len(), pk.trace_len);
    let trace = Trace {
        program: magnetar_cpu::expand::expand_program(&pk.program),
        records,
        final_state,
    };
    prove_padded_trace(pk, pub_in, &trace)
}

/// The segment count is a proving-key parameter (stored in the key's
/// trace geometry: the chunk length defaults to the smallest power of
/// two ≤ 2^13 cycles, the streaming-friendly granularity).
fn segment_count(pk: &KeyGen) -> usize {
    let chunk = pk
        .trace_len
        .min(1usize << 13)
        .next_power_of_two()
        .min(pk.trace_len);
    pk.trace_len / chunk
}

/// Verify a segmented proof (identical to the monolithic verification —
/// the segments are invisible in the artifact).
pub fn verify_segments(
    vk: &VerifyingKey,
    pub_in: &PublicInput,
    proof: &Proof,
) -> Result<(), &'static str> {
    verify(vk, pub_in, proof)
}

/// The measured claim-aggregation report: the shared-commitment proof
/// of `segments × single.cycles` cycles versus `segments` independent
/// proofs of the single-execution size. Returns
/// `(independent_bytes, shared_bytes)`.
pub fn aggregation_report(
    single: &crate::stats::ProofStats,
    shared: &crate::stats::ProofStats,
) -> (u64, u64) {
    let segments = (shared.cycles / single.cycles.max(1)).max(1) as u64;
    (
        segments * single.bytes_total as u64,
        shared.bytes_total as u64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{keygen, prove, PublicInput};

    fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32) -> u32 {
        ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x13
    }

    /// The chunked execution must produce exactly the monolithic
    /// witness, and the segmented proof must verify (the segments share
    /// the commitment; the Twist continuity spans the chunk boundaries).
    #[test]
    fn segments_match_monolithic_and_verify() {
        let bne = |off: i32| -> u32 {
            let imm = off as u16;
            let i12 = ((imm >> 12) & 1) as u32;
            let i11 = ((imm >> 11) & 1) as u32;
            let i10_5 = ((imm >> 5) & 0x3f) as u32;
            let i4_1 = ((imm >> 1) & 0xf) as u32;
            (i12 << 31) | (i10_5 << 25) | (3 << 15) | (1 << 12) | (i4_1 << 8) | (i11 << 7) | 0x63
        };
        // A looping program that crosses the chunk boundary repeatedly.
        let program = vec![
            enc_i(3, 0, 0, 3),
            enc_i(1, 0, 0, 1),
            enc_i(1, 0, 0, 2),
            enc_i(0xfff, 3, 0, 3),
            (2 << 20) | (1 << 15) | (4 << 7) | 0x33,
            enc_i(0, 2, 0, 1),
            enc_i(0, 4, 0, 2),
            bne(-24),
            enc_i(0, 2, 0, 10),
            0x0000_0073,
        ];
        let trace_len = 64usize;
        let chunks = 4usize;
        let out = {
            let t = magnetar_cpu::interp::execute(&program, &[], trace_len);
            t.final_state.output
        };
        // The monolithic witness, padded to the proving key's trace
        // length (the interpreter itself only pads to the next power of
        // two; the proving path pads the rest).
        let mut mono = magnetar_cpu::interp::execute(&program, &[], trace_len);
        mono.records = magnetar_cpu::interp::pad_records(&mono, trace_len);
        // The chunked witness: identical records.
        let chunk_len = trace_len / chunks;
        let (chunked, _) = execute_chunks(&program, &[], chunk_len, chunks);
        let flat: Vec<_> = chunked.into_iter().flatten().collect();
        assert_eq!(flat.len(), mono.records.len());
        for (i, (a, b)) in flat.iter().zip(mono.records.iter()).enumerate() {
            assert_eq!(
                format!("{a:?}"),
                format!("{b:?}"),
                "record {i} differs between the chunked and monolithic witnesses"
            );
        }
        // The segmented proof verifies.
        let (pk, vk) = keygen_segments(program.clone(), chunk_len, chunks);
        assert_eq!(pk.trace_len, trace_len);
        let pub_in = PublicInput {
            input: vec![],
            output: out,
        };
        let proof = prove_segments(&pk, &pub_in);
        assert!(verify_segments(&vk, &pub_in, &proof).is_ok());
        // And equals a monolithic proof's verification over the same key.
        let mono_proof = prove(&pk, &pub_in);
        assert!(verify(&vk, &pub_in, &mono_proof).is_ok());
    }

    /// The measured aggregation saving: one shared commitment and one
    /// evaluation argument versus `S` independent proofs.
    #[test]
    fn aggregation_measured() {
        let program = vec![
            enc_i(1, 0, 0, 1),
            enc_i(1, 0, 0, 2),
            (2u32 << 20) | (1 << 15) | (3 << 7) | 0x33,
            (2u32 << 15) | (1 << 7) | 0x33,
            enc_i(0, 3, 0, 2),
            (3 << 15) | (10 << 7) | 0x33,
            0x0000_0073,
        ];
        // A single 16-cycle proof.
        let (spk, _svk) = keygen(program.clone(), 16);
        let trace = magnetar_cpu::interp::execute(&program, &[], 16);
        let out = trace.final_state.output;
        let pub_in = PublicInput {
            input: vec![],
            output: out,
        };
        let single_proof = prove(&spk, &pub_in);
        let single = crate::stats::proof_stats(&spk, &single_proof);
        // The 4-segment (64-cycle) shared-commitment proof.
        let (pk, vk) = keygen_segments(program, 16, 4);
        let trace64 = magnetar_cpu::interp::execute(&pk.program, &[], 64);
        let pub_in64 = PublicInput {
            input: vec![],
            output: trace64.final_state.output,
        };
        let shared_proof = prove(&pk, &pub_in64);
        assert!(verify(&vk, &pub_in64, &shared_proof).is_ok());
        let shared = crate::stats::proof_stats(&pk, &shared_proof);
        let (independent, shared_bytes) = aggregation_report(&single, &shared);
        // 4 independent 16-cycle proofs carry 4 evaluation arguments;
        // the shared proof carries 1 — the saving must be material.
        assert!(
            shared_bytes < independent,
            "shared {shared_bytes} vs independent {independent}"
        );
    }
}
