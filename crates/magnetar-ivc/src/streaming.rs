//! The **streaming, parallel tree-IVC prover** — the bounded-memory
//! production path (docs/PERFORMANCE.md R7).
//!
//! The sequential [`crate::prove_ivc`] materializes, at once: the full
//! padded trace (`O(T)` records), every segment's witness tables
//! (`O(m·T)` bits — the largest artifact in the stack), the boolean
//! union `g` (`O(m·T)` bits again), the packed vector, and the codeword.
//! This module replaces that with a three-phase pipeline:
//!
//! 1. **Checkpoint pass** (sequential, one segment of records in the
//!    window): the machine runs once; every segment boundary snapshots
//!    the [`StreamCursor`] (registers + RAM + the halt-frozen fetch
//!    chain). The records are discarded as soon as the boundary is
//!    taken — the trace itself is never resident.
//! 2. **Segment proofs** (rayon-parallel over segments,
//!    [`docs/PERFORMANCE.md` R1/R8]): each worker replays its segment
//!    from its snapshot (deterministic ⇒ byte-identical records), builds
//!    the segment witness, proves steps 2–6 into its **own** Fiat–Shamir
//!    transcript (the tree-IVC design: segment independence is what
//!    makes the parallelism free), and packs its tables **straight into
//!    the shared leaf vector** (`pack_tables_slice` — the boolean union
//!    is never materialized). The tables drop immediately after packing:
//!    at most one in-flight segment's tables exist per worker.
//! 3. **Root** (sequential): one codeword encode over the packed vector
//!    (parallel ε-blocks), one parallel Merkle build, the digest chain,
//!    and the single Mosaic discharge.
//!
//! Peak memory is therefore `O(checkpoints + one segment's witness +
//! packed + codeword + tree)` instead of `O(full trace + all tables +
//! g + packed + codeword + tree)`. Every phase's wall-clock and the
//! peak RSS are measured ([`StreamingStats`]) — a regression in either
//! is a bug (R10).
//!
//! The equivalence with the sequential prover is exact, not
//! structural: same statements, same per-segment transcripts, same
//! commitment — `prove_ivc` and `prove_ivc_streaming` produce
//! byte-identical proofs (tested).

use crate::{combined_layout, stmt_bytes};
use magnetar_cpu::interp::{StreamCursor, Trace};
use magnetar_cpu::{expand_program, Word};
use magnetar_hash::{MerkleTree, Transcript};
use magnetar_pcs::code::CodeParams;
use magnetar_pcs::PcsStatement;
use magnetar_zkvm::mosaic::{pack_tables_slice, prove_mosaic, MosaicClaim};
use magnetar_zkvm::{
    build_segment, keygen_with, prove_inner, Boundary, InnerProof, PublicInput, SegmentStatement,
    AMOUNT_BITS,
};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Measured resource profile of a streaming proof run (R10: honest
/// numbers, never extrapolated).
#[derive(Clone, Debug, Default)]
pub struct StreamingStats {
    /// Wall-clock of the checkpoint pass (trace streaming).
    pub wall_checkpoints: Duration,
    /// Wall-clock of the parallel segment-proof phase (max over workers
    /// plus the pack time — the critical path).
    pub wall_segments: Duration,
    /// Wall-clock of the root commitment + discharge.
    pub wall_root: Duration,
    /// Total `prove_ivc_streaming` wall-clock.
    pub wall_total: Duration,
    /// Peak resident set size observed (Linux `VmHWM`), bytes.
    pub peak_rss_bytes: u64,
    /// The packed leaf vector's byte size (the commitment working set).
    pub packed_bytes: usize,
    /// The codeword's byte size (`2^k ×` the packed vector).
    pub codeword_bytes: usize,
    /// The largest in-flight segment's witness-table bytes (the per-worker
    /// bound actually observed).
    pub segment_table_bytes: usize,
    /// The checkpoint store's byte size.
    pub checkpoint_bytes: usize,
    /// Number of rayon threads that ran the segment phase.
    pub threads: usize,
}

/// The current peak RSS (Linux `VmHWM`; 0 where unavailable).
pub fn peak_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| {
                    l.split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse::<u64>().ok())
                })
            })
            .map(|kb| kb * 1024)
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// The boundary state at checkpoint `s`, in the exact shape
/// [`crate::prove_ivc`]'s `boundary_at` replay produces: registers, RAM
/// and the halt latch evolve identically under `step`/`frozen_record`
/// and under the record replay, and the fetch pc is the cursor's
/// `micro_pc` (the halt-frozen row once halted — `step` keeps it
/// modular, and the padding identity preserves it).
fn boundary_of(cursor: &StreamCursor) -> Boundary {
    Boundary {
        regs: cursor.st.regs.to_vec(),
        pc: cursor.st.micro_pc,
        halted: cursor.st.halted,
        ram: cursor.st.ram.clone(),
    }
}

/// Prove a program execution as `segments` tree-IVC segments with
/// **bounded memory and parallel segment proofs**. Same public
/// interface and byte-identical output as [`crate::prove_ivc`]; see the
/// module docs for the pipeline.
#[allow(clippy::too_many_arguments)]
pub fn prove_ivc_streaming(
    program: &[u32],
    input: &[Word],
    output: Word,
    total_len: usize,
    segments: usize,
    pcs: &magnetar_pcs::PcsParams,
) -> (crate::IvcProof, StreamingStats) {
    let t_total = Instant::now();
    assert!(segments.is_power_of_two() && segments >= 1);
    assert!(total_len.is_power_of_two());
    let seg_len = total_len / segments;
    assert!(
        seg_len.is_power_of_two() && seg_len >= 2,
        "segment too short"
    );

    // ---- Phase 1: the checkpoint pass (streaming execution). ----
    let t0 = Instant::now();
    let prog = expand_program(program);
    let mut cursor = StreamCursor::start(input);
    // `segments + 1` snapshots: the segment starts and the final state.
    let mut checkpoints: Vec<StreamCursor> = Vec::with_capacity(segments + 1);
    checkpoints.push(cursor.clone());
    for _ in 0..segments {
        let records = cursor.run_segment(&prog, seg_len);
        // The records die here — only the state snapshot survives (R7).
        drop(records);
        checkpoints.push(cursor.clone());
    }
    assert_eq!(
        checkpoints[segments].st.output, output,
        "execution output mismatch"
    );
    let wall_checkpoints = t0.elapsed();
    let checkpoint_bytes = checkpoints.len()
        * (std::mem::size_of::<Word>() * magnetar_cpu::REGS
            + std::mem::size_of::<Word>() * (1usize << magnetar_cpu::RAM_WORDS_LOG2)
            + 64);

    // ---- The shared proving key + the combined layout. ----
    let (pk, _vk) = keygen_with(program.to_vec(), seg_len, AMOUNT_BITS, pcs.clone());
    let combined = combined_layout(&pk, segments);
    let n_comp = pk.mosaic.components.len();
    let n = combined.ell - 7;

    // ---- Phase 2: the parallel segment proofs + streaming pack. ----
    let t1 = Instant::now();
    let __phase0 = t1;
    let mut packed = vec![magnetar_field::ZERO; 1usize << n];
    // The pack OR-slices share leaf words across segments, so the slice
    // packs serialize (each is a single O(leaves) pass — a rounding error
    // next to the segment proof it follows).
    let packed_lock = Mutex::new(&mut packed);
    let max_tables = AtomicUsize::new(0);

    /// One segment worker's outputs: (segment index, statement, inner
    /// proof, remapped claims, transcript digest).
    type SegResult = (
        usize,
        SegmentStatement,
        InnerProof,
        Vec<MosaicClaim>,
        [u8; 32],
    );
    let results: Vec<SegResult> = (0..segments)
        .into_par_iter()
        .map(|s| {
            let __t_start = std::time::Instant::now();
            let __tid = std::thread::current().id();
            // 2a. Replay this segment from its snapshot — byte-identical
            // to the sequential prover's full-trace slice.
            let mut cur = checkpoints[s].clone();
            let records = cur.run_segment(&prog, seg_len);
            let seg_trace = Trace {
                program: prog.clone(),
                records,
                final_state: cur.st.clone(),
            };
            // 2b. The boundary statement (the final boundary's pc is 0 —
            // the sequential replay's end-of-trace convention).
            let init = boundary_of(&checkpoints[s]);
            let mut final_ = boundary_of(&checkpoints[s + 1]);
            if s + 1 == segments {
                final_.pc = 0;
            }
            let stmt = SegmentStatement {
                input: if s == 0 { input.to_vec() } else { Vec::new() },
                output,
                init,
                final_,
            };
            let pub_in = PublicInput {
                input: Vec::new(),
                output,
            };
            // 2c. Steps 2–6 in the segment's own transcript.
            let mut t = Transcript::new(
                "magnetar-segment",
                &[
                    b"tree-ivc-v1",
                    &program
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<u8>>(),
                    &(s as u64).to_le_bytes(),
                    &stmt_bytes(&stmt),
                ],
            );
            let (wd, seg_tables) = build_segment(&pk, &pub_in, &seg_trace);
            max_tables.fetch_max(
                seg_tables.iter().map(|t| t.len()).sum::<usize>(),
                Ordering::Relaxed,
            );
            // The cross-step's last-cycle successor: the next segment's
            // initial pc, or the halt freeze on the last segment.
            let next_pc = if s + 1 < segments {
                Some(checkpoints[s + 1].st.micro_pc % prog.rows.len())
            } else {
                None
            };
            let (inner, seg_claims) =
                prove_inner(&pk, &pub_in, Some(&stmt), &seg_trace, &wd, next_pc, &mut t);
            let digest = t.state_digest();
            // The witness dies here; only the tables' packed planes live on.
            drop(wd);
            // 2d. Pack this segment's tables into the shared leaf vector
            // (the segment's components are `[s·n_comp, …)` of the
            // combined layout).
            {
                let mut p = packed_lock.lock().expect("pack lock poisoned");
                pack_tables_slice(&combined, s * n_comp, &seg_tables, p.as_mut_slice());
            }
            // The claims remapped into the combined layout.
            let shift = s * n_comp;
            let claims: Vec<MosaicClaim> = seg_claims
                .into_iter()
                .map(|mut c| {
                    c.component += shift;
                    c
                })
                .collect();
            if std::env::var_os("MAGNETAR_TRACE_WORKERS").is_some() {
                eprintln!(
                    "[worker {s}] thread {__tid:?} took {:?} (start offset {:?})",
                    __t_start.elapsed(),
                    __phase0.elapsed() - __t_start.elapsed(),
                );
            }
            (s, stmt, inner, claims, digest)
        })
        .collect();
    let wall_segments = t1.elapsed();

    // Order the workers' results by segment (par_iter completion order).
    let mut ordered: Vec<_> = results;
    ordered.sort_by_key(|(s, ..)| *s);
    let mut statements = Vec::with_capacity(segments);
    let mut inners = Vec::with_capacity(segments);
    let mut digests = Vec::with_capacity(segments);
    let mut claims = Vec::new();
    for (_, stmt, inner, seg_claims, digest) in ordered {
        statements.push(stmt);
        inners.push(inner);
        digests.push(digest);
        claims.extend(seg_claims);
    }
    drop(checkpoints);

    // ---- Phase 3: the root — one commitment, one discharge. ----
    let t2 = Instant::now();
    let code_params = CodeParams::derive(n, pcs.k, &pcs.domain);
    let codeword = magnetar_pcs::code::Codeword::encode(&code_params, &packed);
    let tree = MerkleTree::build(&format!("{}/level/0", pcs.domain), &codeword.values);
    let stmt = PcsStatement {
        root: tree.root(),
        n,
        k: pcs.k,
        domain: pcs.domain.clone(),
    };
    let mut t = Transcript::new(
        "magnetar-ivc",
        &[
            b"tree-ivc-v1",
            &program
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            &input
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            &output.to_le_bytes(),
            &(segments as u64).to_le_bytes(),
        ],
    );
    for (s, d) in digests.iter().enumerate() {
        t.absorb_digest(&format!("seg-digest-{s}"), d);
    }
    t.absorb_digest("cm-g", &stmt.root);
    let commitment = magnetar_zkvm::MosaicCommitment {
        packed,
        codeword,
        tree,
        stmt,
        ell: combined.ell,
    };
    let mosaic = prove_mosaic(pcs, &combined, &commitment, &claims, &mut t);
    let wall_root = t2.elapsed();

    let stats = StreamingStats {
        wall_checkpoints,
        wall_segments,
        wall_root,
        wall_total: t_total.elapsed(),
        peak_rss_bytes: peak_rss_bytes(),
        packed_bytes: (1usize << n) * 16,
        codeword_bytes: commitment.codeword.values.len() * 16,
        segment_table_bytes: max_tables.load(Ordering::Relaxed),
        checkpoint_bytes,
        threads: rayon::current_num_threads(),
    };
    let proof = crate::IvcProof {
        statements,
        inners,
        segment_digests: digests,
        commitment_root: commitment.stmt.root,
        pcs_n: commitment.stmt.n,
        pcs_k: commitment.stmt.k,
        pcs_domain: commitment.stmt.domain.clone(),
        ell: commitment.ell,
        mosaic,
    };
    (proof, stats)
}
