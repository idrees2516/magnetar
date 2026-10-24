//! # magnetar-ivc — tree-IVC for Magnetar
//!
//! The roadmap's §8.2 architecture decision: **tree-IVC with
//! claim-batching at each internal node**, reusing the MosaicPCS
//! machinery Magnetar already has.
//!
//! An execution of `2^n` cycles is split into `k = 2^d` **segments** of
//! equal length. Each segment is proved *independently* by Construction
//! 3's steps 2–6 ([`magnetar_zkvm::prove_inner`]) in its own Fiat–Shamir
//! transcript — so segment proving parallelizes and segments may arrive
//! out of order. The segments **share one commitment**: every segment's
//! Mosaic components are placed on disjoint aligned subcubes of one
//! combined layout (the segment index in the suffix), exactly as
//! MosaicPCS places components within one proof, and the root of the
//! aggregation tree discharges **all** segments' claims with a single
//! batching sum-check, ring switch, and PCS evaluation argument. The
//! final proof therefore carries one commitment and one evaluation
//! argument regardless of `k` — the claim-batching tree.
//!
//! ## Boundary continuity (the paper's Remark 4)
//!
//! Adjacent segments chain on their **public boundary states**:
//!
//! * the initial register file and RAM bind through the Twists' public
//!   initial memories `m0`;
//! * the final register file and RAM bind through the final-state
//!   sum-checks (`magnetar_memory::final_state`);
//! * the initial pc (the fetch one-hot at cycle 0) and the initial/final
//!   halt latches bind through public-point claims on committed
//!   components — the final halt latch, with the R1CS halt-latch
//!   constraint, soundly binds the public output;
//! * the verifier checks adjacent boundaries match in the clear.
//!
//! The within-segment cross-step pc sequencing remains the paper's
//! documented Remark-4 omission (the interpreter enforces it; the
//! constraint-level enforcement is the roadmap's next item).
//!
//! ## Scope notes
//!
//! * v1 aggregates on the prover side (the tree merges pairwise as
//!   segments arrive; the merge is associative). Distributed proving
//!   (each shard holding only its segment) needs the wrap/recursion
//!   horizon.
//! * The per-segment proof work is the full Construction 3 at the
//!   segment length — the aggregation saves the *communication* tail
//!   (the PCS evaluation argument, ~84% of the proof bytes), not the
//!   per-segment prover time.

#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

use magnetar_cpu::interp::{execute, pad_records, Trace};
use magnetar_cpu::Word;
use magnetar_hash::Transcript;
use magnetar_pcs::PcsParams;
use magnetar_zkvm::mosaic::{commit_mosaic, prove_mosaic, verify_mosaic};
use magnetar_zkvm::{
    build_segment, keygen_with, prove_inner, verify_inner, Boundary, InnerProof, KeyGen,
    MosaicClaim, MosaicLayout, MosaicProof, PublicInput, SegmentStatement, AMOUNT_BITS,
};

pub mod streaming;
pub use streaming::{peak_rss_bytes, prove_ivc_streaming, StreamingStats};

/// One aggregated tree node: the union of a subtree's components and
/// claims. Merging is associative, so segments can combine pairwise in
/// any order (the claim-batching of the roadmap's §8.2).
pub struct AggNode {
    /// The component tables in the combined (per-segment) order.
    pub tables: Vec<Vec<bool>>,
    /// The claims (component indices relative to this node's layout).
    pub claims: Vec<MosaicClaim>,
    /// How many segments this node covers.
    pub segments: usize,
}

impl AggNode {
    /// Merge two sibling nodes (the tree's internal-node operation):
    /// concatenates the component tables and claims — the combined
    /// Mosaic layout places the union on disjoint aligned subcubes, so
    /// the claim indices shift by the left child's component count.
    pub fn merge(mut left: AggNode, right: AggNode) -> AggNode {
        let shift = left.tables.len();
        let mut claims = left.claims;
        for mut c in right.claims {
            c.component += shift;
            claims.push(c);
        }
        left.tables.extend(right.tables);
        left.segments += right.segments;
        AggNode {
            tables: left.tables,
            claims,
            segments: left.segments,
        }
    }
}

/// The tree-IVC proof: `k` segment arguments, the public boundary chain,
/// one shared commitment, and one Mosaic discharge.
pub struct IvcProof {
    /// The per-segment public statements (the boundary chain).
    pub statements: Vec<SegmentStatement>,
    /// The per-segment arguments (steps 2–6 + the boundary bindings).
    pub inners: Vec<InnerProof>,
    /// The per-segment transcript digests (bound into the root).
    pub segment_digests: Vec<[u8; 32]>,
    /// The shared commitment.
    pub commitment_root: [u8; 32],
    pub pcs_n: usize,
    pub pcs_k: usize,
    pub pcs_domain: String,
    pub ell: usize,
    /// The single Mosaic discharge for every segment's claims.
    pub mosaic: MosaicProof,
}

/// The machine state at a cycle boundary, **replayed from the padded
/// full trace** (the padding cycles' trash-cell writes are part of the
/// schedule's evolution, so interpreter snapshots at a cut would miss
/// them). `upto` cycles of `records` have executed.
fn boundary_at(
    input: &[Word],
    records: &[magnetar_cpu::interp::CycleRecord],
    upto: usize,
) -> Boundary {
    let mut regs = vec![0u64; magnetar_cpu::REGS];
    let mut ram = vec![0u64; 1usize << magnetar_cpu::RAM_WORDS_LOG2];
    for (i, &w) in input.iter().enumerate() {
        if i < ram.len() {
            ram[i] = w;
        }
    }
    let mut halted = false;
    let mut output = 0u64;
    for rec in records.iter().take(upto) {
        regs[rec.rd] = rec.rd_val;
        if rec.is_store {
            ram[rec.ram_addr] = rec.ram_new;
        }
        if rec.halted_latch {
            halted = true;
            output = rec.rd_val;
        }
    }
    let _ = output;
    let pc = if upto < records.len() {
        records[upto].micro_pc
    } else {
        0
    };
    Boundary {
        regs,
        pc,
        halted,
        ram,
    }
}

/// Prove a program execution as `segments` (a power of two) tree-IVC
/// segments sharing one commitment. `total_len` is the padded trace
/// length (a power of two); each segment covers `total_len / segments`
/// cycles.
#[allow(clippy::too_many_arguments)]
pub fn prove_ivc(
    program: &[u32],
    input: &[Word],
    output: Word,
    total_len: usize,
    segments: usize,
    pcs: &PcsParams,
) -> IvcProof {
    assert!(segments.is_power_of_two() && segments >= 1);
    assert!(total_len.is_power_of_two());
    let seg_len = total_len / segments;
    assert!(
        seg_len.is_power_of_two() && seg_len >= 2,
        "segment too short"
    );
    // The full execution (for the records) — the interpreter is
    // deterministic, so per-boundary state snapshots come from re-runs.
    let mut full = execute(program, input, total_len);
    assert_eq!(full.final_state.output, output, "execution output mismatch");
    // The interpreter pads to the next power of two of the record count;
    // the IVC schedule needs exactly `total_len` cycles.
    pad_records(
        &mut full.records,
        &mut full.final_state,
        &full.program,
        total_len,
    );
    assert_eq!(full.records.len(), total_len);
    let (pk, _vk) = keygen_with(program.to_vec(), seg_len, AMOUNT_BITS, pcs.clone());
    let a_rom = pk.rom.a_rom;

    let mut statements = Vec::with_capacity(segments);
    let mut inners = Vec::with_capacity(segments);
    let mut digests = Vec::with_capacity(segments);
    let mut tables: Vec<Vec<bool>> = Vec::new();
    let mut claims: Vec<MosaicClaim> = Vec::new();

    for s in 0..segments {
        let start = s * seg_len;
        let end = start + seg_len;
        // The boundary states, replayed from the padded full trace.
        let init = if s == 0 {
            Boundary::initial(input)
        } else {
            boundary_at(input, &full.records, start)
        };
        let final_ = boundary_at(input, &full.records, end);
        // The segment's records: the full trace's slice (already padded
        // to total_len, so the slice is full length).
        let records: Vec<_> = full.records[start..end].to_vec();
        // The interpreter state at the segment's end (for the trace's
        // final_state bookkeeping).
        let mut final_state = execute(program, input, end).final_state;
        pad_records(&mut Vec::new(), &mut final_state, &full.program, 0);
        let seg_trace = Trace {
            program: full.program.clone(),
            records,
            final_state,
        };
        let stmt = SegmentStatement {
            input: if s == 0 { input.to_vec() } else { Vec::new() },
            output,
            init: init.clone(),
            final_: final_.clone(),
        };
        // The per-segment public input binding: only the output matters
        // structurally (the inputs bind through the RAM's m0).
        let pub_in = PublicInput {
            input: Vec::new(),
            output,
        };
        // The segment's own transcript, seeded with the full statement.
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
        // The cross-step's last-cycle successor: the next segment's
        // initial pc, or the halt freeze on the last segment.
        let next_pc = if s + 1 < segments {
            Some(if end < full.records.len() {
                full.records[end].micro_pc % full.program.rows.len()
            } else {
                0
            })
        } else {
            None
        };
        let (inner, seg_claims) =
            prove_inner(&pk, &pub_in, Some(&stmt), &seg_trace, &wd, next_pc, &mut t);
        // Remap the claims into the combined layout: the per-segment
        // component lists are identical, so segment s's components start
        // at s · (component count).
        let shift = s * pk.mosaic.components.len();
        for mut c in seg_claims {
            c.component += shift;
            claims.push(c);
        }
        tables.extend(seg_tables);
        statements.push(stmt);
        inners.push(inner);
        digests.push(t.state_digest());
        let _ = a_rom;
    }

    // The root: one commitment over the union, one discharge.
    let combined = combined_layout(&pk, segments);
    let commitment = commit_mosaic(pcs, &combined, &tables);
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
    t.absorb_digest("cm-g", &commitment.stmt.root);
    let mosaic = prove_mosaic(pcs, &combined, &commitment, &claims, false, &mut t);
    IvcProof {
        statements,
        inners,
        segment_digests: digests,
        commitment_root: commitment.stmt.root,
        pcs_n: commitment.stmt.n,
        pcs_k: commitment.stmt.k,
        pcs_domain: commitment.stmt.domain,
        ell: commitment.ell,
        mosaic,
    }
}

/// Verify a tree-IVC proof: the boundary chain, every segment's steps
/// 2–6 replay, and the single shared Mosaic discharge.
pub fn verify_ivc(
    program: &[u32],
    input: &[Word],
    output: Word,
    total_len: usize,
    proof: &IvcProof,
    pcs: &PcsParams,
) -> Result<(), &'static str> {
    let segments = proof.statements.len();
    if !segments.is_power_of_two() || segments < 1 {
        return Err("ivc: segment count");
    }
    let seg_len = total_len / segments;
    if !seg_len.is_power_of_two() || seg_len < 2 {
        return Err("ivc: segment length");
    }
    let (pk, vk) = keygen_with(program.to_vec(), seg_len, AMOUNT_BITS, pcs.clone());
    // ---- The boundary chain.
    if proof.statements[0].init != Boundary::initial(input) {
        return Err("ivc: the first segment must start at the canonical state");
    }
    if proof.statements[0].input != input {
        return Err("ivc: segment 0 public input mismatch");
    }
    for st in proof.statements.iter() {
        if st.output != output {
            return Err("ivc: segment output claim mismatch");
        }
    }
    for w in proof.statements.windows(2) {
        if w[0].final_ != w[1].init {
            return Err("ivc: boundary chain broken");
        }
    }
    let last = proof.statements.last().expect("nonempty");
    if !last.final_.halted {
        return Err("ivc: the execution must halt within the trace");
    }
    if last.final_.output() != output {
        return Err("ivc: the halted boundary's output mismatch");
    }
    // ---- The segments.
    let mut tables_len = 0usize;
    let mut claims: Vec<MosaicClaim> = Vec::new();
    for (s, (stmt, inner)) in proof.statements.iter().zip(proof.inners.iter()).enumerate() {
        let mut t = Transcript::new(
            "magnetar-segment",
            &[
                b"tree-ivc-v1",
                &program
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<u8>>(),
                &(s as u64).to_le_bytes(),
                &stmt_bytes(stmt),
            ],
        );
        let pub_in = PublicInput {
            input: Vec::new(),
            output,
        };
        // The cross-step's public successor: the next statement's
        // initial pc (the halt freeze on the last segment).
        let next_pc = if s + 1 < segments {
            Some(proof.statements[s + 1].init.pc)
        } else {
            None
        };
        let seg_claims =
            verify_inner(&vk, &pub_in, Some(stmt), inner, next_pc, &mut t).map_err(|e| {
                eprintln!("[dbg] verify_inner seg {s}: {e}");
                e
            })?;
        if t.state_digest() != proof.segment_digests[s] {
            return Err("ivc: segment transcript digest mismatch");
        }
        let shift = s * vk.mosaic.components.len();
        for mut c in seg_claims {
            c.component += shift;
            claims.push(c);
        }
        tables_len += vk.mosaic.components.len();
    }
    let _ = tables_len;
    // ---- The shared discharge.
    let combined = combined_layout(&pk, segments);
    let cm_stmt = magnetar_pcs::PcsStatement {
        root: proof.commitment_root,
        n: proof.pcs_n,
        k: proof.pcs_k,
        domain: proof.pcs_domain.clone(),
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
    for (s, d) in proof.segment_digests.iter().enumerate() {
        t.absorb_digest(&format!("seg-digest-{s}"), d);
    }
    t.absorb_digest("cm-g", &proof.commitment_root);
    verify_mosaic(
        pcs,
        &cm_stmt,
        proof.ell,
        &combined,
        &claims,
        &proof.mosaic,
        &mut t,
    )
}

/// The combined Mosaic layout: `segments` copies of the per-segment
/// component list, tagged by the segment index (placed on disjoint
/// aligned subcubes by the alignment rule).
pub(crate) fn combined_layout(pk: &KeyGen, segments: usize) -> MosaicLayout {
    let n = pk.trace_len.trailing_zeros() as usize;
    let a_rom = pk.rom.a_rom;
    let mut comps = Vec::new();
    for s in 0..segments {
        for c in crate_mosaic_components(n, a_rom) {
            comps.push(magnetar_zkvm::mosaic::ComponentSpec {
                name: format!("s{s}-{}", c.name),
                vars: c.vars,
            });
        }
    }
    MosaicLayout::build(comps)
}

/// The per-segment component list (mirrors protocol::mosaic_components,
/// with the tag-free names — the combined layout renames).
fn crate_mosaic_components(n: usize, a_rom: usize) -> Vec<magnetar_zkvm::mosaic::ComponentSpec> {
    let seg = |name: &str, lv: usize| magnetar_zkvm::mosaic::ComponentSpec {
        name: name.to_string(),
        vars: lv + n,
    };
    vec![
        seg("seg-t1", 6),
        seg("seg-t2", 6),
        seg("seg-carry", 6),
        seg("seg-alu", 6),
        seg("seg-shift-src", 6),
        seg("seg-store-old", 6),
        seg("seg-store-src", 6),
        seg("seg-flags", 4),
        seg("seg-off", 3),
        seg("seg-ram-addr", 3),
        seg("seg-eqaux", 6),
        seg("reg-ra-1", 6),
        seg("reg-ra-2", 6),
        seg("reg-wfa", 6),
        seg("reg-inc", 6),
        seg("ram-ra-read", 8),
        seg("ram-ra-write", 8),
        seg("ram-inc", 6),
        seg("fetch-ra", a_rom),
    ]
}

/// The canonical byte encoding of a segment statement (transcript seed).
pub(crate) fn stmt_bytes(st: &SegmentStatement) -> Vec<u8> {
    let mut v = Vec::new();
    for b in [&st.init, &st.final_] {
        for w in b.regs.iter().chain(b.ram.iter()) {
            v.extend_from_slice(&w.to_le_bytes());
        }
        v.extend_from_slice(&(b.pc as u64).to_le_bytes());
        v.extend_from_slice(&(b.halted as u64).to_le_bytes());
    }
    v.extend_from_slice(&st.output.to_le_bytes());
    v
}
