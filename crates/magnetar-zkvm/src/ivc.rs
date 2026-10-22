//! # Tree-IVC: chunked proving with state continuity and claim-batching
//!
//! The roadmap's IVC architecture decision (§8.2 of the accompanying
//! report): for a hash-only system the workable schedules are sequential
//! IVC, binary-tree IVC, and aggregation-then-wrap; **the correct default
//! for Magnetar is tree-IVC with claim-batching at each internal node**,
//! because both mechanisms reuse the MosaicPCS machinery the system
//! already has, and because tree schedules degrade gracefully when chunks
//! arrive out of order — the property distributed proving needs.
//!
//! This module implements that design on the foundation the single-chunk
//! protocol already provides — *segments share one commitment* — by
//! generalizing the sharing one level up: **chunks share one commitment**.
//!
//! ## The design (three layers)
//!
//! **1. Chunking.** A long execution is cut into `k` consecutive chunks of
//! `2^n` cycles each, cut only at RV64I instruction boundaries (never
//! inside a register-shift stage sequence). Every chunk is proven through
//! the shared steps 1–4 core (Virtual Spartan, both Twists with their
//! Ψ-sum-checks, the increment claim, both routers, the bytecode Shout and
//! the selector checks) on one coordinator-ordered transcript.
//!
//! **2. Boundary continuity.** The machine state crosses a chunk boundary
//! through public data plus committed images, each bound by its own
//! argument:
//!
//! * **Registers** — the entry and exit register files (the 32
//!   architectural words plus the trash cell that x0-redirected writes
//!   target) are published per chunk. A **final-state identity
//!   sum-check** binds `exit = entry + Σ_j wfa(k,j)·inc(o,j)` — the
//!   committed write log — to the published tables, so a forged boundary
//!   cannot be consistent with the committed log. The entry of chunk
//!   `c+1` must equal the exit of chunk `c` (a public check).
//! * **Micro-PC** — published per chunk; the fetch one-hot component is
//!   claimed at the first and last cycle's Boolean points
//!   (`e₀ = 0…0`, `e_last = 1…1`), where its value is exactly the
//!   eq-packing of the published entry/exit micro-PC. A forged PC fails
//!   the committed-component discharge.
//! * **RAM** — the large state, **not** published. Each chunk commits two
//!   image components, `ram-init` (the entry state) and `ram-final` (the
//!   exit state), as bit tables over `(k, o)`. The same final-state
//!   identity binds `ram-final = ram-init + Σ wfa·inc` inside the chunk;
//!   chunk 0's identity pins `ram-init` to the **public input image**;
//!   and each boundary adds a **continuity check**: at a random point
//!   `r_b`, chunk `c`'s `ram-final` and chunk `c+1`'s `ram-init` are
//!   claimed with values summing to zero (characteristic 2). The RAM
//!   Twist of an interior chunk runs in *claimed-M₀* mode: its
//!   reconstruction claims `M̂₀(r_k, r_v)` are transmitted and discharged
//!   against the `ram-init` component, so every read is explained by the
//!   chained image.
//!
//! **3. Claim-batching — the tree node operation.** The per-chunk Mosaic
//! component lists concatenate into **one merged layout** over a single
//! cube (the same aligned-subcube machinery that packs segments), and the
//! `k` chunks' components are committed in **one** BLAKE3 Merkle tree.
//! Every claim of every chunk — the step 1–4 discharges, the identity
//! terminals, the PC claims, the continuity claims — lifts into the merged
//! cube and discharges through **one** MosaicPCS evaluation argument
//! (batching sum-check + ring switch + query set). The evaluation layer —
//! 84% of a chunk proof's bytes in the measured cost model — is therefore
//! `O(1)` in `k` instead of `O(k)`.
//!
//! ## Soundness sketch
//!
//! * Every chunk's memory reads are bound (by its Twist) to its committed
//!   entry image; the identity binds entry→exit through the committed
//!   write log; chunk 0's identity binds the entry image to the public
//!   input; the continuity checks chain the images across boundaries at
//!   random points (soundness `≤ 14/2^128` per boundary, multilinear
//!   Schwartz–Zippel). By induction every read is explained by the true
//!   accumulated state and every published boundary equals the committed
//!   images' evolution.
//! * A false entry image at cells a chunk never accesses is inert: the
//!   reads it must explain are checked, and the exit it must produce is
//!   checked, and nothing else consumes it.
//! * The register and PC boundaries are fully public and each is bound to
//!   the committed log (registers) or the committed fetch component (PC).
//! * The merged commitment is absorbed before any challenge; the
//!   coordinator-ordered transcript makes the Fiat–Shamir composition
//!   standard (the coordinator-star topology of §8.3 — hash-only, no
//!   multiset hash).
//!
//! ## The tree schedule
//!
//! The claim-batching operation is associative: merging chunk sets is the
//! same component concatenation, so a distributed schedule can aggregate
//! subtrees as data arrives and the root runs the single step-5 — the
//! binary-tree IVC schedule (the CycleFold-popularized shape) with
//! `log k` depth and out-of-order tolerance. The Fiat–Shamir constraint
//! (§8.3) is that challenges derive from the ordered transcript, so the
//! shards produce partial round messages and a single coordinator
//! serializes them — the star topology, with the Merkle-transcript as the
//! documented scale-up path. The verifier-guest wrap that would collapse
//! the per-chunk step 1–4 transcripts into one final proof is the next
//! milestone (H2); this module's aggregate is exactly the artifact that
//! wrap will verify.

use crate::mosaic::{
    commit_mosaic, prove_mosaic, verify_mosaic, ComponentSpec, MosaicClaim, MosaicLayout,
    MosaicProof,
};
use crate::protocol::{
    build_witness, prove_segment_body, ram_image_bits, regs_image_bits, ChunkCtx, KeyGen,
    PublicInput, SegmentProof, VerifyingKey,
};
use magnetar_cpu::expand::expand_program;
use magnetar_cpu::{execute_chunk, ExecState, RAM_WORDS_LOG2, REGS};
use magnetar_field::{eval_mle, EqBits, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_router::PaddedTableFactor;
use magnetar_sumcheck::engine::SumcheckVerifier;
use magnetar_sumcheck::sparse::{SparseFactor, SparseProver, SparseSpec};
use magnetar_sumcheck::{BindOrder, Term};

/// The published boundary words: `x0..x31` plus the trash cell (word 32).
/// x0 is always zero (the shift-scratch slot is not part of the
/// architectural state and never crosses a boundary); cells 33..64 are
/// never written and stay zero.
pub const BOUNDARY_WORDS: usize = if REGS < 33 { REGS } else { 33 };

/// The cycle-budget margin: a chunk runs at most
/// `trace_len − CHUNK_MARGIN` real cycles before looking for a safe cut,
/// so the instruction-boundary overrun (bounded by the longest expansion,
/// the 6-stage register shifts) never exceeds the padding target.
pub const CHUNK_MARGIN: usize = 8;

/// The public boundary state of one chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryState {
    /// The register file at chunk entry: `[x0..x31, trash]`.
    pub entry_regs: [u64; BOUNDARY_WORDS],
    /// The register file at chunk exit: `[x0..x31, trash]`.
    pub exit_regs: [u64; BOUNDARY_WORDS],
    /// The micro-PC at chunk entry (virtual-instruction units).
    pub entry_pc: u64,
    /// The micro-PC at chunk exit (the frozen PC the padding carries).
    pub exit_pc: u64,
}

/// The aggregate statement: the public data every boundary threads.
#[derive(Clone, Debug)]
pub struct AggregatePublic {
    /// Chunk 0's RAM image (the public input words, addresses `0..`).
    pub input: Vec<u64>,
    /// The claimed output (x10 at the final chunk's halt).
    pub output: u64,
    /// One boundary per chunk, in chunk order.
    pub boundaries: Vec<BoundaryState>,
}

/// The final-state identity sum-check shared by the RAM and the register
/// file:
/// `Σ_{k,o,j} eq(r_k,k)·eq(r_v,o)·wfa(k,j)·inc(o,j) = claim`
/// with `claim = final(r) + init(r)` over the two images.
#[derive(Clone, Debug)]
pub struct StateIdentityProof {
    /// Round messages (degree 4: two dense × two public factors).
    pub rounds: Vec<Vec<Gf128>>,
    /// The write one-hot component's terminal claim.
    pub wfa_claim: Gf128,
    /// The increment component's terminal claim.
    pub inc_claim: Gf128,
}

/// One chunk's proof: the shared steps 1–4 core plus the two boundary
/// identities.
#[derive(Clone, Debug)]
pub struct ChunkProof {
    pub core: SegmentProof,
    pub ram_state: StateIdentityProof,
    pub reg_state: StateIdentityProof,
}

/// The tree-IVC aggregate: `k` chunk cores, one shared commitment, one
/// batched MosaicPCS discharge.
#[derive(Clone, Debug)]
pub struct AggregateProof {
    pub chunks: Vec<ChunkProof>,
    pub commitment_root: [u8; 32],
    pub pcs_n: usize,
    pub pcs_k: usize,
    pub pcs_domain: String,
    /// The merged cube's variable count.
    pub ell: usize,
    /// Per chunk: the transmitted `(ram-final, ram-init)` values at the
    /// identity's random point.
    pub ram_state_values: Vec<[Gf128; 2]>,
    /// Per boundary: `(chunk b's ram-final, chunk b+1's ram-init)` at the
    /// continuity point — must sum to zero.
    pub cont_values: Vec<[Gf128; 2]>,
    /// The single batched evaluation argument.
    pub mosaic: MosaicProof,
}

// ---------------------------------------------------------------------------
// The chunker.
// ---------------------------------------------------------------------------

/// One chunk's execution products.
struct ChunkExec {
    trace: magnetar_cpu::Trace,
    boundary: BoundaryState,
    ram_entry: Vec<u64>,
    ram_exit: Vec<u64>,
}

/// Execute the program in tree-IVC chunks of `trace_len` cycles until the
/// machine halts.
fn run_chunks(
    program: &[u32],
    input: &[u64],
    trace_len: usize,
    max_chunks: usize,
) -> Vec<ChunkExec> {
    let mut state = ExecState {
        regs: [0; REGS],
        ram: vec![0; 1usize << RAM_WORDS_LOG2],
        micro_pc: 0,
        cycles: 0,
        halted: false,
        output: 0,
    };
    for (i, &w) in input.iter().enumerate() {
        if i < state.ram.len() {
            state.ram[i] = w;
        }
    }
    let budget = trace_len - CHUNK_MARGIN.min(trace_len / 2);
    let mut chunks: Vec<ChunkExec> = Vec::new();
    loop {
        let entry = state.clone();
        let trace = execute_chunk(program, entry.clone(), budget, trace_len);
        let mut entry_regs = [0u64; BOUNDARY_WORDS];
        let mut exit_regs = [0u64; BOUNDARY_WORDS];
        entry_regs[1..].copy_from_slice(&entry.regs[1..BOUNDARY_WORDS]);
        exit_regs[1..].copy_from_slice(&trace.final_state.regs[1..BOUNDARY_WORDS]);
        // x0 is the log image (zero); the shift scratch never crosses.
        let boundary = BoundaryState {
            entry_regs,
            exit_regs,
            entry_pc: entry.micro_pc as u64,
            exit_pc: trace.records.last().map(|r| r.next_pc).unwrap_or(0) as u64,
        };
        let halted = trace.final_state.halted;
        let ram_exit = trace.final_state.ram.clone();
        chunks.push(ChunkExec {
            trace,
            boundary,
            ram_entry: entry.ram.clone(),
            ram_exit,
        });
        if halted {
            break;
        }
        // The next entry: the architectural state carries; the scratch
        // slot resets; the counters restart.
        let mut next = chunks.last().unwrap().trace.final_state.clone();
        next.regs[0] = 0;
        next.cycles = 0;
        state = next;
        assert!(
            chunks.len() < max_chunks,
            "the program did not halt within max_chunks"
        );
    }
    chunks
}

// ---------------------------------------------------------------------------
// The final-state identity sum-check.
// ---------------------------------------------------------------------------

/// Prove the identity `Σ eq(r_k,k)·eq(r_v,o)·wfa(k,j)·inc(o,j) = claim`
/// over the joint `(k, o, j)` space with the sparse prover. Returns the
/// proof and the challenges (for the terminal claim points).
#[allow(clippy::too_many_arguments)]
fn prove_state_identity(
    a: usize,
    ell: usize,
    n: usize,
    wfa: &[Gf128],
    inc: &[Gf128],
    r_k: &[Gf128],
    r_v: &[Gf128],
    claim: Gf128,
    label: &str,
    t: &mut Transcript,
) -> (StateIdentityProof, Vec<Gf128>) {
    let jv = a + ell + n;
    let eq_k = EqBits::build(r_k).values;
    let eq_v = EqBits::build(r_v).values;
    let active_wfa: Vec<usize> = (0..a).chain(a + ell..jv).collect();
    let active_inc: Vec<usize> = (a..a + ell).chain(a + ell..jv).collect();
    let spec = SparseSpec {
        num_vars: jv,
        dense: vec![
            SparseFactor::new(wfa.to_vec(), active_wfa),
            SparseFactor::new(inc.to_vec(), active_inc),
        ],
        public: vec![
            SparseFactor::new(eq_k, (0..a).collect()),
            SparseFactor::new(eq_v, (a..a + ell).collect()),
        ],
        terms: vec![Term {
            coeff: ONE,
            dense: vec![0, 1],
            public: vec![0, 1],
        }],
        claim,
    };
    let mut p = SparseProver::new(spec, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(jv);
    for _ in 0..jv {
        let msg = p.message();
        rounds.push(msg.0.clone());
        p.absorb_and_fold(t, label);
    }
    let challenges = p.challenges().to_vec();
    let (claims, _) = p.finish();
    (
        StateIdentityProof {
            rounds,
            wfa_claim: claims[0],
            inc_claim: claims[1],
        },
        challenges,
    )
}

/// Verify the identity sum-check. Returns the challenges (for the terminal
/// claim points). The dense tables are never needed — only the public eq
/// factors and the transmitted terminals.
#[allow(clippy::too_many_arguments)]
fn verify_state_identity(
    a: usize,
    ell: usize,
    n: usize,
    r_k: &[Gf128],
    r_v: &[Gf128],
    claim: Gf128,
    proof: &StateIdentityProof,
    label: &str,
    t: &mut Transcript,
) -> Result<Vec<Gf128>, &'static str> {
    let jv = a + ell + n;
    if proof.rounds.len() != jv {
        return Err("identity: round count mismatch");
    }
    let eq_k = EqBits::build(r_k).values;
    let eq_v = EqBits::build(r_v).values;
    let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> = vec![
        Box::new(PaddedTableFactor::new(eq_k, (0..a).collect(), jv)),
        Box::new(PaddedTableFactor::new(eq_v, (a..a + ell).collect(), jv)),
    ];
    let terms = vec![Term {
        coeff: ONE,
        dense: vec![0, 1],
        public: vec![0, 1],
    }];
    let mut v = SumcheckVerifier::new(jv, claim, &public, terms, BindOrder::Increasing);
    let mut challenges = Vec::with_capacity(jv);
    for msg in &proof.rounds {
        challenges.push(v.round_capturing(&magnetar_sumcheck::RoundMsg(msg.clone()), t, label)?);
    }
    v.finish(&[proof.wfa_claim, proof.inc_claim])
        .map_err(|_| "identity: terminal failed")?;
    Ok(challenges)
}

/// The terminal claim points of the identity's two dense factors, in their
/// components' local coordinates: the write one-hot at `(r_k ‖ r_j)` and
/// the increment at `(r_v ‖ r_j)`.
fn identity_claim_points(
    a: usize,
    ell: usize,
    n: usize,
    challenges: &[Gf128],
) -> (Vec<Gf128>, Vec<Gf128>) {
    let jv = a + ell + n;
    let r_j = &challenges[a + ell..jv];
    let wfa_pt: Vec<Gf128> = [&challenges[..a], r_j].concat();
    let inc_pt: Vec<Gf128> = [&challenges[a..a + ell], r_j].concat();
    (wfa_pt, inc_pt)
}

/// The eq-packing of a micro-PC inside the fetch component's address space.
fn pc_packing(pc: u64, r: &[Gf128], rows_len: usize) -> Gf128 {
    EqBits::build(r).values[(pc as usize) % rows_len]
}

// ---------------------------------------------------------------------------
// The merged layout.
// ---------------------------------------------------------------------------

/// The `k`-chunk merged Mosaic layout: the per-chunk component lists
/// concatenated (name-prefixed) and packed onto one cube by the standard
/// aligned-subcube builder. Claim component `i` of chunk `c` is merged
/// component `c·n_comp + i`.
fn merged_layout(per: &MosaicLayout, k: usize) -> MosaicLayout {
    let comps: Vec<ComponentSpec> = (0..k)
        .flat_map(|c| {
            per.components.iter().map(move |s| ComponentSpec {
                name: format!("c{c}/{}", s.name),
                vars: s.vars,
            })
        })
        .collect();
    MosaicLayout::build(comps)
}

/// Re-index per-chunk claims into the merged layout.
fn reindex_claims(claims: Vec<MosaicClaim>, chunk: usize, n_comp: usize) -> Vec<MosaicClaim> {
    claims
        .into_iter()
        .map(|mut c| {
            c.component += chunk * n_comp;
            c
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The aggregate prove.
// ---------------------------------------------------------------------------

/// The boundary bytes for the transcript binding.
fn boundary_bytes(public: &AggregatePublic) -> Vec<u8> {
    let mut b = Vec::new();
    for bd in &public.boundaries {
        for w in bd.entry_regs.iter().chain(bd.exit_regs.iter()) {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b.extend_from_slice(&bd.entry_pc.to_le_bytes());
        b.extend_from_slice(&bd.exit_pc.to_le_bytes());
    }
    b
}

/// Prove a program in tree-IVC chunks under a chunked keygen. Returns the
/// aggregate proof and the derived public boundary data.
///
/// `max_chunks` bounds the chunk count (the chunker panics if the program
/// does not halt within it).
pub fn prove_chunked(
    pk: &KeyGen,
    pub_in: &PublicInput,
    max_chunks: usize,
) -> (AggregateProof, AggregatePublic) {
    assert!(
        pk.mosaic.index_of("ram-init").is_some(),
        "prove_chunked requires the chunked keygen"
    );
    let trace_len = pk.trace_len;
    let n = trace_len.trailing_zeros() as usize;
    let a_rom = pk.rom.a_rom;
    let rows_len = expand_program(&pk.program).rows.len();
    let n_comp = pk.mosaic.components.len();
    let idx = |name: &str| pk.mosaic.index_of(name).expect("component");

    // ---- 1. Execute the chunks and build every witness.
    let chunks = run_chunks(&pk.program, &pub_in.input, trace_len, max_chunks);
    let k = chunks.len();
    let last = k - 1;
    assert!(
        chunks[last].trace.final_state.halted,
        "the final chunk must halt"
    );
    assert_eq!(
        chunks[last].trace.final_state.output, pub_in.output,
        "execution output mismatch"
    );
    struct ChunkData {
        boundary: BoundaryState,
        ram_init: Vec<Gf128>,
        ram_final: Vec<Gf128>,
        wd: crate::protocol::WitnessData,
    }
    let mut data: Vec<ChunkData> = Vec::with_capacity(k);
    let mut tables: Vec<Vec<bool>> = Vec::with_capacity(k * n_comp);
    for (c, ch) in chunks.iter().enumerate() {
        let ctx = ChunkCtx {
            output: if c == last { pub_in.output } else { 0 },
            input_words: if c == 0 { &pub_in.input } else { &[] },
            ram_words: if c == 0 { &pub_in.input } else { &ch.ram_entry },
            ram_m0_public: false,
            regs_m0: regs_image_bits(&ch.boundary.entry_regs),
        };
        let wd = build_witness(pk, &ctx, &ch.trace);
        // The component tables in Mosaic order, then the two images.
        let mut ct: Vec<Vec<bool>> = wd.direct_tables.clone();
        for comp in wd.reg_wit.ports.iter().flat_map(|p| p.iter()) {
            ct.push(comp.table.iter().map(|&v| v == ONE).collect());
        }
        ct.push(wd.reg_wit.inc.iter().map(|&v| v == ONE).collect());
        for comp in wd.ram_wit.ports.iter().flat_map(|p| p.iter()) {
            ct.push(comp.table.iter().map(|&v| v == ONE).collect());
        }
        ct.push(wd.ram_wit.inc.iter().map(|&v| v == ONE).collect());
        ct.push(wd.fetch_comp.table.iter().map(|&v| v == ONE).collect());
        let ram_init = ram_image_bits(&ch.ram_entry);
        let ram_final = ram_image_bits(&ch.ram_exit);
        ct.push(ram_init.iter().map(|&v| v == ONE).collect());
        ct.push(ram_final.iter().map(|&v| v == ONE).collect());
        tables.extend(ct);
        data.push(ChunkData {
            boundary: ch.boundary.clone(),
            ram_init,
            ram_final,
            wd,
        });
    }

    // ---- 2. The merged commitment (the single shared tree).
    let merged = merged_layout(&pk.mosaic, k);
    let commitment = commit_mosaic(&pk.pcs, &merged, &tables);

    // ---- 3. The coordinator-ordered transcript.
    let boundaries: Vec<BoundaryState> = data.iter().map(|d| d.boundary.clone()).collect();
    let public = AggregatePublic {
        input: pub_in.input.clone(),
        output: pub_in.output,
        boundaries,
    };
    let mut t = Transcript::new(
        "magnetar-ivc",
        &[
            b"tree-ivc-v1",
            &pk.program
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            &pub_in
                .input
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            &pub_in.output.to_le_bytes(),
            &boundary_bytes(&public),
        ],
    );
    t.absorb_digest("cm-g", &commitment.stmt.root);

    // ---- 4. Per chunk: the shared core + the boundary identities.
    let mut all_claims: Vec<MosaicClaim> = Vec::new();
    let mut chunk_proofs: Vec<ChunkProof> = Vec::with_capacity(k);
    let mut ram_state_values: Vec<[Gf128; 2]> = Vec::with_capacity(k);
    let e_first: Vec<Gf128> = vec![ZERO; n];
    let e_last: Vec<Gf128> = vec![ONE; n];
    for (c, d) in data.iter().enumerate() {
        let ram_words: &[u64] = if c == 0 {
            &pub_in.input
        } else {
            &chunks[c].ram_entry
        };
        let ctx = ChunkCtx {
            output: if c == last { pub_in.output } else { 0 },
            input_words: if c == 0 { &pub_in.input } else { &[] },
            ram_words,
            ram_m0_public: false,
            regs_m0: regs_image_bits(&d.boundary.entry_regs),
        };
        let (core, out) = prove_segment_body(pk, &ctx, &d.wd, &chunks[c].trace, &mut t);

        // (a) The RAM final-state identity.
        let r_state = t.challenge_field_vec(&format!("ram-state-point-{c}"), 14);
        let final_val = eval_mle(&d.ram_final, &r_state);
        let init_val = eval_mle(&d.ram_init, &r_state);
        t.absorb_field(&format!("ram-state-claims-{c}"), &[final_val, init_val]);
        let (ram_id, ram_chs) = prove_state_identity(
            8,
            6,
            n,
            &d.wd.ram_wit.ports[1][0].table,
            &d.wd.ram_wit.inc,
            &r_state[..8],
            &r_state[8..],
            final_val + init_val,
            &format!("ram-state-{c}"),
            &mut t,
        );
        ram_state_values.push([final_val, init_val]);

        // (b) The register final-state identity (both ends public).
        let r_regs = t.challenge_field_vec(&format!("reg-state-point-{c}"), 12);
        let entry_img = regs_image_bits(&d.boundary.entry_regs);
        let exit_img = regs_image_bits(&d.boundary.exit_regs);
        let reg_claim = eval_mle(&exit_img, &r_regs) + eval_mle(&entry_img, &r_regs);
        let (reg_id, reg_chs) = prove_state_identity(
            6,
            6,
            n,
            &d.wd.reg_wit.ports[2][0].table,
            &d.wd.reg_wit.inc,
            &r_regs[..6],
            &r_regs[6..],
            reg_claim,
            &format!("reg-state-{c}"),
            &mut t,
        );

        // (c) The PC entry/exit points (claim values are the public
        // eq-packings — no transmission).
        let r_pin = t.challenge_field_vec(&format!("pc-point-in-{c}"), a_rom);
        let r_pout = t.challenge_field_vec(&format!("pc-point-out-{c}"), a_rom);

        // The body's local claims (steps 3-6, collected inside).
        let mut claims = out;
        claims.push(MosaicClaim::new(
            idx("ram-final"),
            r_state.clone(),
            final_val,
        ));
        claims.push(MosaicClaim::new(idx("ram-init"), r_state, init_val));
        let (rwfa_pt, rinc_pt) = identity_claim_points(8, 6, n, &ram_chs);
        claims.push(MosaicClaim::new(
            idx("ram-ra-write"),
            rwfa_pt,
            ram_id.wfa_claim,
        ));
        claims.push(MosaicClaim::new(idx("ram-inc"), rinc_pt, ram_id.inc_claim));
        let (gwfa_pt, ginc_pt) = identity_claim_points(6, 6, n, &reg_chs);
        claims.push(MosaicClaim::new(idx("reg-wfa"), gwfa_pt, reg_id.wfa_claim));
        claims.push(MosaicClaim::new(idx("reg-inc"), ginc_pt, reg_id.inc_claim));
        let mut pt_in = r_pin.clone();
        pt_in.extend_from_slice(&e_first);
        claims.push(MosaicClaim::new(
            idx("fetch-ra"),
            pt_in,
            pc_packing(d.boundary.entry_pc, &r_pin, rows_len),
        ));
        let mut pt_out = r_pout.clone();
        pt_out.extend_from_slice(&e_last);
        claims.push(MosaicClaim::new(
            idx("fetch-ra"),
            pt_out,
            pc_packing(d.boundary.exit_pc, &r_pout, rows_len),
        ));
        all_claims.extend(reindex_claims(claims, c, n_comp));
        chunk_proofs.push(ChunkProof {
            core,
            ram_state: ram_id,
            reg_state: reg_id,
        });
    }

    // ---- 5. The continuity checks (per boundary).
    let mut cont_values: Vec<[Gf128; 2]> = Vec::with_capacity(k - 1);
    for b in 0..k - 1 {
        let r_b = t.challenge_field_vec(&format!("cont-point-{b}"), 14);
        let v_final = eval_mle(&data[b].ram_final, &r_b);
        let v_init = eval_mle(&data[b + 1].ram_init, &r_b);
        t.absorb_field(&format!("cont-claims-{b}"), &[v_final, v_init]);
        assert_eq!(
            v_final + v_init,
            ZERO,
            "the RAM images must chain across the boundary"
        );
        // Chunk b's final image and chunk b+1's entry image, at the same
        // random point, with values summing to zero.
        all_claims.push(MosaicClaim::new(
            b * n_comp + idx("ram-final"),
            r_b.clone(),
            v_final,
        ));
        all_claims.push(MosaicClaim::new(
            (b + 1) * n_comp + idx("ram-init"),
            r_b,
            v_init,
        ));
        cont_values.push([v_final, v_init]);
    }

    // ---- 6. The single batched MosaicPCS discharge.
    let mosaic = prove_mosaic(&pk.pcs, &merged, &commitment, &all_claims, &mut t);
    let proof = AggregateProof {
        chunks: chunk_proofs,
        commitment_root: commitment.stmt.root,
        pcs_n: commitment.stmt.n,
        pcs_k: commitment.stmt.k,
        pcs_domain: commitment.stmt.domain,
        ell: commitment.ell,
        ram_state_values,
        cont_values,
        mosaic,
    };
    (proof, public)
}

// ---------------------------------------------------------------------------
// The aggregate verify.
// ---------------------------------------------------------------------------

/// Verify a tree-IVC aggregate. The statement is `(pub_in, public)`: the
/// public input image and output claim, plus the per-chunk boundary data.
/// The proof must be consistent with all of it — every boundary is bound
/// to the committed write logs, the RAM images chain from the public
/// input through every continuity check, and the single batched MosaicPCS
/// argument discharges every claim against the merged commitment.
pub fn verify_chunked(
    vk: &VerifyingKey,
    pub_in: &PublicInput,
    public: &AggregatePublic,
    proof: &AggregateProof,
) -> Result<(), &'static str> {
    assert!(
        vk.mosaic.index_of("ram-init").is_some(),
        "verify_chunked requires the chunked keygen"
    );
    let k = proof.chunks.len();
    let n = vk.trace_len.trailing_zeros() as usize;
    let a_rom = vk.rom.a_rom;
    let rows_len = expand_program(&vk.program).rows.len();
    let n_comp = vk.mosaic.components.len();
    let idx = |name: &str| vk.mosaic.index_of(name).expect("component");

    // ---- 0. The boundary conventions and chaining.
    if public.boundaries.len() != k
        || proof.ram_state_values.len() != k
        || proof.cont_values.len() != k.saturating_sub(1)
    {
        return Err("aggregate: shape mismatch");
    }
    let zero_words = [0u64; BOUNDARY_WORDS];
    if public.boundaries[0].entry_regs != zero_words || public.boundaries[0].entry_pc != 0 {
        return Err("boundary: chunk 0 must start at the conventional state");
    }
    for b in 0..k - 1 {
        let (a, c) = (&public.boundaries[b], &public.boundaries[b + 1]);
        if a.exit_regs != c.entry_regs || a.exit_pc != c.entry_pc {
            return Err("boundary: the chunks do not chain");
        }
    }

    // ---- 1. The merged layout and statement.
    let merged = merged_layout(&vk.mosaic, k);
    let cm_stmt = magnetar_pcs::PcsStatement {
        root: proof.commitment_root,
        n: proof.pcs_n,
        k: proof.pcs_k,
        domain: proof.pcs_domain.clone(),
    };

    // ---- 2. The coordinator-ordered transcript.
    let mut t = Transcript::new(
        "magnetar-ivc",
        &[
            b"tree-ivc-v1",
            &vk.program
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            &pub_in
                .input
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            &pub_in.output.to_le_bytes(),
            &boundary_bytes(public),
        ],
    );
    t.absorb_digest("cm-g", &proof.commitment_root);

    // ---- 3. Per chunk: the shared core + the boundary identities.
    let mut all_claims: Vec<MosaicClaim> = Vec::new();
    let e_first: Vec<Gf128> = vec![ZERO; n];
    let e_last: Vec<Gf128> = vec![ONE; n];
    let public_init_bits = ram_image_bits(&pub_in.input);
    for c in 0..k {
        let bd = &public.boundaries[c];
        // The verifier does not know the interior entry images — the Twist
        // runs in claimed-M0 mode and the (unused) table argument is empty.
        let ctx = ChunkCtx {
            output: if c == k - 1 { pub_in.output } else { 0 },
            input_words: if c == 0 { &pub_in.input } else { &[] },
            ram_words: &[],
            ram_m0_public: false,
            regs_m0: regs_image_bits(&bd.entry_regs),
        };
        let out = crate::protocol::verify_segment_body(vk, &ctx, &proof.chunks[c].core, &mut t)?;

        // (a) The RAM final-state identity.
        let r_state = t.challenge_field_vec(&format!("ram-state-point-{c}"), 14);
        let [final_val, init_val] = proof.ram_state_values[c];
        t.absorb_field(&format!("ram-state-claims-{c}"), &[final_val, init_val]);
        if c == 0 {
            // Chunk 0's entry image is the public input image.
            if init_val != eval_mle(&public_init_bits, &r_state) {
                return Err("ram-state: chunk 0 entry image is not the public input");
            }
        }
        let ram_chs = verify_state_identity(
            8,
            6,
            n,
            &r_state[..8],
            &r_state[8..],
            final_val + init_val,
            &proof.chunks[c].ram_state,
            &format!("ram-state-{c}"),
            &mut t,
        )?;

        // (b) The register final-state identity (both ends public).
        let r_regs = t.challenge_field_vec(&format!("reg-state-point-{c}"), 12);
        let entry_img = regs_image_bits(&bd.entry_regs);
        let exit_img = regs_image_bits(&bd.exit_regs);
        let reg_claim = eval_mle(&exit_img, &r_regs) + eval_mle(&entry_img, &r_regs);
        let reg_chs = verify_state_identity(
            6,
            6,
            n,
            &r_regs[..6],
            &r_regs[6..],
            reg_claim,
            &proof.chunks[c].reg_state,
            &format!("reg-state-{c}"),
            &mut t,
        )?;

        // (c) The PC entry/exit points; the claim values are the public
        // eq-packings of the published micro-PCs.
        let r_pin = t.challenge_field_vec(&format!("pc-point-in-{c}"), a_rom);
        let r_pout = t.challenge_field_vec(&format!("pc-point-out-{c}"), a_rom);

        // The body's local claims (steps 3-6, verified and collected inside).
        let mut claims = out;
        claims.push(MosaicClaim::new(
            idx("ram-final"),
            r_state.clone(),
            final_val,
        ));
        claims.push(MosaicClaim::new(idx("ram-init"), r_state, init_val));
        let (rwfa_pt, rinc_pt) = identity_claim_points(8, 6, n, &ram_chs);
        claims.push(MosaicClaim::new(
            idx("ram-ra-write"),
            rwfa_pt,
            proof.chunks[c].ram_state.wfa_claim,
        ));
        claims.push(MosaicClaim::new(
            idx("ram-inc"),
            rinc_pt,
            proof.chunks[c].ram_state.inc_claim,
        ));
        let (gwfa_pt, ginc_pt) = identity_claim_points(6, 6, n, &reg_chs);
        claims.push(MosaicClaim::new(
            idx("reg-wfa"),
            gwfa_pt,
            proof.chunks[c].reg_state.wfa_claim,
        ));
        claims.push(MosaicClaim::new(
            idx("reg-inc"),
            ginc_pt,
            proof.chunks[c].reg_state.inc_claim,
        ));
        let mut pt_in = r_pin.clone();
        pt_in.extend_from_slice(&e_first);
        claims.push(MosaicClaim::new(
            idx("fetch-ra"),
            pt_in,
            pc_packing(bd.entry_pc, &r_pin, rows_len),
        ));
        let mut pt_out = r_pout.clone();
        pt_out.extend_from_slice(&e_last);
        claims.push(MosaicClaim::new(
            idx("fetch-ra"),
            pt_out,
            pc_packing(bd.exit_pc, &r_pout, rows_len),
        ));
        all_claims.extend(reindex_claims(claims, c, n_comp));
    }

    // ---- 4. The continuity checks.
    for b in 0..k - 1 {
        let r_b = t.challenge_field_vec(&format!("cont-point-{b}"), 14);
        let [v_final, v_init] = proof.cont_values[b];
        t.absorb_field(&format!("cont-claims-{b}"), &[v_final, v_init]);
        if v_final + v_init != ZERO {
            return Err("continuity: the RAM images do not chain");
        }
        all_claims.push(MosaicClaim::new(
            b * n_comp + idx("ram-final"),
            r_b.clone(),
            v_final,
        ));
        all_claims.push(MosaicClaim::new(
            (b + 1) * n_comp + idx("ram-init"),
            r_b,
            v_init,
        ));
    }

    // ---- 5. The single batched MosaicPCS discharge.
    verify_mosaic(
        &vk.pcs,
        &cm_stmt,
        proof.ell,
        &merged,
        &all_claims,
        &proof.mosaic,
        &mut t,
    )
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::keygen_chunked;

    fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
        ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
    }

    fn enc_s(imm: u32, rs2: u32, rs1: u32, funct3: u32, op: u32) -> u32 {
        (((imm >> 5) & 0x7f) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | ((imm & 0x1f) << 7)
            | op
    }

    fn enc_r(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
    }

    /// A straight-line program whose RAM store lands in chunk 0 and whose
    /// load lands in chunk 1 (the value only exists through the carried
    /// image), with the register file carrying the loop-free accumulator:
    /// `x10 = 20 + 0x5A = 110` at halt.
    fn spanning_program() -> Vec<u32> {
        let mut p = vec![
            enc_i(40, 0, 0, 1, 0x13),    // x1 = 40 (RAM word 5)
            enc_i(0x5A, 0, 0, 10, 0x13), // x10 = 0x5A
            enc_s(0, 10, 1, 3, 0x23),    // sd x10, 0(x1)   — chunk 0
            enc_i(0, 0, 0, 10, 0x13),    // x10 = 0
        ];
        for _ in 0..20 {
            p.push(enc_i(1, 10, 0, 10, 0x13)); // addi x10, x10, 1
        }
        p.push(enc_i(0, 1, 3, 11, 0x03)); // ld x11, 0(x1) — chunk 1
        p.push(enc_r(0, 11, 10, 0, 10, 0x33)); // add x10, x10, x11
        p.push(0x0000_0073); // ecall
        p
    }

    fn prove_spanning(
        trace_len: usize,
        max_chunks: usize,
    ) -> (AggregatePublic, AggregateProof, VerifyingKey, PublicInput) {
        let program = spanning_program();
        let (pk, vk) = keygen_chunked(program, trace_len);
        let pub_in = PublicInput {
            input: vec![],
            output: 110,
        };
        let (proof, public) = prove_chunked(&pk, &pub_in, max_chunks);
        assert!(proof.chunks.len() >= 2, "the program must span chunks");
        (public, proof, vk, pub_in)
    }

    #[test]
    fn chunked_ram_and_register_carryover() {
        let (public, proof, vk, pub_in) = prove_spanning(32, 8);
        assert!(
            verify_chunked(&vk, &pub_in, &public, &proof).is_ok(),
            "the honest aggregate must verify"
        );
    }

    #[test]
    fn chunked_single_chunk_compatible() {
        // A program that halts within one chunk: k = 1, the identities and
        // PC claims still run, the aggregate reduces to one core.
        let program = vec![
            enc_i(1, 0, 0, 1, 0x13),
            enc_i(2, 0, 0, 10, 0x13),
            0x0000_0073,
        ];
        let (pk, vk) = keygen_chunked(program, 8);
        let pub_in = PublicInput {
            input: vec![],
            output: 2,
        };
        let (proof, public) = prove_chunked(&pk, &pub_in, 4);
        assert_eq!(proof.chunks.len(), 1);
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_ok());
    }

    #[test]
    fn chunked_public_input_binding() {
        // Chunk 0's entry RAM image is the public input: a program that
        // loads the input word and outputs it back.
        let program = vec![
            enc_i(0, 0, 3, 10, 0x03), // ld x10, 0(x0) — the input word
            0x0000_0073,
        ];
        let (pk, vk) = keygen_chunked(program, 8);
        let pub_in = PublicInput {
            input: vec![0xABCD_9876],
            output: 0xABCD_9876,
        };
        let (proof, public) = prove_chunked(&pk, &pub_in, 4);
        let v = verify_chunked(&vk, &pub_in, &public, &proof);
        assert!(v.is_ok(), "verify failed: {v:?}");
        // The wrong input statement must fail (chunk 0's identity pins the
        // entry image to the public input).
        let bad = PublicInput {
            input: vec![0xFFFF_FFFF],
            output: 0xABCD_9876,
        };
        assert!(verify_chunked(&vk, &bad, &public, &proof).is_err());
    }

    // ---- Tamper classes ----

    #[test]
    fn tampered_continuity_claim_rejected() {
        let (public, mut proof, vk, pub_in) = prove_spanning(32, 8);
        proof.cont_values[0][0] += ONE; // the images no longer chain
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn tampered_ram_state_values_rejected() {
        let (public, mut proof, vk, pub_in) = prove_spanning(32, 8);
        // A false chunk-1 entry image: the identity no longer holds.
        proof.ram_state_values[1][1] += ONE;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn tampered_chunk0_entry_image_rejected() {
        let (public, mut proof, vk, pub_in) = prove_spanning(32, 8);
        // Chunk 0's transmitted init must equal the public input image.
        proof.ram_state_values[0][1] += ONE;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn tampered_boundary_regs_rejected() {
        let (mut public, proof, vk, pub_in) = prove_spanning(32, 8);
        // A forged exit register for chunk 0: the final-state identity
        // binds it to the committed write log, so verification fails.
        public.boundaries[0].exit_regs[10] += 1;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn tampered_boundary_pc_rejected() {
        let (mut public, proof, vk, pub_in) = prove_spanning(32, 8);
        // A forged exit PC: the fetch component claim at the last cycle
        // no longer matches the published packing.
        public.boundaries[0].exit_pc += 1;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn broken_boundary_chain_rejected() {
        let (mut public, proof, vk, pub_in) = prove_spanning(32, 8);
        // The published chunks no longer chain: chunk 1's entry differs.
        public.boundaries[1].entry_regs[3] ^= 0x77;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn tampered_commitment_rejected() {
        let (public, mut proof, vk, pub_in) = prove_spanning(32, 8);
        proof.commitment_root[0] ^= 0xff;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn tampered_identity_round_rejected() {
        let (public, mut proof, vk, pub_in) = prove_spanning(32, 8);
        proof.chunks[0].ram_state.rounds[0][1] += ONE;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn tampered_identity_terminal_rejected() {
        let (public, mut proof, vk, pub_in) = prove_spanning(32, 8);
        proof.chunks[1].reg_state.inc_claim += ONE;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn tampered_mosaic_discharge_rejected() {
        let (public, mut proof, vk, pub_in) = prove_spanning(32, 8);
        proof.mosaic.terminal += ONE;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn tampered_chunk_core_rejected() {
        let (public, mut proof, vk, pub_in) = prove_spanning(32, 8);
        // A corrupted step 1-4 piece of an interior chunk.
        proof.chunks[1].core.ram_wv_claim += ONE;
        assert!(verify_chunked(&vk, &pub_in, &public, &proof).is_err());
    }

    #[test]
    fn aggregate_cost_model_measured() {
        let (public, proof, vk, _pub_in) = prove_spanning(32, 8);
        assert!(verify_chunked(&vk, &_pub_in, &public, &proof).is_ok());
        let stats = crate::stats::aggregate_stats(&vk, &proof);
        assert!(
            stats.chunks >= 2,
            "the spanning program must use >= 2 chunks"
        );
        assert!(stats.eval_layer_bytes > 0);
        // The claim-batching win: k separate evaluation layers vs one shared.
        assert!(
            stats.separate_eval_bytes > stats.eval_layer_bytes,
            "the shared eval layer must beat {} separate ones",
            stats.chunks
        );
        assert!(stats.identity_layer_bytes > 0);
        let report = stats.report();
        assert!(report.contains("claim-batching"), "{report}");
    }

    #[test]
    fn chunked_fibonacci_spans_chunks() {
        fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
            ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
        }
        let mut program = vec![enc_i(1, 0, 0, 1, 0x13), enc_i(1, 0, 0, 2, 0x13)];
        for _ in 0..32 {
            program.push((2u32 << 20) | (1 << 15) | (3 << 7) | 0x33);
            program.push((2 << 15) | (1 << 7) | 0x33);
            program.push(enc_i(0, 3, 0, 2, 0x13));
        }
        program.push((3 << 15) | (10 << 7) | 0x33);
        program.push(0x0000_0073);
        let (pk, vk) = keygen_chunked(program, 32);
        let pub_in = PublicInput {
            input: vec![],
            output: 5702887,
        };
        let (proof, public) = prove_chunked(&pk, &pub_in, 16);
        eprintln!("chunks = {}", proof.chunks.len());
        let v = verify_chunked(&vk, &pub_in, &public, &proof);
        assert!(v.is_ok(), "verify failed: {v:?}");
    }

    #[test]
    fn boundary_reads_match_write_log() {
        fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
            ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
        }
        let mut program = vec![enc_i(1, 0, 0, 1, 0x13), enc_i(1, 0, 0, 2, 0x13)];
        for _ in 0..32 {
            program.push((2u32 << 20) | (1 << 15) | (3 << 7) | 0x33);
            program.push((2 << 15) | (1 << 7) | 0x33);
            program.push(enc_i(0, 3, 0, 2, 0x13));
        }
        program.push((3 << 15) | (10 << 7) | 0x33);
        program.push(0x0000_0073);
        let trace = magnetar_cpu::execute_chunk(
            &program,
            magnetar_cpu::ExecState {
                regs: [0; magnetar_cpu::REGS],
                ram: vec![0; 1 << magnetar_cpu::RAM_WORDS_LOG2],
                micro_pc: 0,
                cycles: 0,
                halted: false,
                output: 0,
            },
            32 - CHUNK_MARGIN,
            32,
        );
        let mut shadow = [0u64; magnetar_cpu::REGS];
        let rows = &trace.program.rows;
        for (j, rec) in trace.records.iter().enumerate() {
            let row = &rows[rec.micro_pc % rows.len()].row;
            let e1 = shadow[row.rs1 % 64];
            let e2 = shadow[row.rs2 % 64];
            assert_eq!(
                rec.rs1_val, e1,
                "rs1 mismatch at cycle {j} (pc {})",
                rec.micro_pc
            );
            assert_eq!(
                rec.rs2_val, e2,
                "rs2 mismatch at cycle {j} (pc {})",
                rec.micro_pc
            );
            // The write: strict reads precede the cycle's own write.
            if rec.rd != 0 {
                shadow[rec.rd] = rec.rd_val;
            }
        }
    }

    #[test]
    fn boundary_padding_write_consistency() {
        fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
            ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
        }
        let mut program = vec![enc_i(1, 0, 0, 1, 0x13), enc_i(1, 0, 0, 2, 0x13)];
        for _ in 0..32 {
            program.push((2u32 << 20) | (1 << 15) | (3 << 7) | 0x33);
            program.push((2 << 15) | (1 << 7) | 0x33);
            program.push(enc_i(0, 3, 0, 2, 0x13));
        }
        program.push((3 << 15) | (10 << 7) | 0x33);
        program.push(0x0000_0073);
        let (pk, _vk) = keygen_chunked(program.clone(), 32);
        let trace = magnetar_cpu::execute_chunk(
            &program,
            magnetar_cpu::ExecState {
                regs: [0; magnetar_cpu::REGS],
                ram: vec![0; 1 << magnetar_cpu::RAM_WORDS_LOG2],
                micro_pc: 0,
                cycles: 0,
                halted: false,
                output: 0,
            },
            32 - CHUNK_MARGIN,
            32,
        );
        let boundary = BoundaryState {
            entry_regs: [0; BOUNDARY_WORDS],
            exit_regs: [0; BOUNDARY_WORDS],
            entry_pc: 0,
            exit_pc: 0,
        };
        let ctx = ChunkCtx {
            output: 0,
            input_words: &[],
            ram_words: &[],
            ram_m0_public: false,
            regs_m0: regs_image_bits(&boundary.entry_regs),
        };
        let wd = build_witness(&pk, &ctx, &trace);
        // The write one-hot per cycle, from the ROM row (the witness's rule).
        let rows = &trace.program.rows;
        let mut w_target = vec![0usize; 32];
        for (u, rec) in trace.records.iter().enumerate() {
            let row = &rows[rec.micro_pc % rows.len()].row;
            w_target[u] = if row.rd == 0 { 32 } else { row.rd };
        }
        // The shadow post-write content: shadow[k] after cycle j's write.
        let mut shadow = [0u64; 64];
        let mut mismatch = 0usize;
        for (j, rec) in trace.records.iter().enumerate() {
            let post = w_target[j];
            shadow[post] = rec.rd_val;
            // Compare against the witness's inc table at this cycle: the
            // inc bits must be old ^ new.
            for o in 0..64usize {
                let want_inc = ((rec.rd_old ^ rec.rd_val) >> o) & 1;
                let got_inc = wd.reg_wit.inc[o | (j << 6)].0 & 1;
                if want_inc as u128 != got_inc as u128 {
                    if mismatch < 5 {
                        eprintln!(
                            "inc mismatch j={j} o={o}: want {want_inc} got {got_inc} (rd_old={:x} rd_val={:x} rec.rd={} row.rd-ish target={post})",
                            rec.rd_old, rec.rd_val, rec.rd
                        );
                    }
                    mismatch += 1;
                }
            }
        }
        assert_eq!(mismatch, 0, "{mismatch} inc mismatches");
        // Also: the write target vs the record's rd.
        for (j, rec) in trace.records.iter().enumerate() {
            assert_eq!(
                w_target[j], rec.rd,
                "write target mismatch at j={j}: row target {} vs record rd {}",
                w_target[j], rec.rd
            );
        }
    }
}
