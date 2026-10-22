//! The complete protocol (paper §5.5, Construction 3): KeyGen, Prove,
//! Verify — every step of the paper's seven-step online phase, with the
//! virtual-claim discharge (step 4) fully integrated:
//!
//! 1. **Execute** — the interpreter runs the expanded program; the trace
//!    plus the memory-access log define the witness.
//! 2. **Commit** — the direct segments and the auxiliary polynomials (the
//!    one-hot address components and the increments) form one MosaicPCS
//!    polynomial; the protocol's only commitment.
//! 3. **Virtual Spartan** — the two SIMD R1CS relations reduce to one
//!    evaluation claim per witness segment (direct claims held for the
//!    PCS, virtual claims forwarded).
//! 4. **Reduce the virtual claims** — BinaryTwist for registers and for
//!    RAM (with the Ψ-sum-checks binding their address claims),
//!    BinaryRouter for shifts and sub-word stores (the latter on the
//!    verifier-sampled increment claim), BinaryShout for the bytecode
//!    (the bus Φ, the pc Ψ, and the selector checks).
//! 5. **Prove the committed evaluations** — MosaicPCS lifts, batches, and
//!    ring-switches every remaining claim into one PCS evaluation
//!    argument.

use crate::mosaic::{
    commit_mosaic, prove_mosaic, verify_mosaic, MosaicClaim, MosaicCommitment, MosaicLayout,
    MosaicProof,
};
use crate::RomTables;

use magnetar_cpu::circuit::{build_circuit, witness_row, CircuitSpec, WITNESS_LAYOUT};
use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_memory::{
    prove_bus_shout, prove_psi_claim, prove_selector_shout, prove_stage_select, prove_twist,
    verify_bus_shout, verify_psi_claim, verify_selector_shout, verify_stage_select, verify_twist,
    BusShoutProof, BusShoutSpec, SelectorCheckSpec, SelectorShoutProof, SelectorShoutSpec,
    ShoutPcsClaims, StageSelectProof, StageSelectSpec, TwistInstance, TwistPcsClaims, TwistProof,
    TwistWitness,
};
use magnetar_pcs::PcsParams;
use magnetar_router::{
    build_shift_tensor, build_store_tensor, prove_routing, verify_routing, PaddedDense,
    PaddedTableFactor, RouterProof, RoutingLayout,
};
use magnetar_spartan::{
    prove_virtual_spartan, verify_virtual_spartan, DualRelation, SegmentClaim, SegmentLayout,
    SegmentSpec, VirtualSpartanProof,
};

/// The shift-amount routing width (test-scale default; the production
/// envelope uses 6 for the full 64-amount space).
pub const AMOUNT_BITS: usize = 2;

/// The proving key (transparent: derived from the public program).
#[derive(Clone)]
pub struct KeyGen {
    /// The RV64I program (raw instruction words).
    pub program: Vec<u32>,
    /// The dual SIMD R1CS circuit.
    pub circuit: CircuitSpec,
    /// The witness-segment layout (virtual + direct, declared order).
    pub layout: SegmentLayout,
    /// The MosaicPCS component layout.
    pub mosaic: MosaicLayout,
    /// The bytecode ROM's derived tables.
    pub rom: RomTables,
    /// The shift routing tensor `(values, o_vars, u_vars, h_vars)`.
    pub shift_tensor: (Vec<Gf128>, usize, usize, usize),
    /// The store routing tensor `(values, o_vars, u_vars, h_vars, a_vars)`.
    pub store_tensor: (Vec<Gf128>, usize, usize, usize, usize),
    pub amount_bits: usize,
    /// The trace length bound (a power of two).
    pub trace_len: usize,
    /// The claim-aggregation index width (log2 of the segment count;
    /// 0 = a single segment).
    pub idx_bits: usize,
    /// PCS parameters.
    pub pcs: PcsParams,
}

/// The verifying key.
#[derive(Clone)]
pub struct VerifyingKey {
    pub program: Vec<u32>,
    pub circuit: CircuitSpec,
    pub layout: SegmentLayout,
    pub mosaic: MosaicLayout,
    pub rom: RomTables,
    pub shift_tensor: (Vec<Gf128>, usize, usize, usize),
    pub store_tensor: (Vec<Gf128>, usize, usize, usize, usize),
    pub amount_bits: usize,
    pub trace_len: usize,
    pub idx_bits: usize,
    pub pcs: PcsParams,
}

/// Public input/output binding.
#[derive(Clone, Debug)]
pub struct PublicInput {
    /// Input words (loaded into RAM starting at word 0).
    pub input: Vec<u64>,
    /// The claimed output (x10 at halt).
    pub output: u64,
}

/// One segment's auxiliary-argument proofs (Construction 3, steps 3–6).
/// The single-segment proof carries one of these; the aggregated proof
/// carries one per segment, all discharged against the ONE commitment.
#[derive(Clone)]

/// The per-chunk proving/verifying context: everything that differs
/// between a fresh statement segment and a tree-IVC chunk.
///
/// * `output` — the halt-latch output binding (the last chunk's claimed
///   output; interior chunks publish 0).
/// * `input_words` — chunk 0's public input words (interior chunks:
///   empty); they populate the circuit's public columns after the output.
/// * `ram_words` — the chunk's entry RAM image in words (the memory
///   argument's `M0` source). Chunk 0's image is the public input; an
///   interior chunk's image is the previous chunk's exit state, carried
///   by the committed `ram-init` component (not verifier-known).
/// * `ram_m0_public` — whether the entry RAM image is verifier-known.
///   When false the RAM Twist transmits its `M0` claims and the
///   final-state identity binds them to the commitment.
/// * `regs_m0` — the register file's public entry image as a bit table
///   over `(k, o)` (64 cells x 64 bits; x0 and cells 33..64 are zero).
pub struct ChunkCtx<'a> {
    pub output: u64,
    pub input_words: &'a [u64],
    pub ram_words: &'a [u64],
    pub ram_m0_public: bool,
    pub regs_m0: Vec<Gf128>,
}

impl<'a> ChunkCtx<'a> {
    /// The fresh-statement context (a public input image, a zero register
    /// file) — what a single-chunk or aggregated segment runs with.
    pub fn fresh(pub_in: &'a PublicInput) -> Self {
        ChunkCtx {
            output: pub_in.output,
            input_words: &pub_in.input,
            ram_words: &pub_in.input,
            ram_m0_public: true,
            regs_m0: vec![ZERO; 1 << 12],
        }
    }
}

/// Build the RAM image bit table over `(k, o)` (address low, bit high)
/// from the word image.
pub(crate) fn ram_image_bits(words: &[u64]) -> Vec<Gf128> {
    let mut t0 = vec![ZERO; 1usize << 14];
    for (k, &w) in words.iter().enumerate() {
        if k < (1usize << 8) {
            for o in 0..64usize {
                t0[k | (o << 8)] = Gf128(((w >> o) & 1) as u128);
            }
        }
    }
    t0
}

/// Build the register-file image bit table over `(k, o)` (cell low, bit
/// high) from the boundary words `x0..x31` plus the trash cell (word 32).
pub(crate) fn regs_image_bits(words: &[u64]) -> Vec<Gf128> {
    let mut t0 = vec![ZERO; 1usize << 12];
    for (k, &w) in words.iter().enumerate() {
        if k < 64 {
            for o in 0..64usize {
                t0[k | (o << 6)] = Gf128(((w >> o) & 1) as u128);
            }
        }
    }
    t0
}

#[derive(Clone, Debug)]
pub struct SegmentProof {
    /// Virtual Spartan's proof.
    pub vs: VirtualSpartanProof,
    /// Step 3: the register BinaryTwist.
    pub reg_twist: TwistProof,
    /// Step 3: the register-index Ψ-sum-checks (read 1, read 2, write).
    pub reg_psi: [magnetar_memory::PsiProof; 3],
    /// Step 3: the register-index claim values (in the same order).
    pub reg_index_claims: [Gf128; 3],
    /// Step 4: the RAM BinaryTwist.
    pub ram_twist: TwistProof,
    /// Step 4: the RAM address Ψ-sum-checks (read, write).
    pub ram_psi: [magnetar_memory::PsiProof; 2],
    /// Step 4: the RAM address claim values.
    pub ram_addr_claims: [Gf128; 2],
    /// Step 4: the RAM write-value claim.
    pub ram_wv_claim: Gf128,
    /// Step 4: the verifier-sampled Inc point and the prover's claim.
    pub inc_point: Vec<Gf128>,
    pub inc_claim: Gf128,
    /// Step 5: the shift router.
    pub shift_router: RouterProof,
    /// Step 5: the store router.
    pub store_router: RouterProof,
    /// Step 5: the store X split claims (old, src).
    pub store_x_claims: [Gf128; 2],
    /// Step 6: the bytecode Shout.
    pub bus_shout: BusShoutProof,
    /// Step 6: the register-shift stage-selector sum-check (the
    /// stage-select extension).
    pub stage_select: StageSelectProof,
    /// Step 6: the stage one-hot selector check.
    pub sel_stage: SelectorShoutProof,
    /// Step 6: the shift selector check.
    pub sel_shift: SelectorShoutProof,
    /// Step 6: the store selector check.
    pub sel_store: SelectorShoutProof,
}

/// The end-to-end proof: ONE commitment, one segment body per segment, and
/// ONE Mosaic discharge covering every segment's claims (the
/// **claim-aggregation** artifact of roadmap Horizon 2 — the evaluation
/// argument, which dominates proof bytes, is paid once and stays
/// independent of the segment count).
#[derive(Clone)]
pub struct Proof {
    /// The Mosaic commitment root.
    pub commitment_root: [u8; 32],
    pub pcs_n: usize,
    pub pcs_k: usize,
    pub pcs_domain: String,
    /// ℓ (the Mosaic bit-witness variable count, including the segment
    /// index dimension).
    pub ell: usize,
    /// The per-segment protocol bodies (one entry per segment).
    pub segments: Vec<SegmentProof>,
    /// Step 7: the single Mosaic discharge.
    pub mosaic: MosaicProof,
}

/// The W-segment names, in declared order (must match the circuit's
/// offsets: each 64-column block at its aligned offset).
/// The W-segment table (name, length, direct) — public for the stats
/// instrumentation.
pub const fn segments() -> &'static [(&'static str, usize, bool)] {
    SEGMENTS
}

const SEGMENTS: &[(&str, usize, bool)] = &[
    ("rs1", 64, false),
    ("rs2", 64, false),
    ("rd", 64, false),
    ("load-val", 64, false),
    ("shift-out", 64, false),
    ("t1", 64, true),
    ("t2", 64, true),
    ("carry", 64, true),
    ("alu", 64, true),
    ("shift-src", 64, true),
    ("store-old", 64, true),
    ("store-src", 64, true),
    ("flags", 16, true),
    ("off", 8, true),
    ("ram-addr", 8, true),
    ("pc", 8, false),
    ("fetch", 64, false),
    ("eqaux", 64, true),
];

/// Derive the keys for the tree-IVC chunked layout: every chunk commits
/// its entry and exit RAM images (`ram-init` / `ram-final`) alongside the
/// standard components, so the aggregate can chain memory across chunks.
/// The chunk count is dynamic (the program runs until halt), so the
/// per-chunk layout carries no segment-index dimension.
pub fn keygen_chunked(program: Vec<u32>, trace_len: usize) -> (KeyGen, VerifyingKey) {
    keygen_chunked_with(program, trace_len, AMOUNT_BITS, PcsParams::default())
}

/// The chunked keygen with explicit parameters.
pub fn keygen_chunked_with(
    program: Vec<u32>,
    trace_len: usize,
    amount_bits: usize,
    pcs: PcsParams,
) -> (KeyGen, VerifyingKey) {
    let mut out = keygen_with(program, trace_len, amount_bits, pcs);
    out.0.mosaic = MosaicLayout::build(mosaic_components(
        trace_len.trailing_zeros() as usize,
        out.0.rom.a_rom,
        0,
        true,
    ));
    out.1.mosaic = out.0.mosaic.clone();
    out
}

/// Derive the keys from the public program.
pub fn keygen(program: Vec<u32>, trace_len: usize) -> (KeyGen, VerifyingKey) {
    keygen_with(program, trace_len, AMOUNT_BITS, PcsParams::default())
}

/// KeyGen with explicit parameters (the test scale and the production
/// envelope differ in the shift-amount width and the PCS rate).
pub fn keygen_with(
    program: Vec<u32>,
    trace_len: usize,
    amount_bits: usize,
    pcs: PcsParams,
) -> (KeyGen, VerifyingKey) {
    assert!(trace_len.is_power_of_two() && trace_len >= 2);
    let circuit = build_circuit();
    let layout = SegmentLayout::build(
        WITNESS_LAYOUT::M / 2,
        SEGMENTS
            .iter()
            .map(|(name, len, direct)| SegmentSpec {
                name: name.to_string(),
                len: *len,
                direct: *direct,
            })
            .collect(),
    );
    // The ROM tables.
    let expanded = magnetar_cpu::expand::expand_program(&program);
    // Register-shift stages route by 2^k for k < 6, so their amounts must
    // live in the routing one-hot's index space.
    if expanded.rows.iter().any(|r| {
        matches!(
            r.row.kind,
            magnetar_cpu::expand::VirtualKind::ShiftStage { .. }
        )
    }) && amount_bits < crate::rom::STAGES
    {
        panic!("register-shift programs need amount_bits >= 6");
    }
    let rom = RomTables::build(&expanded.rows, amount_bits);
    let n = trace_len.trailing_zeros() as usize;
    let a_rom = rom.a_rom;
    let mosaic = MosaicLayout::build(mosaic_components(n, a_rom, 0, false));
    let shift_tensor = build_shift_tensor(64, amount_bits);
    let store_tensor = build_store_tensor(64);
    let pk = KeyGen {
        program: program.clone(),
        circuit: circuit.clone(),
        layout: layout.clone(),
        mosaic: mosaic.clone(),
        rom: rom.clone(),
        shift_tensor: shift_tensor.clone(),
        store_tensor: store_tensor.clone(),
        amount_bits,
        trace_len,
        idx_bits: 0,
        pcs: pcs.clone(),
    };
    let vk = VerifyingKey {
        program,
        circuit,
        layout,
        mosaic,
        rom,
        shift_tensor,
        store_tensor,
        amount_bits,
        trace_len,
        idx_bits: 0,
        pcs,
    };
    (pk, vk)
}

/// KeyGen for the **claim-aggregated** prover: `segments` executions share
/// ONE Mosaic commitment (the components' subcubes gain the
/// `log2(segments)` index dimension) and ONE Mosaic discharge, so the
/// evaluation argument — the byte-dominant part of every hash-based proof —
/// is paid once regardless of the segment count (roadmap Horizon 2,
/// "aggregation before the wrap"). The segments are independent
/// (input, output) pairs of the same program — the rollup/batch flavor;
/// cross-segment state continuity is the tree-IVC path documented in the
/// roadmap.
pub fn keygen_aggregate(
    program: Vec<u32>,
    trace_len: usize,
    amount_bits: usize,
    pcs: PcsParams,
    segments: usize,
) -> (KeyGen, VerifyingKey) {
    assert!(
        segments.is_power_of_two() && segments >= 1,
        "the segment count must be a power of two"
    );
    assert!(trace_len.is_power_of_two() && trace_len >= 2);
    let idx_bits = segments.trailing_zeros() as usize;
    let circuit = build_circuit();
    let layout = SegmentLayout::build(
        WITNESS_LAYOUT::M / 2,
        SEGMENTS
            .iter()
            .map(|(name, len, direct)| SegmentSpec {
                name: name.to_string(),
                len: *len,
                direct: *direct,
            })
            .collect(),
    );
    let expanded = magnetar_cpu::expand::expand_program(&program);
    if expanded.rows.iter().any(|r| {
        matches!(
            r.row.kind,
            magnetar_cpu::expand::VirtualKind::ShiftStage { .. }
        )
    }) && amount_bits < crate::rom::STAGES
    {
        panic!("register-shift programs need amount_bits >= 6");
    }
    let rom = RomTables::build(&expanded.rows, amount_bits);
    let n = trace_len.trailing_zeros() as usize;
    let a_rom = rom.a_rom;
    let mosaic = MosaicLayout::build(mosaic_components(n, a_rom, idx_bits, false));
    let shift_tensor = build_shift_tensor(64, amount_bits);
    let store_tensor = build_store_tensor(64);
    let pk = KeyGen {
        program: program.clone(),
        circuit: circuit.clone(),
        layout: layout.clone(),
        mosaic: mosaic.clone(),
        rom: rom.clone(),
        shift_tensor: shift_tensor.clone(),
        store_tensor: store_tensor.clone(),
        amount_bits,
        trace_len,
        idx_bits,
        pcs: pcs.clone(),
    };
    let vk = VerifyingKey {
        program,
        circuit,
        layout,
        mosaic,
        rom,
        shift_tensor,
        store_tensor,
        amount_bits,
        trace_len,
        idx_bits,
        pcs,
    };
    (pk, vk)
}

/// The Mosaic component list (the direct segments then the auxiliary
/// polynomials), at a trace length `2^n`.
fn mosaic_components(
    n: usize,
    a_rom: usize,
    idx_bits: usize,
    chunked: bool,
) -> Vec<crate::mosaic::ComponentSpec> {
    use crate::mosaic::ComponentSpec;
    // With claim aggregation, every component's subcube gains the segment
    // index dimension: the component's table becomes the concatenation of
    // the K segments' copies (index `local | (s << local_vars)`), and each
    // segment's claim lifts by appending the segment's Boolean index bits.
    let seg = |name: &str, lv: usize| ComponentSpec {
        name: name.to_string(),
        vars: lv + n + idx_bits,
    };
    let mut v = vec![
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
        // The register BinaryTwist's components (d = 1).
        seg("reg-ra-1", 6),
        seg("reg-ra-2", 6),
        seg("reg-wfa", 6),
        seg("reg-inc", 6),
        // The RAM BinaryTwist's components (d = 1).
        seg("ram-ra-read", 8),
        seg("ram-ra-write", 8),
        seg("ram-inc", 6),
        // The bytecode BinaryShout's fetch-address component.
        seg("fetch-ra", a_rom),
    ];
    if chunked {
        // The tree-IVC state components: the chunk's entry/exit RAM images
        // as bit tables over (k, o) — 14 vars, no cycle dimension.
        v.push(ComponentSpec {
            name: "ram-init".to_string(),
            vars: 14,
        });
        v.push(ComponentSpec {
            name: "ram-final".to_string(),
            vars: 14,
        });
    }
    let _ = &mut v;
    v
}

/// One prover-side witness bundle.
pub(crate) struct WitnessData {
    pub(crate) joint_z: Vec<Gf128>,
    /// The direct component tables (bits), in Mosaic order.
    pub(crate) direct_tables: Vec<Vec<bool>>,
    /// The register Twist's witness.
    pub(crate) reg_wit: TwistWitness,
    /// The RAM Twist's witness.
    pub(crate) ram_wit: TwistWitness,
    /// The fetch component.
    pub(crate) fetch_comp: magnetar_memory::AddressComponent,
}

/// Assemble the witness from the trace.
pub(crate) fn build_witness(
    pk: &KeyGen,
    ctx: &ChunkCtx,
    trace: &magnetar_cpu::interp::Trace,
) -> WitnessData {
    let q = pk.trace_len;
    let m = WITNESS_LAYOUT::M;
    let n = q.trailing_zeros() as usize;
    let rows = &trace.program.rows;
    let a_rom = pk.rom.a_rom;
    // The public x vector per cycle.
    let public_x: Vec<Gf128> = {
        let mut v = vec![ZERO; m / 2 - 1];
        v[0] = Gf128(ctx.output as u128);
        for (i, &w) in ctx.input_words.iter().enumerate() {
            if 1 + i < v.len() {
                v[1 + i] = Gf128(w as u128);
            }
        }
        v
    };
    // The witness rows.
    let mut joint_z = vec![ZERO; q * m];
    for (u, rec) in trace.records.iter().enumerate() {
        let row = &rows[rec.micro_pc % rows.len()].row;
        let z = witness_row(rec, row, &public_x);
        joint_z[u * m..(u + 1) * m].copy_from_slice(&z);
    }
    // The direct segment tables.
    let mut direct_tables = Vec::new();
    for (name, len, _direct) in SEGMENTS.iter() {
        if !*_direct {
            continue;
        }
        let off = pk.layout.offsets[pk.layout.index_of(name).expect("segment")];
        let lv = len.trailing_zeros() as usize;
        let mut bits = vec![false; 1usize << (lv + n)];
        for (u, _) in trace.records.iter().enumerate() {
            for c in 0..*len {
                bits[c | (u << lv)] = joint_z[u * m + off + c].0 & 1 == 1;
            }
        }
        direct_tables.push(bits);
    }
    // The register Twist's witness: the read/write one-hot components and
    // the increment.
    // The write port's one-hot: the ROM row's redirected destination on
    // every cycle (the paper's one-write-per-cycle model; no-op writes
    // carry zero increments).
    let write_one_hot = || -> Vec<bool> {
        let mut b = vec![false; (1usize << 6) * q];
        for (u, rec) in trace.records.iter().enumerate() {
            let row = &rows[rec.micro_pc % rows.len()].row;
            let idx = if row.rd == 0 { 32 } else { row.rd };
            b[idx | (u << 6)] = true;
        }
        b
    };
    let reg_ports = vec![
        // Read 1: rs1's index.
        {
            let mut b = vec![false; (1usize << 6) * q];
            for (u, rec) in trace.records.iter().enumerate() {
                let row = &rows[rec.micro_pc % rows.len()].row;
                b[row.rs1 | (u << 6)] = true;
            }
            vec![magnetar_memory::AddressComponent::from_bits(6, n, &b)]
        },
        // Read 2: rs2's index.
        {
            let mut b = vec![false; (1usize << 6) * q];
            for (u, rec) in trace.records.iter().enumerate() {
                let row = &rows[rec.micro_pc % rows.len()].row;
                b[row.rs2 | (u << 6)] = true;
            }
            vec![magnetar_memory::AddressComponent::from_bits(6, n, &b)]
        },
        // Write: the ROM row's redirected rd on every cycle.
        vec![magnetar_memory::AddressComponent::from_bits(
            6,
            n,
            &write_one_hot(),
        )],
    ];
    // The register increment: XOR of the old and new written value.
    let reg_inc: Vec<Gf128> = {
        let mut t = vec![ZERO; (1usize << 6) * q];
        for (u, rec) in trace.records.iter().enumerate() {
            let xor = rec.rd_old ^ rec.rd_val;
            for o in 0..64 {
                t[o | (u << 6)] = Gf128(((xor >> o) & 1) as u128);
            }
        }
        t
    };
    let reg_wit = TwistWitness {
        ports: reg_ports,
        inc: reg_inc,
    };
    // The RAM Twist's witness.
    let ram_read_bits = {
        let mut b = vec![false; (1usize << 8) * q];
        for (u, rec) in trace.records.iter().enumerate() {
            let idx = if rec.is_load || rec.is_store {
                rec.ram_addr
            } else {
                0
            };
            b[idx | (u << 8)] = true;
        }
        b
    };
    // The write port's one-hot: the cycle's access address on every
    // cycle (loads write the same word back — a zero-increment no-op at
    // the loaded address; non-memory cycles write cell 0's content).
    let ram_write_bits = {
        let mut b = vec![false; (1usize << 8) * q];
        for (u, rec) in trace.records.iter().enumerate() {
            b[rec.ram_addr | (u << 8)] = true;
        }
        b
    };
    let ram_inc: Vec<Gf128> = {
        let mut t = vec![ZERO; (1usize << 6) * q];
        for (u, rec) in trace.records.iter().enumerate() {
            if rec.is_store {
                let xor = rec.ram_old ^ rec.ram_new;
                for o in 0..64 {
                    t[o | (u << 6)] = Gf128(((xor >> o) & 1) as u128);
                }
            }
        }
        t
    };
    let ram_wit = TwistWitness {
        ports: vec![
            vec![magnetar_memory::AddressComponent::from_bits(
                8,
                n,
                &ram_read_bits,
            )],
            vec![magnetar_memory::AddressComponent::from_bits(
                8,
                n,
                &ram_write_bits,
            )],
        ],
        inc: ram_inc,
    };
    // The fetch component.
    let fetch_bits = {
        let mut b = vec![false; (1usize << a_rom) * q];
        for (u, rec) in trace.records.iter().enumerate() {
            b[(rec.micro_pc % rows.len()) | (u << a_rom)] = true;
        }
        b
    };
    let fetch_comp = magnetar_memory::AddressComponent::from_bits(a_rom, n, &fetch_bits);
    WitnessData {
        joint_z,
        direct_tables,
        reg_wit,
        ram_wit,
        fetch_comp,
    }
}

/// The eq weights `w_t = eq(point, pos + t)` for `t < width`.
fn slice_weights(point: &[Gf128], pos: usize, width: usize) -> Vec<Gf128> {
    let eq = magnetar_field::EqBits::build(point).values;
    (0..width).map(|t| eq[pos + t]).collect()
}

/// Prove one execution (the single-segment aggregate).
pub fn prove(pk: &KeyGen, pub_in: &PublicInput) -> Proof {
    prove_multi(pk, std::slice::from_ref(pub_in))
}

/// One segment's committed component tables, in Mosaic order.
fn segment_tables(wd: &WitnessData) -> Vec<Vec<bool>> {
    let mut tables: Vec<Vec<bool>> = wd.direct_tables.clone();
    for comp in wd.reg_wit.ports.iter().flat_map(|p| p.iter()) {
        tables.push(comp.table.iter().map(|&v| v == ONE).collect());
    }
    tables.push(wd.reg_wit.inc.iter().map(|&v| v == ONE).collect());
    for comp in wd.ram_wit.ports.iter().flat_map(|p| p.iter()) {
        tables.push(comp.table.iter().map(|&v| v == ONE).collect());
    }
    tables.push(wd.ram_wit.inc.iter().map(|&v| v == ONE).collect());
    tables.push(wd.fetch_comp.table.iter().map(|&v| v == ONE).collect());
    tables
}

/// Interleave the segments' component tables into the concatenated
/// (index-dimension) tables the shared commitment commits to:
/// `out[local | (s << local_vars)] = tables_s[component][local]`.
fn concat_component_tables(pk: &KeyGen, per_segment: &[Vec<Vec<bool>>]) -> Vec<Vec<bool>> {
    assert_eq!(per_segment.len(), 1usize << pk.idx_bits);
    let n = pk.trace_len.trailing_zeros() as usize;
    (0..pk.mosaic.components.len())
        .map(|i| {
            let local_vars = pk.mosaic.vars(i) - pk.idx_bits;
            let local_len = 1usize << local_vars;
            let _ = n;
            let mut out = vec![false; 1usize << pk.mosaic.vars(i)];
            for (s, tables) in per_segment.iter().enumerate() {
                out[s << local_vars..(s + 1) << local_vars]
                    .copy_from_slice(&tables[i][..local_len]);
            }
            out
        })
        .collect()
}

/// Append the segment's Boolean index bits to a claim's point (the lift
/// through the index dimension: the component's own coordinates are
/// `(local, segment index)`).
fn lift_claim_with_segment(c: MosaicClaim, s: usize, idx_bits: usize) -> MosaicClaim {
    if idx_bits == 0 {
        return c;
    }
    let mut pt = c.point.clone();
    for b in 0..idx_bits {
        pt.push(Gf128(((s >> b) & 1) as u128));
    }
    MosaicClaim::new(c.component, pt, c.value)
}

/// Prove `segments` executions against ONE commitment and ONE Mosaic
/// discharge (Construction 3 with claim aggregation). Each segment runs the
/// full steps 3–6 against the shared commitment; the single evaluation
/// argument covers every segment's claims.
pub fn prove_multi(pk: &KeyGen, pub_ins: &[PublicInput]) -> Proof {
    assert_eq!(
        pub_ins.len(),
        1usize << pk.idx_bits,
        "segment count mismatch"
    );
    // ---- 1. Execute every segment.
    let mut wds = Vec::with_capacity(pub_ins.len());
    for pub_in in pub_ins {
        let trace = magnetar_cpu::interp::execute_padded(
            &pk.program,
            &pub_in.input,
            pk.trace_len,
            pk.trace_len,
        );
        assert_eq!(
            trace.final_state.output, pub_in.output,
            "execution output mismatch"
        );
        assert_eq!(
            trace.records.len(),
            pk.trace_len,
            "the trace must fill the witness cube exactly"
        );
        let ctx = ChunkCtx::fresh(pub_in);
        let wd = build_witness(pk, &ctx, &trace);
        wds.push((trace, wd));
    }
    // ---- 2. Commit: the concatenated component tables, ONE commitment.
    let per_segment_tables: Vec<Vec<Vec<bool>>> =
        wds.iter().map(|(_, wd)| segment_tables(wd)).collect();
    let tables = concat_component_tables(pk, &per_segment_tables);
    let commitment: MosaicCommitment = commit_mosaic(&pk.pcs, &pk.mosaic, &tables);
    // ---- 3. The transcript: the program and every segment's public I/O.
    let mut pub_bytes: Vec<u8> = Vec::new();
    for pub_in in pub_ins {
        pub_bytes.extend_from_slice(b"|");
        for w in &pub_in.input {
            pub_bytes.extend_from_slice(&w.to_le_bytes());
        }
        pub_bytes.extend_from_slice(b"|");
        pub_bytes.extend_from_slice(&pub_in.output.to_le_bytes());
    }
    let mut t = Transcript::new(
        "magnetar",
        &[
            b"construction-3-v3",
            &pk.program
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            &pub_bytes,
        ],
    );
    t.absorb_digest("cm-g", &commitment.stmt.root);
    // ---- 4. Every segment's steps 3–6, then ONE Mosaic discharge.
    let mut seg_proofs = Vec::with_capacity(pub_ins.len());
    let mut claims: Vec<MosaicClaim> = Vec::new();
    for (s, (trace, wd)) in wds.iter().enumerate() {
        let ctx = ChunkCtx::fresh(&pub_ins[s]);
        let (sp, local_claims) = prove_segment_body(pk, &ctx, wd, trace, &mut t);
        for c in local_claims {
            claims.push(lift_claim_with_segment(c, s, pk.idx_bits));
        }
        seg_proofs.push(sp);
    }
    let mosaic_proof = prove_mosaic(&pk.pcs, &pk.mosaic, &commitment, &claims, &mut t);
    Proof {
        commitment_root: commitment.stmt.root,
        pcs_n: commitment.stmt.n,
        pcs_k: commitment.stmt.k,
        pcs_domain: commitment.stmt.domain,
        ell: commitment.ell,
        segments: seg_proofs,
        mosaic: mosaic_proof,
    }
}

/// One segment's steps 3–6 (Virtual Spartan, both Twists, both routers,
/// the bytecode Shout, the stage-selector sum-check, and the selector
/// checks), returning the segment body and its local Mosaic claims (the
/// caller lifts them through the index dimension).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_segment_body(
    pk: &KeyGen,
    ctx: &ChunkCtx,
    wd: &WitnessData,
    trace: &magnetar_cpu::interp::Trace,
    t: &mut Transcript,
) -> (SegmentProof, Vec<MosaicClaim>) {
    let q = pk.trace_len;
    let m = WITNESS_LAYOUT::M;
    let n = q.trailing_zeros() as usize;
    let a_rom = pk.rom.a_rom;
    let rel = DualRelation {
        m,
        a_bin: pk.circuit.a_bin.clone(),
        b_bin: pk.circuit.b_bin.clone(),
        c_bin: pk.circuit.c_bin.clone(),
        a_fld: pk.circuit.a_fld.clone(),
        b_fld: pk.circuit.b_fld.clone(),
        c_fld: pk.circuit.c_fld.clone(),
    };
    let (vs_proof, seg_claims) = prove_virtual_spartan(&rel, &pk.layout, &wd.joint_z, q, t);
    let seg = |name: &str| -> SegmentClaim {
        let i = pk.layout.index_of(name).expect("segment");
        seg_claims[i].clone()
    };
    // The shared points.
    let r_val: Vec<Gf128> = seg("rs1").r_local; // 6 vars
    let r_local3: Vec<Gf128> = seg("pc").r_local; // 3 vars
    let r_inst: Vec<Gf128> = seg("rs1").r_inst;
    // ---- 4a. BinaryTwist for registers.
    let reg_inst = TwistInstance {
        a: 6,
        ell: 6,
        n,
        p: 2,
        m0: ctx.regs_m0.clone(),
        m0_public: true,
        rv_claims: vec![seg("rs1").value, seg("rs2").value],
        wv_claim: seg("rd").value,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let (reg_proof, reg_twist_claims) = prove_twist(&reg_inst, &wd.reg_wit, t);
    // The register-index Ψ-sum-checks (read 1, read 2, write).
    let port_comps = [
        &wd.reg_wit.ports[0],
        &wd.reg_wit.ports[1],
        &wd.reg_wit.ports[2],
    ];
    let positions = [
        WITNESS_LAYOUT::BUS_RS1,
        WITNESS_LAYOUT::BUS_RS2,
        WITNESS_LAYOUT::BUS_RD,
    ];
    let mut reg_psi = Vec::new();
    let mut reg_index_claims = Vec::new();
    for (s, comps) in port_comps.iter().enumerate() {
        let weights = slice_weights(&r_val, positions[s], 6);
        let (claim, proof) = prove_psi_claim(comps, &r_inst, &weights, &format!("reg-psi-{s}"), t);
        reg_psi.push(proof);
        reg_index_claims.push(claim);
    }
    // ---- 4b. BinaryTwist for RAM.
    // The write-value claim: the new word per cycle (stores) or the
    // current content of cell 0 (no-op writes).
    let ram_wv_claim: Gf128 = {
        let mut cell0: u64 = ctx.ram_words.first().copied().unwrap_or(0);
        let mut acc = ZERO;
        let eq_v = magnetar_field::EqBits::build(&r_val).values;
        let eq_c = magnetar_field::EqBits::build(&r_inst).values;
        for (u, rec) in trace.records.iter().enumerate() {
            // The write value: the written cell's post-write content —
            // the new word on stores, the current content at the access
            // address otherwise (the no-op write).
            let word = if rec.is_store {
                rec.ram_new
            } else if rec.is_load {
                rec.ram_word
            } else {
                cell0
            };
            for o in 0..64usize {
                acc += mul(mul(eq_v[o], eq_c[u]), Gf128(((word >> o) & 1) as u128));
            }
            if rec.is_store && rec.ram_addr == 0 {
                cell0 = rec.ram_new;
            }
        }
        acc
    };
    let ram_m0: Vec<Gf128> = ram_image_bits(ctx.ram_words);
    let ram_inst = TwistInstance {
        a: 8,
        ell: 6,
        n,
        p: 1,
        m0: ram_m0,
        m0_public: ctx.ram_m0_public,
        rv_claims: vec![seg("load-val").value],
        wv_claim: ram_wv_claim,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let (ram_proof, ram_twist_claims) = prove_twist(&ram_inst, &wd.ram_wit, t);
    // The RAM address Ψ-sum-checks (read, write) — checked against the
    // ram-addr segment's VS claim.
    let v_ramaddr = seg("ram-addr").value;
    let mut ram_psi = Vec::new();
    let mut ram_addr_claims = Vec::new();
    for (s, comps) in wd.ram_wit.ports.iter().enumerate() {
        let weights = slice_weights(&r_local3, 0, 8);
        let (claim, proof) = prove_psi_claim(comps, &r_inst, &weights, &format!("ram-psi-{s}"), t);
        ram_psi.push(proof);
        ram_addr_claims.push(claim);
    }
    // Both address claims must equal the segment claim (the eq identity).
    debug_assert_eq!(ram_addr_claims[0], v_ramaddr);
    debug_assert_eq!(ram_addr_claims[1], v_ramaddr);
    // ---- 4c. The verifier-sampled Inc point and claim.
    let inc_point: Vec<Gf128> = t.challenge_field_vec("ram-inc-point", 6 + n);
    let inc_claim: Gf128 = magnetar_field::eval_mle(&wd.ram_wit.inc, &inc_point);
    t.absorb_field("ram-inc-claim", &[inc_claim]);
    // ---- 5a. BinaryRouter for shifts.
    // The X table: the shift_src segment's bits.
    let shift_src_off = pk.layout.offsets[pk.layout.index_of("shift-src").expect("segment")];
    let x_shift: Vec<Gf128> = {
        let mut x = vec![ZERO; (1usize << 6) * q];
        for (u, _) in trace.records.iter().enumerate() {
            for b in 0..64usize {
                x[b | (u << 6)] = joint_bit(&wd.joint_z, u * m, shift_src_off + b);
            }
        }
        x
    };
    // The selector tables (over (h, j)) from the ROM at the fetched rows,
    // with the register-shift stage conditional and the stage sum-check's
    // dense inputs.
    let (sel_lo, sel_hi, sel_st, sel_oh, sel_b, sel_amt) =
        shift_selector_tables(&pk.rom, trace, &wd.joint_z, pk.amount_bits, q);
    let (_, o_v, u_v, h_v) = pk.shift_tensor.clone();
    let shift_layout = RoutingLayout {
        o_vars: o_v,
        j_vars: n,
        u_vars: u_v,
        h_vars: h_v,
        a_vars: 0,
    };
    let shift_r_pub = {
        // The R tensor's own table over (o, u, h) with the joint active
        // positions: o | u | h (j skipped).
        let [_, _j_off, u_off, h_off, _] = shift_layout.offsets();
        let mut active = (0..o_v).collect::<Vec<usize>>();
        active.extend(u_off..u_off + u_v);
        active.extend(h_off..h_off + h_v);
        PaddedTableFactor::new(pk.shift_tensor.0.clone(), active, shift_layout.joint_vars())
    };
    let shift_dense = vec![
        PaddedDense::new(
            x_shift.clone(),
            {
                let [_, j_off, u_off, _, _] = shift_layout.offsets();
                let mut a = (u_off..u_off + u_v).collect::<Vec<usize>>();
                a.extend(j_off..j_off + n);
                a
            },
            shift_layout.joint_vars(),
        ),
        PaddedDense::new(
            sel_lo.clone(),
            {
                let [_, j_off, _, h_off, _] = shift_layout.offsets();
                // Q_lo over the h-block's low `amount_bits` positions.
                let mut a: Vec<usize> = (h_off..h_off + pk.amount_bits).collect();
                a.extend(j_off..j_off + n);
                a
            },
            shift_layout.joint_vars(),
        ),
        PaddedDense::new(
            sel_hi.clone(),
            {
                let [_, j_off, _, h_off, _] = shift_layout.offsets();
                // Q_hi over the h-block's top 2 positions.
                let mut a: Vec<usize> =
                    (h_off + pk.amount_bits..h_off + pk.amount_bits + 2).collect();
                a.extend(j_off..j_off + n);
                a
            },
            shift_layout.joint_vars(),
        ),
    ];
    let shift_claim = seg("shift-out").value;
    let shift_router = prove_routing(
        &shift_layout,
        shift_claim,
        &r_val,
        &r_inst,
        &shift_r_pub,
        &shift_dense,
        2,
        t,
    );
    // ---- 5b. BinaryRouter for sub-word stores.
    let (o_vs, u_vs, h_vs, a_vs) = (
        pk.store_tensor.1,
        pk.store_tensor.2,
        pk.store_tensor.3,
        pk.store_tensor.4,
    );
    let store_layout = RoutingLayout {
        o_vars: o_vs,
        j_vars: n,
        u_vars: u_vs,
        h_vars: h_vs,
        a_vars: a_vs,
    };
    // The X table: (old, src) with u = bit + 64·half.
    let old_off = pk.layout.offsets[pk.layout.index_of("store-old").expect("segment")];
    let src_off = pk.layout.offsets[pk.layout.index_of("store-src").expect("segment")];
    let x_store: Vec<Gf128> = {
        let mut x = vec![ZERO; (1usize << u_vs) * q];
        for (u, _) in trace.records.iter().enumerate() {
            for b in 0..64usize {
                x[b | (u << u_vs)] = joint_bit(&wd.joint_z, u * m, old_off + b);
                x[(b + 64) | (u << u_vs)] = joint_bit(&wd.joint_z, u * m, src_off + b);
            }
        }
        x
    };
    // The offset-link tables.
    let off_off = pk.layout.offsets[pk.layout.index_of("off").expect("segment")];
    let links: Vec<Vec<Gf128>> = (0..3)
        .map(|k| {
            let mut lk = vec![ZERO; 2 * q];
            for (u, _) in trace.records.iter().enumerate() {
                let bit = joint_bit(&wd.joint_z, u * m, off_off + k);
                // eq(a_k, â_k(j)) = 1 + a_k + â_k.
                lk[u << 1] = ONE + bit;
                lk[1 | (u << 1)] = bit;
            }
            lk
        })
        .collect();
    let store_r_pub = {
        let [_, _j_off, u_off, h_off, a_off] = store_layout.offsets();
        let mut active = (0..o_vs).collect::<Vec<usize>>();
        active.extend(u_off..u_off + u_vs);
        active.extend(h_off..h_off + h_vs);
        active.extend(a_off..a_off + a_vs);
        PaddedTableFactor::new(pk.store_tensor.0.clone(), active, store_layout.joint_vars())
    };
    let mut store_dense = vec![
        PaddedDense::new(
            x_store.clone(),
            {
                let [_, j_off, u_off, _, _] = store_layout.offsets();
                let mut a = (u_off..u_off + u_vs).collect::<Vec<usize>>();
                a.extend(j_off..j_off + n);
                a
            },
            store_layout.joint_vars(),
        ),
        PaddedDense::new(
            sel_st.clone(),
            {
                let [_, j_off, _, h_off, _] = store_layout.offsets();
                let mut a = (h_off..h_off + h_vs).collect::<Vec<usize>>();
                a.extend(j_off..j_off + n);
                a
            },
            store_layout.joint_vars(),
        ),
    ];
    for (k, lk) in links.iter().enumerate() {
        let [_, j_off, _, _, a_off] = store_layout.offsets();
        let mut a = vec![a_off + k];
        a.extend(j_off..j_off + n);
        store_dense.push(PaddedDense::new(lk.clone(), a, store_layout.joint_vars()));
    }
    // The output claim: the Inc claim at the verifier-sampled point.
    let (r_inc_o, r_inc_j) = inc_point.split_at(6);
    let store_router = prove_routing(
        &store_layout,
        inc_claim,
        r_inc_o,
        r_inc_j,
        &store_r_pub,
        &store_dense,
        1,
        t,
    );
    // The store X split claims (the two segment MLEs at the router's
    // terminal point).
    let store_x_claims: [Gf128; 2] = {
        let [p_old_pt, p_src_pt] = terminal_store_src_points(&store_layout, &store_router.terminal);
        let old_mle = {
            let mut x = vec![ZERO; (1usize << 6) * q];
            for (u, _) in trace.records.iter().enumerate() {
                for b in 0..64usize {
                    x[b | (u << 6)] = joint_bit(&wd.joint_z, u * m, old_off + b);
                }
            }
            x
        };
        let src_mle = {
            let mut x = vec![ZERO; (1usize << 6) * q];
            for (u, _) in trace.records.iter().enumerate() {
                for b in 0..64usize {
                    x[b | (u << 6)] = joint_bit(&wd.joint_z, u * m, src_off + b);
                }
            }
            x
        };
        [
            magnetar_field::eval_mle(&old_mle, &p_old_pt),
            magnetar_field::eval_mle(&src_mle, &p_src_pt),
        ]
    };
    t.absorb_field("store-x", &store_x_claims);
    // ---- 6. BinaryShout for the bytecode.
    let bus_spec = BusShoutSpec {
        a: a_rom,
        ell: 6,
        n,
        rom_bus: pk.rom.bus.clone(),
        bus_claim: seg("fetch").value,
        index_claims: vec![
            (WITNESS_LAYOUT::BUS_RS1, 6, reg_index_claims[0]),
            (WITNESS_LAYOUT::BUS_RS2, 6, reg_index_claims[1]),
            (WITNESS_LAYOUT::BUS_RD, 6, reg_index_claims[2]),
        ],
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
        pc_r_local: r_local3.clone(),
        pc_claim: seg("pc").value,
    };
    let (bus_shout, bus_shout_claims) =
        prove_bus_shout(&bus_spec, std::slice::from_ref(&wd.fetch_comp), t);
    // ---- 6b. The register-shift stage-selector sum-check (the
    // stage-select extension): binds the shift router's conditional
    // amount selectors to the ROM's stage one-hot and the amount
    // register's bits, and yields the claim `q_stage` that the `sel_lo`
    // selector check composes additively (characteristic two).
    let shift_terminal = shift_router.terminal.clone();
    let stage_spec = {
        let [_, j_off, _, h_off, _] = shift_layout.offsets();
        StageSelectSpec {
            stage_bits: crate::rom::STAGE_BITS,
            n,
            amount_bits: pk.amount_bits,
            r_amount: shift_terminal[h_off..h_off + pk.amount_bits].to_vec(),
            r_j: shift_terminal[j_off..j_off + n].to_vec(),
        }
    };
    let (stage_sel, stage_pts) = prove_stage_select(&stage_spec, &sel_oh, &sel_b, &sel_amt, t);
    // The stage one-hot's selector check: discharges the fetched
    // `sel_stage` claim against the ROM + the fetch one-hot, exactly like
    // the routers' other selectors.
    let sel_stage = SelectorShoutSpec {
        a: a_rom,
        n,
        r_j: stage_pts.r_j.clone(),
        factors: vec![SelectorCheckSpec {
            h_vars: crate::rom::STAGE_BITS,
            r_h: stage_pts.rho_k.clone(),
            claim: stage_sel.oh_terminal,
            rom_sel: pk.rom.sel_stage.clone(),
        }],
    };
    let (sel_stage_proof, sel_stage_claims) = prove_selector_shout(
        &sel_stage,
        std::slice::from_ref(&wd.fetch_comp),
        "sel-stage",
        t,
    );
    // The selector checks.
    let sel_shift = {
        let [_, j_off, _, h_off, _] = shift_layout.offsets();
        let r_j = shift_terminal[j_off..j_off + n].to_vec();
        SelectorShoutSpec {
            a: a_rom,
            n,
            r_j,
            factors: vec![
                SelectorCheckSpec {
                    h_vars: pk.amount_bits,
                    r_h: shift_terminal[h_off..h_off + pk.amount_bits].to_vec(),
                    // The ROM part's claim: the router's Q_lo terminal
                    // plus the structured stage part's claim (the char-2
                    // difference composes Q_lo = ROM(sel_lo) + STAGE).
                    claim: shift_router.q_values[0] + stage_sel.claim,
                    rom_sel: pk.rom.sel_lo.clone(),
                },
                SelectorCheckSpec {
                    h_vars: 2,
                    r_h: shift_terminal[h_off + pk.amount_bits..h_off + pk.amount_bits + 2]
                        .to_vec(),
                    claim: shift_router.q_values[1],
                    rom_sel: pk.rom.sel_hi.clone(),
                },
            ],
        }
    };
    let (sel_shift_proof, sel_shift_claims) = prove_selector_shout(
        &sel_shift,
        std::slice::from_ref(&wd.fetch_comp),
        "sel-shift",
        t,
    );
    let store_terminal = store_router.terminal.clone();
    let sel_store = {
        let [_, j_off, _, h_off, _] = store_layout.offsets();
        let r_j = store_terminal[j_off..j_off + n].to_vec();
        SelectorShoutSpec {
            a: a_rom,
            n,
            r_j,
            factors: vec![SelectorCheckSpec {
                h_vars: h_vs,
                r_h: store_terminal[h_off..h_off + h_vs].to_vec(),
                claim: store_router.q_values[0],
                rom_sel: pk.rom.sel_st.clone(),
            }],
        }
    };
    let (sel_store_proof, sel_store_claims) = prove_selector_shout(
        &sel_store,
        std::slice::from_ref(&wd.fetch_comp),
        "sel-store",
        t,
    );
    // ---- 7. MosaicPCS.
    let stage_mosaic = StageMosaicInputs {
        rho_k: stage_pts.rho_k.clone(),
        r_j: stage_pts.r_j.clone(),
        b: stage_sel.b_terminal,
        amt: stage_sel.amt_terminal,
    };
    let mosaic_claims = collect_mosaic_claims(
        &pk.layout,
        &pk.mosaic,
        n,
        &seg_claims,
        &reg_twist_claims,
        &ram_twist_claims,
        &inc_point,
        inc_claim,
        &bus_shout_claims,
        &sel_shift_claims,
        &sel_store_claims,
        &sel_stage_claims,
        &stage_mosaic,
        &shift_layout,
        &shift_router,
        &shift_terminal,
        &store_layout,
        &store_router,
        &store_terminal,
        &store_x_claims,
    );
    let sp = SegmentProof {
        vs: vs_proof,
        reg_twist: reg_proof,
        reg_psi: [reg_psi[0].clone(), reg_psi[1].clone(), reg_psi[2].clone()],
        reg_index_claims: [
            reg_index_claims[0],
            reg_index_claims[1],
            reg_index_claims[2],
        ],
        ram_twist: ram_proof,
        ram_psi: [ram_psi[0].clone(), ram_psi[1].clone()],
        ram_addr_claims: [ram_addr_claims[0], ram_addr_claims[1]],
        ram_wv_claim,
        inc_point: inc_point.clone(),
        inc_claim,
        shift_router,
        store_router,
        store_x_claims,
        bus_shout,
        stage_select: stage_sel,
        sel_stage: sel_stage_proof,
        sel_shift: sel_shift_proof,
        sel_store: sel_store_proof,
    };
    (sp, mosaic_claims)
}

#[inline]
fn joint_bit(z: &[Gf128], base: usize, col: usize) -> Gf128 {
    z[base + col]
}

/// The X-claim point `(u-slice ++ j-slice)` from a terminal point.
fn terminal_x_point(layout: &RoutingLayout, terminal: &[Gf128]) -> Vec<Gf128> {
    let [_, j_off, u_off, _, _] = layout.offsets();
    let mut pt = terminal[u_off..u_off + layout.u_vars].to_vec();
    pt.extend_from_slice(&terminal[j_off..j_off + layout.j_vars]);
    pt
}

/// The two store source-segment claim points: the u-slice's low 6 vars
/// (the bit index) ++ the j-slice, for each half.
fn terminal_store_src_points(layout: &RoutingLayout, terminal: &[Gf128]) -> [Vec<Gf128>; 2] {
    let x = terminal_x_point(layout, terminal);
    let bits = x[..x.len() - layout.j_vars - 1].to_vec();
    let j = x[x.len() - layout.j_vars..].to_vec();
    let mut p0 = bits.clone();
    p0.extend_from_slice(&j);
    let mut p1 = bits;
    p1.extend_from_slice(&j);
    [p0, p1]
}

/// The six selector-related tables: the shift router's `(q_lo, q_hi)`, the
/// store router's `q_st`, and the stage-selector sum-check's `(oh, b, amt)`.
type SelectorTables = (
    Vec<Gf128>,
    Vec<Gf128>,
    Vec<Gf128>,
    Vec<Gf128>,
    Vec<Gf128>,
    Vec<Gf128>,
);

/// Build the shift router's witness selector tables — the amount `Q_lo`
/// (the ROM's immediate-shift one-hot **plus** the register-shift stage
/// conditional `{identity, shift-by-2^k}` selected by the shamt flag), the
/// kind `Q_hi`, the store width `Q_st` — and the stage-selector
/// sum-check's inputs: the fetched `sel_stage` one-hot `oh`, the
/// per-cycle shamt flags `b`, and the amount register's low-byte bits
/// `amt` (the `store-src` copy segment's first byte, which the R1CS binds
/// to the rs2 virtual segment in every cycle).
fn shift_selector_tables(
    rom: &RomTables,
    trace: &magnetar_cpu::interp::Trace,
    joint_z: &[Gf128],
    amount_bits: usize,
    q: usize,
) -> SelectorTables {
    let m = WITNESS_LAYOUT::M;
    let sb = crate::rom::STAGE_BITS;
    let mut lo = vec![ZERO; (1usize << amount_bits) * q];
    let mut hi = vec![ZERO; 4 * q];
    let mut st = vec![ZERO; 4 * q];
    let mut oh = vec![ZERO; (1usize << sb) * q];
    let mut b = vec![ZERO; q];
    let mut amt = vec![ZERO; (1usize << sb) * q];
    for (j, rec) in trace.records.iter().enumerate() {
        let k = rec.micro_pc % (1usize << rom.a_rom);
        // The fetched stage one-hot (the row's sel_stage column).
        let mut stage: Option<usize> = None;
        for k_slot in 0..(1usize << sb) {
            let v = rom.sel_stage[k_slot | (k << sb)];
            oh[k_slot | (j << sb)] = v;
            if v == ONE {
                stage = Some(k_slot);
            }
        }
        // The witness shamt flag and the amount register's low byte.
        b[j] = joint_z[j * m + WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_SHAMT_BIT];
        for k_slot in 0..(1usize << sb) {
            amt[k_slot | (j << sb)] = joint_z[j * m + WITNESS_LAYOUT::STORE_SRC + k_slot];
        }
        // Q_lo: the stage conditional or the ROM's immediate one-hot.
        if let Some(s) = stage {
            let bit = b[j];
            lo[j << amount_bits] = ONE + bit;
            if (1usize << s) < (1usize << amount_bits) {
                lo[(1usize << s) | (j << amount_bits)] = bit;
            }
        } else {
            for h in 0..(1usize << amount_bits) {
                lo[h | (j << amount_bits)] = rom.sel_lo[h | (k << amount_bits)];
            }
        }
        for h in 0..4usize {
            hi[h | (j << 2)] = rom.sel_hi[h | (k << 2)];
            st[h | (j << 2)] = rom.sel_st[h | (k << 2)];
        }
    }
    (lo, hi, st, oh, b, amt)
}

/// The stage-selector sum-check's Mosaic inputs (symmetric on both sides:
/// the prover's bound points and terminals, the verifier's recomputed
/// claims).
pub(crate) struct StageMosaicInputs {
    rho_k: Vec<Gf128>,
    r_j: Vec<Gf128>,
    b: Gf128,
    amt: Gf128,
}

/// Collect every Mosaic claim (the same collection on both sides; the
/// claim values and points come from symmetric proof structures).
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_mosaic_claims(
    layout: &SegmentLayout,
    mosaic: &MosaicLayout,
    n: usize,
    seg_claims: &[SegmentClaim],
    reg_claims: &TwistPcsClaims,
    ram_claims: &TwistPcsClaims,
    inc_point: &[Gf128],
    inc_claim: Gf128,
    bus_claims: &ShoutPcsClaims,
    sel_shift_claims: &magnetar_memory::SelectorPcsClaims,
    sel_store_claims: &magnetar_memory::SelectorPcsClaims,
    sel_stage_claims: &magnetar_memory::SelectorPcsClaims,
    stage: &StageMosaicInputs,
    shift_layout: &RoutingLayout,
    shift_router: &RouterProof,
    shift_terminal: &[Gf128],
    store_layout: &RoutingLayout,
    store_router: &RouterProof,
    store_terminal: &[Gf128],
    store_x_claims: &[Gf128; 2],
) -> Vec<MosaicClaim> {
    let idx = |name: &str| mosaic.index_of(name).expect("component");
    let mut out = Vec::new();
    // 1. The direct segments.
    for (name, _, direct) in SEGMENTS.iter() {
        if !*direct {
            continue;
        }
        let i = layout.index_of(name).expect("segment");
        let c = &seg_claims[i];
        out.push(MosaicClaim::new(
            idx(&format!("seg-{name}")),
            {
                let mut pt = c.r_local.clone();
                pt.extend_from_slice(&c.r_inst);
                pt
            },
            c.value,
        ));
    }
    // 2. The register Twist's claims.
    let reg_comp = |port: usize| match port {
        0 => idx("reg-ra-1"),
        1 => idx("reg-ra-2"),
        _ => idx("reg-wfa"),
    };
    for (s, per) in reg_claims.read_addr.iter().enumerate() {
        for c in per {
            out.push(MosaicClaim::new(reg_comp(s), c.point.clone(), c.value));
        }
    }
    for c in &reg_claims.write_addr {
        out.push(MosaicClaim::new(idx("reg-wfa"), c.point.clone(), c.value));
    }
    for per in &reg_claims.recon_wfa {
        for c in per {
            out.push(MosaicClaim::new(idx("reg-wfa"), c.point.clone(), c.value));
        }
    }
    for c in &reg_claims.inc {
        out.push(MosaicClaim::new(idx("reg-inc"), c.point.clone(), c.value));
    }
    for (port, per) in reg_claims.st1.iter().enumerate() {
        for c in per {
            out.push(MosaicClaim::new(reg_comp(port), c.point.clone(), c.value));
        }
    }
    for (port, per) in reg_claims.onehot.iter().enumerate() {
        for pair in per {
            for c in pair {
                out.push(MosaicClaim::new(reg_comp(port), c.point.clone(), c.value));
            }
        }
    }
    // 3. The RAM Twist's claims.
    let ram_comp = |port: usize| match port {
        0 => idx("ram-ra-read"),
        _ => idx("ram-ra-write"),
    };
    for (s, per) in ram_claims.read_addr.iter().enumerate() {
        for c in per {
            out.push(MosaicClaim::new(ram_comp(s), c.point.clone(), c.value));
        }
    }
    for c in &ram_claims.write_addr {
        out.push(MosaicClaim::new(
            idx("ram-ra-write"),
            c.point.clone(),
            c.value,
        ));
    }
    for per in &ram_claims.recon_wfa {
        for c in per {
            out.push(MosaicClaim::new(
                idx("ram-ra-write"),
                c.point.clone(),
                c.value,
            ));
        }
    }
    for c in &ram_claims.inc {
        out.push(MosaicClaim::new(idx("ram-inc"), c.point.clone(), c.value));
    }
    // The transmitted entry-image claims (the tree-IVC chunks; empty when
    // the entry image is public).
    for c in &ram_claims.m0 {
        out.push(MosaicClaim::new(idx("ram-init"), c.point.clone(), c.value));
    }
    for (port, per) in ram_claims.st1.iter().enumerate() {
        for c in per {
            out.push(MosaicClaim::new(ram_comp(port), c.point.clone(), c.value));
        }
    }
    for (port, per) in ram_claims.onehot.iter().enumerate() {
        for pair in per {
            for c in pair {
                out.push(MosaicClaim::new(ram_comp(port), c.point.clone(), c.value));
            }
        }
    }
    // 4. The verifier-sampled Inc claim.
    out.push(MosaicClaim::new(
        idx("ram-inc"),
        inc_point.to_vec(),
        inc_claim,
    ));
    // 5. The bytecode Shout's claims.
    for c in &bus_claims.phi {
        out.push(MosaicClaim::new(idx("fetch-ra"), c.point.clone(), c.value));
    }
    for c in &bus_claims.psi {
        out.push(MosaicClaim::new(idx("fetch-ra"), c.point.clone(), c.value));
    }
    for pair in &bus_claims.onehot {
        for c in pair {
            out.push(MosaicClaim::new(idx("fetch-ra"), c.point.clone(), c.value));
        }
    }
    // 6. The selector checks' claims (the shift, store, and stage
    // one-hots — all discharged through the fetch component).
    for src in [sel_shift_claims, sel_store_claims, sel_stage_claims] {
        for c in &src.components {
            out.push(MosaicClaim::new(idx("fetch-ra"), c.point.clone(), c.value));
        }
    }
    // 7. The shift router's X claim → the shift-src segment.
    {
        let x_pt = terminal_x_point(shift_layout, shift_terminal);
        // The u-slice IS the bit index (6 vars) ++ the j-slice.
        let mut pt = x_pt[..6].to_vec();
        pt.extend_from_slice(&x_pt[shift_layout.u_vars..]);
        out.push(MosaicClaim::new(
            idx("seg-shift-src"),
            pt,
            shift_router.x_value,
        ));
    }
    // 8. The store router's X split claims.
    {
        let [p_old, p_src] = terminal_store_src_points(store_layout, store_terminal);
        out.push(MosaicClaim::new(
            idx("seg-store-old"),
            p_old,
            store_x_claims[0],
        ));
        out.push(MosaicClaim::new(
            idx("seg-store-src"),
            p_src,
            store_x_claims[1],
        ));
    }
    // 9. The store router's offset-bit claims → the off segment at the
    // Boolean unit points.
    {
        let [_, j_off, _, _, a_off] = store_layout.offsets();
        let r_j = store_terminal[j_off..j_off + n].to_vec();
        for k in 0..3usize {
            let mut pt = vec![ZERO; 3];
            pt[k] = ONE;
            pt.extend_from_slice(&r_j);
            let r_ak = store_terminal[a_off + k];
            let value = store_router.offset_values[k] + ONE + r_ak;
            out.push(MosaicClaim::new(idx("seg-off"), pt, value));
        }
    }
    // 10. The stage-selector sum-check's claims: the shamt flag on the
    // flags segment (the Boolean point e_{F_SHAMT_BIT} in the segment's
    // local coordinates) and the amount low byte on the store-src copy
    // segment (the sub-cube with the high bits zero).
    {
        let mut pt: Vec<Gf128> = (0..4)
            .map(|i| Gf128(((WITNESS_LAYOUT::F_SHAMT_BIT >> i) & 1) as u128))
            .collect();
        pt.extend_from_slice(&stage.r_j);
        out.push(MosaicClaim::new(idx("seg-flags"), pt, stage.b));
        let mut pt = vec![ZERO; 6];
        for (i, &r) in stage.rho_k.iter().enumerate() {
            pt[i] = r;
        }
        pt.extend_from_slice(&stage.r_j);
        out.push(MosaicClaim::new(idx("seg-store-src"), pt, stage.amt));
    }
    out
}

/// Verify one execution's proof (the single-segment aggregate).
pub fn verify(vk: &VerifyingKey, pub_in: &PublicInput, proof: &Proof) -> Result<(), &'static str> {
    verify_multi(vk, std::slice::from_ref(pub_in), proof)
}

/// Verify the claim-aggregated proof: every segment's body against the ONE
/// commitment, then the single Mosaic discharge.
pub fn verify_multi(
    vk: &VerifyingKey,
    pub_ins: &[PublicInput],
    proof: &Proof,
) -> Result<(), &'static str> {
    assert_eq!(
        pub_ins.len(),
        1usize << vk.idx_bits,
        "segment count mismatch"
    );
    if proof.segments.len() != pub_ins.len() {
        return Err("proof: segment count mismatch");
    }
    let mut pub_bytes: Vec<u8> = Vec::new();
    for pub_in in pub_ins {
        pub_bytes.extend_from_slice(b"|");
        for w in &pub_in.input {
            pub_bytes.extend_from_slice(&w.to_le_bytes());
        }
        pub_bytes.extend_from_slice(b"|");
        pub_bytes.extend_from_slice(&pub_in.output.to_le_bytes());
    }
    let mut t = Transcript::new(
        "magnetar",
        &[
            b"construction-3-v3",
            &vk.program
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            &pub_bytes,
        ],
    );
    t.absorb_digest("cm-g", &proof.commitment_root);
    // Every segment's body, then the lifted claims.
    let mut claims: Vec<MosaicClaim> = Vec::new();
    for (s, (pub_in, seg)) in pub_ins.iter().zip(proof.segments.iter()).enumerate() {
        let ctx = ChunkCtx::fresh(pub_in);
        let local = verify_segment_body(vk, &ctx, seg, &mut t)?;
        for c in local {
            claims.push(lift_claim_with_segment(c, s, vk.idx_bits));
        }
    }
    let cm_stmt = magnetar_pcs::PcsStatement {
        root: proof.commitment_root,
        n: proof.pcs_n,
        k: proof.pcs_k,
        domain: proof.pcs_domain.clone(),
    };
    verify_mosaic(
        &vk.pcs,
        &cm_stmt,
        proof.ell,
        &vk.mosaic,
        &claims,
        &proof.mosaic,
        &mut t,
    )
}

/// Verify one segment's steps 3–6; returns the segment's local Mosaic
/// claims (lifted by the caller).
pub(crate) fn verify_segment_body(
    vk: &VerifyingKey,
    ctx: &ChunkCtx,
    segp: &SegmentProof,
    t: &mut Transcript,
) -> Result<Vec<MosaicClaim>, &'static str> {
    let q = vk.trace_len;
    let m = WITNESS_LAYOUT::M;
    let n = q.trailing_zeros() as usize;
    let a_rom = vk.rom.a_rom;
    let public: Vec<Vec<Gf128>> = {
        let mut v = vec![ZERO; m / 2 - 1];
        v[0] = Gf128(ctx.output as u128);
        for (i, &w) in ctx.input_words.iter().enumerate() {
            if 1 + i < v.len() {
                v[1 + i] = Gf128(w as u128);
            }
        }
        vec![v; q]
    };
    let rel = DualRelation {
        m,
        a_bin: vk.circuit.a_bin.clone(),
        b_bin: vk.circuit.b_bin.clone(),
        c_bin: vk.circuit.c_bin.clone(),
        a_fld: vk.circuit.a_fld.clone(),
        b_fld: vk.circuit.b_fld.clone(),
        c_fld: vk.circuit.c_fld.clone(),
    };
    // ---- 3. Virtual Spartan.
    let seg_claims = verify_virtual_spartan(&rel, &vk.layout, q, &public, &segp.vs, t)?;
    let seg = |name: &str| -> SegmentClaim {
        let i = vk.layout.index_of(name).expect("segment");
        seg_claims[i].clone()
    };
    let r_val: Vec<Gf128> = seg("rs1").r_local;
    let r_local3: Vec<Gf128> = seg("pc").r_local;
    let r_inst: Vec<Gf128> = seg("rs1").r_inst;
    // ---- 4a. The register Twist.
    let reg_inst = TwistInstance {
        a: 6,
        ell: 6,
        n,
        p: 2,
        m0: ctx.regs_m0.clone(),
        m0_public: true,
        rv_claims: vec![seg("rs1").value, seg("rs2").value],
        wv_claim: seg("rd").value,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let reg_claims = verify_twist(&reg_inst, 1, &segp.reg_twist, t)?;
    // The register-index Ψ-sum-checks.
    let positions = [
        WITNESS_LAYOUT::BUS_RS1,
        WITNESS_LAYOUT::BUS_RS2,
        WITNESS_LAYOUT::BUS_RD,
    ];
    for s in 0..3 {
        let weights = slice_weights(&r_val, positions[s], 6);
        verify_psi_claim(
            1,
            6,
            n,
            &r_inst,
            &weights,
            segp.reg_index_claims[s],
            &segp.reg_psi[s],
            &format!("reg-psi-{s}"),
            t,
        )?;
    }
    // ---- 4b. The RAM Twist.
    let ram_m0: Vec<Gf128> = ram_image_bits(ctx.ram_words);
    let ram_inst = TwistInstance {
        a: 8,
        ell: 6,
        n,
        p: 1,
        m0: ram_m0,
        m0_public: ctx.ram_m0_public,
        rv_claims: vec![seg("load-val").value],
        wv_claim: segp.ram_wv_claim,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let ram_claims = verify_twist(&ram_inst, 1, &segp.ram_twist, t)?;
    // The RAM address Ψ-sum-checks — the claims must equal the ram-addr
    // segment's VS claim (the eq identity; a single eq-folded claim
    // avoids the combination leak).
    let v_ramaddr = seg("ram-addr").value;
    for s in 0..2 {
        let weights = slice_weights(&r_local3, 0, 8);
        verify_psi_claim(
            1,
            8,
            n,
            &r_inst,
            &weights,
            segp.ram_addr_claims[s],
            &segp.ram_psi[s],
            &format!("ram-psi-{s}"),
            t,
        )?;
        if segp.ram_addr_claims[s] != v_ramaddr {
            return Err("ram: address claim does not match the segment");
        }
    }
    // The Inc point and claim.
    let inc_point: Vec<Gf128> = t.challenge_field_vec("ram-inc-point", 6 + n);
    if inc_point != segp.inc_point {
        return Err("ram: inc point mismatch");
    }
    t.absorb_field("ram-inc-claim", &[segp.inc_claim]);
    // ---- 5a. The shift router.
    let (_, o_v, u_v, h_v) = vk.shift_tensor.clone();
    let shift_layout = RoutingLayout {
        o_vars: o_v,
        j_vars: n,
        u_vars: u_v,
        h_vars: h_v,
        a_vars: 0,
    };
    let shift_r_pub = {
        let [_, _j_off, u_off, h_off, _] = shift_layout.offsets();
        let mut active = (0..o_v).collect::<Vec<usize>>();
        active.extend(u_off..u_off + u_v);
        active.extend(h_off..h_off + h_v);
        PaddedTableFactor::new(vk.shift_tensor.0.clone(), active, shift_layout.joint_vars())
    };
    let shift_claims = verify_routing(
        &shift_layout,
        seg("shift-out").value,
        &r_val,
        &r_inst,
        &shift_r_pub,
        2,
        0,
        &segp.shift_router,
        t,
    )?;
    // ---- 5b. The store router.
    let (o_vs, u_vs, h_vs, a_vs) = (
        vk.store_tensor.1,
        vk.store_tensor.2,
        vk.store_tensor.3,
        vk.store_tensor.4,
    );
    let store_layout = RoutingLayout {
        o_vars: o_vs,
        j_vars: n,
        u_vars: u_vs,
        h_vars: h_vs,
        a_vars: a_vs,
    };
    let store_r_pub = {
        let [_, _j_off, u_off, h_off, a_off] = store_layout.offsets();
        let mut active = (0..o_vs).collect::<Vec<usize>>();
        active.extend(u_off..u_off + u_vs);
        active.extend(h_off..h_off + h_vs);
        active.extend(a_off..a_off + a_vs);
        PaddedTableFactor::new(vk.store_tensor.0.clone(), active, store_layout.joint_vars())
    };
    let (r_inc_o, r_inc_j) = segp.inc_point.split_at(6);
    let store_claims = verify_routing(
        &store_layout,
        segp.inc_claim,
        r_inc_o,
        r_inc_j,
        &store_r_pub,
        1,
        3,
        &segp.store_router,
        t,
    )?;
    // The store X split: (1 + r_u[top])·v_old + r_u[top]·v_src == the X
    // claim, at the *replayed* terminal point.
    {
        let x_pt = store_claims.x_point(&store_layout);
        let w_old = ONE + x_pt[u_vs - 1];
        let w_src = x_pt[u_vs - 1];
        let combo = mul(w_old, segp.store_x_claims[0]) + mul(w_src, segp.store_x_claims[1]);
        if combo != store_claims.x_value {
            return Err("store: X split mismatch");
        }
    }
    t.absorb_field("store-x", &segp.store_x_claims);
    // ---- 6. BinaryShout for the bytecode.
    let bus_spec = BusShoutSpec {
        a: a_rom,
        ell: 6,
        n,
        rom_bus: vk.rom.bus.clone(),
        bus_claim: seg("fetch").value,
        index_claims: vec![
            (WITNESS_LAYOUT::BUS_RS1, 6, segp.reg_index_claims[0]),
            (WITNESS_LAYOUT::BUS_RS2, 6, segp.reg_index_claims[1]),
            (WITNESS_LAYOUT::BUS_RD, 6, segp.reg_index_claims[2]),
        ],
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
        pc_r_local: r_local3.clone(),
        pc_claim: seg("pc").value,
    };
    let bus_claims = verify_bus_shout(&bus_spec, 1, &segp.bus_shout, t)?;
    // ---- 6b. The register-shift stage-selector sum-check (the
    // stage-select extension).
    let [_, j_off_st, _, h_off_st, _] = shift_layout.offsets();
    let stage_spec = StageSelectSpec {
        stage_bits: crate::rom::STAGE_BITS,
        n,
        amount_bits: vk.amount_bits,
        r_amount: shift_claims.r_joint[h_off_st..h_off_st + vk.amount_bits].to_vec(),
        r_j: shift_claims.r_joint[j_off_st..j_off_st + n].to_vec(),
    };
    let stage_claims = verify_stage_select(&stage_spec, &segp.stage_select, t)?;
    // The stage one-hot's selector check.
    let sel_stage = SelectorShoutSpec {
        a: a_rom,
        n,
        r_j: stage_claims.r_j.clone(),
        factors: vec![SelectorCheckSpec {
            h_vars: crate::rom::STAGE_BITS,
            r_h: stage_claims.rho_k.clone(),
            claim: stage_claims.oh,
            rom_sel: vk.rom.sel_stage.clone(),
        }],
    };
    let sel_stage_pcs = verify_selector_shout(&sel_stage, 1, &segp.sel_stage, "sel-stage", t)?;
    // The selector checks.
    let sel_shift = {
        let [_, j_off, _, h_off, _] = shift_layout.offsets();
        let r_j = shift_claims.r_joint[j_off..j_off + n].to_vec();
        SelectorShoutSpec {
            a: a_rom,
            n,
            r_j,
            factors: vec![
                SelectorCheckSpec {
                    h_vars: vk.amount_bits,
                    r_h: shift_claims.r_joint[h_off..h_off + vk.amount_bits].to_vec(),
                    // The ROM part's claim: the router's Q_lo terminal plus
                    // the structured stage part's claim (char-2 difference).
                    claim: segp.shift_router.q_values[0] + segp.stage_select.claim,
                    rom_sel: vk.rom.sel_lo.clone(),
                },
                SelectorCheckSpec {
                    h_vars: 2,
                    r_h: shift_claims.r_joint[h_off + vk.amount_bits..h_off + vk.amount_bits + 2]
                        .to_vec(),
                    claim: segp.shift_router.q_values[1],
                    rom_sel: vk.rom.sel_hi.clone(),
                },
            ],
        }
    };
    let sel_shift_claims = verify_selector_shout(&sel_shift, 1, &segp.sel_shift, "sel-shift", t)?;
    let sel_store = {
        let [_, j_off, _, h_off, _] = store_layout.offsets();
        let r_j = store_claims.r_joint[j_off..j_off + n].to_vec();
        SelectorShoutSpec {
            a: a_rom,
            n,
            r_j,
            factors: vec![SelectorCheckSpec {
                h_vars: h_vs,
                r_h: store_claims.r_joint[h_off..h_off + h_vs].to_vec(),
                claim: segp.store_router.q_values[0],
                rom_sel: vk.rom.sel_st.clone(),
            }],
        }
    };
    let sel_store_claims = verify_selector_shout(&sel_store, 1, &segp.sel_store, "sel-store", t)?;
    // The segment's local Mosaic claims (the caller lifts them).
    let stage_mosaic = StageMosaicInputs {
        rho_k: stage_claims.rho_k.clone(),
        r_j: stage_claims.r_j.clone(),
        b: stage_claims.b,
        amt: stage_claims.amt,
    };
    Ok(collect_mosaic_claims(
        &vk.layout,
        &vk.mosaic,
        n,
        &seg_claims,
        &reg_claims,
        &ram_claims,
        &segp.inc_point,
        segp.inc_claim,
        &bus_claims,
        &sel_shift_claims,
        &sel_store_claims,
        &sel_stage_pcs,
        &stage_mosaic,
        &shift_layout,
        &segp.shift_router,
        &shift_claims.r_joint,
        &store_layout,
        &segp.store_router,
        &store_claims.r_joint,
        &segp.store_x_claims,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn end_to_end_fibonacci() {
        // addi x1, x0, 1; addi x2, x0, 1; add x3, x1, x2; add x4, x2, x3;
        // ecall (output x10 = 0).
        let program = vec![
            enc_i(1, 0, 0, 1, 0x13),
            enc_i(1, 0, 0, 2, 0x13),
            (2u32 << 20) | (1 << 15) | (3 << 7) | 0x33,
            (3u32 << 20) | (2 << 15) | (4 << 7) | 0x33,
            0x0000_0073,
        ];
        let (pk, vk) = keygen(program, 8);
        let pub_in = PublicInput {
            input: vec![],
            output: 0,
        };
        let proof = prove(&pk, &pub_in);
        assert!(
            verify(&vk, &pub_in, &proof).is_ok(),
            "honest proof must verify"
        );
    }

    #[test]
    fn end_to_end_load_store() {
        // addi x1, x0, 16; addi x5, x0, 0xAB; sb x5, 0(x1);
        // lb x3, 0(x1); lh x4, 0(x1); ecall.
        let program = vec![
            enc_i(16, 0, 0, 1, 0x13),
            enc_i(0xAB, 0, 0, 5, 0x13),
            enc_s(0, 5, 1, 0, 0x23),                 // sb x5, 0(x1)
            (1 << 15) | (3 << 7) | 0x03,             // lb x3, 0(x1)
            (1 << 15) | (1 << 12) | (4 << 7) | 0x03, // lh x4, 0(x1)
            0x0000_0073,
        ];
        let (pk, vk) = keygen(program, 8);
        let pub_in = PublicInput {
            input: vec![],
            output: 0,
        };
        let proof = prove(&pk, &pub_in);
        assert!(
            verify(&vk, &pub_in, &proof).is_ok(),
            "honest load/store proof must verify"
        );
    }

    #[test]
    fn end_to_end_shift() {
        // addi x1, x0, 5; slli x3, x1, 3; ecall — output x10 = 0.
        let program = vec![
            enc_i(5, 0, 0, 1, 0x13),
            (3u32 << 20) | (1 << 15) | (1 << 12) | (3 << 7) | 0x13, // slli x3, x1, 3
            0x0000_0073,
        ];
        let (pk, vk) = keygen_with(program, 4, 2, PcsParams::default());
        let pub_in = PublicInput {
            input: vec![],
            output: 0,
        };
        let proof = prove(&pk, &pub_in);
        assert!(
            verify(&vk, &pub_in, &proof).is_ok(),
            "honest shift proof must verify"
        );
    }

    /// The register-shift stage-select extension: SLL/SRL/SRA by register
    /// expand into 6 conditional routing stages; the stage-selector
    /// sum-check binds the amount bits, and the whole chain (through the
    /// trash cell) is proven end to end.
    #[test]
    fn end_to_end_register_shift() {
        // addi x1, x0, -16; addi x2, x0, 3;
        // sll  x3, x1, x2; srl x4, x3, x2; sra x5, x1, x2;
        // add  x10, x4, x0; ecall.
        let program = vec![
            enc_i((-16i32) as u32 & 0xfff, 0, 0, 1, 0x13), // addi x1, x0, -16
            enc_i(3, 0, 0, 2, 0x13),                       // addi x2, x0, 3
            (2u32 << 20) | (1 << 15) | (1 << 12) | (3 << 7) | 0x33, // sll x3, x1, x2
            (2u32 << 20) | (3 << 15) | (5 << 12) | (4 << 7) | 0x33, // srl x4, x3, x2
            (0x20u32 << 25) | (2 << 20) | (1 << 15) | (5 << 12) | (5 << 7) | 0x33, // sra x5, x1, x2
            (4 << 15) | (10 << 7) | 0x33,                  // add x10, x4, x0
            0x0000_0073,
        ];
        // The native semantics (checked against the interpreter first):
        // x1 = 0xFFFF..FFF0; sll by 3 → 0xFF..FF80; srl by 3 →
        // 0x1FFF_FFFF_FFFF_FFF0.
        let expect: u64 = 0x1FFF_FFFF_FFFF_FFF0;
        {
            let trace = magnetar_cpu::interp::execute(&program, &[], 64);
            assert_eq!(trace.final_state.regs[10], expect);
            // The stage chain uses the trash cell; x0 stays zero.
            assert_eq!(trace.final_state.regs[0], 0);
        }
        // Register shifts need the full 6-bit amount space.
        let (pk, vk) = keygen_with(program, 32, 6, PcsParams::default());
        let pub_in = PublicInput {
            input: vec![],
            output: expect,
        };
        let proof = prove(&pk, &pub_in);
        assert!(
            verify(&vk, &pub_in, &proof).is_ok(),
            "honest register-shift proof must verify"
        );
        // Stage tamper classes.
        // 1. The stage claim: breaks the sel_lo composition
        //    (Q_lo = ROM + STAGE fails at the router's point).
        let mut p = proof.clone();
        p.segments[0].stage_select.claim += Gf128(1);
        assert!(verify(&vk, &pub_in, &p).is_err());
        // 2. A stage sum-check round message.
        let mut p = proof.clone();
        p.segments[0].stage_select.rounds[0][1] += Gf128(1);
        assert!(verify(&vk, &pub_in, &p).is_err());
        // 3. The shamt flag's terminal: the bit-binding identity
        //    (b = the ROM-selected rs2 bit) no longer balances.
        let mut p = proof.clone();
        p.segments[0].stage_select.b_terminal += Gf128(1);
        assert!(verify(&vk, &pub_in, &p).is_err());
        // 4. The amount low-byte terminal.
        let mut p = proof.clone();
        p.segments[0].stage_select.amt_terminal += Gf128(1);
        assert!(verify(&vk, &pub_in, &p).is_err());
        // 5. The stage one-hot's selector check.
        let mut p = proof.clone();
        p.segments[0].sel_stage.per_factor[0].rounds[0][1] += Gf128(1);
        assert!(verify(&vk, &pub_in, &p).is_err());
    }

    #[test]
    fn end_to_end_hot_cell_with_input() {
        // Input word 0 = 7 (the hot cell); the program doubles it via
        // load/add/store:
        //   ld x3, 0(x0); addi x5, x0, 7; add x3, x3, x5; sd x3, 0(x0);
        //   ld x10, 0(x0); ecall  → output 14.
        let program = vec![
            (3 << 12) | (3 << 7) | 0x03, // ld x3, 0(x0)
            enc_i(7, 0, 0, 5, 0x13),
            (5u32 << 20) | (3 << 15) | (3 << 7) | 0x33, // add x3, x3, x5
            enc_s(0, 3, 0, 3, 0x23),                    // sd x3, 0(x0)
            (3 << 12) | (10 << 7) | 0x03,               // ld x10, 0(x0)
            0x0000_0073,
        ];
        let (pk, vk) = keygen(program, 8);
        let pub_in = PublicInput {
            input: vec![7],
            output: 14,
        };
        let proof = prove(&pk, &pub_in);
        assert!(
            verify(&vk, &pub_in, &proof).is_ok(),
            "honest hot-cell proof must verify"
        );
        // A wrong output must fail to even produce a proof (the prover
        // asserts the execution matches).
    }

    /// Claim aggregation (roadmap Horizon 2): K segments share ONE
    /// commitment and ONE evaluation argument; the byte-dominant part of
    /// the proof stays constant while the segment count grows.
    #[test]
    fn end_to_end_claim_aggregation() {
        // The doubling program: ld x3, 0(x0); addi x5, x0, k; add x3, x3, x5;
        // sd x3, 0(x0); ld x10, 0(x0); ecall.
        let prog = vec![
            (3 << 12) | (3 << 7) | 0x03,                // ld x3, 0(x0)
            enc_i(7, 0, 0, 5, 0x13),                    // addi x5, x0, 7
            (5u32 << 20) | (3 << 15) | (3 << 7) | 0x33, // add x3, x3, x5
            enc_s(0, 3, 0, 3, 0x23),                    // sd x3, 0(x0)
            (3 << 12) | (10 << 7) | 0x03,               // ld x10, 0(x0)
            0x0000_0073,
        ];
        let pub_ins: Vec<PublicInput> = [1u64, 2, 3, 4]
            .iter()
            .map(|&k| PublicInput {
                input: vec![k],
                output: k + 7,
            })
            .collect();
        let (pk, vk) = keygen_aggregate(prog, 8, 2, PcsParams::default(), 4);
        let proof = prove_multi(&pk, &pub_ins);
        assert!(
            verify_multi(&vk, &pub_ins, &proof).is_ok(),
            "honest aggregated proof must verify"
        );
        // The cost model: the amortization ledger.
        let stats = crate::stats::proof_stats(&pk, &proof);
        assert_eq!(stats.segments, 4);
        assert!(stats.bytes_eval_argument > 0);
        // The standalone counterfactual must exceed the aggregated proof:
        // four evaluation arguments cost more than one.
        assert!(
            stats.bytes_standalone_counterfactual > stats.bytes_total,
            "aggregation must amortize the evaluation argument"
        );
        // Tamper one segment's VS terminal: rejected.
        let mut p = proof.clone();
        p.segments[1].vs.segment_values[0] += Gf128(1);
        assert!(verify_multi(&vk, &pub_ins, &p).is_err());
        // Tamper the shared Mosaic terminal: rejected.
        let mut p = proof.clone();
        p.mosaic.terminal += Gf128(1);
        assert!(verify_multi(&vk, &pub_ins, &p).is_err());
        // Tamper one segment's stage claim: rejected (the sel_lo
        // composition checks each segment's body).
        let mut p = proof.clone();
        p.segments[3].stage_select.claim += Gf128(1);
        assert!(verify_multi(&vk, &pub_ins, &p).is_err());
        // A swapped public input: rejected.
        let mut bad = pub_ins.clone();
        bad.swap(0, 1);
        assert!(verify_multi(&vk, &bad, &proof).is_err());
    }

    /// The dual-rate profiles: the chunk rate (1/2) and the wrap rate (1/4)
    /// both prove and verify end to end.
    #[test]
    fn end_to_end_dual_rate_profiles() {
        let program = vec![
            enc_i(16, 0, 0, 1, 0x13),
            enc_i(0xAB, 0, 0, 5, 0x13),
            enc_s(0, 5, 1, 0, 0x23),
            (1 << 15) | (3 << 7) | 0x03,
            0x0000_0073,
        ];
        for (name, pcs) in [("chunk", PcsParams::chunk()), ("wrap", PcsParams::wrap())] {
            let (pk, vk) = keygen_with(program.clone(), 8, 2, pcs);
            let pub_in = PublicInput {
                input: vec![],
                output: 0,
            };
            let proof = prove(&pk, &pub_in);
            assert!(
                verify(&vk, &pub_in, &proof).is_ok(),
                "honest {name}-profile proof must verify"
            );
            let stats = crate::stats::proof_stats(&pk, &proof);
            assert_eq!(stats.pcs_k, pk.pcs.k, "the profile's rate must be used");
        }
    }

    #[test]
    fn tamper_rejected() {
        let program = vec![
            enc_i(16, 0, 0, 1, 0x13),
            enc_i(0xAB, 0, 0, 5, 0x13),
            enc_s(0, 5, 1, 0, 0x23),
            (1 << 15) | (3 << 7) | 0x03,
            0x0000_0073,
        ];
        let (pk, vk) = keygen(program, 8);
        let pub_in = PublicInput {
            input: vec![],
            output: 0,
        };
        let proof = prove(&pk, &pub_in);
        assert!(verify(&vk, &pub_in, &proof).is_ok());
        // 1. The Mosaic terminal.
        let mut p = proof.clone();
        p.mosaic.terminal += ONE;
        assert!(
            verify(&vk, &pub_in, &p).is_err(),
            "tampered mosaic terminal"
        );
        // 2. A VS terminal.
        let mut p = proof.clone();
        p.segments[0].vs.terminals[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered vs terminal");
        // 3. A VS segment value (breaks Equation (1)).
        let mut p = proof.clone();
        p.segments[0].vs.segment_values[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered segment value");
        // 4. A register-index claim (breaks the bytecode Shout).
        let mut p = proof.clone();
        p.segments[0].reg_index_claims[0] += ONE;
        assert!(
            verify(&vk, &pub_in, &p).is_err(),
            "tampered reg index claim"
        );
        // 5. The RAM address claim (breaks the segment identity).
        let mut p = proof.clone();
        p.segments[0].ram_addr_claims[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered ram addr claim");
        // 6. The Inc claim (breaks the store router's output).
        let mut p = proof.clone();
        p.segments[0].inc_claim += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered inc claim");
        // 7. A shift router round.
        let mut p = proof.clone();
        p.segments[0].shift_router.rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered shift router");
        // 8. A store router round.
        let mut p = proof.clone();
        p.segments[0].store_router.rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered store router");
        // 9. The store X split claim.
        let mut p = proof.clone();
        p.segments[0].store_x_claims[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered store X");
        // 10. The bus Shout's Φ round.
        let mut p = proof.clone();
        p.segments[0].bus_shout.phi_rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered bus shout");
        // 11. A selector round.
        let mut p = proof.clone();
        p.segments[0].sel_shift.per_factor[0].rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered selector");
        // 12. The commitment root.
        let mut p = proof.clone();
        p.commitment_root[0] ^= 0xFF;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered commitment");
        // 13. The RAM write-value claim.
        let mut p = proof.clone();
        p.segments[0].ram_wv_claim += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered ram wv");
        // 14. A register Twist read round.
        let mut p = proof.clone();
        p.segments[0].reg_twist.read_runs[0].rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered reg twist");
        // 15. A register Ψ round.
        let mut p = proof.clone();
        p.segments[0].reg_psi[0].rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered reg psi");
    }
}
