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
    commit_mosaic, prove_mosaic_opt, verify_mosaic, MosaicClaim, MosaicCommitment, MosaicLayout,
    MosaicProof,
};
use crate::RomTables;

use magnetar_cpu::circuit::{build_circuit, witness_row, CircuitSpec, WITNESS_LAYOUT};
use magnetar_cpu::interp::execute;
use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_memory::{
    prove_bus_shout, prove_psi_claim, prove_selector_shout, prove_twist, verify_bus_shout,
    verify_psi_claim, verify_selector_shout, verify_twist, BusShoutProof, BusShoutSpec,
    SelectorCheckSpec, SelectorShoutProof, SelectorShoutSpec, ShoutPcsClaims, TwistInstance,
    TwistPcsClaims, TwistProof, TwistWitness,
};
use magnetar_pcs::PcsParams;
use magnetar_router::{
    build_shift_tensor, build_store_tensor, prove_routing_join, verify_routing, JoinWitness,
    PaddedTableFactor, RouterProof, RoutingLayout, RoutingTensor, SelCol,
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
    /// PCS parameters.
    pub pcs: PcsParams,
    /// Zero-knowledge masking of the Mosaic batching discharge (roadmap
    /// H1): per-round mask polynomials hide the witness-derived round
    /// messages (see `magnetar_sumcheck::zk`).
    pub zk: bool,
}

impl KeyGen {
    /// Enable ZK masking of the Mosaic batching discharge.
    pub fn enable_zk(&mut self) {
        self.zk = true;
    }
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

/// The public machine state at a segment boundary (the Remark-4
/// initial/final-state binding): the full register file, the micro-PC,
/// the halt latch, and the whole RAM.
#[derive(Clone, Debug, PartialEq)]
pub struct Boundary {
    /// The 64 register cells (x0 hardwired to zero).
    pub regs: Vec<u64>,
    /// The micro-PC (in virtual-instruction units).
    pub pc: usize,
    /// The halt latch.
    pub halted: bool,
    /// The whole RAM (`2^{RAM_WORDS_LOG2}` words).
    pub ram: Vec<u64>,
}

impl Boundary {
    /// The canonical machine start: zero registers and RAM, with the
    /// public input words loaded at word 0.
    pub fn initial(input: &[u64]) -> Self {
        let mut ram = vec![0u64; 1usize << magnetar_cpu::RAM_WORDS_LOG2];
        for (i, &w) in input.iter().enumerate() {
            if i < ram.len() {
                ram[i] = w;
            }
        }
        Boundary {
            regs: vec![0; magnetar_cpu::REGS],
            pc: 0,
            halted: false,
            ram,
        }
    }

    /// The output register (x10) — meaningful when halted.
    pub fn output(&self) -> u64 {
        self.regs.get(10).copied().unwrap_or(0)
    }

    /// The register-file m0 table (address low, bit high): 2^{12} bits.
    fn reg_m0(&self) -> Vec<Gf128> {
        let mut m0 = vec![ZERO; 1usize << 12];
        for (k, &w) in self.regs.iter().enumerate().take(64) {
            for o in 0..64usize {
                m0[k | (o << 6)] = Gf128(((w >> o) & 1) as u128);
            }
        }
        m0
    }

    /// The RAM m0 table (address low, bit high): 2^{14} bits.
    fn ram_m0(&self) -> Vec<Gf128> {
        let mut m0 = vec![ZERO; 1usize << 14];
        for (k, &w) in self.ram.iter().enumerate() {
            for o in 0..64usize {
                m0[k | (o << 8)] = Gf128(((w >> o) & 1) as u128);
            }
        }
        m0
    }
}

/// One segment's public statement: the public input/output binding plus
/// the two boundary states (initial and final).
#[derive(Clone, Debug)]
pub struct SegmentStatement {
    /// The program's public input words (segment 0 only; empty for the
    /// continuation segments).
    pub input: Vec<u64>,
    /// The claimed output — bound only when the segment halts.
    pub output: u64,
    /// The initial boundary (registers and RAM bind via the Twists' m0;
    /// the initial pc and halt latch via public-point claims).
    pub init: Boundary,
    /// The final boundary (registers and RAM bind via the final-state
    /// sum-checks; the final halt latch via a public-point claim).
    pub final_: Boundary,
}

/// The segment argument: every sub-proof of Construction 3's steps 2–6
/// — everything **before** the (shared) commitment and its MosaicPCS
/// discharge. The single-segment [`Proof`] wraps this; tree-IVC shares
/// one commitment across many segments.
#[derive(Clone, Debug)]
pub struct InnerProof {
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
    /// Step 6: the shift selector check.
    pub sel_shift: SelectorShoutProof,
    /// Step 6: the store selector check.
    pub sel_store: SelectorShoutProof,
    /// The Remark-4 final-state bindings (present iff the statement
    /// carries boundary states): registers then RAM.
    pub final_reg: Option<magnetar_memory::FinalStateProof>,
    pub final_ram: Option<magnetar_memory::FinalStateProof>,
    /// The Remark-4 cross-step transition argument.
    pub cross_step: magnetar_memory::CrossStepProof,
}

/// The end-to-end proof.
#[derive(Clone)]
pub struct Proof {
    /// The Mosaic commitment root.
    pub commitment_root: [u8; 32],
    pub pcs_n: usize,
    pub pcs_k: usize,
    pub pcs_domain: String,
    /// ℓ (the Mosaic bit-witness variable count).
    pub ell: usize,
    /// The segment argument (steps 2–6).
    pub inner: InnerProof,
    /// Step 7: the Mosaic discharge.
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
    // The flag-derivation aux block (Remark 4): virtual — its values are
    // R1CS-constrained against the bus and the ALU, so its VS claim needs
    // no external discharge.
    ("crossaux", 1024, false),
];

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
    // JALR's target is data-dependent; the cross-step transition table is
    // static — reject the program (docs/STATUS.md).
    assert!(
        !program.iter().any(|w| w & 0x7f == 0x67),
        "JALR is not supported (see docs/STATUS.md): the cross-step transition table is static"
    );
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
    let rom = RomTables::build(&expanded.rows, amount_bits);
    let n = trace_len.trailing_zeros() as usize;
    let a_rom = rom.a_rom;
    let mosaic = MosaicLayout::build(mosaic_components(n, a_rom));
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
        pcs: pcs.clone(),
        zk: false,
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
        pcs,
    };
    (pk, vk)
}

/// The Mosaic component list (the direct segments then the auxiliary
/// polynomials), at a trace length `2^n`.
fn mosaic_components(n: usize, a_rom: usize) -> Vec<crate::mosaic::ComponentSpec> {
    use crate::mosaic::ComponentSpec;
    let seg = |name: &str, lv: usize| ComponentSpec {
        name: name.to_string(),
        vars: lv + n,
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
    let _ = &mut v;
    v
}

/// One prover-side witness bundle.
pub struct WitnessData {
    joint_z: Vec<Gf128>,
    /// The direct component tables (bits), in Mosaic order.
    direct_tables: Vec<Vec<bool>>,
    /// The register Twist's witness.
    reg_wit: TwistWitness,
    /// The RAM Twist's witness.
    ram_wit: TwistWitness,
    /// The fetch component.
    fetch_comp: magnetar_memory::AddressComponent,
}

/// Assemble the witness from the trace.
fn build_witness(
    pk: &KeyGen,
    pub_in: &PublicInput,
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
        v[0] = Gf128(pub_in.output as u128);
        for (i, &w) in pub_in.input.iter().enumerate() {
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

/// Prove the execution (Construction 3, steps 1–7).
/// Run steps 1–6 (everything but the commitment and its discharge) for
/// one segment. `stmt` carries the boundary states; when `None` the
/// canonical single-segment statement is assumed (zero initial memory
/// with the public input loaded, no final-state binding — the original
/// Construction-3 shape). Returns the segment argument, the Mosaic
/// component tables (for the caller's **shared** commitment), and the
/// Mosaic claims.
#[allow(clippy::too_many_arguments)]
pub fn prove_inner(
    pk: &KeyGen,
    pub_in: &PublicInput,
    stmt: Option<&SegmentStatement>,
    trace: &magnetar_cpu::interp::Trace,
    wd: &WitnessData,
    next_pc_public: Option<usize>,
    t: &mut Transcript,
) -> (InnerProof, Vec<MosaicClaim>) {
    let q = pk.trace_len;
    let m = WITNESS_LAYOUT::M;
    let n = q.trailing_zeros() as usize;
    let a_rom = pk.rom.a_rom;
    let _rows = &trace.program.rows;
    let rel = DualRelation {
        m,
        a_bin: pk.circuit.a_bin.clone(),
        b_bin: pk.circuit.b_bin.clone(),
        c_bin: pk.circuit.c_bin.clone(),
        a_fld: pk.circuit.a_fld.clone(),
        b_fld: pk.circuit.b_fld.clone(),
        c_fld: pk.circuit.c_fld.clone(),
    };
    let _pt = PhaseTimer::new("3a-virtual-spartan");
    // Column bit-ness of `z = (w, 1, x)`: the w-half and the 1-column are
    // bits; the public x columns hold packed words (output + inputs) and
    // stay generic. This is what makes the select-based application and
    // the small-value message paths sound.
    let z_bit_cols: Vec<bool> = {
        let w_cols = magnetar_cpu::WITNESS_LAYOUT::W_COLS;
        let m = magnetar_cpu::WITNESS_LAYOUT::M;
        let mut v = vec![false; m];
        for c in v.iter_mut().take(w_cols + 1) {
            *c = true;
        }
        v
    };
    let (vs_proof, seg_claims) =
        prove_virtual_spartan(&rel, &pk.layout, &wd.joint_z, q, &z_bit_cols, t);
    drop(_pt);
    let seg = |name: &str| -> SegmentClaim {
        let i = pk.layout.index_of(name).expect("segment");
        seg_claims[i].clone()
    };
    // The shared points.
    let r_val: Vec<Gf128> = seg("rs1").r_local; // 6 vars
    let r_local3: Vec<Gf128> = seg("pc").r_local; // 3 vars
    let r_inst: Vec<Gf128> = seg("rs1").r_inst;
    // ---- 4a. BinaryTwist for registers. The initial memory is the
    // boundary's register file when a segment statement is present
    // (the Remark-4 initial-state binding), else the canonical zeros.
    let reg_inst = TwistInstance {
        a: 6,
        ell: 6,
        n,
        p: 2,
        m0: stmt
            .map(|st| st.init.reg_m0())
            .unwrap_or_else(|| vec![ZERO; 1 << 12]),
        rv_claims: vec![seg("rs1").value, seg("rs2").value],
        wv_claim: seg("rd").value,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let _pt = PhaseTimer::new("3b-reg-twist");
    let (reg_proof, reg_twist_claims) = prove_twist(&reg_inst, &wd.reg_wit, t);
    drop(_pt);
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
        // The no-op write value on non-memory cycles: the current content
        // of cell 0 — from the boundary's initial RAM for segments, from
        // the public input word 0 otherwise.
        let mut cell0: u64 = match stmt {
            Some(st) => st.init.ram.first().copied().unwrap_or(0),
            None => pub_in.input.first().copied().unwrap_or(0),
        };
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
    // The RAM's initial memory: the boundary's RAM for segments, else the
    // canonical zeros with the public input words loaded.
    let ram_m0: Vec<Gf128> = match stmt {
        Some(st) => st.init.ram_m0(),
        None => {
            let mut t0 = vec![ZERO; 1usize << 14];
            for (k, &w) in pub_in.input.iter().enumerate() {
                if k < (1usize << 8) {
                    for o in 0..64usize {
                        t0[k | (o << 8)] = Gf128(((w >> o) & 1) as u128);
                    }
                }
            }
            t0
        }
    };
    let ram_inst = TwistInstance {
        a: 8,
        ell: 6,
        n,
        p: 1,
        m0: ram_m0.clone(),
        rv_claims: vec![seg("load-val").value],
        wv_claim: ram_wv_claim,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let _pt = PhaseTimer::new("3c-ram-twist");
    let (ram_proof, ram_twist_claims) = prove_twist(&ram_inst, &wd.ram_wit, t);
    drop(_pt);
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
    // The selector tables (over (h, j)) from the ROM at the fetched rows.
    let (sel_lo, sel_hi, sel_st) = selector_tables_from_trace(&pk.rom, trace, pk.amount_bits, q);
    let (_, o_v, u_v, h_v) = pk.shift_tensor.clone();
    let shift_layout = RoutingLayout {
        o_vars: o_v,
        j_vars: n,
        u_vars: u_v,
        h_vars: h_v,
        a_vars: 0,
    };
    // The shift router runs the join-enumeration prover (linear in T):
    // the per-column X and selector tables, never a joint materialization.
    let shift_wit = JoinWitness {
        x: x_shift.clone(),
        selectors: vec![
            SelCol {
                table: sel_lo.clone(),
                h_off: 0,
                h_vars: pk.amount_bits,
            },
            SelCol {
                table: sel_hi.clone(),
                h_off: pk.amount_bits,
                h_vars: 2,
            },
        ],
        offset_bits: Vec::new(),
    };
    let shift_claim = seg("shift-out").value;
    let _pt = PhaseTimer::new("4a-shift-router");
    let shift_router = prove_routing_join(
        &shift_layout,
        shift_claim,
        &r_val,
        &r_inst,
        &RoutingTensor::from_parts(
            pk.shift_tensor.0.clone(),
            pk.shift_tensor.1,
            pk.shift_tensor.2,
            pk.shift_tensor.3,
            0,
        ),
        &shift_wit,
        t,
    );
    drop(_pt);
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
    // The raw offset bits â_k(j) (per offset variable, one bool per cycle)
    // — the join prover derives the eq-links (1 + a_k + â_k) itself.
    let off_off = pk.layout.offsets[pk.layout.index_of("off").expect("segment")];
    let off_bits: Vec<Vec<bool>> = (0..3)
        .map(|k| {
            trace
                .records
                .iter()
                .map(|rec| (rec.byte_offset >> k) & 1 == 1)
                .collect()
        })
        .collect();
    let _ = off_off;
    // The store router runs the join-enumeration prover as well.
    let store_wit = JoinWitness {
        x: x_store.clone(),
        selectors: vec![SelCol {
            table: sel_st.clone(),
            h_off: 0,
            h_vars: h_vs,
        }],
        offset_bits: off_bits,
    };
    // The output claim: the Inc claim at the verifier-sampled point.
    let (r_inc_o, r_inc_j) = inc_point.split_at(6);
    let _pt = PhaseTimer::new("4b-store-router");
    let store_router = prove_routing_join(
        &store_layout,
        inc_claim,
        r_inc_o,
        r_inc_j,
        &RoutingTensor::from_parts(
            pk.store_tensor.0.clone(),
            pk.store_tensor.1,
            pk.store_tensor.2,
            pk.store_tensor.3,
            pk.store_tensor.4,
        ),
        &store_wit,
        t,
    );
    drop(_pt);
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
    let _pt = PhaseTimer::new("5-bus-shout");
    let (bus_shout, bus_shout_claims) =
        prove_bus_shout(&bus_spec, std::slice::from_ref(&wd.fetch_comp), t);
    drop(_pt);
    // The selector checks.
    let shift_terminal = shift_router.terminal.clone();
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
                    claim: shift_router.q_values[0],
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
    // ---- 7. The Mosaic claims (the caller commits and discharges).
    let mut mosaic_claims = collect_mosaic_claims(
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
        &shift_layout,
        &shift_router,
        &shift_terminal,
        &store_layout,
        &store_router,
        &store_terminal,
        &store_x_claims,
    );
    // ---- 8. The Remark-4 boundary bindings (segments only).
    //
    // (a) The final-state sum-checks bind the public final register file
    //     and RAM to the committed increments and write addresses (the
    //     paper's Remark-4 mechanism).
    // (b) Three public-point Mosaic claims pin the committed fetch
    //     one-hot at cycle 0 to the public initial pc (value 1), and the
    //     committed halt-flag bit at cycles 0 and T−1 to the public
    //     initial/final halt latches. Together with the halt-latch R1CS
    //     constraint this also soundly binds the public output.
    let mut final_reg = None;
    let mut final_ram = None;
    if let Some(st) = stmt {
        let reg_fs = magnetar_memory::FinalStateInstance {
            a: 6,
            ell: 6,
            n,
            m0: st.init.reg_m0(),
            pub_final: st.final_.reg_m0(),
        };
        let (reg_fs_proof, reg_fs_claims) =
            magnetar_memory::prove_final_state(&reg_fs, &wd.reg_wit, t);
        let ram_fs = magnetar_memory::FinalStateInstance {
            a: 8,
            ell: 6,
            n,
            m0: st.init.ram_m0(),
            pub_final: st.final_.ram_m0(),
        };
        let (ram_fs_proof, ram_fs_claims) =
            magnetar_memory::prove_final_state(&ram_fs, &wd.ram_wit, t);
        // Discharge the bindings' claims through the Mosaic.
        let (reg_wfa_pt, reg_wfa_v) = &reg_fs_claims.component_points[0];
        mosaic_claims.push(MosaicClaim::new(
            pk.mosaic.index_of("reg-wfa").expect("component"),
            reg_wfa_pt.clone(),
            *reg_wfa_v,
        ));
        let (reg_inc_pt, reg_inc_v) = &reg_fs_claims.inc_point;
        mosaic_claims.push(MosaicClaim::new(
            pk.mosaic.index_of("reg-inc").expect("component"),
            reg_inc_pt.clone(),
            *reg_inc_v,
        ));
        let (ram_w_pt, ram_w_v) = &ram_fs_claims.component_points[0];
        mosaic_claims.push(MosaicClaim::new(
            pk.mosaic.index_of("ram-ra-write").expect("component"),
            ram_w_pt.clone(),
            *ram_w_v,
        ));
        let (ram_inc_pt, ram_inc_v) = &ram_fs_claims.inc_point;
        mosaic_claims.push(MosaicClaim::new(
            pk.mosaic.index_of("ram-inc").expect("component"),
            ram_inc_pt.clone(),
            *ram_inc_v,
        ));
        // The public-point claims.
        for c in boundary_point_claims(&pk.mosaic, a_rom, n, st) {
            mosaic_claims.push(c);
        }
        final_reg = Some(reg_fs_proof);
        final_ram = Some(ram_fs_proof);
    } else {
        // The single-segment shape: the initial pc (row 0, cycle 0) and
        // the final halt latch (cycle T−1) pin the chain's ends.
        let zero_pt: Vec<Gf128> = vec![ZERO; a_rom + n];
        mosaic_claims.push(MosaicClaim::new(
            pk.mosaic.index_of("fetch-ra").expect("component"),
            zero_pt,
            ONE,
        ));
        let halt_sel: Vec<Gf128> = {
            let h = magnetar_cpu::circuit::WITNESS_LAYOUT::F_HALT;
            let mut v: Vec<Gf128> = (0..4).map(|i| Gf128(((h >> i) & 1) as u128)).collect();
            v.extend(vec![ONE; n]);
            v
        };
        mosaic_claims.push(MosaicClaim::new(
            pk.mosaic.index_of("seg-flags").expect("component"),
            halt_sel,
            ONE,
        ));
    }
    // ---- 6b. The cross-step transition argument (Remark 4): the fetch
    // chain pc(0) = init, fetch(j+1) = TT[fetch(j), taken(j), halt(j)].
    let a_rom_cs = pk.rom.a_rom;
    let n_rows = 1usize << a_rom_cs;
    let taken_col: Vec<Gf128> = trace
        .records
        .iter()
        .map(|r| Gf128(r.branch_taken as u128))
        .collect();
    let halt_col: Vec<Gf128> = trace
        .records
        .iter()
        .map(|r| Gf128(r.halted_latch as u128))
        .collect();
    let cs_spec = magnetar_memory::CrossStepSpec {
        a: a_rom_cs,
        n,
        tt: pk.rom.tt.clone(),
        next_pc_public,
    };
    let (cs_proof, cs_claims) =
        magnetar_memory::prove_cross_step(&cs_spec, &wd.fetch_comp, &taken_col, &halt_col, t);
    let _ = n_rows;

    // The cross-step claims: the two fetch-component evaluations and the
    // taken/halt flag bits at S1's cycle point.
    {
        let fetch_idx = pk.mosaic.index_of("fetch-ra").expect("component");
        mosaic_claims.push(MosaicClaim::new(
            fetch_idx,
            cs_claims.s1_a.point.clone(),
            cs_claims.s1_a.value,
        ));
        mosaic_claims.push(MosaicClaim::new(
            fetch_idx,
            cs_claims.s2_a.point.clone(),
            cs_claims.s2_a.value,
        ));
        let flags_idx = pk.mosaic.index_of("seg-flags").expect("component");
        let flag_pt = |flag: usize, r_j: &[Gf128]| -> Vec<Gf128> {
            let mut v: Vec<Gf128> = (0..4).map(|i| Gf128(((flag >> i) & 1) as u128)).collect();
            v.extend_from_slice(r_j);
            v
        };
        mosaic_claims.push(MosaicClaim::new(
            flags_idx,
            flag_pt(
                magnetar_cpu::circuit::WITNESS_LAYOUT::F_TAKEN,
                &cs_claims.r_j,
            ),
            cs_claims.taken_value,
        ));
        mosaic_claims.push(MosaicClaim::new(
            flags_idx,
            flag_pt(
                magnetar_cpu::circuit::WITNESS_LAYOUT::F_HALT,
                &cs_claims.r_j,
            ),
            cs_claims.halt_value,
        ));
    }

    let inner = InnerProof {
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
        sel_shift: sel_shift_proof,
        sel_store: sel_store_proof,
        final_reg,
        final_ram,
        cross_step: cs_proof,
    };
    (inner, mosaic_claims)
}

/// Build one segment's witness and Mosaic component tables (pure — no
/// transcript interaction, so the caller can commit before proving).
pub fn build_segment(
    pk: &KeyGen,
    pub_in: &PublicInput,
    trace: &magnetar_cpu::interp::Trace,
) -> (WitnessData, Vec<Vec<bool>>) {
    let wd = build_witness(pk, pub_in, trace);
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
    (wd, tables)
}

/// The single-segment protocol (paper §5.5, Construction 3): execute,
/// run steps 2–6, then commit and discharge once.
/// Phase timing (MAGNETAR_TIME=1 prints the per-phase wall clock to
/// stderr — the honest-measurement rule of docs/PERFORMANCE.md).
pub struct PhaseTimer {
    enabled: bool,
    label: &'static str,
    start: std::time::Instant,
}
impl PhaseTimer {
    pub fn new(label: &'static str) -> Self {
        PhaseTimer {
            enabled: std::env::var("MAGNETAR_TIME").is_ok(),
            label,
            start: std::time::Instant::now(),
        }
    }
}
impl Drop for PhaseTimer {
    fn drop(&mut self) {
        if self.enabled {
            eprintln!(
                "[time] {:<24} {:>9.1} ms",
                self.label,
                self.start.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

pub fn prove(pk: &KeyGen, pub_in: &PublicInput) -> Proof {
    prove_with_wrap(pk, pub_in, None)
}

/// Prove, optionally **wrapping** the public evaluation tail at a deep
/// rate (roadmap H2's dual-rate wrap): the chunk side keeps its fast
/// profile; the discharge runs at `deep`'s low rate with the consistency
/// spot-checks binding it to the chunk commitment.
pub fn prove_with_wrap(
    pk: &KeyGen,
    pub_in: &PublicInput,
    deep: Option<magnetar_pcs::PcsParams>,
) -> Proof {
    // ---- 1. Execute (padded to exactly `trace_len` cycles).
    let mut trace = execute(&pk.program, &pub_in.input, pk.trace_len);
    assert_eq!(
        trace.final_state.output, pub_in.output,
        "execution output mismatch"
    );
    assert!(trace.records.len() <= pk.trace_len, "trace exceeds bound");
    magnetar_cpu::interp::pad_records(
        &mut trace.records,
        &mut trace.final_state,
        &trace.program,
        pk.trace_len,
    );
    assert_eq!(trace.records.len(), pk.trace_len);
    let _pt_exec = PhaseTimer::new("1-execute+witness");
    let (wd, tables) = build_segment(pk, pub_in, &trace);
    drop(_pt_exec);
    // ---- 2. Commit (the protocol's only commitment), then bind it into
    // the transcript before any challenge is drawn.
    let _pt_commit = PhaseTimer::new("2-commit");
    let commitment: MosaicCommitment = commit_mosaic(&pk.pcs, &pk.mosaic, &tables);
    drop(_pt_commit);
    let mut t = Transcript::new(
        "magnetar",
        &[
            b"construction-3-v2",
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
        ],
    );
    t.absorb_digest("cm-g", &commitment.stmt.root);
    assert!(
        trace.final_state.halted,
        "the execution must halt within the trace bound"
    );
    let _pt_inner = PhaseTimer::new("3-prove_inner");
    let (inner, mosaic_claims) = prove_inner(pk, pub_in, None, &trace, &wd, None, &mut t);
    drop(_pt_inner);
    // ---- 7. Discharge.
    let _pt_mosaic = PhaseTimer::new("7-mosaic-discharge");
    let mosaic_proof = prove_mosaic_opt(
        &pk.pcs,
        &pk.mosaic,
        &commitment,
        &mosaic_claims,
        pk.zk,
        deep.as_ref(),
        &mut t,
    );
    drop(_pt_mosaic);
    Proof {
        commitment_root: commitment.stmt.root,
        pcs_n: commitment.stmt.n,
        pcs_k: commitment.stmt.k,
        pcs_domain: commitment.stmt.domain,
        ell: commitment.ell,
        inner,
        mosaic: mosaic_proof,
    }
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

/// Build the per-cycle selector tables from the trace's fetch sequence.
fn selector_tables_from_trace(
    rom: &RomTables,
    trace: &magnetar_cpu::interp::Trace,
    amount_bits: usize,
    q: usize,
) -> (Vec<Gf128>, Vec<Gf128>, Vec<Gf128>) {
    // (sel_lo, sel_hi, sel_st) over (h, j).
    let mut lo = vec![ZERO; (1usize << amount_bits) * q];
    let mut hi = vec![ZERO; 4 * q];
    let mut st = vec![ZERO; 4 * q];
    for (j, rec) in trace.records.iter().enumerate() {
        let k = rec.micro_pc % (1usize << rom.a_rom);
        for h in 0..(1usize << amount_bits) {
            lo[h | (j << amount_bits)] = rom.sel_lo[h | (k << amount_bits)];
        }
        for h in 0..4usize {
            hi[h | (j << 2)] = rom.sel_hi[h | (k << 2)];
            st[h | (j << 2)] = rom.sel_st[h | (k << 2)];
        }
    }
    (lo, hi, st)
}

/// The public-point Mosaic claims of the boundary binding:
/// * the fetch one-hot at cycle 0 at the public initial pc — value 1;
/// * the halt flag bit at cycles 0 and T−1 — the public halt latches.
pub fn boundary_point_claims(
    mosaic: &MosaicLayout,
    a_rom: usize,
    n: usize,
    st: &SegmentStatement,
) -> Vec<MosaicClaim> {
    let mut out = Vec::with_capacity(3);
    // fetch-ra at (init pc, cycle 0) = 1.
    let mut pt: Vec<Gf128> = (0..a_rom)
        .map(|i| Gf128(((st.init.pc >> i) & 1) as u128))
        .collect();
    pt.extend(vec![ZERO; n]);
    out.push(MosaicClaim::new(
        mosaic.index_of("fetch-ra").expect("component"),
        pt,
        ONE,
    ));
    // The halt flag at cycles 0 and T−1. The 16 flag bits are 4 MLE
    // variables (little-endian), so the selector for flag bit F_HALT is
    // the **bits of F_HALT**, not the index comparison.
    let halt = magnetar_cpu::circuit::WITNESS_LAYOUT::F_HALT;
    let sel: Vec<Gf128> = (0..4).map(|i| Gf128(((halt >> i) & 1) as u128)).collect();
    let mut pt0 = sel.clone();
    pt0.extend(vec![ZERO; n]);
    out.push(MosaicClaim::new(
        mosaic.index_of("seg-flags").expect("component"),
        pt0,
        Gf128(st.init.halted as u128),
    ));
    let mut ptl = sel;
    ptl.extend(vec![ONE; n]);
    out.push(MosaicClaim::new(
        mosaic.index_of("seg-flags").expect("component"),
        ptl,
        Gf128(st.final_.halted as u128),
    ));
    out
}

/// Collect every Mosaic claim (the same collection on both sides; the
/// claim values and points come from symmetric proof structures).
#[allow(clippy::too_many_arguments)]
fn collect_mosaic_claims(
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
    // 6. The selector checks' claims.
    for src in [sel_shift_claims, sel_store_claims] {
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
            // The point selects the k-th bit COLUMN: the little-endian
            // binary of k (a unit point e_k would select column 2^k).
            let mut pt: Vec<Gf128> = (0..3).map(|i| Gf128(((k >> i) & 1) as u128)).collect();
            pt.extend_from_slice(&r_j);
            let r_ak = store_terminal[a_off + k];
            let value = store_router.offset_values[k] + ONE + r_ak;
            out.push(MosaicClaim::new(idx("seg-off"), pt, value));
        }
    }
    out
}

/// Replay steps 2–6 of one segment and derive its Mosaic claims. The
/// mirror of [`prove_inner`]. The caller seeds the transcript and
/// discharges the returned claims against the shared commitment.
pub fn verify_inner(
    vk: &VerifyingKey,
    pub_in: &PublicInput,
    stmt: Option<&SegmentStatement>,
    inner: &InnerProof,
    next_pc_public: Option<usize>,
    t: &mut Transcript,
) -> Result<Vec<MosaicClaim>, &'static str> {
    let q = vk.trace_len;
    let m = WITNESS_LAYOUT::M;
    let n = q.trailing_zeros() as usize;
    let a_rom = vk.rom.a_rom;
    // The public x per cycle (the I/O binding).
    let public: Vec<Vec<Gf128>> = {
        let mut v = vec![ZERO; m / 2 - 1];
        v[0] = Gf128(pub_in.output as u128);
        for (i, &w) in pub_in.input.iter().enumerate() {
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
    let seg_claims = verify_virtual_spartan(&rel, &vk.layout, q, &public, &inner.vs, t)?;
    let seg = |name: &str| -> SegmentClaim {
        let i = vk.layout.index_of(name).expect("segment");
        seg_claims[i].clone()
    };
    let r_val: Vec<Gf128> = seg("rs1").r_local;
    let r_local3: Vec<Gf128> = seg("pc").r_local;
    let r_inst: Vec<Gf128> = seg("rs1").r_inst;
    // ---- 4a. The register Twist. The initial memory comes from the
    // boundary statement when present.
    let reg_inst = TwistInstance {
        a: 6,
        ell: 6,
        n,
        p: 2,
        m0: stmt
            .map(|st| st.init.reg_m0())
            .unwrap_or_else(|| vec![ZERO; 1 << 12]),
        rv_claims: vec![seg("rs1").value, seg("rs2").value],
        wv_claim: seg("rd").value,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let reg_claims = verify_twist(&reg_inst, 1, &inner.reg_twist, t)?;
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
            inner.reg_index_claims[s],
            &inner.reg_psi[s],
            &format!("reg-psi-{s}"),
            t,
        )?;
    }
    // ---- 4b. The RAM Twist.
    let ram_m0: Vec<Gf128> = match stmt {
        Some(st) => st.init.ram_m0(),
        None => {
            let mut t0 = vec![ZERO; 1usize << 14];
            for (k, &w) in pub_in.input.iter().enumerate() {
                if k < (1usize << 8) {
                    for o in 0..64usize {
                        t0[k | (o << 8)] = Gf128(((w >> o) & 1) as u128);
                    }
                }
            }
            t0
        }
    };
    let ram_inst = TwistInstance {
        a: 8,
        ell: 6,
        n,
        p: 1,
        m0: ram_m0,
        rv_claims: vec![seg("load-val").value],
        wv_claim: inner.ram_wv_claim,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let ram_claims = verify_twist(&ram_inst, 1, &inner.ram_twist, t)?;
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
            inner.ram_addr_claims[s],
            &inner.ram_psi[s],
            &format!("ram-psi-{s}"),
            t,
        )?;
        if inner.ram_addr_claims[s] != v_ramaddr {
            return Err("ram: address claim does not match the segment");
        }
    }
    // The Inc point and claim.
    let inc_point: Vec<Gf128> = t.challenge_field_vec("ram-inc-point", 6 + n);
    if inc_point != inner.inc_point {
        return Err("ram: inc point mismatch");
    }
    t.absorb_field("ram-inc-claim", &[inner.inc_claim]);
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
        let [o_off, _j_off, u_off, h_off, _] = shift_layout.offsets();
        let mut active = (0..o_v).map(|i| o_off + i).collect::<Vec<usize>>();
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
        &inner.shift_router,
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
        let [o_off, _j_off, u_off, h_off, a_off] = store_layout.offsets();
        let mut active = (0..o_vs).map(|i| o_off + i).collect::<Vec<usize>>();
        active.extend(u_off..u_off + u_vs);
        active.extend(h_off..h_off + h_vs);
        active.extend(a_off..a_off + a_vs);
        PaddedTableFactor::new(vk.store_tensor.0.clone(), active, store_layout.joint_vars())
    };
    let (r_inc_o, r_inc_j) = inner.inc_point.split_at(6);
    let store_claims = verify_routing(
        &store_layout,
        inner.inc_claim,
        r_inc_o,
        r_inc_j,
        &store_r_pub,
        1,
        3,
        &inner.store_router,
        t,
    )?;
    // The store X split: (1 + r_u[top])·v_old + r_u[top]·v_src == the X
    // claim, at the *replayed* terminal point.
    {
        let x_pt = store_claims.x_point(&store_layout);
        let w_old = ONE + x_pt[u_vs - 1];
        let w_src = x_pt[u_vs - 1];
        let combo = mul(w_old, inner.store_x_claims[0]) + mul(w_src, inner.store_x_claims[1]);
        if combo != store_claims.x_value {
            return Err("store: X split mismatch");
        }
    }
    t.absorb_field("store-x", &inner.store_x_claims);
    // ---- 6. BinaryShout for the bytecode.
    let bus_spec = BusShoutSpec {
        a: a_rom,
        ell: 6,
        n,
        rom_bus: vk.rom.bus.clone(),
        bus_claim: seg("fetch").value,
        index_claims: vec![
            (WITNESS_LAYOUT::BUS_RS1, 6, inner.reg_index_claims[0]),
            (WITNESS_LAYOUT::BUS_RS2, 6, inner.reg_index_claims[1]),
            (WITNESS_LAYOUT::BUS_RD, 6, inner.reg_index_claims[2]),
        ],
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
        pc_r_local: r_local3.clone(),
        pc_claim: seg("pc").value,
    };
    let bus_claims = verify_bus_shout(&bus_spec, 1, &inner.bus_shout, t)?;
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
                    claim: inner.shift_router.q_values[0],
                    rom_sel: vk.rom.sel_lo.clone(),
                },
                SelectorCheckSpec {
                    h_vars: 2,
                    r_h: shift_claims.r_joint[h_off + vk.amount_bits..h_off + vk.amount_bits + 2]
                        .to_vec(),
                    claim: inner.shift_router.q_values[1],
                    rom_sel: vk.rom.sel_hi.clone(),
                },
            ],
        }
    };
    let sel_shift_claims = verify_selector_shout(&sel_shift, 1, &inner.sel_shift, "sel-shift", t)?;
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
                claim: inner.store_router.q_values[0],
                rom_sel: vk.rom.sel_st.clone(),
            }],
        }
    };
    let sel_store_claims = verify_selector_shout(&sel_store, 1, &inner.sel_store, "sel-store", t)?;
    // ---- 7. MosaicPCS.
    let mut mosaic_claims = collect_mosaic_claims(
        &vk.layout,
        &vk.mosaic,
        n,
        &seg_claims,
        &reg_claims,
        &ram_claims,
        &inner.inc_point,
        inner.inc_claim,
        &bus_claims,
        &sel_shift_claims,
        &sel_store_claims,
        &shift_layout,
        &inner.shift_router,
        &shift_claims.r_joint,
        &store_layout,
        &inner.store_router,
        &store_claims.r_joint,
        &inner.store_x_claims,
    );
    // ---- 8. The Remark-4 boundary bindings (segments only).
    if let Some(st) = stmt {
        let inner_ref = inner;
        let reg_fs = magnetar_memory::FinalStateInstance {
            a: 6,
            ell: 6,
            n,
            m0: st.init.reg_m0(),
            pub_final: st.final_.reg_m0(),
        };
        let reg_fs_claims = magnetar_memory::verify_final_state(
            &reg_fs,
            inner_ref.final_reg.as_ref().expect("reg binding"),
            t,
        )?;
        let ram_fs = magnetar_memory::FinalStateInstance {
            a: 8,
            ell: 6,
            n,
            m0: st.init.ram_m0(),
            pub_final: st.final_.ram_m0(),
        };
        let ram_fs_claims = magnetar_memory::verify_final_state(
            &ram_fs,
            inner_ref.final_ram.as_ref().expect("ram binding"),
            t,
        )?;
        for (name, claims) in [
            ("reg-wfa", &reg_fs_claims),
            ("ram-ra-write", &ram_fs_claims),
        ] {
            let comp = vk.mosaic.index_of(name).expect("component");
            let (pt, v) = &claims.component_points[0];
            mosaic_claims.push(MosaicClaim::new(comp, pt.clone(), *v));
            let idx_inc = vk
                .mosaic
                .index_of(if name == "reg-wfa" {
                    "reg-inc"
                } else {
                    "ram-inc"
                })
                .expect("component");
            let (ipt, iv) = &claims.inc_point;
            mosaic_claims.push(MosaicClaim::new(idx_inc, ipt.clone(), *iv));
        }
        for c in boundary_point_claims(&vk.mosaic, a_rom, n, st) {
            mosaic_claims.push(c);
        }
    } else {
        // The single-segment shape: the initial pc and the final halt.
        let zero_pt: Vec<Gf128> = vec![ZERO; a_rom + n];
        mosaic_claims.push(MosaicClaim::new(
            vk.mosaic.index_of("fetch-ra").expect("component"),
            zero_pt,
            ONE,
        ));
        let halt_sel: Vec<Gf128> = {
            let h = magnetar_cpu::circuit::WITNESS_LAYOUT::F_HALT;
            let mut v: Vec<Gf128> = (0..4).map(|i| Gf128(((h >> i) & 1) as u128)).collect();
            v.extend(vec![ONE; n]);
            v
        };
        mosaic_claims.push(MosaicClaim::new(
            vk.mosaic.index_of("seg-flags").expect("component"),
            halt_sel,
            ONE,
        ));
    }
    // ---- 6b. The cross-step transition argument.
    let cs_spec = magnetar_memory::CrossStepSpec {
        a: vk.rom.a_rom,
        n,
        tt: vk.rom.tt.clone(),
        next_pc_public,
    };
    let cs_claims = magnetar_memory::verify_cross_step(
        &cs_spec,
        vk.rom.a_rom,
        n,
        ZERO,
        ZERO,
        &inner.cross_step,
        t,
    )?;
    {
        let fetch_idx = vk.mosaic.index_of("fetch-ra").expect("component");
        mosaic_claims.push(MosaicClaim::new(
            fetch_idx,
            cs_claims.s1_a.point.clone(),
            cs_claims.s1_a.value,
        ));
        mosaic_claims.push(MosaicClaim::new(
            fetch_idx,
            cs_claims.s2_a.point.clone(),
            cs_claims.s2_a.value,
        ));
        let flags_idx = vk.mosaic.index_of("seg-flags").expect("component");
        let flag_pt = |flag: usize, r_j: &[Gf128]| -> Vec<Gf128> {
            let mut v: Vec<Gf128> = (0..4).map(|i| Gf128(((flag >> i) & 1) as u128)).collect();
            v.extend_from_slice(r_j);
            v
        };
        mosaic_claims.push(MosaicClaim::new(
            flags_idx,
            flag_pt(
                magnetar_cpu::circuit::WITNESS_LAYOUT::F_TAKEN,
                &cs_claims.r_j,
            ),
            cs_claims.taken_value,
        ));
        mosaic_claims.push(MosaicClaim::new(
            flags_idx,
            flag_pt(
                magnetar_cpu::circuit::WITNESS_LAYOUT::F_HALT,
                &cs_claims.r_j,
            ),
            cs_claims.halt_value,
        ));
    }

    Ok(mosaic_claims)
}

/// Verify a single-segment proof.
pub fn verify(vk: &VerifyingKey, pub_in: &PublicInput, proof: &Proof) -> Result<(), &'static str> {
    let mut t = Transcript::new(
        "magnetar",
        &[
            b"construction-3-v2",
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
        ],
    );
    t.absorb_digest("cm-g", &proof.commitment_root);
    let mosaic_claims = verify_inner(vk, pub_in, None, &proof.inner, None, &mut t)?;
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
        &mosaic_claims,
        &proof.mosaic,
        &mut t,
    )
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
    fn enc_b(imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
        let imm = imm as u32;
        (((imm >> 12) & 1) << 31)
            | (((imm >> 5) & 0x3f) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | (((imm >> 1) & 0xf) << 8)
            | (((imm >> 11) & 1) << 7)
            | 0x63
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

    /// The ZK-masked Mosaic batching discharge (roadmap H1): the round
    /// messages are blinded by per-round mask polynomials; the uniform
    /// finish offset closes the terminal identity. The masked proof must
    /// verify, carry a nonzero offset, differ from the plain proof's
    /// transcript, and reject a tampered offset.
    #[test]
    fn end_to_end_zk_masked_batching() {
        let program = vec![
            enc_i(1, 0, 0, 1, 0x13),
            enc_i(1, 0, 0, 2, 0x13),
            (2u32 << 20) | (1 << 15) | (3 << 7) | 0x33,
            (3u32 << 20) | (2 << 15) | (4 << 7) | 0x33,
            0x0000_0073,
        ];
        let (mut pk, vk) = keygen(program, 8);
        pk.enable_zk();
        let pub_in = PublicInput {
            input: vec![],
            output: 0,
        };
        let proof = prove(&pk, &pub_in);
        assert_ne!(
            proof.mosaic.zk_offset,
            magnetar_field::ZERO,
            "the finish offset must be random-looking"
        );
        assert!(verify(&vk, &pub_in, &proof).is_ok(), "ZK proof verifies");
        // A tampered offset breaks the terminal identity.
        let mut p = proof.clone();
        p.mosaic.zk_offset += magnetar_field::ONE;
        assert!(verify(&vk, &pub_in, &p).is_err());
        // A tampered masked round breaks the round check.
        let mut p = proof.clone();
        p.mosaic.local_rounds[0][0][1] += magnetar_field::ONE;
        assert!(verify(&vk, &pub_in, &p).is_err());
    }

    /// The dual-rate wrap (roadmap H2): the chunk profile stays fast; the
    /// public evaluation tail is discharged at `DeepWrap` with the
    /// consistency spot-checks. The wrapped proof verifies, is SMALLER
    /// than the plain proof's tail, and rejects tampered consistency
    /// values, tampered deep roots, and a tampered ring round.
    #[test]
    fn end_to_end_wrapped_deep() {
        use magnetar_pcs::PcsProfile;
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
        let plain = prove(&pk, &pub_in);
        assert!(verify(&vk, &pub_in, &plain).is_ok());
        let deep = PcsProfile::DeepWrap.params();
        let wrapped = prove_with_wrap(&pk, &pub_in, Some(deep.clone()));
        assert!(
            verify(&vk, &pub_in, &wrapped).is_ok(),
            "wrapped proof verifies"
        );
        // The size win: the deep ring (14 queries) + consistency replaces
        // the chunk ring (40 queries).
        let tail_bytes = |p: &Proof| -> usize {
            // The full byte inventory of the evaluation-argument tail:
            // round messages + level roots + queries + tail + anchors
            // (+ the wrap's consistency openings).
            let ring = |r: &magnetar_pcs::RingProof| -> usize {
                let mut n = r.rounds.iter().map(|m| m.len() * 16).sum::<usize>();
                n += r.level_roots.len() * 32;
                for q in &r.queries {
                    n += q.levels.len() * 3 * (16 + 32);
                }
                for (_v, path) in &r.tail {
                    n += 16 + path.len() * 32;
                }
                for a in &r.anchors {
                    n += 16 + a.main_path.len() * 32 + 16 + a.c_path.len() * 32;
                }
                n
            };
            let mut n = p.mosaic.ring.as_ref().map(ring).unwrap_or(0);
            if let Some(w) = &p.mosaic.wrap {
                n += ring(&w.ring);
                n += w.consistency.len()
                    * (16
                        + (w.consistency[0].chunk_path.len() + w.consistency[0].deep_path.len())
                            * 32);
            }
            n
        };
        let plain_tail = tail_bytes(&plain);
        let wrapped_tail = tail_bytes(&wrapped);
        assert!(
            wrapped_tail < plain_tail,
            "the deep tail ({wrapped_tail} B) must beat the chunk tail ({plain_tail} B)"
        );
        // A tampered consistency value breaks the dual opening.
        let mut p = wrapped.clone();
        if let Some(w) = p.mosaic.wrap.as_mut() {
            w.consistency[0].value += magnetar_field::ONE;
        }
        assert!(verify(&vk, &pub_in, &p).is_err());
        // A tampered deep root breaks the ring switch.
        let mut p = wrapped.clone();
        if let Some(w) = p.mosaic.wrap.as_mut() {
            w.deep.root[0] ^= 0xff;
        }
        assert!(verify(&vk, &pub_in, &p).is_err());
        // A tampered deep ring round breaks the round check.
        let mut p = wrapped.clone();
        if let Some(w) = p.mosaic.wrap.as_mut() {
            w.ring.rounds[0][1] += magnetar_field::ONE;
        }
        assert!(verify(&vk, &pub_in, &p).is_err());
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
    fn end_to_end_erc20_mixed_block() {
        // The paper's Table-6 workload shape: balance transfers between
        // two accounts, every 16th failing the underflow check. Input:
        // ram[0] = 1_000_000 (from), ram[1] = 0 (to). 20 transfers, 1
        // skipped (i = 16), each moving 7: from = 1_000_000 − 7·19.
        let sub = |rs2: u32, rs1: u32, rd: u32| {
            (0x20u32 << 25) | (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33
        };
        let add = |rs2: u32, rs1: u32, rd: u32| (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33;
        let sd = |imm: u32, rs2: u32, rs1: u32| {
            (((imm >> 5) & 0x7f) << 25)
                | (rs2 << 20)
                | (rs1 << 15)
                | (3 << 12)
                | ((imm & 0x1f) << 7)
                | 0x23
        };
        let ld = |imm: u32, rd: u32| (imm << 20) | (3 << 12) | (rd << 7) | 0x03;
        let program = vec![
            enc_i(20, 0, 0, 1, 0x13),           // x1 = 20
            enc_i(7, 0, 0, 2, 0x13),            // x2 = 7
            ld(0, 4),                           // x4 = ram[0]
            ld(8, 5),                           // x5 = ram[1]
            enc_i(15, 1, 7, 6, 0x13),           // x6 = x1 & 15
            enc_i(15, 0, 0, 7, 0x13),           // x7 = 15
            enc_b(24, 7, 6, 0),                 // beq x6, x7, skip
            enc_b(20, 2, 4, 6),                 // bltu x4, x2, skip
            sub(2, 4, 4),                       // x4 -= x2
            add(2, 5, 5),                       // x5 += x2
            sd(0, 4, 0),                        // ram[0] = x4
            sd(8, 5, 0),                        // ram[1] = x5
            enc_i(-1i32 as u32, 1, 0, 1, 0x13), // skip: x1 -= 1
            enc_b(-44, 0, 1, 1),                // bne x1, x0, loop
            ld(0, 10),                          // x10 = ram[0]
            0x0000_0073,                        // ecall
        ];
        let input = vec![1_000_000u64, 0u64];
        let (pk, vk) = keygen_with(program.clone(), 256, 2, PcsParams::default());
        let pub_in = PublicInput {
            input: input.clone(),
            output: 1_000_000 - 7 * 19,
        };
        let proof = prove(&pk, &pub_in);
        assert!(
            verify(&vk, &pub_in, &proof).is_ok(),
            "honest erc20 mixed-block proof must verify"
        );
        // Tamper: claim the wrong final balance.
        let bad = PublicInput {
            input: input.clone(),
            output: 1_000_000 - 7 * 18,
        };
        assert!(
            verify(&vk, &bad, &proof).is_err(),
            "a wrong public output must be rejected"
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
        p.inner.vs.terminals[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered vs terminal");
        // 3. A VS segment value (breaks Equation (1)).
        let mut p = proof.clone();
        p.inner.vs.segment_values[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered segment value");
        // 4. A register-index claim (breaks the bytecode Shout).
        let mut p = proof.clone();
        p.inner.reg_index_claims[0] += ONE;
        assert!(
            verify(&vk, &pub_in, &p).is_err(),
            "tampered reg index claim"
        );
        // 5. The RAM address claim (breaks the segment identity).
        let mut p = proof.clone();
        p.inner.ram_addr_claims[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered ram addr claim");
        // 6. The Inc claim (breaks the store router's output).
        let mut p = proof.clone();
        p.inner.inc_claim += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered inc claim");
        // 7. A shift router round.
        let mut p = proof.clone();
        p.inner.shift_router.rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered shift router");
        // 8. A store router round.
        let mut p = proof.clone();
        p.inner.store_router.rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered store router");
        // 9. The store X split claim.
        let mut p = proof.clone();
        p.inner.store_x_claims[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered store X");
        // 10. The bus Shout's Φ round.
        let mut p = proof.clone();
        p.inner.bus_shout.phi_rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered bus shout");
        // 11. A selector round.
        let mut p = proof.clone();
        p.inner.sel_shift.per_factor[0].rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered selector");
        // 12. The commitment root.
        let mut p = proof.clone();
        p.commitment_root[0] ^= 0xFF;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered commitment");
        // 13. The RAM write-value claim.
        let mut p = proof.clone();
        p.inner.ram_wv_claim += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered ram wv");
        // 14. A register Twist read round.
        let mut p = proof.clone();
        p.inner.reg_twist.read_runs[0].rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered reg twist");
        // 15. A register Ψ round.
        let mut p = proof.clone();
        p.inner.reg_psi[0].rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered reg psi");
    }
}

#[cfg(test)]
mod cross_step_tests {
    use super::*;

    fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
        ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
    }

    fn enc_b(imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
        let imm = imm as u32;
        (((imm >> 12) & 1) << 31)
            | (((imm >> 5) & 0x3f) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | ((imm >> 1) & 0xf) << 8
            | (((imm >> 11) & 1) << 7)
            | 0x63
    }

    /// A jumped-pc trace: swap two order-independent instructions — the
    /// memory arguments stay consistent but the cross-step chain breaks.
    #[test]
    fn tamper_jumped_pc_rejected() {
        // addi x1, x0, 1; addi x2, x0, 2; addi x3, x0, 3; ecall.
        let program = vec![
            enc_i(1, 0, 0, 1, 0x13),
            enc_i(2, 0, 0, 2, 0x13),
            enc_i(3, 0, 0, 3, 0x13),
            0x0000_0073,
        ];
        let (pk, vk) = keygen(program.clone(), 8);
        let pub_in = PublicInput {
            input: vec![],
            output: 0,
        };
        // The honest proof.
        let honest = prove(&pk, &pub_in);
        assert!(verify(&vk, &pub_in, &honest).is_ok());
        // The jumped trace: swap records 1 and 2 (fetch order 0, 2, 1, 3).
        let mut trace = execute(&pk.program, &pub_in.input, pk.trace_len);
        magnetar_cpu::interp::pad_records(
            &mut trace.records,
            &mut trace.final_state,
            &trace.program,
            pk.trace_len,
        );
        trace.records.swap(1, 2);
        let (wd, tables) = build_segment(&pk, &pub_in, &trace);
        let mut t = Transcript::new(
            "magnetar",
            &[
                b"construction-3-v2",
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
            ],
        );
        let commitment = commit_mosaic(&pk.pcs, &pk.mosaic, &tables);
        t.absorb_digest("cm-g", &commitment.stmt.root);
        let (inner, mosaic_claims) = prove_inner(&pk, &pub_in, None, &trace, &wd, None, &mut t);
        let _pt_mosaic = PhaseTimer::new("7-mosaic-discharge");
        let mosaic_proof = prove_mosaic_opt(
            &pk.pcs,
            &pk.mosaic,
            &commitment,
            &mosaic_claims,
            pk.zk,
            None,
            &mut t,
        );
        drop(_pt_mosaic);
        let proof = Proof {
            commitment_root: commitment.stmt.root,
            pcs_n: commitment.stmt.n,
            pcs_k: commitment.stmt.k,
            pcs_domain: commitment.stmt.domain,
            ell: commitment.ell,
            inner,
            mosaic: mosaic_proof,
        };
        assert!(
            verify(&vk, &pub_in, &proof).is_err(),
            "a jumped pc must be rejected by the cross-step argument"
        );
    }

    /// A branch program: the honest proof verifies, and a tampered
    /// cross-step round is rejected.
    #[test]
    fn tamper_branch_cross_step_rejected() {
        // li x1, 1; beq x0, x0, +8 (skip one); addi x2, x0, 99; addi x10, x0, 5;
        // ecall → output 5.
        let program = vec![
            enc_i(1, 0, 0, 1, 0x13),  // addi x1, x0, 1
            enc_b(8, 0, 0, 0),        // beq x0, x0, +8 → the addi x10
            enc_i(99, 0, 0, 2, 0x13), // addi x2, x0, 99 (skipped)
            enc_i(5, 0, 0, 10, 0x13), // addi x10, x0, 5
            0x0000_0073,
        ];
        let (pk, vk) = keygen(program.clone(), 8);
        let pub_in = PublicInput {
            input: vec![],
            output: 5,
        };
        let proof = prove(&pk, &pub_in);
        assert!(verify(&vk, &pub_in, &proof).is_ok());
        // A tampered cross-step round.
        let mut p = proof.clone();
        p.inner.cross_step.s1_rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err());
        // A tampered taken-link claim.
        let mut p = proof.clone();
        p.inner.cross_step.s1_link_claims[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err());
        // A tampered S2 claim.
        let mut p = proof.clone();
        p.inner.cross_step.s2_a_claim += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err());
    }

    /// JALR is rejected at keygen (the static transition table).
    #[test]
    #[should_panic(expected = "JALR")]
    fn jalr_rejected_at_keygen() {
        let program = vec![
            enc_i(4, 0, 0, 1, 0x13), // addi x1, x0, 4
            (1 << 15) | 0x67,        // jalr x0, 0(x1)
            0x0000_0073,
        ];
        let _ = keygen(program, 8);
    }
}
