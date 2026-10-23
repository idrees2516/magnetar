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
    build_shift_tensor, build_store_tensor, prove_routing, verify_routing, PaddedDense,
    PaddedTableFactor, RouterProof, RoutingLayout,
};
use magnetar_spartan::{
    prove_virtual_spartan, verify_virtual_spartan, DualRelation, SegmentClaim, SegmentLayout,
    SegmentSpec, VirtualSpartanProof,
};

/// The shift-amount routing width: 6 covers the full 64-entry shamt space
/// (RV64I shamt is 6 bits). The routing tensor grows to 2^(6+6+8) entries
/// and the ROM's sel_lo columns to 64 per row — both trivial against the
/// other commitments.
pub const AMOUNT_BITS: usize = 6;

/// The (kind × stage) one-hot width (5 bits: 4 kinds × 8 stages).
fn magnetar_zkvm_rom_ks_bits() -> usize {
    crate::rom::SEL_KS_BITS
}

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
    /// The register-shift stage routing tensor
    /// `(values, o_vars, u_vars, h_vars)`.
    pub stage_tensor: (Vec<Gf128>, usize, usize, usize),
    /// The store routing tensor `(values, o_vars, u_vars, h_vars, a_vars)`.
    pub store_tensor: (Vec<Gf128>, usize, usize, usize, usize),
    pub amount_bits: usize,
    /// The trace length bound (a power of two).
    pub trace_len: usize,
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
    pub stage_tensor: (Vec<Gf128>, usize, usize, usize),
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
    /// Step 5: the register-shift stage router (the 6-stage chain's
    /// routing validation; the b-select factor's claim is
    /// `stage_g_claim`).
    pub stage_router: RouterProof,
    /// The stage routing's b-select factor claim `Ĝ'(r_b, r_s, r_j)` —
    /// the verifier derives the shamt-segment claim
    /// `Ŝ(r_s, r_j) = stage_g_claim + 1 + r_b` from it.
    pub stage_g_claim: Gf128,
    /// Step 5: the store router.
    pub store_router: RouterProof,
    /// Step 5: the store X split claims (old, src).
    pub store_x_claims: [Gf128; 2],
    /// Step 6: the bytecode Shout.
    pub bus_shout: BusShoutProof,
    /// Step 6: the shift selector check.
    pub sel_shift: SelectorShoutProof,
    /// Step 6: the stage selector check (the (kind, stage) one-hot
    /// against the ROM's sel_ks columns).
    pub sel_stage: SelectorShoutProof,
    /// Step 6: the class-selector check (the decode-class one-hot
    /// against the ROM's class columns — the flag folds' anchor).
    pub sel_class: SelectorShoutProof,
    /// Step 6: the store selector check.
    pub sel_store: SelectorShoutProof,
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
    ("stage-out", 64, true),
    ("t2", 64, true),
    ("carry", 64, true),
    ("alu", 64, true),
    ("pand", 64, true),
    ("shift-src", 64, true),
    ("store-old", 64, true),
    ("store-src", 64, true),
    ("class", 32, true),
    ("flags", 16, true),
    ("off", 8, true),
    ("ram-addr", 8, true),
    ("pc", 8, false),
    ("aux", 16, true),
    ("fetch", 64, false),
    ("eqaux", 64, true),
    ("shamt", 8, true),
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
    let stage_tensor = magnetar_router::build_stage_tensor(64);
    let store_tensor = build_store_tensor(64);
    let pk = KeyGen {
        program: program.clone(),
        circuit: circuit.clone(),
        layout: layout.clone(),
        mosaic: mosaic.clone(),
        rom: rom.clone(),
        shift_tensor: shift_tensor.clone(),
        stage_tensor: stage_tensor.clone(),
        store_tensor: store_tensor.clone(),
        amount_bits,
        trace_len,
        pcs: pcs.clone(),
    };
    let vk = VerifyingKey {
        program,
        circuit,
        layout,
        mosaic,
        rom,
        shift_tensor,
        stage_tensor,
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
        seg("seg-stage-out", 6),
        seg("seg-t2", 6),
        seg("seg-carry", 6),
        seg("seg-alu", 6),
        seg("seg-pand", 6),
        seg("seg-shift-src", 6),
        seg("seg-store-old", 6),
        seg("seg-store-src", 6),
        seg("seg-class", 5),
        seg("seg-flags", 4),
        seg("seg-off", 3),
        seg("seg-ram-addr", 3),
        seg("seg-aux", 4),
        seg("seg-eqaux", 6),
        seg("seg-shamt", 3),
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
struct WitnessData {
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
pub fn prove(pk: &KeyGen, pub_in: &PublicInput) -> Proof {
    let time_phases = std::env::var("MAGNETAR_TIME").is_ok();
    let t0 = std::time::Instant::now();
    let mark = |time_phases: bool, label: &str, t0: &std::time::Instant| {
        if time_phases {
            eprintln!("[phase] {label}: {:?}s", t0.elapsed().as_secs_f64());
        }
    };
    // ---- 1. Execute.
    let mut trace = execute(&pk.program, &pub_in.input, pk.trace_len);
    assert_eq!(
        trace.final_state.output, pub_in.output,
        "execution output mismatch"
    );
    assert!(trace.records.len() <= pk.trace_len, "trace exceeds bound");
    // Pad the records up to the proving key's trace length (the witness
    // covers every cycle of the trace).
    trace.records = magnetar_cpu::interp::pad_records(&trace, pk.trace_len);
    let q = pk.trace_len;
    let m = WITNESS_LAYOUT::M;
    let n = q.trailing_zeros() as usize;
    let a_rom = pk.rom.a_rom;
    let _rows = &trace.program.rows;
    let wd = build_witness(pk, pub_in, &trace);
    mark(time_phases, "witness", &t0);
    // ---- 2. Commit.
    // The component tables in Mosaic order.
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
    let commitment: MosaicCommitment = commit_mosaic(&pk.pcs, &pk.mosaic, &tables);
    mark(time_phases, "commit", &t0);
    // ---- 3. Virtual Spartan.
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
    let rel = DualRelation {
        m,
        a_bin: pk.circuit.a_bin.clone(),
        b_bin: pk.circuit.b_bin.clone(),
        c_bin: pk.circuit.c_bin.clone(),
        a_fld: pk.circuit.a_fld.clone(),
        b_fld: pk.circuit.b_fld.clone(),
        c_fld: pk.circuit.c_fld.clone(),
    };
    let (vs_proof, seg_claims) = prove_virtual_spartan(&rel, &pk.layout, &wd.joint_z, q, &mut t);
    mark(time_phases, "virtual-spartan", &t0);
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
        m0: vec![ZERO; 1 << 12],
        rv_claims: vec![seg("rs1").value, seg("rs2").value],
        wv_claim: seg("rd").value,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let (reg_proof, reg_twist_claims) = prove_twist(&reg_inst, &wd.reg_wit, &mut t);
    mark(time_phases, "reg-twist", &t0);
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
        let (claim, proof) =
            prove_psi_claim(comps, &r_inst, &weights, &format!("reg-psi-{s}"), &mut t);
        reg_psi.push(proof);
        reg_index_claims.push(claim);
    }
    // ---- 4b. BinaryTwist for RAM.
    // The write-value claim: the new word per cycle (stores) or the
    // current content of cell 0 (no-op writes).
    let ram_wv_claim: Gf128 = {
        let mut cell0: u64 = pub_in.input.first().copied().unwrap_or(0);
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
    let ram_m0: Vec<Gf128> = {
        let mut t0 = vec![ZERO; 1usize << 14];
        for (k, &w) in pub_in.input.iter().enumerate() {
            if k < (1usize << 8) {
                for o in 0..64usize {
                    t0[k | (o << 8)] = Gf128(((w >> o) & 1) as u128);
                }
            }
        }
        t0
    };
    let ram_inst = TwistInstance {
        a: 8,
        ell: 6,
        n,
        p: 1,
        m0: ram_m0,
        rv_claims: vec![seg("load-val").value],
        wv_claim: ram_wv_claim,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let (ram_proof, ram_twist_claims) = prove_twist(&ram_inst, &wd.ram_wit, &mut t);
    mark(time_phases, "ram-twist", &t0);
    // The RAM address Ψ-sum-checks (read, write) — checked against the
    // ram-addr segment's VS claim.
    let v_ramaddr = seg("ram-addr").value;
    let mut ram_psi = Vec::new();
    let mut ram_addr_claims = Vec::new();
    for (s, comps) in wd.ram_wit.ports.iter().enumerate() {
        let weights = slice_weights(&r_local3, 0, 8);
        let (claim, proof) =
            prove_psi_claim(comps, &r_inst, &weights, &format!("ram-psi-{s}"), &mut t);
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
    // The X table: the shift-src copy segment's bits.
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
    let (sel_lo, sel_hi, sel_st) = selector_tables_from_trace(&pk.rom, &trace, pk.amount_bits, q);
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
    mark(time_phases, "pre-shift-router", &t0);
    let shift_router = prove_routing(
        &shift_layout,
        shift_claim,
        &r_val,
        &r_inst,
        &shift_r_pub,
        &shift_dense,
        2,
        &mut t,
    );
    // ---- 5a'. BinaryRouter for register-shift stages.
    // Each stage row is a static routing (paper §5.4): the h space is
    // kind(2) | stage(3) | b(1); the runtime amount bit b selects the
    // identity (b=0) or the static 2^stage shift (b=1). Q_ks is the
    // (kind, stage) one-hot from the ROM (the selector check
    // authenticates it); the b-select factor G' = 1 + h_b + shamt(stage, j)
    // reads the committed shamt segment (rs2's low 6 bits), so its
    // terminal claim discharges to a shamt-segment claim.
    let (st_o, st_u, st_h) = (pk.stage_tensor.1, pk.stage_tensor.2, pk.stage_tensor.3);
    let stage_layout = RoutingLayout {
        o_vars: st_o,
        j_vars: n,
        u_vars: st_u,
        h_vars: st_h,
        a_vars: 0,
    };
    // The Q_ks table over (h_ks, j) from the ROM at the fetched rows.
    let sel_ks_j: Vec<Gf128> = {
        let w = 1usize << magnetar_zkvm_rom_ks_bits();
        let mut tab = vec![ZERO; w * q];
        for (u, rec) in trace.records.iter().enumerate() {
            let k = rec.micro_pc % (1usize << a_rom);
            for h in 0..w {
                tab[h | (u << magnetar_zkvm_rom_ks_bits())] =
                    pk.rom.sel_ks[h | (k << magnetar_zkvm_rom_ks_bits())];
            }
        }
        tab
    };
    // The G' table over (b, stage, j): 1 + b + shamt(stage, j).
    let (g_tab, shamt_cols) = {
        let shamt_off = pk.layout.offsets[pk.layout.index_of("shamt").expect("segment")];
        let cols: Vec<Vec<Gf128>> = (0..8)
            .map(|s| {
                let mut c = vec![ZERO; q];
                for (u, _) in trace.records.iter().enumerate() {
                    c[u] = joint_bit(&wd.joint_z, u * m, shamt_off + s);
                }
                c
            })
            .collect();
        let mut tab = vec![ZERO; 16 * q];
        for u in 0..q {
            for b in 0..2usize {
                for s in 0..8usize {
                    tab[b | (s << 1) | (u << 4)] = Gf128(b as u128) + cols[s][u] + ONE;
                }
            }
        }
        (tab, cols)
    };
    let _ = shamt_cols;
    let stage_r_pub = {
        let [_, j_off, u_off, h_off, _] = stage_layout.offsets();
        let mut active = (0..st_o).collect::<Vec<usize>>();
        active.extend(u_off..u_off + st_u);
        active.extend(h_off..h_off + st_h);
        PaddedTableFactor::new(pk.stage_tensor.0.clone(), active, stage_layout.joint_vars())
    };
    let stage_dense = vec![
        PaddedDense::new(
            x_shift.clone(),
            {
                let [_, j_off, u_off, _, _] = stage_layout.offsets();
                let mut a = (u_off..u_off + st_u).collect::<Vec<usize>>();
                a.extend(j_off..j_off + n);
                a
            },
            stage_layout.joint_vars(),
        ),
        PaddedDense::new(
            sel_ks_j.clone(),
            {
                let [_, j_off, _, h_off, _] = stage_layout.offsets();
                // Q_ks over the h block's low 5 positions (kind|stage).
                let mut a: Vec<usize> = (h_off..h_off + 5).collect();
                a.extend(j_off..j_off + n);
                a
            },
            stage_layout.joint_vars(),
        ),
        PaddedDense::new(
            g_tab,
            {
                let [_, j_off, _, h_off, _] = stage_layout.offsets();
                // G' over (b = h bit 5, stage = h bits 2..5, j): the
                // table's own order is (b, stage, j).
                let mut a: Vec<usize> = vec![h_off + 5];
                a.extend(h_off + 2..h_off + 5);
                a.extend(j_off..j_off + n);
                a
            },
            stage_layout.joint_vars(),
        ),
    ];
    let stage_claim = seg("stage-out").value;
    mark(time_phases, "shift-router", &t0);
    let stage_router = prove_routing(
        &stage_layout,
        stage_claim,
        &r_val,
        &r_inst,
        &stage_r_pub,
        &stage_dense,
        2,
        &mut t,
    );
    let stage_g_claim = stage_router.q_values[1];
    let stage_terminal = stage_router.terminal.clone();
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
    mark(time_phases, "stage-router", &t0);
    let store_router = prove_routing(
        &store_layout,
        inc_claim,
        r_inc_o,
        r_inc_j,
        &store_r_pub,
        &store_dense,
        1,
        &mut t,
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
    mark(time_phases, "store-router", &t0);
    let (bus_shout, bus_shout_claims) =
        prove_bus_shout(&bus_spec, std::slice::from_ref(&wd.fetch_comp), &mut t);
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
    mark(time_phases, "bus-shout", &t0);
    let (sel_shift_proof, sel_shift_claims) = prove_selector_shout(
        &sel_shift,
        std::slice::from_ref(&wd.fetch_comp),
        "sel-shift",
        &mut t,
    );
    // The class-selector check: the decode-class one-hot vs the ROM's
    // class columns (the anchor of every flag fold).
    let class_sel_spec = {
        let cls = seg("class");
        SelectorShoutSpec {
            a: a_rom,
            n,
            r_j: r_inst.clone(),
            factors: vec![SelectorCheckSpec {
                h_vars: 5,
                r_h: cls.r_local.clone(),
                claim: cls.value,
                rom_sel: pk.rom.class.clone(),
            }],
        }
    };
    mark(time_phases, "sel-shift", &t0);
    let (sel_class_proof, sel_class_claims) = prove_selector_shout(
        &class_sel_spec,
        std::slice::from_ref(&wd.fetch_comp),
        "sel-class",
        &mut t,
    );
    // The stage selector check (the (kind, stage) one-hot vs sel_ks).
    let sel_stage_spec = {
        let terminal = stage_router.terminal.clone();
        let [_, j_off, _, h_off, _] = stage_layout.offsets();
        let r_j = terminal[j_off..j_off + n].to_vec();
        SelectorShoutSpec {
            a: a_rom,
            n,
            r_j,
            factors: vec![SelectorCheckSpec {
                h_vars: 5,
                r_h: terminal[h_off..h_off + 5].to_vec(),
                claim: stage_router.q_values[0],
                rom_sel: pk.rom.sel_ks.clone(),
            }],
        }
    };
    mark(time_phases, "sel-class", &t0);
    let (sel_stage_proof, sel_stage_claims) = prove_selector_shout(
        &sel_stage_spec,
        std::slice::from_ref(&wd.fetch_comp),
        "sel-stage",
        &mut t,
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
    mark(time_phases, "sel-stage", &t0);
    let (sel_store_proof, sel_store_claims) = prove_selector_shout(
        &sel_store,
        std::slice::from_ref(&wd.fetch_comp),
        "sel-store",
        &mut t,
    );
    // ---- 7. MosaicPCS.
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
        &sel_stage_claims,
        &sel_class_claims,
        &sel_store_claims,
        &shift_layout,
        &shift_router,
        &shift_terminal,
        &stage_layout,
        &stage_router,
        &stage_terminal,
        stage_g_claim,
        &store_layout,
        &store_router,
        &store_terminal,
        &store_x_claims,
    );
    mark(time_phases, "sel-store", &t0);
    let mosaic_proof = prove_mosaic(&pk.pcs, &pk.mosaic, &commitment, &mosaic_claims, &mut t);
    mark(time_phases, "mosaic", &t0);
    Proof {
        commitment_root: commitment.stmt.root,
        pcs_n: commitment.stmt.n,
        pcs_k: commitment.stmt.k,
        pcs_domain: commitment.stmt.domain,
        ell: commitment.ell,
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
        stage_router,
        stage_g_claim,
        store_router,
        store_x_claims,
        bus_shout,
        sel_shift: sel_shift_proof,
        sel_stage: sel_stage_proof,
        sel_class: sel_class_proof,
        sel_store: sel_store_proof,
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

/// Collect every Mosaic claim (the same collection on both sides; the
/// claim values and points come from symmetric proof structures).
#[allow(clippy::too_many_arguments)]
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
    sel_stage_claims: &magnetar_memory::SelectorPcsClaims,
    sel_class_claims: &magnetar_memory::SelectorPcsClaims,
    sel_store_claims: &magnetar_memory::SelectorPcsClaims,
    shift_layout: &RoutingLayout,
    shift_router: &RouterProof,
    shift_terminal: &[Gf128],
    stage_layout: &RoutingLayout,
    stage_router: &RouterProof,
    stage_terminal: &[Gf128],
    stage_g_claim: Gf128,
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
    for src in [
        sel_shift_claims,
        sel_stage_claims,
        sel_class_claims,
        sel_store_claims,
    ] {
        for c in &src.components {
            out.push(MosaicClaim::new(idx("fetch-ra"), c.point.clone(), c.value));
        }
    }
    // 6b. The stage routing's claims: the X claim lifts onto the
    // shift-src copy; the b-select factor's claim gives the shamt
    // segment's claim: Ŝ(r_s, r_j) = Ĝ' + 1 + r_b.
    {
        let x_pt = terminal_x_point(stage_layout, stage_terminal);
        // The u-slice IS the bit index (6 vars) ++ the j-slice.
        let mut pt = x_pt[..6].to_vec();
        pt.extend_from_slice(&x_pt[stage_layout.u_vars..]);
        out.push(MosaicClaim::new(
            idx("seg-shift-src"),
            pt,
            stage_router.x_value,
        ));
        let [_, j_off, _, h_off, _] = stage_layout.offsets();
        let r_b = stage_terminal[h_off + 5];
        let r_s: Vec<Gf128> = stage_terminal[h_off + 2..h_off + 5].to_vec();
        let r_j: Vec<Gf128> = stage_terminal[j_off..j_off + n].to_vec();
        let mut sh_pt = r_s;
        sh_pt.extend_from_slice(&r_j);
        let sh_val = stage_g_claim + ONE + r_b;
        out.push(MosaicClaim::new(idx("seg-shamt"), sh_pt, sh_val));
    }
    // 7. The shift router's X claim → the shift-src copy segment.
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
    // 8. The store router's X split claims → the copy segments.
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
    out
}

/// Verify a proof.
pub fn verify(vk: &VerifyingKey, pub_in: &PublicInput, proof: &Proof) -> Result<(), &'static str> {
    let q = vk.trace_len;
    let m = WITNESS_LAYOUT::M;
    let n = q.trailing_zeros() as usize;
    let a_rom = vk.rom.a_rom;
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
    let seg_claims = verify_virtual_spartan(&rel, &vk.layout, q, &public, &proof.vs, &mut t)?;
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
        m0: vec![ZERO; 1 << 12],
        rv_claims: vec![seg("rs1").value, seg("rs2").value],
        wv_claim: seg("rd").value,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let reg_claims = verify_twist(&reg_inst, 1, &proof.reg_twist, &mut t)?;
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
            proof.reg_index_claims[s],
            &proof.reg_psi[s],
            &format!("reg-psi-{s}"),
            &mut t,
        )?;
    }
    // ---- 4b. The RAM Twist.
    let ram_m0: Vec<Gf128> = {
        let mut t0 = vec![ZERO; 1usize << 14];
        for (k, &w) in pub_in.input.iter().enumerate() {
            if k < (1usize << 8) {
                for o in 0..64usize {
                    t0[k | (o << 8)] = Gf128(((w >> o) & 1) as u128);
                }
            }
        }
        t0
    };
    let ram_inst = TwistInstance {
        a: 8,
        ell: 6,
        n,
        p: 1,
        m0: ram_m0,
        rv_claims: vec![seg("load-val").value],
        wv_claim: proof.ram_wv_claim,
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
    };
    let ram_claims = verify_twist(&ram_inst, 1, &proof.ram_twist, &mut t)?;
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
            proof.ram_addr_claims[s],
            &proof.ram_psi[s],
            &format!("ram-psi-{s}"),
            &mut t,
        )?;
        if proof.ram_addr_claims[s] != v_ramaddr {
            return Err("ram: address claim does not match the segment");
        }
    }
    // The Inc point and claim.
    let inc_point: Vec<Gf128> = t.challenge_field_vec("ram-inc-point", 6 + n);
    if inc_point != proof.inc_point {
        return Err("ram: inc point mismatch");
    }
    t.absorb_field("ram-inc-claim", &[proof.inc_claim]);
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
        &proof.shift_router,
        &mut t,
    )?;
    // ---- 5a'. The stage router (the register-shift chain).
    let (st_o, st_u, st_h) = (vk.stage_tensor.1, vk.stage_tensor.2, vk.stage_tensor.3);
    let stage_layout = RoutingLayout {
        o_vars: st_o,
        j_vars: n,
        u_vars: st_u,
        h_vars: st_h,
        a_vars: 0,
    };
    let stage_r_pub = {
        let [_, _j_off, u_off, h_off, _] = stage_layout.offsets();
        let mut active = (0..st_o).collect::<Vec<usize>>();
        active.extend(u_off..u_off + st_u);
        active.extend(h_off..h_off + st_h);
        PaddedTableFactor::new(vk.stage_tensor.0.clone(), active, stage_layout.joint_vars())
    };
    let stage_claims = verify_routing(
        &stage_layout,
        seg("stage-out").value,
        &r_val,
        &r_inst,
        &stage_r_pub,
        2,
        0,
        &proof.stage_router,
        &mut t,
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
    let (r_inc_o, r_inc_j) = proof.inc_point.split_at(6);
    let store_claims = verify_routing(
        &store_layout,
        proof.inc_claim,
        r_inc_o,
        r_inc_j,
        &store_r_pub,
        1,
        3,
        &proof.store_router,
        &mut t,
    )?;
    // The store X split: (1 + r_u[top])·v_old + r_u[top]·v_src == the X
    // claim, at the *replayed* terminal point.
    {
        let x_pt = store_claims.x_point(&store_layout);
        let w_old = ONE + x_pt[u_vs - 1];
        let w_src = x_pt[u_vs - 1];
        let combo = mul(w_old, proof.store_x_claims[0]) + mul(w_src, proof.store_x_claims[1]);
        if combo != store_claims.x_value {
            return Err("store: X split mismatch");
        }
    }
    t.absorb_field("store-x", &proof.store_x_claims);
    // ---- 6. BinaryShout for the bytecode.
    let bus_spec = BusShoutSpec {
        a: a_rom,
        ell: 6,
        n,
        rom_bus: vk.rom.bus.clone(),
        bus_claim: seg("fetch").value,
        index_claims: vec![
            (WITNESS_LAYOUT::BUS_RS1, 6, proof.reg_index_claims[0]),
            (WITNESS_LAYOUT::BUS_RS2, 6, proof.reg_index_claims[1]),
            (WITNESS_LAYOUT::BUS_RD, 6, proof.reg_index_claims[2]),
        ],
        r_val: r_val.clone(),
        r_cycle: r_inst.clone(),
        pc_r_local: r_local3.clone(),
        pc_claim: seg("pc").value,
    };
    let bus_claims = verify_bus_shout(&bus_spec, 1, &proof.bus_shout, &mut t)?;
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
                    claim: proof.shift_router.q_values[0],
                    rom_sel: vk.rom.sel_lo.clone(),
                },
                SelectorCheckSpec {
                    h_vars: 2,
                    r_h: shift_claims.r_joint[h_off + vk.amount_bits..h_off + vk.amount_bits + 2]
                        .to_vec(),
                    claim: proof.shift_router.q_values[1],
                    rom_sel: vk.rom.sel_hi.clone(),
                },
            ],
        }
    };
    let sel_shift_claims =
        verify_selector_shout(&sel_shift, 1, &proof.sel_shift, "sel-shift", &mut t)?;
    // The class-selector check (the decode-class one-hot vs the ROM).
    let sel_class = {
        let cls = seg("class");
        SelectorShoutSpec {
            a: a_rom,
            n,
            r_j: r_inst.clone(),
            factors: vec![SelectorCheckSpec {
                h_vars: 5,
                r_h: cls.r_local.clone(),
                claim: cls.value,
                rom_sel: vk.rom.class.clone(),
            }],
        }
    };
    let sel_class_claims =
        verify_selector_shout(&sel_class, 1, &proof.sel_class, "sel-class", &mut t)?;
    // The stage selector check (the (kind, stage) one-hot vs sel_ks).
    let sel_stage = {
        let [_, j_off, _, h_off, _] = stage_layout.offsets();
        let r_j = stage_claims.r_joint[j_off..j_off + n].to_vec();
        SelectorShoutSpec {
            a: a_rom,
            n,
            r_j,
            factors: vec![SelectorCheckSpec {
                h_vars: 5,
                r_h: stage_claims.r_joint[h_off..h_off + 5].to_vec(),
                claim: proof.stage_router.q_values[0],
                rom_sel: vk.rom.sel_ks.clone(),
            }],
        }
    };
    let sel_stage_claims =
        verify_selector_shout(&sel_stage, 1, &proof.sel_stage, "sel-stage", &mut t)?;
    // The store selector check.
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
                claim: proof.store_router.q_values[0],
                rom_sel: vk.rom.sel_st.clone(),
            }],
        }
    };
    let sel_store_claims =
        verify_selector_shout(&sel_store, 1, &proof.sel_store, "sel-store", &mut t)?;
    // ---- 7. MosaicPCS.
    let mosaic_claims = collect_mosaic_claims(
        &vk.layout,
        &vk.mosaic,
        n,
        &seg_claims,
        &reg_claims,
        &ram_claims,
        &proof.inc_point,
        proof.inc_claim,
        &bus_claims,
        &sel_shift_claims,
        &sel_stage_claims,
        &sel_class_claims,
        &sel_store_claims,
        &shift_layout,
        &proof.shift_router,
        &shift_claims.r_joint,
        &stage_layout,
        &proof.stage_router,
        &stage_claims.r_joint,
        proof.stage_g_claim,
        &store_layout,
        &proof.store_router,
        &store_claims.r_joint,
        &proof.store_x_claims,
    );
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

    #[test]
    fn witness_rom_consistency_pointwise() {
        // A control-flow program (a backward BNE — a negative immediate)
        // checked point-wise: (a) the store-routing identity over (o, j),
        // (b) the eq-weighted routing total vs the Inc claim, and (c) the
        // witness fetch segment vs the ROM bus word at the fetched row.
        // Regression guard for the truncated-ROM-immediate bug (the ROM
        // bus must carry the full sign-extended 36-bit immediate, or every
        // negative immediate desyncs the bus Shout).
        let bne = |off: i32| -> u32 {
            let imm = off as u16;
            let i12 = ((imm >> 12) & 1) as u32;
            let i11 = ((imm >> 11) & 1) as u32;
            let i10_5 = ((imm >> 5) & 0x3f) as u32;
            let i4_1 = ((imm >> 1) & 0xf) as u32;
            (i12 << 31)
                | (i10_5 << 25)
                | (0 << 20)
                | (3 << 15)
                | (1 << 12)
                | (i4_1 << 8)
                | (i11 << 7)
                | 0x63
        };
        let program = vec![
            enc_i(2, 0, 0, 3, 0x13),
            enc_i(1, 0, 0, 1, 0x13),
            enc_i(1, 0, 0, 2, 0x13),
            enc_i(0xfff, 3, 0, 3, 0x13),
            (2 << 20) | (1 << 15) | (4 << 7) | 0x33,
            enc_i(0, 2, 0, 1, 0x13),
            enc_i(0, 4, 0, 2, 0x13),
            bne(-24),
            enc_i(0, 2, 0, 10, 0x13),
            0x0000_0073,
        ];
        let q = 32usize;
        let trace = execute(&program, &[], q);
        let out = trace.final_state.regs[10];
        let (pk, _vk) = keygen(program, q);
        let pub_in = PublicInput {
            input: vec![],
            output: out,
        };
        let wd = build_witness(&pk, &pub_in, &trace);
        let (_, _, sel_st) = selector_tables_from_trace(&pk.rom, &trace, pk.amount_bits, q);
        // The X bits: old = store-old, src = store-src (from the joint z).
        let m = magnetar_cpu::circuit::WITNESS_LAYOUT::M;
        let old_off = pk.layout.offsets[pk.layout.index_of("store-old").unwrap()];
        let src_off = pk.layout.offsets[pk.layout.index_of("store-src").unwrap()];
        let off_off = pk.layout.offsets[pk.layout.index_of("off").unwrap()];
        let x = |u: usize, j: usize| -> Gf128 {
            let col = if u < 64 {
                old_off + u
            } else {
                src_off + u - 64
            };
            joint_bit(&wd.joint_z, j * m, col)
        };
        let ahat = |k: usize, j: usize| -> Gf128 { joint_bit(&wd.joint_z, j * m, off_off + k) };
        let (tensor, o_vars, u_vars, h_vars, a_vars) = (
            &pk.store_tensor.0,
            pk.store_tensor.1,
            pk.store_tensor.2,
            pk.store_tensor.3,
            pk.store_tensor.4,
        );
        let mut bad = 0usize;
        for j in 0..q {
            for o in 0..64usize {
                // lhs = Σ_{u,h,a} R·X·sel·Π eq(a_k, â_k)
                let mut lhs = Gf128(0);
                for h in 0..(1usize << h_vars) {
                    let sel = sel_st[h | (j << h_vars)];
                    if sel == Gf128(0) {
                        continue;
                    }
                    for a in 0..(1usize << a_vars) {
                        let mut link = Gf128(1);
                        for k in 0..a_vars {
                            let ak = Gf128(((a >> k) & 1) as u128);
                            link = magnetar_field::mul(link, Gf128(1) + ak + ahat(k, j));
                        }
                        if link == Gf128(0) {
                            continue;
                        }
                        let mut routed_sum = Gf128(0);
                        for u in 0..(1usize << u_vars) {
                            let idx = o
                                | (u << o_vars)
                                | (h << (o_vars + u_vars))
                                | (a << (o_vars + u_vars + h_vars));
                            if tensor[idx] == Gf128(1) {
                                routed_sum += x(u, j);
                            }
                        }
                        lhs += magnetar_field::mul(magnetar_field::mul(sel, link), routed_sum);
                    }
                }
                let rhs = wd.ram_wit.inc[o | (j << 6)];
                if lhs != rhs {
                    if bad < 12 {
                        eprintln!("VIOLATION o={o} j={j}: routing={lhs:?} inc={rhs:?} is_store={:?} pc={}",
                            trace.records[j].is_store, trace.records[j].micro_pc);
                    }
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "{bad} store-identity violations");

        // Now the eq-weighted total vs the claim: the exact round-0 check.
        let n = q.trailing_zeros() as usize;
        let inc_point: Vec<Gf128> = {
            let mut t = magnetar_hash::Transcript::new("magnetar/protocol/v1", &[b"dbg"]);
            let _ = &mut t;
            t.challenge_field_vec("ram-inc-point", 6 + n)
        };
        let r_o = &inc_point[..6];
        let r_j = &inc_point[6..];
        let eq_o = magnetar_field::EqBits::build(r_o).values;
        let eq_j = magnetar_field::EqBits::build(r_j).values;
        let mut total = Gf128(0);
        for j in 0..q {
            for o in 0..64usize {
                let mut lhs = Gf128(0);
                for h in 0..(1usize << h_vars) {
                    let sel = sel_st[h | (j << h_vars)];
                    if sel == Gf128(0) {
                        continue;
                    }
                    for a in 0..(1usize << a_vars) {
                        let mut link = Gf128(1);
                        for k in 0..a_vars {
                            let ak = Gf128(((a >> k) & 1) as u128);
                            link = magnetar_field::mul(link, Gf128(1) + ak + ahat(k, j));
                        }
                        if link == Gf128(0) {
                            continue;
                        }
                        let mut routed_sum = Gf128(0);
                        for u in 0..(1usize << u_vars) {
                            let idx = o
                                | (u << o_vars)
                                | (h << (o_vars + u_vars))
                                | (a << (o_vars + u_vars + h_vars));
                            if tensor[idx] == Gf128(1) {
                                routed_sum += x(u, j);
                            }
                        }
                        lhs += magnetar_field::mul(magnetar_field::mul(sel, link), routed_sum);
                    }
                }
                total += magnetar_field::mul(magnetar_field::mul(eq_o[o], eq_j[j]), lhs);
            }
        }
        let inc_claim = magnetar_field::eval_mle(&wd.ram_wit.inc, &inc_point);
        eprintln!("eq-weighted routing total = {total:?}, inc_claim = {inc_claim:?}");
        assert_eq!(
            total, inc_claim,
            "the store-routing total must equal the Inc claim"
        );

        // The fundamental bus equality: the witness's fetch segment per
        // cycle must equal the ROM bus word at the fetched row, and the
        // register read/write one-hots must equal the bus's index slices.
        let fetch_off = pk.layout.offsets[pk.layout.index_of("fetch").unwrap()];
        let rows = &trace.program.rows;
        for (j, rec) in trace.records.iter().enumerate() {
            let k = rec.micro_pc % rows.len();
            for c in 0..64usize {
                let w_bit = joint_bit(&wd.joint_z, j * m, fetch_off + c);
                let r_bit = pk.rom.bus[c | (k << 6)];
                if w_bit != r_bit {
                    eprintln!(
                        "FETCH MISMATCH j={j} k={k} bit={c}: witness={w_bit:?} rom={r_bit:?}"
                    );
                }
            }
            let row = &rows[k].row;
            let slice = |pos: usize, width: usize| -> u64 {
                let mut v = 0u64;
                for t in 0..width {
                    if pk.rom.bus[(pos + t) | (k << 6)] == Gf128(1) {
                        v |= 1 << t;
                    }
                }
                v
            };
            let rs1_bus = slice(magnetar_cpu::circuit::WITNESS_LAYOUT::BUS_RS1, 6);
            let rs2_bus = slice(magnetar_cpu::circuit::WITNESS_LAYOUT::BUS_RS2, 6);
            let rd_bus = slice(magnetar_cpu::circuit::WITNESS_LAYOUT::BUS_RD, 6);
            let rs1_w = (row.rs1 & 63) as u64;
            let rs2_w = (row.rs2 & 63) as u64;
            let rd_w = if row.rd == 0 {
                32
            } else {
                (row.rd & 63) as u64
            };
            if rs1_bus != rs1_w {
                eprintln!("RS1 MISMATCH j={j} k={k}: bus={rs1_bus} row={rs1_w}");
            }
            if rs2_bus != rs2_w {
                eprintln!("RS2 MISMATCH j={j} k={k}: bus={rs2_bus} row={rs2_w}");
            }
            if rd_bus != rd_w {
                eprintln!("RD MISMATCH j={j} k={k}: bus={rd_bus} row={rd_w}");
            }
        }
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
    fn end_to_end_logic_and_control() {
        // Logic ops (XOR/OR/AND through T2 + the PAND aux), a large
        // immediate shift (shamt 40 > 3, exercising the full 6-bit amount
        // space), a register shift by a large amount (the 6-stage chain
        // through the scratch cell), JAL/JALR (the expansion's link rows),
        // and LUI/AUIPC (the pre-adjusted immediates).
        let lui = |imm: u32, rd: u32| (imm << 12) | (rd << 7) | 0x37;
        let slli = |sh: u32, rs1: u32, rd: u32| {
            ((sh & 0x3f) << 20) | (rs1 << 15) | (1 << 12) | (rd << 7) | 0x13
        };
        let xor =
            |rs2: u32, rs1: u32, rd: u32| (rs2 << 20) | (rs1 << 15) | (4 << 12) | (rd << 7) | 0x33;
        let or =
            |rs2: u32, rs1: u32, rd: u32| (rs2 << 20) | (rs1 << 15) | (6 << 12) | (rd << 7) | 0x33;
        let and =
            |rs2: u32, rs1: u32, rd: u32| (rs2 << 20) | (rs1 << 15) | (7 << 12) | (rd << 7) | 0x33;
        let sll =
            |rs2: u32, rs1: u32, rd: u32| (rs2 << 20) | (rs1 << 15) | (1 << 12) | (rd << 7) | 0x33;
        let srl =
            |rs2: u32, rs1: u32, rd: u32| (rs2 << 20) | (rs1 << 15) | (5 << 12) | (rd << 7) | 0x33;
        let program = vec![
            lui(0x12345, 1),              // x1 = 0x12345_000
            enc_i(0x67, 0, 0, 2, 0x13),   // x2 = 0x67
            xor(2, 1, 3),                 // x3 = x1 ^ x2
            or(2, 1, 4),                  // x4 = x1 | x2
            and(2, 1, 5),                 // x5 = x1 & x2
            slli(40, 1, 6),               // x6 = x1 << 40
            enc_i(41, 0, 0, 7, 0x13),     // x7 = 41 (a big shift amount)
            sll(7, 1, 8),                 // x8 = x1 << 41 (the stage chain)
            srl(7, 6, 9),                 // x9 = x6 >> 41
            (9 << 15) | (10 << 7) | 0x33, // x10 = x9 + x0
            0x0000_0073,
        ];
        let trace = magnetar_cpu::interp::execute(&program, &[], 256);
        let out = trace.final_state.regs[10];
        // Sanity: the semantics.
        assert_eq!(trace.final_state.regs[3], 0x12345_000 ^ 0x67);
        assert_eq!(trace.final_state.regs[8], 0x12345_000u64 << 41);
        let (pk, vk) = keygen(program, 256);
        let pub_in = PublicInput {
            input: vec![],
            output: out,
        };
        let proof = prove(&pk, &pub_in);
        assert!(
            verify(&vk, &pub_in, &proof).is_ok(),
            "honest logic/control proof must verify"
        );
    }

    #[test]
    fn tamper_class_and_stage_rejected() {
        // A register shift exercises the class one-hot, the stage routing,
        // and the shamt machinery; each tamper must be rejected.
        let program = vec![
            enc_i(0xF0, 0, 0, 1, 0x13),                          // x1 = 0xF0
            enc_i(5, 0, 0, 2, 0x13),                             // x2 = 5 (the amount)
            (2 << 20) | (1 << 15) | (1 << 12) | (3 << 7) | 0x33, // sll x3, x1, x2
            (3 << 15) | (10 << 7) | 0x33,                        // x10 = x3
            0x0000_0073,
        ];
        let (pk, vk) = keygen(program, 64);
        let pub_in = PublicInput {
            input: vec![],
            output: 0xF0 << 5,
        };
        let proof = prove(&pk, &pub_in);
        assert!(verify(&vk, &pub_in, &proof).is_ok());
        let cls_idx = pk.layout.index_of("class").expect("class segment");
        // 1. A tampered class segment value: the class-selector check
        //    binds it to the ROM — the claim can no longer match.
        let mut p = proof.clone();
        p.vs.segment_values[cls_idx] += ONE;
        assert!(
            verify(&vk, &pub_in, &p).is_err(),
            "tampered class segment value"
        );
        // 2. A tampered stage-router X claim.
        let mut p = proof.clone();
        p.stage_router.x_value += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered stage X claim");
        // 3. A tampered b-select claim (breaks the shamt identity).
        let mut p = proof.clone();
        p.stage_g_claim += ONE;
        assert!(
            verify(&vk, &pub_in, &p).is_err(),
            "tampered stage b-select claim"
        );
        // 4. A tampered stage selector round message.
        let mut p = proof.clone();
        if let Some(r0) = p.sel_stage.per_factor.first_mut() {
            if let Some(c0) = r0.rounds.first_mut() {
                if let Some(cc) = c0.first_mut() {
                    *cc += ONE;
                }
            }
        }
        assert!(
            verify(&vk, &pub_in, &p).is_err(),
            "tampered stage selector round"
        );
        // 5. A tampered class selector round message.
        let mut p = proof.clone();
        if let Some(r0) = p.sel_class.per_factor.first_mut() {
            if let Some(c0) = r0.rounds.first_mut() {
                if let Some(cc) = c0.first_mut() {
                    *cc += ONE;
                }
            }
        }
        assert!(
            verify(&vk, &pub_in, &p).is_err(),
            "tampered class selector round"
        );
        // 6. A tampered stage routing round message.
        let mut p = proof.clone();
        if let Some(c0) = p.stage_router.rounds.first_mut() {
            if let Some(cc) = c0.first_mut() {
                *cc += ONE;
            }
        }
        assert!(
            verify(&vk, &pub_in, &p).is_err(),
            "tampered stage routing round"
        );
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
        p.vs.terminals[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered vs terminal");
        // 3. A VS segment value (breaks Equation (1)).
        let mut p = proof.clone();
        p.vs.segment_values[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered segment value");
        // 4. A register-index claim (breaks the bytecode Shout).
        let mut p = proof.clone();
        p.reg_index_claims[0] += ONE;
        assert!(
            verify(&vk, &pub_in, &p).is_err(),
            "tampered reg index claim"
        );
        // 5. The RAM address claim (breaks the segment identity).
        let mut p = proof.clone();
        p.ram_addr_claims[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered ram addr claim");
        // 6. The Inc claim (breaks the store router's output).
        let mut p = proof.clone();
        p.inc_claim += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered inc claim");
        // 7. A shift router round.
        let mut p = proof.clone();
        p.shift_router.rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered shift router");
        // 8. A store router round.
        let mut p = proof.clone();
        p.store_router.rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered store router");
        // 9. The store X split claim.
        let mut p = proof.clone();
        p.store_x_claims[0] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered store X");
        // 10. The bus Shout's Φ round.
        let mut p = proof.clone();
        p.bus_shout.phi_rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered bus shout");
        // 11. A selector round.
        let mut p = proof.clone();
        p.sel_shift.per_factor[0].rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered selector");
        // 12. The commitment root.
        let mut p = proof.clone();
        p.commitment_root[0] ^= 0xFF;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered commitment");
        // 13. The RAM write-value claim.
        let mut p = proof.clone();
        p.ram_wv_claim += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered ram wv");
        // 14. A register Twist read round.
        let mut p = proof.clone();
        p.reg_twist.read_runs[0].rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered reg twist");
        // 15. A register Ψ round.
        let mut p = proof.clone();
        p.reg_psi[0].rounds[0][1] += ONE;
        assert!(verify(&vk, &pub_in, &p).is_err(), "tampered reg psi");
    }
}
