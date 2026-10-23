//! The dual SIMD R1CS circuit (paper §5.1) — witness layout **v2** with the
//! virtual/direct segment split of §5.3–§5.5.
//!
//! One SIMD instance covers one virtual cycle; the witness matrix `W` has
//! one row per cycle, and the same structure `(A₁,B₁,C₁)` over `B` and
//! `(A₂,B₂,C₂)` over `K` constrain every row.
//!
//! ## Witness layout (per cycle, columns of `z = (w, 1, x)`)
//!
//! **Virtual segments** (read by the R1CS, discharged by the auxiliary
//! arguments — never committed):
//! * `rs1[64]`, `rs2[64]` — BinaryTwist-for-registers read ports,
//! * `rd[64]` — the register-Twist write port,
//! * `load_val[64]` — BinaryTwist-for-RAM's read port (the full word at
//!   the access address, loads and stores alike),
//! * `shift_out[64]` — BinaryRouter-for-shifts' output,
//! * `pc[8]` — the micro-PC (BinaryShout's fetch address bits),
//! * `fetch[64]` — the instruction bus `op‖rd‖rs1‖rs2‖funct3‖imm`
//!   (BinaryShout's read value).
//!
//! **Direct segments** (committed via MosaicPCS):
//! * `t1[64]`, `t2[64]`, `carry[64]`, `alu[64]` — the adder datapath,
//! * `shift_src[64]`, `store_old[64]`, `store_src[64]` — the router
//!   source copies, bound to the virtual originals by linear constraints
//!   (paper §5.4: "W also has a direct segment that holds a copy of the
//!   two words. The R1CS binds this copy to the originals"),
//! * `flags[16]`, `off[8]` (the store byte offset), `ram_addr[8]` (the
//!   word address, bound to `alu[3..11]`),
//! * the byte-lane `eq`/`product`/`conditional` aux columns for the
//!   sub-word-load mux.
//!
//! ## The relations
//!
//! *F₂ relation*: booleanity for every used column; the adder carry
//! recurrence; the copy bindings (`shift_src = rs1`, `store_old =
//! load_val`, `store_src = rs2`, `ram_addr = alu[3..11]`); the eq-column
//! product chains; the sub-word zero-extension.
//!
//! *K relation*: the packed write mux `we·(Σβ·alu + Σβ·rd) = 0`; the
//! masked adder sum; `c₀ = sub`; the halt-latch I/O binding; the shift
//! select `is_shift·(Σβ·alu + Σβ·shift_out) = 0`; the word-load copy; the
//! one-hot byte-lane muxes `cA_a·(B_a + M) = 0` (§5.1's packed-select
//! idiom: the selector times the *difference*, vacuous when the selector
//! is zero).

use crate::expand::RomRow;
use crate::interp::CycleRecord;
use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_spartan::SparseMatrix;

/// Witness layout constants: one row of `z = (w, 1, x)`.
#[allow(non_camel_case_types)] // a constants namespace, not a type
pub struct WITNESS_LAYOUT;

impl WITNESS_LAYOUT {
    pub const W_BITS: usize = 64;
    /// The fetch instruction bus layout (bit positions).
    pub const OP_BITS: usize = 7;
    pub const IDX_BITS: usize = 6;
    pub const FUNCT3_BITS: usize = 3;
    pub const IMM_BITS: usize = 13;
    // Total witness columns (before padding). 2048 leaves room for the
    // comparison/flag-derivation aux (the Remark-4 cross-step support).
    pub const W_COLS: usize = 2048;
    /// The `1` column index in `z`.
    pub const ONE_COL: usize = Self::W_COLS;
    /// Public inputs per cycle.
    pub const X_COLS: usize = Self::M / 2 - 1;
    /// Total columns `m = 2·|w|` (power of two).
    pub const M: usize = 4096;

    // ---- Virtual segments ----
    pub const RS1: usize = 0;
    pub const RS2: usize = 64;
    pub const RD: usize = 128;
    pub const LOAD_VAL: usize = 192;
    pub const SHIFT_OUT: usize = 256;
    // ---- Direct segments ----
    pub const T1: usize = 320;
    pub const T2: usize = 384;
    pub const CARRY_OFF: usize = 448;
    pub const ALU: usize = 512;
    pub const SHIFT_SRC: usize = 576;
    pub const STORE_OLD: usize = 640;
    pub const STORE_SRC: usize = 704;
    pub const FLAGS: usize = 768; // 16 flag bits
    pub const OFF: usize = 784; // 3 offset bits + padding
    pub const RAM_ADDR: usize = 792; // 8 address bits
                                     // ---- Virtual (cont'd) ----
    pub const PC: usize = 800; // 8 micro-PC bits
    /// The fetch bus at a 64-aligned offset (808..832 is zero padding).
    pub const FETCH: usize = 832; // 64-bit instruction bus
                                  // ---- Byte-lane mux aux (direct, 896..960) ----
    pub const EQA: usize = 896; // 8: [off == a]
    pub const PA: usize = 904; // 8: partial products
    pub const EQB: usize = 912; // 8: [off+1 ≡ a (mod 8)]
    pub const PB: usize = 920; // 8
    pub const CA: usize = 928; // 8: EQA·is_load_byte
    pub const CH: usize = 936; // 8: EQA·is_load_half
    pub const CH2: usize = 944; // 8: EQB·is_load_half
    pub const NH: usize = 952; // (1+is_word)(1+is_half)
    pub const WNL: usize = 953; // we·(1+is_load) — the write mux gate
                                // ---- Cross-step / flag-derivation aux (1024.., one 1024-wide
                                // virtual segment; the 960..1024 gap stays zero so the segments tile
                                // the witness half exactly) ----
    /// The running zero-product of the ALU word: z_i = Π_{j≤i}(1+alu_j);
    /// z_63 = [alu == 0] (the branch equality indicator).
    pub const ZERO: usize = 1024; // 64
    /// The running [opcode == SYSTEM (0x73)] product (7 bits).
    pub const OPSYS: usize = 1088; // 7
    /// The running [opcode == BRANCH (0x63)] product.
    pub const OPBR: usize = 1095; // 7
    /// The funct3 equality intermediates: m_f = (1+f3_0+f_0)(1+f3_1+f_1).
    pub const EQFM: usize = 1102; // 8
    /// The funct3 equalities: e_f = m_f·(1+f3_2+f_2).
    pub const EQF: usize = 1110; // 8
    /// The signed-lt products (see build_circuit).
    pub const LTSA: usize = 1118;
    pub const LTSB: usize = 1119;
    /// The unsigned-lt (borrow) products.
    pub const LTUA: usize = 1120;
    pub const LTUB: usize = 1121;
    /// The comparison mux: (e_0+e_1)·z_63, (e_4+e_5)·lt_s, (e_6+e_7)·lt_u.
    pub const CEQ: usize = 1122;
    pub const CLT: usize = 1123;
    pub const CLU: usize = 1124;
    /// The gated zero-extension: ZE16 = (1+ld_word)·is_load,
    /// ZE8 = NH·is_load (the load zero-extension applies ONLY on loads).
    pub const ZE16: usize = 1125;
    pub const ZE8: usize = 1126;
    /// First unused witness column.
    pub const USED: usize = 1127;

    // Flag bit positions (within FLAGS).
    pub const F_SUB: usize = 0;
    pub const F_WE: usize = 1;
    pub const F_IS_LOAD: usize = 2;
    pub const F_IS_STORE: usize = 3;
    pub const F_IS_ARITH: usize = 4;
    pub const F_HALT: usize = 5;
    pub const F_IS_SHIFT: usize = 6;
    pub const F_IS_LOGIC: usize = 7;
    pub const F_LD_BYTE: usize = 8;
    pub const F_LD_HALF: usize = 9;
    pub const F_LD_WORD: usize = 10;
    /// The branch-taken flag (derived: `branch_row ∧ comparison`).
    pub const F_TAKEN: usize = 11;

    // The fetch bus field positions.
    pub const BUS_OP: usize = 0;
    pub const BUS_RD: usize = 7;
    pub const BUS_RS1: usize = 13;
    pub const BUS_RS2: usize = 19;
    pub const BUS_FUNCT3: usize = 25;
    pub const BUS_IMM: usize = 28;
}

/// The dual-relation circuit spec: matrices over the M columns.
#[derive(Clone, Debug)]
pub struct CircuitSpec {
    pub a_bin: SparseMatrix,
    pub b_bin: SparseMatrix,
    pub c_bin: SparseMatrix,
    pub a_fld: SparseMatrix,
    pub b_fld: SparseMatrix,
    pub c_fld: SparseMatrix,
    pub m: usize,
}

/// Assemble one witness row `z = (w, 1, x)` from a cycle record and its
/// bytecode row.
pub fn witness_row(rec: &CycleRecord, row: &RomRow, public_x: &[Gf128]) -> Vec<Gf128> {
    let mut z = vec![ZERO; WITNESS_LAYOUT::M];
    put_word(&mut z, WITNESS_LAYOUT::RS1, rec.rs1_val);
    put_word(&mut z, WITNESS_LAYOUT::RS2, rec.rs2_val);
    put_word(&mut z, WITNESS_LAYOUT::RD, rec.rd_val);
    put_word(&mut z, WITNESS_LAYOUT::LOAD_VAL, rec.ram_word);
    put_word(&mut z, WITNESS_LAYOUT::SHIFT_OUT, rec.shift_out);
    // The adder operands.
    let sub = (rec.is_arith && rec.is_sub_op) as u64;
    let t1 = rec.alu_a;
    let t2 = rec.alu_b ^ if sub != 0 { u64::MAX } else { 0 };
    put_word(&mut z, WITNESS_LAYOUT::T1, t1);
    put_word(&mut z, WITNESS_LAYOUT::T2, t2);
    let mut carry = 0u64;
    let mut c = sub;
    for i in 0..64 {
        carry |= c << i;
        let a = (t1 >> i) & 1;
        let b = (t2 >> i) & 1;
        c = (a & b) | (a & c) | (b & c);
    }
    put_word(&mut z, WITNESS_LAYOUT::CARRY_OFF, carry);
    let alu = if rec.is_arith {
        t1 ^ t2 ^ carry
    } else {
        rec.alu_out
    };
    put_word(&mut z, WITNESS_LAYOUT::ALU, alu);
    // The router source copies.
    put_word(&mut z, WITNESS_LAYOUT::SHIFT_SRC, rec.rs1_val);
    put_word(&mut z, WITNESS_LAYOUT::STORE_OLD, rec.ram_word);
    put_word(&mut z, WITNESS_LAYOUT::STORE_SRC, rec.rs2_val);
    // Flags.
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_SUB] = Gf128(sub as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_WE] = Gf128(rec.we as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_LOAD] = Gf128(rec.is_load as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_STORE] = Gf128(rec.is_store as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_ARITH] = Gf128(rec.is_arith as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_HALT] = Gf128(rec.halted_latch as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_SHIFT] = Gf128(rec.is_shift as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_LOGIC] = Gf128(rec.is_logic as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_LD_BYTE] = Gf128(rec.ld_byte as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_LD_HALF] = Gf128(rec.ld_half as u128);
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_LD_WORD] = Gf128(rec.ld_word as u128);
    // The store offset bits.
    for k in 0..3 {
        z[WITNESS_LAYOUT::OFF + k] = Gf128(((rec.byte_offset >> k) & 1) as u128);
    }
    // The RAM word address (the low 8 bits of the word address; the
    // interpreter masks into the RAM anyway).
    for k in 0..8 {
        z[WITNESS_LAYOUT::RAM_ADDR + k] = Gf128(((rec.ram_addr >> k) & 1) as u128);
    }
    // The micro-PC.
    for k in 0..8 {
        z[WITNESS_LAYOUT::PC + k] = Gf128(((rec.micro_pc >> k) & 1) as u128);
    }
    // The instruction bus.
    put_bus(&mut z, WITNESS_LAYOUT::FETCH, row);
    // The branch-taken flag (the derivation constraints check it equals
    // `branch_row ∧ comparison`).
    z[WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_TAKEN] = Gf128(rec.branch_taken as u128);
    // ---- The cross-step / flag-derivation aux ----
    // The running zero-product of the ALU word.
    {
        let mut zp = ONE;
        for i in 0..64usize {
            zp = Gf128(((zp.0 == 1) && ((alu >> i) & 1 == 0)) as u128);
            z[WITNESS_LAYOUT::ZERO + i] = zp;
        }
    }
    // The opcode indicators (running products).
    let mut bus_bits = [false; 7];
    for (i, b) in bus_bits.iter_mut().enumerate() {
        *b = z[WITNESS_LAYOUT::FETCH + i] == ONE;
    }
    let mut f3_bits = [false; 3];
    for (i, b) in f3_bits.iter_mut().enumerate() {
        *b = z[WITNESS_LAYOUT::FETCH + WITNESS_LAYOUT::BUS_FUNCT3 + i] == ONE;
    }
    for (target, opcode) in [
        (WITNESS_LAYOUT::OPSYS, 0x73u32),
        (WITNESS_LAYOUT::OPBR, 0x63),
    ] {
        let mut acc = true;
        for i in 0..7usize {
            acc = acc && (bus_bits[i] == ((opcode >> i) & 1 == 1));
            z[target + i] = Gf128(acc as u128);
        }
    }
    // The funct3 equalities.
    for f in 0..8usize {
        let m = (f3_bits[0] == (f & 1 == 1)) && (f3_bits[1] == ((f >> 1) & 1 == 1));
        z[WITNESS_LAYOUT::EQFM + f] = Gf128(m as u128);
        let e = m && (f3_bits[2] == ((f >> 2) & 1 == 1));
        z[WITNESS_LAYOUT::EQF + f] = Gf128(e as u128);
    }
    // The comparison products.
    let rs1_63 = rec.rs1_val >> 63 & 1 == 1;
    let rs2_63 = rec.rs2_val >> 63 & 1 == 1;
    let alu_63 = alu >> 63 & 1 == 1;
    let lt_s = if rs1_63 != rs2_63 { rs1_63 } else { alu_63 };
    // LTSA = (rs1_63+rs2_63)·rs1_63 = [signs differ]·rs1_63;
    // LTSB = (1+rs1_63+rs2_63)·alu_63 = [signs equal]·alu_63;
    // lt_s = LTSA + LTSB.
    z[WITNESS_LAYOUT::LTSA] = Gf128(((rs1_63 != rs2_63) && rs1_63) as u128);
    z[WITNESS_LAYOUT::LTSB] = Gf128(((rs1_63 == rs2_63) && alu_63) as u128);
    // lt_u = 1 + carry_out of t1 + t2 + c0 (the borrow of rs1 - rs2).
    let t1_63 = t1 >> 63 & 1 == 1;
    let t2_63 = t2 >> 63 & 1 == 1;
    let c63 = carry >> 63 & 1 == 1;
    let ltua = t1_63 && t2_63;
    let ltub = c63 && (t1_63 != t2_63);
    z[WITNESS_LAYOUT::LTUA] = Gf128(ltua as u128);
    z[WITNESS_LAYOUT::LTUB] = Gf128(ltub as u128);
    let lt_u = !(ltua ^ ltub);
    // The comparison mux: (e0+e1)·z63, (e4+e5)·lt_s, (e6+e7)·lt_u.
    let mut zf = [false; 8];
    for (i, b) in zf.iter_mut().enumerate() {
        *b = z[WITNESS_LAYOUT::EQF + i] == ONE;
    }
    let zz = z[WITNESS_LAYOUT::ZERO + 63] == ONE;
    z[WITNESS_LAYOUT::CEQ] = Gf128(((zf[0] || zf[1]) && zz) as u128);
    z[WITNESS_LAYOUT::CLT] = Gf128(((zf[4] || zf[5]) && lt_s) as u128);
    z[WITNESS_LAYOUT::CLU] = Gf128(((zf[6] || zf[7]) && lt_u) as u128);
    // The gated load zero-extension.
    let ld_word_b = rec.ld_word;
    let is_load_b = rec.is_load;
    let nh_val = !rec.ld_word && !rec.ld_half;
    z[WITNESS_LAYOUT::ZE16] = Gf128((!ld_word_b && is_load_b) as u128);
    z[WITNESS_LAYOUT::ZE8] = Gf128((nh_val && is_load_b) as u128);
    // The byte-lane eq aux (over the store/load offset).
    let off = rec.byte_offset as usize & 7;
    for a in 0..8usize {
        // pA_a = (1+o0+a0)(1+o1+a1), EQA_a = pA_a·(1+o2+a2).
        let bit =
            |k: usize, v: usize| (((rec.byte_offset >> k) & 1) as usize) ^ ((v >> k) & 1) == 0;
        let p = bit(0, a) && bit(1, a);
        z[WITNESS_LAYOUT::PA + a] = Gf128(p as u128);
        z[WITNESS_LAYOUT::EQA + a] = Gf128((p && bit(2, a)) as u128);
        // The +1 lane (mod 8): EQB_a = [off ≡ a−1 (mod 8)].
        let b = (a + 7) & 7;
        let p2 = bit(0, b) && bit(1, b);
        z[WITNESS_LAYOUT::PB + a] = Gf128(p2 as u128);
        z[WITNESS_LAYOUT::EQB + a] = Gf128((p2 && bit(2, b)) as u128);
        // Conditionals.
        z[WITNESS_LAYOUT::CA + a] =
            Gf128(((z[WITNESS_LAYOUT::EQA + a] == ONE) && rec.ld_byte) as u128);
        z[WITNESS_LAYOUT::CH + a] =
            Gf128(((z[WITNESS_LAYOUT::EQA + a] == ONE) && rec.ld_half) as u128);
        z[WITNESS_LAYOUT::CH2 + a] =
            Gf128(((z[WITNESS_LAYOUT::EQB + a] == ONE) && rec.ld_half) as u128);
    }
    let _ = off;
    z[WITNESS_LAYOUT::NH] = Gf128((!rec.ld_word && !rec.ld_half) as u128);
    z[WITNESS_LAYOUT::WNL] = Gf128((rec.we && !rec.is_load) as u128);
    z[WITNESS_LAYOUT::ONE_COL] = ONE;
    for (j, &xv) in public_x.iter().enumerate() {
        z[WITNESS_LAYOUT::ONE_COL + 1 + j] = xv;
    }
    z
}

#[inline]
fn put_word(z: &mut [Gf128], off: usize, w: u64) {
    for i in 0..64 {
        z[off + i] = Gf128(((w >> i) & 1) as u128);
    }
}

/// Put the instruction bus: `op‖rd‖rs1‖rs2‖funct3‖imm` (zero-padded).
#[inline]
fn put_bus(z: &mut [Gf128], off: usize, row: &RomRow) {
    let op = row.raw & 0x7f;
    for i in 0..WITNESS_LAYOUT::OP_BITS {
        z[off + WITNESS_LAYOUT::BUS_OP + i] = Gf128(((op >> i) & 1) as u128);
    }
    let rd6 = if row.rd == 0 { 32 } else { row.rd };
    for i in 0..WITNESS_LAYOUT::IDX_BITS {
        z[off + WITNESS_LAYOUT::BUS_RD + i] = Gf128((((rd6 as u64) >> i) & 1) as u128);
        z[off + WITNESS_LAYOUT::BUS_RS1 + i] = Gf128((((row.rs1 as u64) >> i) & 1) as u128);
        z[off + WITNESS_LAYOUT::BUS_RS2 + i] = Gf128((((row.rs2 as u64) >> i) & 1) as u128);
    }
    for i in 0..WITNESS_LAYOUT::FUNCT3_BITS {
        z[off + WITNESS_LAYOUT::BUS_FUNCT3 + i] = Gf128(((row.funct3 >> i) & 1) as u128);
    }
    for i in 0..WITNESS_LAYOUT::IMM_BITS {
        z[off + WITNESS_LAYOUT::BUS_IMM + i] = Gf128(((row.imm >> i) & 1) as u128);
    }
}

/// Build the dual circuit.
#[allow(clippy::type_complexity)]
pub fn build_circuit() -> CircuitSpec {
    let m = WITNESS_LAYOUT::M;
    let mut a1 = SparseMatrix::empty(0, m);
    let mut b1 = SparseMatrix::empty(0, m);
    let mut c1 = SparseMatrix::empty(0, m);
    let mut rows = 0usize;
    let mut add = |a: &mut SparseMatrix,
                   b: &mut SparseMatrix,
                   c: &mut SparseMatrix,
                   ra: &[(usize, Gf128)],
                   rb: &[(usize, Gf128)],
                   rc: &[(usize, Gf128)]| {
        let r = rows;
        for &(col, v) in ra {
            a.entries.push((r, col, v));
        }
        for &(col, v) in rb {
            b.entries.push((r, col, v));
        }
        for &(col, v) in rc {
            c.entries.push((r, col, v));
        }
        rows += 1;
    };
    // 1. Booleanity for every used column.
    for col in 0..WITNESS_LAYOUT::USED {
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(col, ONE)],
            &[(col, ONE)],
            &[(col, ONE)],
        );
    }
    // 2. The adder carry recurrence: (t1_i + c_i)(t2_i + c_i) = c_{i+1} + c_i.
    for i in 0..63 {
        let t1i = WITNESS_LAYOUT::T1 + i;
        let t2i = WITNESS_LAYOUT::T2 + i;
        let ci = WITNESS_LAYOUT::CARRY_OFF + i;
        let cn = WITNESS_LAYOUT::CARRY_OFF + i + 1;
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(t1i, ONE), (ci, ONE)],
            &[(t2i, ONE), (ci, ONE)],
            &[(cn, ONE), (ci, ONE)],
        );
    }
    // 3. Copy bindings (linear): (x + y)·1 = 0.
    for i in 0..64 {
        for (x, y) in [
            (WITNESS_LAYOUT::SHIFT_SRC + i, WITNESS_LAYOUT::RS1 + i),
            (WITNESS_LAYOUT::STORE_OLD + i, WITNESS_LAYOUT::LOAD_VAL + i),
            (WITNESS_LAYOUT::STORE_SRC + i, WITNESS_LAYOUT::RS2 + i),
        ] {
            add(
                &mut a1,
                &mut b1,
                &mut c1,
                &[(x, ONE), (y, ONE)],
                &[(WITNESS_LAYOUT::ONE_COL, ONE)],
                &[],
            );
        }
    }
    // ram_addr_t = alu_{3+t} on memory cycles (the effective address's
    // low bits); on other cycles the segment is zero and the memory
    // argument's address claim binds it.
    for t in 0..8 {
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[
                (WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_LOAD, ONE),
                (WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_STORE, ONE),
            ],
            &[
                (WITNESS_LAYOUT::RAM_ADDR + t, ONE),
                (WITNESS_LAYOUT::ALU + 3 + t, ONE),
            ],
            &[],
        );
    }
    // 4. The eq chains: PA_a = (1+o0+a0)(1+o1+a1); EQA_a = PA_a·(1+o2+a2).
    for a in 0..8usize {
        let (o0, o1, o2) = (
            WITNESS_LAYOUT::OFF,
            WITNESS_LAYOUT::OFF + 1,
            WITNESS_LAYOUT::OFF + 2,
        );
        let pa = WITNESS_LAYOUT::PA + a;
        let ea = WITNESS_LAYOUT::EQA + a;
        let pb = WITNESS_LAYOUT::PB + a;
        let eb = WITNESS_LAYOUT::EQB + a;
        let a0 = (a & 1) as u128;
        let a1b = ((a >> 1) & 1) as u128;
        let a2 = ((a >> 2) & 1) as u128;
        // PA_a = (1+o0+a0)(1+o1+a1): A = [o0, 1, a0], B = [o1, 1, a1b].
        let lin1 = |x: usize, c: u128| vec![(x, ONE), (WITNESS_LAYOUT::ONE_COL, Gf128(c))];
        let l1 = if a0 == 0 {
            lin1(o0, 1)
        } else {
            vec![(o0, ONE)]
        };
        let l2 = if a1b == 0 {
            lin1(o1, 1)
        } else {
            vec![(o1, ONE)]
        };
        add(&mut a1, &mut b1, &mut c1, &l1, &l2, &[(pa, ONE)]);
        // EQA_a = PA_a·(1+o2+a2).
        let l3 = if a2 == 0 {
            lin1(o2, 1)
        } else {
            vec![(o2, ONE)]
        };
        add(&mut a1, &mut b1, &mut c1, &[(pa, ONE)], &l3, &[(ea, ONE)]);
        // The +1 lane: PB/EB with the target (a+7)&7.
        let b_ = (a + 7) & 7;
        let b0 = (b_ & 1) as u128;
        let b1_ = ((b_ >> 1) & 1) as u128;
        let b2 = ((b_ >> 2) & 1) as u128;
        let m1 = if b0 == 0 {
            lin1(o0, 1)
        } else {
            vec![(o0, ONE)]
        };
        let m2 = if b1_ == 0 {
            lin1(o1, 1)
        } else {
            vec![(o1, ONE)]
        };
        add(&mut a1, &mut b1, &mut c1, &m1, &m2, &[(pb, ONE)]);
        let m3 = if b2 == 0 {
            lin1(o2, 1)
        } else {
            vec![(o2, ONE)]
        };
        add(&mut a1, &mut b1, &mut c1, &[(pb, ONE)], &m3, &[(eb, ONE)]);
    }
    // 5. The conditionals: CA_a = EQA_a·ld_byte, CH/CH2, NH.
    for a in 0..8usize {
        for (cst, flag) in [
            (WITNESS_LAYOUT::CA + a, WITNESS_LAYOUT::F_LD_BYTE),
            (WITNESS_LAYOUT::CH + a, WITNESS_LAYOUT::F_LD_HALF),
            (WITNESS_LAYOUT::CH2 + a, WITNESS_LAYOUT::F_LD_HALF),
        ] {
            let src = if flag == WITNESS_LAYOUT::F_LD_HALF && cst == WITNESS_LAYOUT::CH2 + a {
                WITNESS_LAYOUT::EQB + a
            } else {
                WITNESS_LAYOUT::EQA + a
            };
            add(
                &mut a1,
                &mut b1,
                &mut c1,
                &[(src, ONE)],
                &[(WITNESS_LAYOUT::FLAGS + flag, ONE)],
                &[(cst, ONE)],
            );
        }
    }
    // NH = (1+ld_word)(1+ld_half): the witness column carries the product
    // (the R1CS row C-side equals the Boolean product).
    {
        let w = WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_LD_WORD;
        let h = WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_LD_HALF;
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(w, ONE), (WITNESS_LAYOUT::ONE_COL, ONE)],
            &[(h, ONE), (WITNESS_LAYOUT::ONE_COL, ONE)],
            &[(WITNESS_LAYOUT::NH, ONE)],
        );
    }
    // 6. The gated load zero-extension: on byte/half loads the unloaded
    // high bits of rd are zero. ZE16 = (1+ld_word)·is_load and
    // ZE8 = NH·is_load gate it (an ungated rule would wrongly zero ALU
    // results with high bits set).
    add(
        &mut a1,
        &mut b1,
        &mut c1,
        &[
            (WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_LD_WORD, ONE),
            (WITNESS_LAYOUT::ONE_COL, ONE),
        ],
        &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_LOAD, ONE)],
        &[(WITNESS_LAYOUT::ZE16, ONE)],
    );
    add(
        &mut a1,
        &mut b1,
        &mut c1,
        &[(WITNESS_LAYOUT::NH, ONE)],
        &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_LOAD, ONE)],
        &[(WITNESS_LAYOUT::ZE8, ONE)],
    );
    for b in 16..64 {
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(WITNESS_LAYOUT::ZE16, ONE)],
            &[(WITNESS_LAYOUT::RD + b, ONE)],
            &[],
        );
    }
    for b in 8..16 {
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(WITNESS_LAYOUT::ZE8, ONE)],
            &[(WITNESS_LAYOUT::RD + b, ONE)],
            &[],
        );
    }

    // ---- The cross-step / flag-derivation constraints (Remark 4) ----
    // The linear-combo helper over (column, coeff) pairs.
    {
        let one = WITNESS_LAYOUT::ONE_COL;
        // The zero-chain: z_0 = 1+alu_0 (linear); z_i = z_{i-1}·(1+alu_i).
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[
                (WITNESS_LAYOUT::ZERO, ONE),
                (WITNESS_LAYOUT::ALU, ONE),
                (one, ONE),
            ],
            &[(one, ONE)],
            &[],
        );
        for i in 1..64 {
            add(
                &mut a1,
                &mut b1,
                &mut c1,
                &[(WITNESS_LAYOUT::ZERO + i - 1, ONE)],
                &[(WITNESS_LAYOUT::ALU + i, ONE), (one, ONE)],
                &[(WITNESS_LAYOUT::ZERO + i, ONE)],
            );
        }
        // The opcode running products: ops_0/opbr_0 linear, then products.
        for (target, opcode) in [
            (WITNESS_LAYOUT::OPSYS, 0x73u32),
            (WITNESS_LAYOUT::OPBR, 0x63),
        ] {
            let bit0 = (opcode & 1) as u128;
            add(
                &mut a1,
                &mut b1,
                &mut c1,
                &[
                    (target, ONE),
                    (WITNESS_LAYOUT::FETCH, ONE),
                    (one, Gf128(1 ^ bit0)),
                ],
                &[(one, ONE)],
                &[],
            );
            for i in 1..7 {
                let bit = (opcode >> i) & 1;
                add(
                    &mut a1,
                    &mut b1,
                    &mut c1,
                    &[(target + i - 1, ONE)],
                    &[
                        (WITNESS_LAYOUT::FETCH + i, ONE),
                        (one, Gf128(1 ^ (bit as u128))),
                    ],
                    &[(target + i, ONE)],
                );
            }
        }
        // F_HALT ⟺ opcode == SYSTEM: halt·(1+ops_6) = 0 and
        // (1+halt)·ops_6 = 0.
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_HALT, ONE)],
            &[(WITNESS_LAYOUT::OPSYS + 6, ONE), (one, ONE)],
            &[],
        );
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[
                (WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_HALT, ONE),
                (one, ONE),
            ],
            &[(WITNESS_LAYOUT::OPSYS + 6, ONE)],
            &[],
        );
        // The funct3 equalities: m_f = (1+f3_0+f_0)(1+f3_1+f_1),
        // e_f = m_f·(1+f3_2+f_2).
        let f3b = |i: usize| WITNESS_LAYOUT::FETCH + WITNESS_LAYOUT::BUS_FUNCT3 + i;
        for f in 0..8usize {
            let b0 = (f & 1) as u128;
            let b1_ = ((f >> 1) & 1) as u128;
            let b2 = ((f >> 2) & 1) as u128;
            add(
                &mut a1,
                &mut b1,
                &mut c1,
                &[(f3b(0), ONE), (one, Gf128(1 ^ b0))],
                &[(f3b(1), ONE), (one, Gf128(1 ^ b1_))],
                &[(WITNESS_LAYOUT::EQFM + f, ONE)],
            );
            add(
                &mut a1,
                &mut b1,
                &mut c1,
                &[(WITNESS_LAYOUT::EQFM + f, ONE)],
                &[(f3b(2), ONE), (one, Gf128(1 ^ b2))],
                &[(WITNESS_LAYOUT::EQF + f, ONE)],
            );
        }
        // The comparison products.
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[
                (WITNESS_LAYOUT::RS1 + 63, ONE),
                (WITNESS_LAYOUT::RS2 + 63, ONE),
            ],
            &[(WITNESS_LAYOUT::RS1 + 63, ONE)],
            &[(WITNESS_LAYOUT::LTSA, ONE)],
        );
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[
                (WITNESS_LAYOUT::RS1 + 63, ONE),
                (WITNESS_LAYOUT::RS2 + 63, ONE),
                (one, ONE),
            ],
            &[(WITNESS_LAYOUT::ALU + 63, ONE)],
            &[(WITNESS_LAYOUT::LTSB, ONE)],
        );
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(WITNESS_LAYOUT::T1 + 63, ONE)],
            &[(WITNESS_LAYOUT::T2 + 63, ONE)],
            &[(WITNESS_LAYOUT::LTUA, ONE)],
        );
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(WITNESS_LAYOUT::CARRY_OFF + 63, ONE)],
            &[
                (WITNESS_LAYOUT::T1 + 63, ONE),
                (WITNESS_LAYOUT::T2 + 63, ONE),
            ],
            &[(WITNESS_LAYOUT::LTUB, ONE)],
        );
        // The comparison mux: CEQ = (e_0+e_1)·z_63, CLT = (e_4+e_5)·lt_s
        // with lt_s = LTSA+LTSB, CLU = (e_6+e_7)·lt_u with
        // lt_u = 1+LTUA+LTUB.
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(WITNESS_LAYOUT::EQF, ONE), (WITNESS_LAYOUT::EQF + 1, ONE)],
            &[(WITNESS_LAYOUT::ZERO + 63, ONE)],
            &[(WITNESS_LAYOUT::CEQ, ONE)],
        );
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[
                (WITNESS_LAYOUT::EQF + 4, ONE),
                (WITNESS_LAYOUT::EQF + 5, ONE),
            ],
            &[(WITNESS_LAYOUT::LTSA, ONE), (WITNESS_LAYOUT::LTSB, ONE)],
            &[(WITNESS_LAYOUT::CLT, ONE)],
        );
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[
                (WITNESS_LAYOUT::EQF + 6, ONE),
                (WITNESS_LAYOUT::EQF + 7, ONE),
            ],
            &[
                (WITNESS_LAYOUT::LTUA, ONE),
                (WITNESS_LAYOUT::LTUB, ONE),
                (one, ONE),
            ],
            &[(WITNESS_LAYOUT::CLU, ONE)],
        );
        // F_TAKEN = opbr_6·comp with comp = CEQ + e_1 + CLT + e_5 + CLU + e_7.
        add(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(WITNESS_LAYOUT::OPBR + 6, ONE)],
            &[
                (WITNESS_LAYOUT::CEQ, ONE),
                (WITNESS_LAYOUT::EQF + 1, ONE),
                (WITNESS_LAYOUT::CLT, ONE),
                (WITNESS_LAYOUT::EQF + 5, ONE),
                (WITNESS_LAYOUT::CLU, ONE),
                (WITNESS_LAYOUT::EQF + 7, ONE),
            ],
            &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_TAKEN, ONE)],
        );
    }

    // ---- K constraints ----
    let mut a2 = SparseMatrix::empty(0, m);
    let mut b2 = SparseMatrix::empty(0, m);
    let mut c2 = SparseMatrix::empty(0, m);
    let mut rows2 = 0usize;
    let mut add_k = |a: &mut SparseMatrix,
                     b: &mut SparseMatrix,
                     c: &mut SparseMatrix,
                     ra: &[(usize, Gf128)],
                     rb: &[(usize, Gf128)],
                     rc: &[(usize, Gf128)]| {
        let r = rows2;
        for &(col, v) in ra {
            a.entries.push((r, col, v));
        }
        for &(col, v) in rb {
            b.entries.push((r, col, v));
        }
        for &(col, v) in rc {
            c.entries.push((r, col, v));
        }
        rows2 += 1;
    };
    // The write-mux gate: WNL = we·(1+is_load) (loads write the memory
    // value, not the ALU output).
    add(
        &mut a1,
        &mut b1,
        &mut c1,
        &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_WE, ONE)],
        &[
            (WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_LOAD, ONE),
            (WITNESS_LAYOUT::ONE_COL, ONE),
        ],
        &[(WITNESS_LAYOUT::WNL, ONE)],
    );
    // The packed write mux: WNL·(Σβ·alu + Σβ·rd) = 0.
    {
        let mut b_entries: Vec<(usize, Gf128)> = Vec::new();
        for i in 0..64 {
            b_entries.push((WITNESS_LAYOUT::ALU + i, Gf128(1u128 << i)));
            b_entries.push((WITNESS_LAYOUT::RD + i, Gf128(1u128 << i)));
        }
        add_k(
            &mut a2,
            &mut b2,
            &mut c2,
            &[(WITNESS_LAYOUT::WNL, ONE)],
            &b_entries,
            &[],
        );
    }
    // The masked adder sum: is_arith·Σβ(t1+t2+c+alu) = 0.
    {
        let mut a_entries: Vec<(usize, Gf128)> = Vec::new();
        for i in 0..64 {
            a_entries.push((WITNESS_LAYOUT::T1 + i, Gf128(1u128 << i)));
            a_entries.push((WITNESS_LAYOUT::T2 + i, Gf128(1u128 << i)));
            a_entries.push((WITNESS_LAYOUT::CARRY_OFF + i, Gf128(1u128 << i)));
            a_entries.push((WITNESS_LAYOUT::ALU + i, Gf128(1u128 << i)));
        }
        add_k(
            &mut a2,
            &mut b2,
            &mut c2,
            &a_entries,
            &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_ARITH, ONE)],
            &[],
        );
    }
    // c_0 = sub.
    add_k(
        &mut a2,
        &mut b2,
        &mut c2,
        &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_SUB, ONE)],
        &[(WITNESS_LAYOUT::ONE_COL, ONE)],
        &[(WITNESS_LAYOUT::CARRY_OFF, ONE)],
    );
    // The halt-latch I/O bindings: halt·(Σβ·rd + x_out) = 0 (the public
    // output) and halt·(Σβ·(rd + rs1)) = 0 (the halt row reads x10 — the
    // expansion sets its rs1 = 10 — so rd = the register file's x10,
    // authenticated by the register Twist: x_out = x10 at halt).
    {
        let mut b_entries: Vec<(usize, Gf128)> = Vec::new();
        for i in 0..64 {
            b_entries.push((WITNESS_LAYOUT::RD + i, Gf128(1u128 << i)));
        }
        b_entries.push((WITNESS_LAYOUT::ONE_COL + 1, ONE));
        add_k(
            &mut a2,
            &mut b2,
            &mut c2,
            &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_HALT, ONE)],
            &b_entries,
            &[],
        );
        let mut b2_entries: Vec<(usize, Gf128)> = Vec::new();
        for i in 0..64 {
            b2_entries.push((WITNESS_LAYOUT::RD + i, Gf128(1u128 << i)));
            b2_entries.push((WITNESS_LAYOUT::RS1 + i, Gf128(1u128 << i)));
        }
        add_k(
            &mut a2,
            &mut b2,
            &mut c2,
            &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_HALT, ONE)],
            &b2_entries,
            &[],
        );
    }
    // The shift select: is_shift·(Σβ·alu + Σβ·shift_out) = 0.
    {
        let mut b_entries: Vec<(usize, Gf128)> = Vec::new();
        for i in 0..64 {
            b_entries.push((WITNESS_LAYOUT::ALU + i, Gf128(1u128 << i)));
            b_entries.push((WITNESS_LAYOUT::SHIFT_OUT + i, Gf128(1u128 << i)));
        }
        add_k(
            &mut a2,
            &mut b2,
            &mut c2,
            &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_IS_SHIFT, ONE)],
            &b_entries,
            &[],
        );
    }
    // The word-load copy: ld_word·(Σβ·rd + Σβ·load_val) = 0.
    {
        let mut b_entries: Vec<(usize, Gf128)> = Vec::new();
        for i in 0..64 {
            b_entries.push((WITNESS_LAYOUT::RD + i, Gf128(1u128 << i)));
            b_entries.push((WITNESS_LAYOUT::LOAD_VAL + i, Gf128(1u128 << i)));
        }
        add_k(
            &mut a2,
            &mut b2,
            &mut c2,
            &[(WITNESS_LAYOUT::FLAGS + WITNESS_LAYOUT::F_LD_WORD, ONE)],
            &b_entries,
            &[],
        );
    }
    // The byte-lane muxes: cA_a·(B_a + M0) = 0, B_a = Σβ_{2^b}·load_{8a+b},
    // M0 = the packed low byte of rd. Same shape for the half (CH: low
    // byte; CH2: byte 1).
    let mk_mux = |_add_k: &mut dyn FnMut(
        &mut SparseMatrix,
        &mut SparseMatrix,
        &mut SparseMatrix,
        &[(usize, Gf128)],
        &[(usize, Gf128)],
        &[(usize, Gf128)],
    )| {};
    let _ = mk_mux;
    for a in 0..8usize {
        // CA_a · (B_a + M0) = 0.
        let mut lin: Vec<(usize, Gf128)> = Vec::new();
        for b in 0..8 {
            lin.push((WITNESS_LAYOUT::LOAD_VAL + 8 * a + b, Gf128(1u128 << b)));
            lin.push((WITNESS_LAYOUT::RD + b, Gf128(1u128 << b)));
        }
        add_k(
            &mut a2,
            &mut b2,
            &mut c2,
            &[(WITNESS_LAYOUT::CA + a, ONE)],
            &lin,
            &[],
        );
        // CH_a · (B_a + M0) = 0 (the half's low byte).
        let mut lin2: Vec<(usize, Gf128)> = Vec::new();
        for b in 0..8 {
            lin2.push((WITNESS_LAYOUT::LOAD_VAL + 8 * a + b, Gf128(1u128 << b)));
            lin2.push((WITNESS_LAYOUT::RD + b, Gf128(1u128 << b)));
        }
        add_k(
            &mut a2,
            &mut b2,
            &mut c2,
            &[(WITNESS_LAYOUT::CH + a, ONE)],
            &lin2,
            &[],
        );
        // CH2_a · (B_a + M1) = 0 (the half's byte 1).
        let mut lin3: Vec<(usize, Gf128)> = Vec::new();
        for b in 0..8 {
            lin3.push((WITNESS_LAYOUT::LOAD_VAL + 8 * a + b, Gf128(1u128 << b)));
            lin3.push((WITNESS_LAYOUT::RD + 8 + b, Gf128(1u128 << b)));
        }
        add_k(
            &mut a2,
            &mut b2,
            &mut c2,
            &[(WITNESS_LAYOUT::CH2 + a, ONE)],
            &lin3,
            &[],
        );
    }
    // Both relations share the same padded row count (paper §5.3).
    let total_rows = rows.max(rows2).next_power_of_two().max(1);
    a1.rows = total_rows;
    b1.rows = total_rows;
    c1.rows = total_rows;
    a2.rows = total_rows;
    b2.rows = total_rows;
    c2.rows = total_rows;
    CircuitSpec {
        a_bin: a1,
        b_bin: b1,
        c_bin: c1,
        a_fld: a2,
        b_fld: b2,
        c_fld: c2,
        m,
    }
}

/// Check that a witness row satisfies both relations (test helper).
pub fn check_row(spec: &CircuitSpec, z: &[Gf128]) -> Result<(), &'static str> {
    for (name, (a, b, c)) in [
        ("bin", (&spec.a_bin, &spec.b_bin, &spec.c_bin)),
        ("fld", (&spec.a_fld, &spec.b_fld, &spec.c_fld)),
    ] {
        let az = a.apply(z);
        let bz = b.apply(z);
        let cz = c.apply(z);
        for r in 0..az.len() {
            if mul(az[r], bz[r]) != cz[r] {
                return Err(name);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_satisfied_by_trace_rows() {
        let prog = vec![
            0x00100093,                                // addi x1, x0, 1
            0x00100113,                                // addi x2, x0, 1
            ((2 << 20) | (1 << 15)) | (3 << 7) | 0x33, // add x3
            0x0000_0073,
        ];
        let trace = crate::interp::execute(&prog, &[], 64);
        let spec = build_circuit();
        for rec in &trace.records {
            let row = &trace.program.rows[rec.micro_pc % trace.program.rows.len()].row;
            let z = witness_row(rec, row, &[Gf128(0)]);
            assert!(check_row(&spec, &z).is_ok(), "row pc={}", rec.micro_pc);
        }
    }

    #[test]
    fn circuit_satisfied_by_load_store_rows() {
        // addi x1, x0, 16; addi x2, x0, 0xAB; sb x2, 0(x1); lb x3, 0(x1);
        // lh x4, 0(x1); ecall.
        let prog = vec![
            0x01000093,                              // addi x1, x0, 16
            0x0ab00113,                              // addi x2, x0, 0xAB
            ((2u32 << 20) | (1 << 15)) | 0x23,       // sb x2, 0(x1)
            (1 << 15) | (3 << 7) | 0x03,             // lb x3, 0(x1)
            (1 << 15) | (1 << 12) | (4 << 7) | 0x03, // lh x4, 0(x1)
            0x0000_0073,
        ];
        let trace = crate::interp::execute(&prog, &[], 128);
        assert_eq!(trace.final_state.regs[3], 0xAB);
        assert_eq!(trace.final_state.regs[4], 0xAB);
        let spec = build_circuit();
        for rec in &trace.records {
            let row = &trace.program.rows[rec.micro_pc % trace.program.rows.len()].row;
            let z = witness_row(rec, row, &[Gf128(0)]);
            assert!(
                check_row(&spec, &z).is_ok(),
                "row pc={} kind={:?}",
                rec.micro_pc,
                trace.program.rows[rec.micro_pc % trace.program.rows.len()]
                    .row
                    .kind
            );
        }
    }

    #[test]
    fn circuit_satisfied_by_shift_rows() {
        // addi x1, x0, 5; slli x3, x1, 3; ecall.
        let prog = vec![
            0x00500093,                                             // addi x1, x0, 5
            (3u32 << 20) | (1 << 15) | (1 << 12) | (3 << 7) | 0x13, // slli x3, x1, 3
            0x0000_0073,
        ];
        let trace = crate::interp::execute(&prog, &[], 64);
        assert_eq!(trace.final_state.regs[3], 5u64 << 3);
        let spec = build_circuit();
        for rec in &trace.records {
            let row = &trace.program.rows[rec.micro_pc % trace.program.rows.len()].row;
            let z = witness_row(rec, row, &[Gf128(0)]);
            assert!(check_row(&spec, &z).is_ok(), "row pc={}", rec.micro_pc);
        }
    }
}
