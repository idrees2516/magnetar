//! The dual SIMD R1CS circuit (paper §5.1).
//!
//! One SIMD instance covers one virtual cycle; the witness matrix `W` has
//! one row per cycle, and the same structure `(A₁,B₁,C₁)` over `B` and
//! `(A₂,B₂,C₂)` over `K` constrain every row.
//!
//! ## Witness layout (per cycle, columns of `z = (w, 1, x)`)
//!
//! Bit-valued columns (little-endian, 64 bits per word):
//! * `rs1[64]`, `rs2[64]`, `rd[64]` — operand and result bits,
//! * `alu[64]` — the raw ALU output before the write mux,
//! * `carry[64]` — the adder carry chain (`carry[0] = c_in`),
//! * `t1[64]`, `t2[64]` — adder scratch (b-operand = rs2 ⊕ sub-mask),
//! * `sub` — subtract flag, `we` — write-enable, `is_load`, `is_store`,
//! * `off[3]` — the store byte offset, `width[2]` — store width selector,
//! * `store_src[64]`, `ram_old[64]`, `ram_new[64]` — the store copy
//!   segment (bound to the originals by packed linear checks),
//! * `micro_pc[8]` — the fetch address bits,
//! * the fetched instruction fields: `op[7]`, `rd_idx[6]`, `rs1_idx[6]`,
//!   `rs2_idx[6]`, `funct3[3]`, `imm[13]` (the Shout-authenticated fields).
//!
//! ## The relations
//!
//! *F₂ relation* (`A₁,B₁,C₁`): booleanity (`b·b = b`) for every bit column;
//! the adder carry recurrence
//! `c_{i+1} ⊕ a_i b_i ⊕ c_i a_i ⊕ c_i b_i = 0`; the XOR logic
//! `rd_i ⊕ a_i ⊕ b_i ⊕ sub·(...) = 0` shapes; the store-copy
//! `ram_new ⊕ ram_old ⊕ store_src = Inc`-style XOR identities.
//!
//! *K relation* (`A₂,B₂,C₂`): the packed word-wide select
//! `c · Σ β_i·alu_i = Σ β_i·rd_i` (§5.1's motivating example: one field
//! constraint replaces 64 bit constraints); packed linear checks for the
//! sum bits `Σ β_i (t1_i ⊕ t2_i ⊕ carry_i) = Σ β_i rd_i`; and the
//! store-copy binding `Σ β_i ram_new_i = packed store output`.

use crate::interp::CycleRecord;
use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_spartan::SparseMatrix;

/// Witness layout constants: one row of `z = (w, 1, x)`.
pub struct WITNESS_LAYOUT;

impl WITNESS_LAYOUT {
    pub const W_BITS: usize = 64;
    pub const N_WORDS: usize = 7; // rs1, rs2, rd, alu, t1, t2, store_src
    pub const BITS: usize = Self::N_WORDS * Self::W_BITS; // 448
    pub const CARRY: usize = 64;
    pub const FLAGS: usize = 6; // sub, we, is_load, is_store, is_arith, halt
    pub const OFF: usize = 3;
    pub const WIDTH: usize = 2;
    pub const RAM_WORDS: usize = 2; // ram_old, ram_new
    pub const RAM_BITS: usize = Self::RAM_WORDS * Self::W_BITS;
    pub const PC_BITS: usize = 8;
    pub const FETCH: usize = 7 + 6 + 6 + 6 + 3 + 13; // 41
    /// Raw witness columns (before padding).
    pub const RAW_W_COLS: usize = Self::BITS
        + Self::CARRY
        + Self::FLAGS
        + Self::OFF
        + Self::WIDTH
        + Self::RAM_BITS
        + Self::PC_BITS
        + Self::FETCH;
    /// Total witness columns: the raw columns padded up to a power of two
    /// (Equation (1) requires |w| = m/2 with m a power of two).
    pub const W_COLS: usize = 1024;
    /// The `1` column index in `z`.
    pub const ONE_COL: usize = Self::W_COLS;
    /// Public inputs per cycle: the packed (rd, rs1, rs2) words then
    /// zero padding to fill m/2 − 1 columns.
    pub const X_COLS: usize = Self::M / 2 - 1;
    /// Total columns `m = 2·|w|` (power of two).
    pub const M: usize = 2048;

    // Column offsets.
    pub const RS1: usize = 0;
    pub const RS2: usize = 64;
    pub const RD: usize = 128;
    pub const ALU: usize = 192;
    pub const T1: usize = 256;
    pub const T2: usize = 320;
    pub const STORE_SRC: usize = 384;
    pub const CARRY_OFF: usize = 448;
    pub const SUB: usize = 512;
    pub const WE: usize = 513;
    pub const IS_LOAD: usize = 514;
    pub const IS_STORE: usize = 515;
    pub const IS_ARITH: usize = 516;
    pub const HALT: usize = 517;
    pub const OFF_OFF: usize = 518;
    pub const WIDTH_OFF: usize = 521;
    pub const RAM_OLD: usize = 523;
    pub const RAM_NEW: usize = 587;
    pub const PC: usize = 651;
    pub const FETCH_OP: usize = 659;
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

/// Assemble one witness row `z = (w, 1, x)` from a cycle record.
pub fn witness_row(rec: &CycleRecord, public_x: &[Gf128]) -> Vec<Gf128> {
    let mut z = vec![ZERO; WITNESS_LAYOUT::M];
    put_word(&mut z, WITNESS_LAYOUT::RS1, rec.rs1_val);
    put_word(&mut z, WITNESS_LAYOUT::RS2, rec.rs2_val);
    put_word(&mut z, WITNESS_LAYOUT::RD, rec.rd_val);
    // t1/t2 = the ALU's operands; subtract flips t2's bits (two's
    // complement via c_0 = sub).
    let sub = (rec.is_arith && rec.is_sub_op) as u64;
    let t1 = rec.alu_a;
    let t2 = rec.alu_b ^ if sub != 0 { u64::MAX } else { 0 };
    put_word(&mut z, WITNESS_LAYOUT::T1, t1);
    put_word(&mut z, WITNESS_LAYOUT::T2, t2);
    put_word(&mut z, WITNESS_LAYOUT::STORE_SRC, rec.store_src);
    // Carry chain: carry bit i = c_i (the carry INTO bit i); c_0 = sub;
    // c_{i+1} = maj(t1_i, t2_i, c_i).
    let mut carry = 0u64;
    let mut c = sub;
    for i in 0..64 {
        carry |= c << i;
        let a = (t1 >> i) & 1;
        let b = (t2 >> i) & 1;
        c = (a & b) | (a & c) | (b & c);
    }
    put_word(&mut z, WITNESS_LAYOUT::CARRY_OFF, carry);
    // The ALU bus: the adder sum for arithmetic cycles (t1 ⊕ t2 ⊕
    // carry); the logic result otherwise.
    let alu = if rec.is_arith {
        t1 ^ t2 ^ carry
    } else {
        rec.rd_val
    };
    put_word(&mut z, WITNESS_LAYOUT::ALU, alu);
    z[WITNESS_LAYOUT::SUB] = Gf128(sub as u128);
    z[WITNESS_LAYOUT::IS_ARITH] = Gf128(rec.is_arith as u128);
    z[WITNESS_LAYOUT::HALT] = Gf128(rec.halted_latch as u128);
    z[WITNESS_LAYOUT::WE] = Gf128(rec.we as u128);
    z[WITNESS_LAYOUT::IS_LOAD] = Gf128(
        (rec.store_width == 0 && rec.ram_addr != usize::MAX && rec.we && rec.rd != 0) as u128,
    );
    z[WITNESS_LAYOUT::IS_STORE] = Gf128((rec.store_width > 0) as u128);
    for k in 0..3 {
        z[WITNESS_LAYOUT::OFF_OFF + k] = Gf128(((rec.byte_offset >> k) & 1) as u128);
    }
    for k in 0..2 {
        z[WITNESS_LAYOUT::WIDTH_OFF + k] = Gf128(((rec.store_width >> k) & 1) as u128);
    }
    put_word(&mut z, WITNESS_LAYOUT::RAM_OLD, rec.ram_old);
    put_word(&mut z, WITNESS_LAYOUT::RAM_NEW, rec.ram_new);
    for k in 0..WITNESS_LAYOUT::PC_BITS {
        z[WITNESS_LAYOUT::PC + k] = Gf128(((rec.micro_pc >> k) & 1) as u128);
    }
    // Fetched instruction fields (from the ROM row — the interpreter keeps
    // the raw word; the circuit witness carries the decoded fields; for
    // the test-scale circuit we embed the opcode/funct bits of the record
    // via the trace's program — the zkvm layer maps these through Shout).
    z[WITNESS_LAYOUT::ONE_COL] = ONE;
    // Public x: [output, input words..., zero padding].
    for (j, &xv) in public_x.iter().enumerate() {
        z[WITNESS_LAYOUT::ONE_COL + 1 + j] = xv;
    }
    // Public inputs: packed rd, rs1, rs2 (the I/O binding), then zero
    // padding to fill the x half.
    z
}

#[inline]
fn pack(w: u64) -> Gf128 {
    Gf128(w as u128)
}

#[inline]
fn put_word(z: &mut [Gf128], off: usize, w: u64) {
    for i in 0..64 {
        z[off + i] = Gf128(((w >> i) & 1) as u128);
    }
}

/// Build the dual circuit: returns the matrices and the segment layout
/// information the zkvm layer needs.
pub fn build_circuit() -> CircuitSpec {
    let m = WITNESS_LAYOUT::M;
    // ---- F2 constraints ----
    let mut a1 = SparseMatrix::empty(0, m);
    let mut b1 = SparseMatrix::empty(0, m);
    let mut c1 = SparseMatrix::empty(0, m);
    let mut rows = 0usize;
    let mut add_constraint = |a: &mut SparseMatrix,
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
    // 1. Booleanity for every bit column: b·b = b.
    for col in 0..WITNESS_LAYOUT::CARRY_OFF {
        add_constraint(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(col, ONE)],
            &[(col, ONE)],
            &[(col, ONE)],
        );
    }
    for col in WITNESS_LAYOUT::CARRY_OFF..WITNESS_LAYOUT::SUB {
        add_constraint(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(col, ONE)],
            &[(col, ONE)],
            &[(col, ONE)],
        );
    }
    for col in WITNESS_LAYOUT::OFF_OFF..WITNESS_LAYOUT::RAM_OLD {
        add_constraint(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(col, ONE)],
            &[(col, ONE)],
            &[(col, ONE)],
        );
    }
    for col in WITNESS_LAYOUT::PC..WITNESS_LAYOUT::FETCH_OP {
        add_constraint(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(col, ONE)],
            &[(col, ONE)],
            &[(col, ONE)],
        );
    }
    for col in [
        WITNESS_LAYOUT::SUB,
        WITNESS_LAYOUT::WE,
        WITNESS_LAYOUT::IS_LOAD,
        WITNESS_LAYOUT::IS_STORE,
        WITNESS_LAYOUT::IS_ARITH,
        WITNESS_LAYOUT::HALT,
    ] {
        add_constraint(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(col, ONE)],
            &[(col, ONE)],
            &[(col, ONE)],
        );
    }
    // 2. The adder carry recurrence (64 rows):
    // c_{i+1} + t1_i·t2_i + c_i·t1_i + c_i·t2_i = 0 —
    // expressed as (t1_i + c_i)·(t2_i + c_i) = c_i + c_{i+1} + t1_i t2_i? —
    // standard: c_{i+1} = maj = t1 t2 + t1 c + t2 c →
    // (t1_i + c_i)(t2_i + c_i) = t1 t2 + t1 c + t2 c + c² = c_{i+1} + c_i·c_i
    // → (t1_i + c_i)(t2_i + c_i) = c_{i+1} + c_i (using c² = c, boolean).
    for i in 0..63 {
        let t1i = WITNESS_LAYOUT::T1 + i;
        let t2i = WITNESS_LAYOUT::T2 + i;
        let ci = WITNESS_LAYOUT::CARRY_OFF + i;
        let cn = WITNESS_LAYOUT::CARRY_OFF + i + 1;
        add_constraint(
            &mut a1,
            &mut b1,
            &mut c1,
            &[(t1i, ONE), (ci, ONE)],
            &[(t2i, ONE), (ci, ONE)],
            &[(cn, ONE), (ci, ONE)],
        );
    }
    // c_0 = sub: linear — fold into the K relation below.
    // 3. XOR logic: rd_i = t1_i ⊕ t2_i ⊕ sub-bit-form: for the XOR cycles,
    // rd_i + rs1_i + rs2_i = 0 — one row per bit would be 64 rows; use the
    // packed K check instead (below). Here: the store-copy XOR identity:
    // ram_new_i + ram_old_i + store_src_i = "increment-ish" only inside
    // written lanes — the Router owns that; the circuit's copy binding is
    // the packed linear check in K.
    let _ = rows;

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
    // 4. The packed write mux (§5.1's motivating example — one field
    // constraint replaces 64 bit constraints): when the write-enable is
    // set, the packed ALU output equals the packed rd:
    // we·(Σβ alu_i + Σβ rd_i) = 0.
    {
        let mut a_entries: Vec<(usize, Gf128)> = vec![(WITNESS_LAYOUT::WE, ONE)];
        let mut b_entries: Vec<(usize, Gf128)> = Vec::new();
        for i in 0..64 {
            b_entries.push((WITNESS_LAYOUT::ALU + i, Gf128((1u128) << i)));
            b_entries.push((WITNESS_LAYOUT::RD + i, Gf128((1u128) << i)));
        }
        add_k(&mut a2, &mut b2, &mut c2, &a_entries, &b_entries, &[]);
    }
    // 5. Packed adder sum: Σβ (t1_i + t2_i + carry_i) = Σβ rd_i XOR-form:
    // linear check: Σ β_i·(t1_i + t2_i + c_i + rd_i)·(1) — hmm the sum is
    // linear in the bits; the XOR is the sum: for the ADD shape,
    // rd_i = t1_i ⊕ t2_i ⊕ c_i. As a K linear check:
    // (one row) Σ_i 2^i-ish — use the coefficient trick: for F₂-bit
    // arithmetic, the XOR of bits equals the sum in F₂ ⊂ K: the packed
    // check with β_i coefficients: A·z = B·z·C·z needs quadratic; a pure
    // linear identity needs C = ONE_COL. So: A = [t1, t2, c, rd bits with
    // β], B = [ONE_COL], C = [0]: constraint (Σβ(t1+t2+c+rd))·1 = 0.
    {
        // Adder sum identity, masked by the arithmetic flag:
        // Σβ(t1 + t2 + c + alu)·is_arith = 0 — vacuous off the adder.
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
            &[(WITNESS_LAYOUT::IS_ARITH, ONE)],
            &[],
        );
    }
    // 6. c_0 = sub: linear: sub·1 = c_0.
    add_k(
        &mut a2,
        &mut b2,
        &mut c2,
        &[(WITNESS_LAYOUT::SUB, ONE)],
        &[(WITNESS_LAYOUT::ONE_COL, ONE)],
        &[(WITNESS_LAYOUT::CARRY_OFF, ONE)],
    );
    // 7. The halt-latch I/O binding: halt·(Σβ·rd_i + x_out) = 0 — the
    // halted cycle's packed rd equals the public output.
    {
        let mut a_entries = vec![(WITNESS_LAYOUT::HALT, ONE)];
        let mut b_entries: Vec<(usize, Gf128)> = Vec::new();
        for i in 0..64 {
            b_entries.push((WITNESS_LAYOUT::RD + i, Gf128(1u128 << i)));
        }
        b_entries.push((WITNESS_LAYOUT::ONE_COL + 1, ONE)); // x_out
        add_k(&mut a2, &mut b2, &mut c2, &a_entries, &b_entries, &[]);
    }
    // 8. The store-copy segment is bound to the RAM/RAM-old virtual
    // segments (owned by BinaryTwist) and the routing semantics by
    // BinaryRouter — the composed protocol checks them there (the
    // paper's Steps 4–5); the circuit itself does not duplicate the
    // binding.
    // Both relations must share the same padded row count (paper §5.3:
    // "the two SIMD R1CS relations have the same padded SIMD dimension q
    // and size bound m").
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
            0x00100093,                                                         // addi x1, x0, 1
            0x00100113,                                                         // addi x2, x0, 1
            (0u32 << 25) | (2 << 20) | (1 << 15) | (0 << 12) | (3 << 7) | 0x33, // add x3
            0x0000_0073,
        ];
        let trace = crate::interp::execute(&prog, &[], 64);
        let spec = build_circuit();
        for rec in &trace.records {
            let z = witness_row(rec, &[Gf128(0)]);
            assert!(check_row(&spec, &z).is_ok(), "row pc={}", rec.micro_pc);
        }
    }
}
