//! The RV64I interpreter: executes the expanded program and records the
//! trace of virtual cycles (paper §2.1 step 1) — including the full
//! memory-access log the auxiliary arguments need (the per-cycle register
//! indices and old/new values, the RAM word at the access address, the
//! shift routing output, and the control flags).
//!
//! Branch targets are resolved in **micro-PC units**: the branch immediate
//! (in bytes) is divided by 4 (instruction units) and lands on the target
//! instruction's first virtual instruction.
//!
//! Sub-word loads use the machine's byte-lane semantics — `LB/LBU` loads
//! the byte at `off`, `LH/LHU` the two bytes at `off` and `(off+1) mod 8`
//! (well-defined for every offset), `LW/LD` the full word.

use crate::expand::{expand_program, ExpandedProgram, VirtualKind, SHIFT_STAGES};
use crate::{Word, RAM_WORDS_LOG2, REGS};

/// Machine state (Harvard architecture: separate ROM and RAM).
#[derive(Clone, Debug)]
pub struct ExecState {
    /// Registers (x0 hardwired to zero; padding cells 32..64 unused).
    pub regs: [Word; REGS],
    /// The scratch cell 32 (the redirected x0 write slot) — the
    /// shift-stage chain's accumulator.
    pub scratch: Word,
    /// Data memory.
    pub ram: Vec<Word>,
    /// The micro-PC (in virtual-instruction units).
    pub micro_pc: usize,
    /// Cycle counter.
    pub cycles: usize,
    /// Halt flag (ECALL/EBREAK halts in this configuration).
    pub halted: bool,
    /// The final output (x10 at halt, the "a0" convention).
    pub output: Word,
}

/// The per-cycle trace record (one per virtual cycle) — the CPU circuit's
/// logical witness plus the memory-access log.
#[derive(Clone, Debug, Default)]
pub struct CycleRecord {
    /// The micro-PC at the start of the cycle (the fetch address).
    pub micro_pc: usize,
    /// Register-read values (rs1, rs2).
    pub rs1_val: Word,
    pub rs2_val: Word,
    /// The register value written back (pre-x0-redirect).
    pub rd_val: Word,
    /// The written register's value before the write (the increment log).
    pub rd_old: Word,
    /// The write-enable (is this a writing virtual instruction).
    pub we: bool,
    /// The destination register (after x0 redirect to the trash cell).
    pub rd: usize,
    /// The RAM word address (for loads/stores).
    pub ram_addr: usize,
    /// The full RAM word at the access address (the RAM-Twist read value).
    pub ram_word: Word,
    /// The old RAM word (stores only; equals `ram_word`).
    pub ram_old: Word,
    /// The new RAM word (stores only).
    pub ram_new: Word,
    /// The byte offset (the low 3 bits of the byte address).
    pub byte_offset: u8,
    /// The store width in bytes (0 = not a store).
    pub store_width: u8,
    /// The store-source word (rs2 for stores).
    pub store_src: Word,
    /// The branch-taken flag.
    pub branch_taken: bool,
    /// The next micro-PC.
    pub next_pc: usize,
    /// The ALU's A operand (rs1 or zero).
    pub alu_a: Word,
    /// The ALU's B operand (rs2 or the immediate).
    pub alu_b: Word,
    /// The ALU output (the pre-write-mux value).
    pub alu_out: Word,
    /// The shift-routing output (immediate shifts; 0 otherwise).
    pub shift_out: Word,
    /// Whether this cycle is an adder-family cycle (ADD/SUB/ADDI).
    pub is_arith: bool,
    /// Whether this cycle is a logic-family cycle (XOR/OR/AND).
    pub is_logic: bool,
    /// Whether the adder computes a subtraction (SUB/SLTI/compares).
    pub is_sub_op: bool,
    /// Whether this cycle is an immediate shift (a static routing).
    pub is_shift: bool,
    /// Whether this cycle is a memory load.
    pub is_load: bool,
    /// Whether this cycle is a memory store.
    pub is_store: bool,
    /// The load-family selectors (byte / half / word).
    pub ld_byte: bool,
    pub ld_half: bool,
    pub ld_word: bool,
    /// The halt latch (the ECALL cycle and the padding cycles).
    pub halted_latch: bool,
}

/// The execution trace.
#[derive(Clone, Debug)]
pub struct Trace {
    pub program: ExpandedProgram,
    pub records: Vec<CycleRecord>,
    pub final_state: ExecState,
}

/// Extract a sub-word load under the byte-lane semantics.
fn extract_load(word: Word, funct3: u32, off: u8) -> Word {
    let byte = |i: u8| (word >> (8 * (i & 7))) & 0xff;
    let sign_extend = |v: u64, bits: u32| -> Word {
        let shift = 64 - bits;
        ((v << shift) as i64 >> shift) as Word
    };
    match funct3 {
        0 => sign_extend(byte(off), 8), // LB (sign-extend)
        4 => byte(off),                 // LBU (zero-extend)
        1 => sign_extend(byte(off) | (byte(off.wrapping_add(1)) << 8), 16), // LH
        5 => byte(off) | (byte(off.wrapping_add(1)) << 8), // LHU
        2 => sign_extend(word & 0xffff_ffff, 32), // LW
        6 => word & 0xffff_ffff,        // LWU
        _ => word,                      // LD (full word)
    }
}

/// The register file cell read: cell 32 is the **scratch** (the redirected
/// x0 write slot) used by the shift-stage chain; cells 0-31 are the RV64I
/// registers (cell 0 always reads 0 — writes to x0 redirect to 32).
#[inline]
fn read_cell(st: &ExecState, idx: usize) -> Word {
    if idx == 32 {
        st.scratch
    } else {
        st.regs[idx % REGS]
    }
}

/// The register file cell write (the scratch at 32).
#[inline]
fn write_cell(st: &mut ExecState, idx: usize, val: Word) {
    if idx == 32 {
        st.scratch = val;
    } else {
        st.regs[idx % REGS] = val;
    }
}

/// Execute an execution **in chunks** with explicit state handoff: the
/// segmented-proving interface (the roadmap's Horizon-2 "segments share
/// the commitment" mechanism — the chunks concatenate into exactly the
/// monolithic witness, and the memory arguments' cross-cycle continuity
/// spans the whole row space, so the segment boundaries are invisible to
/// the proof). Runs `chunks` chunks of `chunk_len` cycles, returning the
/// per-chunk records plus the carried final state. The chunk boundaries
/// are pure bookkeeping: halting inside a chunk latches the halt state
/// and subsequent chunks emit halt-latch cycles.
pub fn execute_chunks(
    program: &[u32],
    input: &[Word],
    chunk_len: usize,
    chunks: usize,
) -> (Vec<Vec<CycleRecord>>, ExecState) {
    let prog = expand_program(program);
    let mut st = ExecState {
        regs: [0; REGS],
        scratch: 0,
        ram: vec![0; 1usize << RAM_WORDS_LOG2],
        micro_pc: 0,
        cycles: 0,
        halted: false,
        output: 0,
    };
    for (i, &w) in input.iter().enumerate() {
        if i < st.ram.len() {
            st.ram[i] = w;
        }
    }
    let mut all = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        let mut recs = Vec::with_capacity(chunk_len);
        while recs.len() < chunk_len && !st.halted {
            let rec = step(&prog, &mut st);
            recs.push(rec);
            st.cycles += 1;
        }
        // Halted (or the chunk filled): pad the remaining chunk budget
        // with halt-latch records re-fetching the halting row.
        while recs.len() < chunk_len {
            let last_pc = recs.last().map(|r| r.next_pc).unwrap_or(st.micro_pc);
            let k = last_pc % prog.rows.len().max(1);
            let row = &prog.rows[k].row;
            let write_addr = if row.rd == 0 { 32 } else { row.rd };
            let out = st.output;
            let rec = CycleRecord {
                micro_pc: k,
                rs1_val: read_cell(&st, row.rs1),
                rs2_val: read_cell(&st, row.rs2),
                next_pc: last_pc,
                rd: write_addr,
                rd_old: read_cell(&st, write_addr),
                rd_val: out,
                alu_out: out,
                we: true,
                halted_latch: true,
                ram_addr: 0,
                ram_word: st.ram[0],
                ..Default::default()
            };
            recs.push(rec);
            write_cell(&mut st, write_addr, out);
        }
        all.push(recs);
    }
    (all, st)
}

/// Execute up to `max_cycles` virtual cycles.
pub fn execute(program: &[u32], input: &[Word], max_cycles: usize) -> Trace {
    let prog = expand_program(program);
    let mut st = ExecState {
        regs: [0; REGS],
        scratch: 0,
        ram: vec![0; 1usize << RAM_WORDS_LOG2],
        micro_pc: 0,
        cycles: 0,
        halted: false,
        output: 0,
    };
    // Input convention: input words loaded into RAM starting at word 0.
    for (i, &w) in input.iter().enumerate() {
        if i < st.ram.len() {
            st.ram[i] = w;
        }
    }
    let mut records = Vec::new();
    while !st.halted && st.cycles < max_cycles {
        let rec = step(&prog, &mut st);
        records.push(rec);
        st.cycles += 1;
    }
    // Pad the trace to a power of two with halt-latch records. The
    // padding cycles fetch the **halting row** (the ecall/system row that
    // latched `next_pc`), keep the halt latch set, and re-write the
    // output — so the class one-hot (authenticated against the ROM), the
    // flag folds, and the one-write register model all stay consistent.
    while records.len().next_power_of_two() != records.len() {
        let last_pc = records.last().map(|r| r.next_pc).unwrap_or(0);
        let out = st.output;
        let k = last_pc % prog.rows.len().max(1);
        let row = &prog.rows[k].row;
        let write_addr = if row.rd == 0 { 32 } else { row.rd };
        let rec = CycleRecord {
            micro_pc: k,
            rs1_val: read_cell(&st, row.rs1),
            rs2_val: read_cell(&st, row.rs2),
            next_pc: last_pc,
            rd: write_addr,
            rd_old: read_cell(&st, write_addr),
            rd_val: out,
            alu_out: out,
            we: true,
            halted_latch: true,
            ram_addr: 0,
            ram_word: st.ram[0],
            ..Default::default()
        };
        records.push(rec);
        // Apply the halt-latch write so later padding reads agree with the
        // memory argument's evolution.
        write_cell(&mut st, write_addr, out);
    }
    Trace {
        program: prog,
        records,
        final_state: st,
    }
}

fn step(prog: &ExpandedProgram, st: &mut ExecState) -> CycleRecord {
    let mp = st.micro_pc;
    let inst = &prog.rows[mp % prog.rows.len()];
    let row = &inst.row;
    let rs1_val = read_cell(st, row.rs1);
    let rs2_val = read_cell(st, row.rs2);
    let mut rec = CycleRecord {
        micro_pc: mp,
        rs1_val,
        rs2_val,
        alu_a: rs1_val,
        alu_b: rs2_val,
        next_pc: mp + 1,
        ..Default::default()
    };
    let imm = row.imm as i64;
    let is_reg = (row.raw & 0x7f) == 0x33;
    match row.kind {
        VirtualKind::Alu => {
            // Immediate forms: the B operand is the immediate; register
            // forms: rs2.
            let b_op = if is_reg { rs2_val } else { imm as Word };
            rec.alu_a = rs1_val;
            rec.alu_b = b_op;
            let val: Word;
            let we = true;
            match row.funct3 {
                0 => {
                    rec.is_arith = true;
                    rec.is_sub_op = is_reg && (row.raw >> 25) == 0x20;
                    val = if rec.is_sub_op {
                        rs1_val.wrapping_sub(rs2_val)
                    } else {
                        rs1_val.wrapping_add(b_op)
                    };
                }
                4 => {
                    rec.is_logic = true;
                    val = rs1_val ^ b_op;
                }
                6 => {
                    rec.is_logic = true;
                    val = rs1_val | b_op;
                }
                7 => {
                    rec.is_logic = true;
                    val = rs1_val & b_op;
                }
                1 => {
                    // SLLI: a static shift routing (the amount is a ROM
                    // constant).
                    rec.is_shift = true;
                    let sh = b_op & 63;
                    val = rs1_val.wrapping_shl(sh as u32);
                    rec.shift_out = val;
                }
                5 => {
                    // SRLI/SRAI: static routings.
                    rec.is_shift = true;
                    let sh = b_op & 63;
                    val = if (row.raw >> 25) == 0x20 {
                        ((rs1_val as i64) >> sh) as Word
                    } else {
                        rs1_val >> sh
                    };
                    rec.shift_out = val;
                }
                3 => {
                    rec.is_arith = true;
                    rec.is_sub_op = true; // SLTI(U): the adder computes a − b
                    let unsigned = (row.raw >> 12) & 0x7 == 3;
                    val = if unsigned {
                        (rs1_val < b_op) as Word
                    } else {
                        ((rs1_val as i64) < b_op as i64) as Word
                    };
                }
                _ => {
                    val = 0;
                }
            }
            rec.alu_out = val;
            rec.rd_val = val;
            rec.we = we;
            rec.rd = if row.rd == 0 { 32 } else { row.rd };
            st.micro_pc = (mp + 1) % prog_len(prog);
            rec.next_pc = st.micro_pc;
            commit(st, &mut rec)
        }
        VirtualKind::ShiftStage { stage } => {
            // Static stage: shift the scratch chain by 2^stage if bit
            // `stage` of the amount (rs2) is set. Stage 0 reads rs1,
            // later stages read the scratch cell 32; every stage writes
            // the chain (cell 32), and the last stage writes rd — all
            // through the register file so the BinaryTwist binds the
            // chain, and the stage routing validates each written value.
            let acc = if stage == 0 { rs1_val } else { st.scratch };
            let bit = (rs2_val >> stage) & 1;
            let sh = (1u64 << stage) * bit;
            let val = match row.funct3 {
                1 => acc.wrapping_shl(sh as u32),
                5 => {
                    if (row.raw >> 25) == 0x20 {
                        ((acc as i64) >> sh) as Word
                    } else {
                        acc >> sh
                    }
                }
                _ => acc,
            };
            // Stage chaining through the register file: stages 0-4 write
            // the scratch cell; the last stage writes rd. we=1 on all
            // stage cycles (the class fold demands it).
            if stage == SHIFT_STAGES - 1 {
                rec.rd_val = val;
                rec.alu_out = val;
                rec.we = true;
                rec.rd = if row.rd == 0 { 32 } else { row.rd };
            } else {
                rec.rd_val = val;
                rec.alu_out = val;
                rec.we = true;
                rec.rd = 32;
            }
            st.micro_pc = (mp + 1) % prog_len(prog);
            rec.next_pc = st.micro_pc;
            commit(st, &mut rec)
        }
        VirtualKind::Load => {
            let addr_word = rs1_val.wrapping_add(imm as Word);
            let byte_addr = addr_word as usize;
            let waddr = (byte_addr / 8) % st.ram.len();
            let off = (byte_addr % 8) as u8;
            let word = st.ram[waddr];
            let val = extract_load(word, row.funct3, off);
            rec.ram_addr = waddr;
            rec.ram_word = word;
            rec.byte_offset = off;
            rec.is_load = true;
            rec.ld_byte = matches!(row.funct3, 0 | 4);
            rec.ld_half = matches!(row.funct3, 1 | 5);
            rec.ld_word = row.funct3 == 3; // LD: the full 64-bit word
            rec.rd_val = val;
            // The ALU computes the effective address (rs1 + imm): the
            // adder identity applies and the circuit binds ram_addr to
            // its low bits.
            rec.is_arith = true;
            rec.is_sub_op = false;
            rec.alu_out = addr_word;
            rec.alu_a = rs1_val;
            rec.alu_b = imm as Word;
            rec.we = true;
            rec.rd = if row.rd == 0 { 32 } else { row.rd };
            st.micro_pc = (mp + 1) % prog_len(prog);
            rec.next_pc = st.micro_pc;
            commit(st, &mut rec)
        }
        VirtualKind::Store => {
            let addr_word = rs1_val.wrapping_add(imm as Word);
            let byte_addr = addr_word as usize;
            let waddr = (byte_addr / 8) % st.ram.len();
            let off = (byte_addr % 8) as u8;
            let old = st.ram[waddr];
            let src = rs2_val;
            let width: u8 = match row.funct3 {
                0 => 1, // SB
                1 => 2, // SH
                2 => 4, // SW
                _ => 8, // SD
            };
            let mask: Word = match width {
                1 => 0xff,
                2 => 0xffff,
                4 => 0xffff_ffff,
                _ => u64::MAX,
            } << (8 * off as u32);
            let new = (old & !mask) | ((src << (8 * off as u32)) & mask);
            rec.ram_addr = waddr;
            rec.ram_word = old;
            rec.ram_old = old;
            rec.ram_new = new;
            rec.byte_offset = off;
            rec.store_width = width;
            rec.store_src = src;
            rec.is_store = true;
            // The effective address (rs1 + imm) runs through the adder:
            // the class fold sets is_arith for stores so the
            // masked-adder row validates the address the memory argument
            // consumes.
            rec.is_arith = true;
            rec.is_sub_op = false;
            rec.alu_a = rs1_val;
            rec.alu_b = imm as Word;
            rec.alu_out = addr_word; // the effective address
            st.ram[waddr] = new;
            rec.we = false;
            rec.next_pc = (mp + 1) % prog_len(prog);
            st.micro_pc = rec.next_pc;
            commit(st, &mut rec)
        }
        VirtualKind::Branch => {
            rec.is_arith = true;
            rec.is_sub_op = true; // compares compute the difference
            rec.alu_out = rs1_val.wrapping_sub(rs2_val);
            let taken = match row.funct3 {
                0 => rs1_val == rs2_val,
                1 => rs1_val != rs2_val,
                4 => (rs1_val as i64) < rs2_val as i64,
                5 => (rs1_val as i64) >= rs2_val as i64,
                6 => rs1_val < rs2_val,
                7 => rs1_val >= rs2_val,
                _ => false,
            };
            rec.branch_taken = taken;
            if taken {
                // Branch target: the immediate is in bytes; convert to
                // virtual-instruction units.
                let target = (mp as i64 + imm / 4).max(0) as usize;
                st.micro_pc = target % prog_len(prog);
                rec.next_pc = st.micro_pc;
            } else {
                st.micro_pc = (mp + 1) % prog_len(prog);
                rec.next_pc = st.micro_pc;
            }
            commit(st, &mut rec)
        }
        VirtualKind::Upper => {
            // LUI/AUIPC and the JAL/JALR link rows: the expansion
            // pre-adjusted the immediate (AUIPC += 4·k; the link rows =
            // 4·k + 4) and forced rs1 = 0, so the adder — t1 = cell 0's
            // content (always 0), t2 = the bus immediate — validates the
            // value (the class fold sets is_arith).
            let val = row.imm;
            rec.is_arith = true;
            rec.is_sub_op = false;
            rec.alu_a = rs1_val; // rs1 = 0 => the cell-0 read (0)
            rec.alu_b = row.imm;
            rec.alu_out = val;
            rec.rd_val = val;
            rec.we = true;
            rec.rd = if row.rd == 0 { 32 } else { row.rd };
            rec.next_pc = (mp + 1) % prog_len(prog);
            st.micro_pc = rec.next_pc;
            commit(st, &mut rec)
        }
        VirtualKind::Jump => {
            // The pc-jump row (JAL/JALR's second row): no datapath
            // effect (class None — no writes, no adder); the target
            // computation stays the paper's Remark-4 omission.
            let is_jalr = (row.raw & 0x7f) == 0x67;
            let target: usize = if is_jalr {
                ((rs1_val.wrapping_add(imm as Word)) / 4) as usize
            } else {
                (mp as i64 + imm / 4).max(0) as usize
            };
            let t = target % prog_len(prog);
            st.micro_pc = t;
            rec.next_pc = t;
            commit(st, &mut rec)
        }
        VirtualKind::System => {
            st.halted = true;
            st.output = st.regs[10];
            // The halt latch carries the output for the I/O binding
            // (the K constraint halt·(packed rd + x_out) = 0). The
            // class fold sets we=1: the cycle (and every padding cycle
            // that re-fetches this row) writes the output at the row's
            // redirected destination.
            rec.rd_val = st.output;
            rec.alu_out = st.output;
            rec.we = true;
            rec.halted_latch = true;
            rec.next_pc = mp; // halt latch
            commit(st, &mut rec)
        }
    }
}

/// Normalize the write port and write back. The BinaryTwist register model
/// performs **one write every cycle** at the ROM row's redirected
/// destination (`rd == 0` maps to the trash cell 32): on write-enabled
/// cycles the record's value lands; on other cycles the write returns the
/// cell's current content (a no-op with zero increment); the halt-latch
/// cycles write the output. The interpreter's state therefore always
/// matches the memory argument's evolution.
fn commit(st: &mut ExecState, rec: &mut CycleRecord) -> CycleRecord {
    if rec.rd == 0 {
        // Handlers that never set a destination write to the trash cell.
        rec.rd = 32;
    }
    if !rec.is_load && !rec.is_store {
        // The RAM-Twist read port reads the access address's current
        // content every cycle (cell 0's word on non-memory cycles) — the
        // load-val segment logs exactly that.
        rec.ram_word = st.ram[rec.ram_addr];
    }
    if !rec.we && !rec.halted_latch {
        // The no-op write's value is the current content.
        rec.rd_val = read_cell(st, rec.rd);
    }
    rec.rd_old = read_cell(st, rec.rd);
    write_cell(st, rec.rd, rec.rd_val);
    rec.clone()
}

fn prog_len(prog: &ExpandedProgram) -> usize {
    prog.rows.len().max(1)
}

/// Extend a trace's halt-latch padding up to `target` records (a power of
/// two ≥ the current count). The proving key fixes the trace length; the
/// interpreter's own padding only reaches the next power of two, so the
/// protocol pads the remainder with the same halt-latch rows (re-fetching
/// the halting row and re-writing the output, exactly as the in-interpreter
/// padding does).
pub fn pad_records(trace: &Trace, target: usize) -> Vec<CycleRecord> {
    let mut records = trace.records.clone();
    let mut st = trace.final_state.clone();
    let rows = &trace.program.rows;
    while records.len() < target {
        let last_pc = records.last().map(|r| r.next_pc).unwrap_or(0);
        let out = st.output;
        let k = last_pc % rows.len().max(1);
        let row = &rows[k].row;
        let write_addr = if row.rd == 0 { 32 } else { row.rd };
        let rec = CycleRecord {
            micro_pc: k,
            rs1_val: read_cell(&st, row.rs1),
            rs2_val: read_cell(&st, row.rs2),
            next_pc: last_pc,
            rd: write_addr,
            rd_old: read_cell(&st, write_addr),
            rd_val: out,
            alu_out: out,
            we: true,
            halted_latch: true,
            ram_addr: 0,
            ram_word: st.ram[0],
            ..Default::default()
        };
        records.push(rec);
        write_cell(&mut st, write_addr, out);
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
        ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
    }

    #[test]
    fn fibonacci() {
        let prog = vec![
            enc_i(1, 0, 0, 1, 0x13),
            enc_i(1, 0, 0, 2, 0x13),
            ((2 << 20) | (1 << 15)) | (3 << 7) | 0x33,
            ((3 << 20) | (2 << 15)) | (4 << 7) | 0x33,
            0x0000_0073,
        ];
        let trace = execute(&prog, &[], 64);
        assert_eq!(trace.final_state.regs[3], 2);
        assert_eq!(trace.final_state.regs[4], 3);
        assert!(trace.final_state.halted);
        assert!(trace.records.len().is_power_of_two());
    }

    #[test]
    fn store_load_roundtrip() {
        let prog = vec![
            enc_i(8, 0, 0, 1, 0x13),
            enc_i(0xAB, 0, 0, 2, 0x13),
            ((2u32 << 20) | (1 << 15)) | 0x23, // sb x2, 0(x1)
            0x0000_0073,
        ];
        let trace = execute(&prog, &[], 64);
        assert_eq!(trace.final_state.ram[1] & 0xff, 0xAB);
    }

    #[test]
    fn register_shift_expansion_runs() {
        let prog = vec![
            enc_i(5, 0, 0, 1, 0x13),
            enc_i(3, 0, 0, 2, 0x13),
            (2 << 20) | (1 << 15) | (1 << 12) | (3 << 7) | 0x33, // sll x3, x1, x2
            0x0000_0073,
        ];
        let trace = execute(&prog, &[], 128);
        assert_eq!(trace.final_state.regs[3], 5u64 << 3);
        assert!(trace.records.len() >= 3 + SHIFT_STAGES);
    }

    #[test]
    fn sub_word_load_semantics() {
        // Store 0xAB at word 2 offset 1, then LB/LH at offset 1.
        let prog = vec![
            enc_i(16, 0, 0, 1, 0x13),   // x1 = 16 (word 2)
            enc_i(0xAB, 0, 0, 5, 0x13), // x5 = 0xAB
            // sb x5, 1(x1): the byte at byte 1 of word 2.
            ((5 << 20) | (1 << 15)) | (1 << 7) | 0x23,
            // lb x3, 1(x1) = 0xAB.
            ((1u32 << 20) | (1 << 15)) | (3 << 7) | 0x03,
            // lh x4, 1(x1) = 0x00AB (bytes 1 and 2).
            (1u32 << 20) | (1 << 15) | (1 << 12) | (4 << 7) | 0x03,
            0x0000_0073,
        ];
        let trace = execute(&prog, &[], 128);
        // RV64I semantics: LB sign-extends (0xAB is negative as a byte);
        // LH of 0x00AB keeps it positive (bit 15 = 0).
        assert_eq!(trace.final_state.regs[3], 0xFFFF_FFFF_FFFF_FFAB);
        assert_eq!(trace.final_state.regs[4], 0xAB);
    }

    #[test]
    fn load_extension_semantics() {
        // Build word 0 = 0x8080_8080 with byte stores, then exercise every
        // load width's extension rule (RV64I).
        let prog = vec![
            enc_i(0, 0, 0, 1, 0x13),                      // x1 = 0
            enc_i(0x80, 0, 0, 5, 0x13),                   // x5 = 0x80
            ((5u32 << 20) | (1 << 15)) | 0x23,            // sb x5, 0(x1)
            ((5u32 << 20) | (1 << 15)) | (1 << 7) | 0x23, // sb x5, 1(x1)
            ((5u32 << 20) | (1 << 15)) | (2 << 7) | 0x23, // sb x5, 2(x1)
            ((5u32 << 20) | (1 << 15)) | (3 << 7) | 0x23, // sb x5, 3(x1)
            (1 << 15) | (3 << 7) | 0x03,                  // lb  x3, 0(x1)
            (1 << 15) | (4 << 12) | (6 << 7) | 0x03,      // lbu x6, 0(x1)
            (1 << 15) | (1 << 12) | (7 << 7) | 0x03,      // lh  x7, 0(x1)
            (1 << 15) | (5 << 12) | (8 << 7) | 0x03,      // lhu x8, 0(x1)
            (1 << 15) | (2 << 12) | (10 << 7) | 0x03,     // lw  x10, 0(x1)
            (1 << 15) | (6 << 12) | (11 << 7) | 0x03,     // lwu x11, 0(x1)
            (1 << 15) | (3 << 12) | (12 << 7) | 0x03,     // ld  x12, 0(x1)
            0x0000_0073,
        ];
        let trace = execute(&prog, &[], 128);
        // word 0 = 0x8080_8080: every width's top bit is set.
        assert_eq!(trace.final_state.regs[3], 0xFFFF_FFFF_FFFF_FF80);
        assert_eq!(trace.final_state.regs[6], 0x80);
        assert_eq!(trace.final_state.regs[7], 0xFFFF_FFFF_FFFF_8080);
        assert_eq!(trace.final_state.regs[8], 0x8080);
        assert_eq!(trace.final_state.regs[10], 0xFFFF_FFFF_8080_8080);
        assert_eq!(trace.final_state.regs[11], 0x8080_8080);
        assert_eq!(trace.final_state.regs[12], 0x0000_0000_8080_8080); // bytes 4-7 unstored
    }
}
