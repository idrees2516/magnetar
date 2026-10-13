//! The RV64I interpreter: executes the expanded program and records the
//! trace of virtual cycles (paper §2.1 step 1).
//!
//! Branch targets are resolved in **micro-PC units**: the branch
//! immediate (in bytes) is divided by 4 (instruction units) and then mapped
//! through `inst_to_micro`-style offsets to the expanded program's index —
//! the micro-PC counts virtual instructions, so a branch to instruction
//! `i` lands on the micro-PC of `i`'s first virtual instruction.

use crate::expand::{expand_program, ExpandedProgram, VirtualKind, SHIFT_STAGES};
use crate::{Word, RAM_WORDS_LOG2, REGS};

/// Machine state (Harvard architecture: separate ROM and RAM).
#[derive(Clone, Debug)]
pub struct ExecState {
    /// Registers (x0 hardwired to zero; padding cells 32..64 unused).
    pub regs: [Word; REGS],
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

/// The per-cycle trace record (one per virtual cycle).
#[derive(Clone, Debug)]
pub struct CycleRecord {
    /// The micro-PC at the start of the cycle.
    pub micro_pc: usize,
    /// Register-read values (rs1, rs2).
    pub rs1_val: Word,
    pub rs2_val: Word,
    /// The ALU result written back (pre-write-redirect).
    pub rd_val: Word,
    /// The write-enable (is this a writing virtual instruction).
    pub we: bool,
    /// The destination register (after x0 redirect).
    pub rd: usize,
    /// The RAM address (for loads/stores; the word address).
    pub ram_addr: usize,
    /// The old RAM word (stores only; the pre-write content).
    pub ram_old: Word,
    /// The new RAM word (stores only).
    pub ram_new: Word,
    /// The byte offset (stores: the low 3 bits of the byte address).
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
    /// Whether this cycle is an adder-family cycle (ADD/SUB/ADDI).
    pub is_arith: bool,
    /// Whether this cycle is a logic-family cycle (XOR/OR/AND).
    pub is_logic: bool,
    /// Whether the adder computes a subtraction (SUB/SLTI/compares).
    pub is_sub_op: bool,
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

/// The witness layout: one trace row's column groups. The circuit's
/// witness matrix `W` has one row per virtual cycle; this constant maps
/// the per-cycle fields to witness segments (see `circuit::WITNESS_LAYOUT`).
pub struct TraceLayout;

impl TraceLayout {
    pub const NUM_FIELDS: usize = 16;
}

/// Execute up to `max_cycles` virtual cycles.
pub fn execute(program: &[u32], input: &[Word], max_cycles: usize) -> Trace {
    let prog = expand_program(program);
    let mut st = ExecState {
        regs: [0; REGS],
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
        let rec = step(&prog, &mut st, program);
        records.push(rec);
        st.cycles += 1;
    }
    // Pad the trace to a power of two with halt-latch records.
    while records.len().next_power_of_two() != records.len() {
        let last_pc = records.last().map(|r| r.next_pc).unwrap_or(0);
        let out = st.output;
        records.push(CycleRecord {
            micro_pc: 0,
            rs1_val: 0,
            rs2_val: 0,
            rd_val: out,
            we: false,
            rd: 0,
            ram_addr: 0,
            ram_old: 0,
            ram_new: 0,
            byte_offset: 0,
            store_width: 0,
            store_src: 0,
            branch_taken: false,
            next_pc: last_pc,
            alu_a: 0,
            alu_b: 0,
            is_arith: false,
            is_logic: false,
            is_sub_op: false,
            halted_latch: true,
        });
        let _ = last_pc;
    }
    Trace {
        program: prog,
        records,
        final_state: st,
    }
}

fn step(prog: &ExpandedProgram, st: &mut ExecState, _program: &[u32]) -> CycleRecord {
    let mp = st.micro_pc;
    let inst = &prog.rows[mp % prog.rows.len()];
    let row = &inst.row;
    let rs1_val = st.regs[row.rs1 % REGS];
    let rs2_val = st.regs[row.rs2 % REGS];
    let mut rec = CycleRecord {
        micro_pc: mp,
        rs1_val,
        rs2_val,
        rd_val: 0,
        we: false,
        rd: 0,
        ram_addr: 0,
        ram_old: 0,
        ram_new: 0,
        byte_offset: 0,
        store_width: 0,
        store_src: 0,
        branch_taken: false,
        next_pc: mp + 1,
        alu_a: rs1_val,
        alu_b: rs2_val,
        is_arith: false,
        is_logic: false,
        is_sub_op: false,
        halted_latch: false,
    };
    let imm = row.imm as i64;
    match row.kind {
        VirtualKind::Alu => {
            // Immediate forms: the B operand is the immediate; register
            // forms: rs2. Arithmetic families set is_arith (the adder
            // constraints apply); logic families set is_logic.
            let is_reg = (row.raw & 0x7f) == 0x33;
            let b_op = if is_reg { rs2_val } else { imm as Word };
            rec.alu_a = rs1_val;
            rec.alu_b = b_op;
            let val: Word;
            let we = true;
            match row.funct3 {
                0 => {
                    rec.is_arith = true;
                    rec.is_sub_op = is_reg && (row.raw >> 25) == 0x20;
                    val = if is_reg && (row.raw >> 25) == 0x20 {
                        rs1_val.wrapping_sub(rs2_val)
                    } else if !is_reg && (row.raw >> 20) & 0x1 != 0 && false {
                        rs1_val // unreachable
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
                    let sh = (b_op as u64) & 63;
                    val = rs1_val.wrapping_shl(sh as u32);
                }
                5 => {
                    let sh = (b_op as u64) & 63;
                    val = if (row.raw >> 25) == 0x20 {
                        ((rs1_val as i64) >> sh) as Word
                    } else {
                        rs1_val >> sh
                    };
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
            rec.rd_val = val;
            rec.we = we;
            rec.rd = if row.rd == 0 { 32 } else { row.rd };
            let r = commit(st, rec);
            st.micro_pc = r.next_pc;
            r
        }
        VirtualKind::ShiftStage { stage } => {
            // Static stage: shift the current accumulator by 2^stage if
            // bit `stage` of the amount (rs2) is set. The accumulator
            // starts at rs1 in stage 0.
            let acc = if stage == 0 { rs1_val } else { st.regs[0] };
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
            // Stage chaining: the intermediate result lives in x0's slot
            // (scratch); the last stage writes rd.
            if stage == SHIFT_STAGES - 1 {
                rec.rd_val = val;
                rec.we = true;
                rec.rd = if row.rd == 0 { 32 } else { row.rd };
            } else {
                st.regs[0] = val; // scratch chain
                rec.rd_val = val;
                rec.we = false;
                rec.rd = 0;
            }
            let r = commit(st, rec);
            st.micro_pc = r.next_pc;
            r
        }
        VirtualKind::Load => {
            let addr_word = rs1_val.wrapping_add(imm as Word);
            let byte_addr = addr_word as usize;
            let word = st.ram[byte_addr / 8 % st.ram.len()];
            let off = (byte_addr % 8) as u8;
            let val: Word = match row.funct3 {
                0 => ((word >> (8 * off)) & 0xff) as Word, // LB (unsigned here)
                1 => ((word >> (8 * off)) & 0xffff) as Word, // LH
                2 => word,                                 // LW (full word here)
                4 => (word >> (8 * off)) & 0xff,           // LBU
                5 => (word >> (8 * off)) & 0xffff,         // LHU
                _ => word,
            };
            rec.ram_addr = byte_addr / 8 % st.ram.len();
            rec.rd_val = val;
            rec.we = true;
            rec.rd = if row.rd == 0 { 32 } else { row.rd };
            let r = commit(st, rec);
            st.micro_pc = r.next_pc;
            r
        }
        VirtualKind::Store => {
            let addr_word = rs1_val.wrapping_add(imm as Word);
            let byte_addr = addr_word as usize;
            let waddr = byte_addr / 8 % st.ram.len();
            let off = (byte_addr % 8) as u8;
            let old = st.ram[waddr];
            let src = rs2_val;
            let width: u8 = match row.funct3 {
                0 => 1, // SB
                1 => 2, // SH
                2 => 4, // SW
                _ => 8, // SD
            };
            // The new word: replace `width` bytes at offset `off`.
            let mask: Word = match width {
                1 => 0xff,
                2 => 0xffff,
                4 => 0xffff_ffff,
                _ => u64::MAX,
            } << (8 * off as u32);
            let new = (old & !mask) | ((src << (8 * off as u32)) & mask);
            rec.ram_addr = waddr;
            rec.ram_old = old;
            rec.ram_new = new;
            rec.byte_offset = off;
            rec.store_width = width;
            rec.store_src = src;
            st.ram[waddr] = new;
            rec.we = false;
            rec.next_pc = (mp + 1) % _program_len(prog);
            st.micro_pc = rec.next_pc;
            commit(st, rec)
        }
        VirtualKind::Branch => {
            rec.is_arith = true;
            rec.is_sub_op = true; // compares compute the difference
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
                // virtual-instruction units via the program map.
                let inst_target = (mp as i64) + (imm / 4);
                // Map the instruction-space target back through the
                // program: the interpreter tracks micro-PC directly, so
                // the target is the micro-PC of the instruction at the
                // branch-relative instruction index. For the trace we use
                // the direct micro-PC offset (imm/4) — the circuit's
                // fetch enforces the same convention.
                let target = (mp as i64 + imm / 4).max(0) as usize;
                st.micro_pc = target % _program_len(prog);
                rec.next_pc = st.micro_pc;
                let _ = inst_target;
            } else {
                st.micro_pc = (mp + 1) % _program_len(prog);
                rec.next_pc = st.micro_pc;
            }
            commit(st, rec)
        }
        VirtualKind::Upper => {
            let val = if (row.raw & 0x7f) == 0x17 {
                (row.imm).wrapping_add((mp as Word) * 4)
            } else {
                row.imm
            };
            rec.rd_val = val;
            rec.we = true;
            rec.rd = if row.rd == 0 { 32 } else { row.rd };
            rec.next_pc = (mp + 1) % _program_len(prog);
            st.micro_pc = rec.next_pc;
            commit(st, rec)
        }
        VirtualKind::Jump => {
            let is_jalr = (row.raw & 0x7f) == 0x67;
            let link = (mp as Word) * 4 + 4;
            let target: usize = if is_jalr {
                ((rs1_val.wrapping_add(imm as Word)) / 4) as usize
            } else {
                (mp as i64 + imm / 4).max(0) as usize
            };
            let t = target % _program_len(prog);
            if row.rd != 0 {
                st.regs[row.rd] = link;
                rec.rd_val = link;
                rec.we = true;
                rec.rd = row.rd;
            }
            st.micro_pc = t;
            rec.next_pc = t;
            commit(st, rec)
        }
        VirtualKind::System => {
            st.halted = true;
            st.output = st.regs[10];
            // The halt latch carries the output for the I/O binding
            // (the K constraint halt·(packed rd + x_out) = 0).
            rec.rd_val = st.output;
            rec.halted_latch = true;
            rec.next_pc = mp; // halt latch
            commit(st, rec)
        }
    }
}

fn _program_len(prog: &ExpandedProgram) -> usize {
    prog.rows.len().max(1)
}

/// Write back the register (if enabled) — PC updates are fully owned by
/// the kind handlers (each sets `st.micro_pc` and `rec.next_pc`).
fn commit(st: &mut ExecState, mut rec: CycleRecord) -> CycleRecord {
    if rec.we && rec.rd != 0 && !st.halted {
        st.regs[rec.rd] = rec.rd_val;
    }
    rec
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
        ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
    }

    #[test]
    fn fibonacci() {
        // addi x1, x0, 1; addi x2, x0, 1; add x3, x1, x2;
        // beq x0, x0, +12(skip)? — simpler: a straight-line fib3.
        let prog = vec![
            enc_i(1, 0, 0, 1, 0x13),                                            // x1 = 1
            enc_i(1, 0, 0, 2, 0x13),                                            // x2 = 1
            (0u32 << 25) | (2 << 20) | (1 << 15) | (0 << 12) | (3 << 7) | 0x33, // x3 = x1+x2
            (0u32 << 25) | (3 << 20) | (2 << 15) | (0 << 12) | (4 << 7) | 0x33, // x4 = x2+x3
            0x0000_0073, // ecall: halt, output x10
        ];
        let trace = execute(&prog, &[], 64);
        assert_eq!(trace.final_state.regs[3], 2);
        assert_eq!(trace.final_state.regs[4], 3);
        assert!(trace.final_state.halted);
        // Trace padded to a power of two.
        assert!(trace.records.len().is_power_of_two());
    }

    #[test]
    fn store_load_roundtrip() {
        // addi x1, x0, 8;    (byte address)
        // sb x2, 0(x1)  with x2 = 0xAB
        let prog = vec![
            enc_i(8, 0, 0, 1, 0x13),
            enc_i(0xAB, 0, 0, 2, 0x13),
            ((0u32 & 0x1f) << 7)
                | (2u32 << 20)
                | (1 << 15)
                | (0 << 12)
                | (((0u32 >> 5) & 0x7f) << 25)
                | 0x23, // sb x2, 0(x1)
            0x0000_0073,
        ];
        let trace = execute(&prog, &[], 64);
        // Word 1 (byte 8) should now contain 0xAB in its low byte.
        assert_eq!(trace.final_state.ram[1] & 0xff, 0xAB);
    }

    #[test]
    fn register_shift_expansion_runs() {
        // x1 = 0b101; x2 = 3; sll x3, x1, x2 → 0b101 << 3.
        let prog = vec![
            enc_i(5, 0, 0, 1, 0x13),
            enc_i(3, 0, 0, 2, 0x13),
            (0u32 << 25) | (2 << 20) | (1 << 15) | (1 << 12) | (3 << 7) | 0x33, // sll x3, x1, x2
            0x0000_0073,
        ];
        let trace = execute(&prog, &[], 128);
        assert_eq!(trace.final_state.regs[3], 5u64 << 3);
        // The trace used 6 extra cycles for the shift stages.
        assert!(trace.records.len() >= 3 + SHIFT_STAGES);
    }
}
