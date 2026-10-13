//! Microcode expansion: each RV64I instruction becomes one or more virtual
//! instructions forming the public ROM (paper §2.1).
//!
//! Register-shift instructions (`SLL/SRL/SRA` by register) expand into a
//! fixed sequence of **static routing stages** (§5.4): a 6-stage binary
//! decomposition of the shift amount, each stage a static shift by
//! `2^k·bit_k(amount)`, with the shift output selected stage by stage.
//! Immediate shifts are already static routings and stay single.
//!
//! The micro-PC indexes the expanded program; branch targets are mapped to
//! the micro-PC of the target instruction's first virtual instruction.

use crate::decode::{decode, Opcode};
use crate::Word;

/// The kind of a virtual instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VirtualKind {
    /// A full ALU operation (register or immediate form).
    Alu,
    /// One static shift-routing stage.
    ShiftStage {
        stage: usize,
    },
    /// Load (width in the row).
    Load,
    /// Store (width + offset bits in the row).
    Store,
    /// Branch.
    Branch,
    /// LUI / AUIPC / JAL / JALR.
    Upper,
    Jump,
    /// ECALL / EBREAK.
    System,
}

/// One row of the public bytecode table ROM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RomRow {
    pub kind: VirtualKind,
    /// The source instruction's fields (already decoded).
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub funct3: u32,
    pub imm: Word,
    /// The immediate's operand role per kind (store data register, etc.).
    pub raw: u32,
}

/// A virtual instruction (for the interpreter).
#[derive(Clone, Debug)]
pub struct VirtualInst {
    pub row: RomRow,
}

/// The expanded program.
#[derive(Clone, Debug)]
pub struct ExpandedProgram {
    pub rows: Vec<VirtualInst>,
    /// `inst_to_micro[i]` = the micro-PC of program instruction `i`'s first
    /// virtual instruction.
    pub inst_to_micro: Vec<usize>,
    /// Total virtual instructions.
    pub len: usize,
}

/// The number of shift-routing stages (6 for 64-bit amounts).
pub const SHIFT_STAGES: usize = 6;

/// Expand a program (a list of raw instruction words) into virtual
/// instructions.
pub fn expand_program(program: &[u32]) -> ExpandedProgram {
    let mut rows = Vec::new();
    let mut inst_to_micro = Vec::with_capacity(program.len());
    for &raw in program {
        inst_to_micro.push(rows.len());
        let inst = decode(raw);
        match inst.opcode {
            Opcode::Op | Opcode::OpImm | Opcode::Op32 | Opcode::OpImm32 => {
                // Register shifts expand into static-routing stages.
                // Register shifts: SLL (funct3=1), SRL/SRA (funct3=5) in
                // the register form (Opcode::Op).
                let is_reg_shift = inst.opcode == Opcode::Op && matches!(inst.funct3, 1 | 5);
                let is_imm_shift = inst.opcode == Opcode::OpImm && matches!(inst.funct3, 1 | 5);
                if is_reg_shift {
                    // 6 static stages: stage k shifts by 2^k if bit k of
                    // the amount is set (amount = rs2's value).
                    for stage in 0..SHIFT_STAGES {
                        rows.push(VirtualInst {
                            row: RomRow {
                                kind: VirtualKind::ShiftStage { stage },
                                rd: if stage == SHIFT_STAGES - 1 {
                                    inst.rd
                                } else {
                                    0 // intermediate stages use a scratch
                                      // (the interpreter chains stages)
                                },
                                rs1: inst.rs1,
                                rs2: inst.rs2,
                                funct3: inst.funct3,
                                imm: inst.imm as Word,
                                raw,
                            },
                        });
                    }
                } else if is_imm_shift {
                    // Immediate shifts: single virtual instruction (static
                    // routing — the amount is a ROM constant).
                    rows.push(VirtualInst {
                        row: RomRow {
                            kind: VirtualKind::Alu,
                            rd: inst.rd,
                            rs1: inst.rs1,
                            rs2: inst.rs2,
                            funct3: inst.funct3,
                            imm: inst.imm as Word,
                            raw,
                        },
                    });
                } else {
                    rows.push(VirtualInst {
                        row: RomRow {
                            kind: VirtualKind::Alu,
                            rd: inst.rd,
                            rs1: inst.rs1,
                            rs2: inst.rs2,
                            funct3: inst.funct3,
                            imm: inst.imm as Word,
                            raw,
                        },
                    });
                }
            }
            Opcode::Load => rows.push(VirtualInst {
                row: RomRow {
                    kind: VirtualKind::Load,
                    rd: inst.rd,
                    rs1: inst.rs1,
                    rs2: 0,
                    funct3: inst.funct3,
                    imm: inst.imm as Word,
                    raw,
                },
            }),
            Opcode::Store => rows.push(VirtualInst {
                row: RomRow {
                    kind: VirtualKind::Store,
                    rd: 0,
                    rs1: inst.rs1,
                    rs2: inst.rs2,
                    funct3: inst.funct3,
                    imm: inst.imm as Word,
                    raw,
                },
            }),
            Opcode::Branch => rows.push(VirtualInst {
                row: RomRow {
                    kind: VirtualKind::Branch,
                    rd: 0,
                    rs1: inst.rs1,
                    rs2: inst.rs2,
                    funct3: inst.funct3,
                    imm: inst.imm as Word,
                    raw,
                },
            }),
            Opcode::Lui | Opcode::Auipc => rows.push(VirtualInst {
                row: RomRow {
                    kind: VirtualKind::Upper,
                    rd: inst.rd,
                    rs1: 0,
                    rs2: 0,
                    funct3: 0,
                    imm: inst.imm as Word,
                    raw,
                },
            }),
            Opcode::Jal | Opcode::Jalr => rows.push(VirtualInst {
                row: RomRow {
                    kind: VirtualKind::Jump,
                    rd: inst.rd,
                    rs1: inst.rs1,
                    rs2: 0,
                    funct3: 0,
                    imm: inst.imm as Word,
                    raw,
                },
            }),
            Opcode::Fence | Opcode::Ecall | Opcode::Ebreak | Opcode::Invalid => {
                rows.push(VirtualInst {
                    row: RomRow {
                        kind: VirtualKind::System,
                        rd: 0,
                        rs1: 0,
                        rs2: 0,
                        funct3: 0,
                        imm: 0,
                        raw,
                    },
                })
            }
        }
    }
    let len = rows.len();
    ExpandedProgram {
        rows,
        inst_to_micro,
        len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_counts() {
        // ADDI: 1 virtual; SLL (register): 6 stages; JAL: 1.
        let addi = crate::decode::decode(0x00100093); // addi x1, x0, 1
        let sll = crate::decode::decode((2 << 20) | (1 << 15) | (1 << 12) | (3 << 7) | 0x33);
        let jal = crate::decode::decode(0x008000ef); // jal x1, +8
        let prog = expand_program(&[addi.raw, sll.raw, jal.raw]);
        assert_eq!(prog.len, 1 + SHIFT_STAGES + 1);
        assert_eq!(prog.inst_to_micro, vec![0, 1, 7]);
    }
}
