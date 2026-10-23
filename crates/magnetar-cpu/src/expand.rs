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
                    // the amount (rs2's value) is set. The chain runs
                    // through the scratch CELL 32 (the redirected x0
                    // write slot): stage 0 reads rs1, later stages read
                    // the scratch; stages 0-4 write the scratch and the
                    // last stage writes rd — all through the register
                    // file (the BinaryTwist covers all 64 cells).
                    for stage in 0..SHIFT_STAGES {
                        let (rs1, rd) = if stage == 0 {
                            (inst.rs1, 32)
                        } else if stage == SHIFT_STAGES - 1 {
                            (32, inst.rd)
                        } else {
                            (32, 32)
                        };
                        rows.push(VirtualInst {
                            row: RomRow {
                                kind: VirtualKind::ShiftStage { stage },
                                rd,
                                rs1,
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
            Opcode::Lui | Opcode::Auipc => {
                // AUIPC pre-adjusts the immediate by 4·(own row index) so the
                // adder (t1 = cell 0 = 0, t2 = the bus imm) validates the
                // value — the row's rs1 is forced to 0 for the same reason.
                let k = rows.len();
                let imm = if matches!(inst.opcode, Opcode::Auipc) {
                    (inst.imm as Word).wrapping_add((k as Word) * 4)
                } else {
                    inst.imm as Word
                };
                rows.push(VirtualInst {
                    row: RomRow {
                        kind: VirtualKind::Upper,
                        rd: inst.rd,
                        rs1: 0,
                        rs2: 0,
                        funct3: 0,
                        imm,
                        raw,
                    },
                })
            }
            Opcode::Jal | Opcode::Jalr => {
                // Two rows: the link write (an Upper-style row with imm =
                // 4·k + 4 so the adder validates it), then the pc jump (the
                // class-None row; the target semantics stay the paper's
                // Remark-4 omission).
                let k = rows.len();
                let link = (k as Word) * 4 + 4;
                rows.push(VirtualInst {
                    row: RomRow {
                        kind: VirtualKind::Upper,
                        rd: inst.rd,
                        rs1: 0,
                        rs2: 0,
                        funct3: 0,
                        imm: link,
                        raw,
                    },
                });
                rows.push(VirtualInst {
                    row: RomRow {
                        kind: VirtualKind::Jump,
                        rd: 0,
                        rs1: inst.rs1,
                        rs2: 0,
                        funct3: 0,
                        imm: inst.imm as Word,
                        raw,
                    },
                });
            }
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

/// The decode classes (the class one-hot's slots). The class of a ROM row
/// is its *constraint pattern*: which flag-fold coefficients fire and which
/// datapath rows apply. The class-selector check binds the committed
/// per-cycle class one-hot to the ROM row's class.
///
/// Class index space (32 slots, 23 used):
/// 0 AddReg, 1 AddImm, 2 SubSltReg, 3 SubSltImm, 4 XorReg, 5 XorImm,
/// 6 OrReg, 7 OrImm, 8 AndReg, 9 AndImm, 10 ShiftImm, 11 Stage,
/// 12 LB, 13 LBU, 14 LH, 15 LHU, 16 LW, 17 LWU, 18 LD, 19 Store,
/// 20 Branch, 21 System, 22 None (no datapath effect), 23-31 zero.
pub mod class {
    pub const ADD_REG: usize = 0;
    pub const ADD_IMM: usize = 1;
    pub const SUBSLT_REG: usize = 2;
    pub const SUBSLT_IMM: usize = 3;
    pub const XOR_REG: usize = 4;
    pub const XOR_IMM: usize = 5;
    pub const OR_REG: usize = 6;
    pub const OR_IMM: usize = 7;
    pub const AND_REG: usize = 8;
    pub const AND_IMM: usize = 9;
    pub const SHIFT_IMM: usize = 10;
    pub const STAGE: usize = 11;
    pub const LB: usize = 12;
    pub const LBU: usize = 13;
    pub const LH: usize = 14;
    pub const LHU: usize = 15;
    pub const LW: usize = 16;
    pub const LWU: usize = 17;
    pub const LD: usize = 18;
    pub const STORE: usize = 19;
    pub const BRANCH: usize = 20;
    pub const SYSTEM: usize = 21;
    pub const NONE: usize = 22;
    pub const COUNT: usize = 32;
}

/// The row's decode class (see [`class`]).
pub fn row_class(row: &RomRow) -> usize {
    use crate::decode::Opcode;
    let op = crate::decode::decode(row.raw).opcode;
    let is_reg = op == Opcode::Op;
    match row.kind {
        VirtualKind::Alu => match row.funct3 {
            0 => {
                if is_reg && (row.raw >> 25) == 0x20 {
                    class::SUBSLT_REG
                } else if is_reg {
                    class::ADD_REG
                } else {
                    class::ADD_IMM
                }
            }
            2 | 3 => {
                if is_reg {
                    class::SUBSLT_REG
                } else {
                    class::SUBSLT_IMM
                }
            }
            4 => {
                if is_reg {
                    class::XOR_REG
                } else {
                    class::XOR_IMM
                }
            }
            6 => {
                if is_reg {
                    class::OR_REG
                } else {
                    class::OR_IMM
                }
            }
            7 => {
                if is_reg {
                    class::AND_REG
                } else {
                    class::AND_IMM
                }
            }
            1 | 5 => class::SHIFT_IMM,
            _ => class::NONE,
        },
        VirtualKind::ShiftStage { .. } => class::STAGE,
        VirtualKind::Load => match row.funct3 {
            0 => class::LB,
            4 => class::LBU,
            1 => class::LH,
            5 => class::LHU,
            2 => class::LW,
            6 => class::LWU,
            _ => class::LD,
        },
        VirtualKind::Store => class::STORE,
        VirtualKind::Branch => class::BRANCH,
        VirtualKind::Upper => class::ADD_IMM,
        VirtualKind::Jump => class::NONE,
        VirtualKind::System => class::SYSTEM,
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
        // JAL expands into the link row + the pc-jump row.
        assert_eq!(prog.len, 1 + SHIFT_STAGES + 2);
        assert_eq!(prog.inst_to_micro, vec![0, 1, 7]);
    }
}
