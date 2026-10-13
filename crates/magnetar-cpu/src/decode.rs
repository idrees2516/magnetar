//! RV64I decoder — the complete base integer instruction set.

/// Major opcodes (bits 0..6, little-endian per the RISC-V spec).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    Lui,
    Auipc,
    Jal,
    Jalr,
    Branch,
    Load,
    Store,
    OpImm,
    Op,
    OpImm32,
    Op32,
    Fence,
    Ecall,
    Ebreak,
    Invalid,
}

/// A decoded instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inst {
    pub opcode: Opcode,
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    /// The three funct bits fields: (funct3, funct7_lo, funct7_hi as flags).
    pub funct3: u32,
    pub funct7: u32,
    /// Immediates as decoded per format.
    pub imm: i64,
    /// The raw encoding (kept for the circuit's instruction-word witness).
    pub raw: u32,
}

#[inline]
fn sext(val: u32, bits: u32) -> i64 {
    let shift = 32 - bits;
    ((val << shift) as i32 >> shift) as i64
}

/// Decode one 32-bit instruction word (RV64I base).
pub fn decode(raw: u32) -> Inst {
    let opcode = raw & 0x7f;
    let rd = ((raw >> 7) & 0x1f) as usize;
    let funct3 = (raw >> 12) & 0x7;
    let rs1 = ((raw >> 15) & 0x1f) as usize;
    let rs2 = ((raw >> 20) & 0x1f) as usize;
    let funct7 = raw >> 25;
    // Immediates.
    let i_imm = sext(((raw >> 20) & 0xfff), 12);
    let s_imm = sext((((raw >> 25) & 0x7f) << 5) | ((raw >> 7) & 0x1f), 12);
    let b_imm = sext(
        (((raw >> 31) & 1) << 12)
            | (((raw >> 7) & 1) << 11)
            | (((raw >> 25) & 0x3f) << 5)
            | (((raw >> 8) & 0xf) << 1),
        13,
    );
    let u_imm = ((raw & 0xffff_f000) as u32 as i64) as i64;
    let j_imm = sext(
        (((raw >> 31) & 1) << 20)
            | (((raw >> 12) & 0xff) << 12)
            | (((raw >> 20) & 1) << 11)
            | (((raw >> 21) & 0x3ff) << 1),
        21,
    );
    let op = match opcode {
        0x37 => Opcode::Lui,
        0x17 => Opcode::Auipc,
        0x6f => Opcode::Jal,
        0x67 => Opcode::Jalr,
        0x63 => Opcode::Branch,
        0x03 => Opcode::Load,
        0x23 => Opcode::Store,
        0x13 => Opcode::OpImm,
        0x33 => Opcode::Op,
        0x1b => Opcode::OpImm32,
        0x3b => Opcode::Op32,
        0x0f => Opcode::Fence,
        0x73 => {
            if raw == 0x0000_0073 {
                Opcode::Ecall
            } else if raw == 0x0010_0073 {
                Opcode::Ebreak
            } else {
                Opcode::Invalid
            }
        }
        _ => Opcode::Invalid,
    };
    let imm = match op {
        Opcode::Lui | Opcode::Auipc => u_imm,
        Opcode::Jal => j_imm,
        Opcode::Jalr => i_imm,
        Opcode::Branch => b_imm,
        Opcode::Load | Opcode::OpImm | Opcode::OpImm32 => i_imm,
        Opcode::Store => s_imm,
        _ => 0,
    };
    Inst {
        opcode: op,
        rd,
        rs1,
        rs2,
        funct3,
        funct7,
        imm,
        raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_r(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
    }
    fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
        ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
    }

    #[test]
    fn decodes_r_type() {
        // ADD x3, x1, x2
        let raw = enc_r(0x00, 2, 1, 0, 3, 0x33);
        let i = decode(raw);
        assert_eq!(i.opcode, Opcode::Op);
        assert_eq!((i.rd, i.rs1, i.rs2), (3, 1, 2));
        assert_eq!((i.funct3, i.funct7), (0, 0));
        // SUB x3, x1, x2
        let raw = enc_r(0x20, 2, 1, 0, 3, 0x33);
        assert_eq!(decode(raw).funct7, 0x20);
    }

    #[test]
    fn decodes_i_type_with_sign_extension() {
        // ADDI x1, x0, -1
        let raw = enc_i(0xfff, 0, 0, 1, 0x13);
        let i = decode(raw);
        assert_eq!(i.opcode, Opcode::OpImm);
        assert_eq!(i.imm, -1);
        // ADDI x1, x0, 2047
        let raw = enc_i(0x7ff, 0, 0, 1, 0x13);
        assert_eq!(decode(raw).imm, 2047);
    }

    #[test]
    fn decodes_branch_and_jump_immediates() {
        // BEQ x1, x2, +8: imm[12]=0, imm[10:5]=0, imm[4:1]=0100 (bits
        // 11:8), imm[11]=0.
        let raw = (0b0 << 31)
            | (0b000000 << 25)
            | (2 << 20)
            | (1 << 15)
            | (0 << 12)
            | (0b0100 << 8)
            | (0 << 7)
            | 0x63;
        assert_eq!(decode(raw).imm, 8);
        // JAL x1, +16: imm[20]=0, imm[10:1]=0000001000 (8), imm[11]=0,
        // imm[19:12]=0.
        let raw = (0b0 << 31) | (8u32 << 21) | (0b0 << 20) | (0 << 12) | (0b0 << 7) | 0x6f;
        assert_eq!(decode(raw).imm, 16);
    }

    #[test]
    fn decodes_u_and_s() {
        // LUI x1, 0xDEADB
        let raw = 0xDEADB000 | (1 << 7) | 0x37;
        let i = decode(raw);
        assert_eq!(i.opcode, Opcode::Lui);
        assert_eq!(i.imm, 0xDEADB000u32 as i64);
        // SW x1, 12(x2)
        let raw = ((12u32 & 0x1f) << 7)
            | (2 << 15)
            | (2 << 12)
            | (((12 >> 5) as u32 & 0x7f) << 25)
            | 0x23;
        let i = decode(raw);
        assert_eq!(i.opcode, Opcode::Store);
        assert_eq!(i.imm, 12);
        // SB x1, -4(x2)
        let raw = ((0x1cu32) << 7) | (2 << 15) | (0 << 12) | (0x7f << 25) | 0x23;
        assert_eq!(decode(raw).imm, -4);
    }

    #[test]
    fn system() {
        assert_eq!(decode(0x0000_0073).opcode, Opcode::Ecall);
        assert_eq!(decode(0x0010_0073).opcode, Opcode::Ebreak);
    }
}
