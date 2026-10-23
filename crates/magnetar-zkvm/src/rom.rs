//! The bytecode ROM's derived tables (paper §5.4/§5.5): the instruction
//! bus bits and the one-hot routing-selector columns the BinaryShout
//! selector checks authenticate.

use magnetar_cpu::expand::{VirtualInst, VirtualKind};
use magnetar_field::{Gf128, ONE, ZERO};

/// The ROM's public tables.
#[derive(Clone, Debug)]
pub struct RomTables {
    /// The ROM address width (rows padded to `2^a_rom`).
    pub a_rom: usize,
    /// The instruction bus bits: `bus[o | (k << 6)]` = bit `o` of row
    /// `k`'s bus `op‖rd‖rs1‖rs2‖funct3‖imm`.
    pub bus: Vec<Gf128>,
    /// The cross-step transition table `TT(k, k', t, h)` (Remark 4):
    /// 1 exactly when the next fetch address is `k'` given the current
    /// fetch `k`, the branch-taken flag `t`, and the halt latch `h` —
    /// `h = 1` freezes (`k' = k`); `h = 0` follows the branch direction.
    /// Indexed `k | k'<<a | t<<2a | h<<(2a+1)`.
    pub tt: Vec<Gf128>,
    /// Whether the program contains `JALR` (rejected at keygen).
    pub has_jalr: bool,
    /// The shift amount one-hot columns: `sel_lo[h | (k << amount_bits)]`
    /// (zero on non-shift rows).
    pub sel_lo: Vec<Gf128>,
    /// The shift kind one-hot columns: `sel_hi[h | (k << 2)]`.
    pub sel_hi: Vec<Gf128>,
    /// The store width one-hot columns: `sel_st[h | (k << 2)]`.
    pub sel_st: Vec<Gf128>,
}

impl RomTables {
    /// Build from the expanded program with a shift-amount width.
    pub fn build(rows: &[VirtualInst], amount_bits: usize) -> Self {
        let a_rom = rows.len().next_power_of_two().max(1).trailing_zeros() as usize;
        let row_count = 1usize << a_rom;
        let mut bus = vec![ZERO; (1usize << 6) * row_count];
        let mut sel_lo = vec![ZERO; (1usize << amount_bits) * row_count];
        let mut sel_hi = vec![ZERO; 4 * row_count];
        let mut sel_st = vec![ZERO; 4 * row_count];
        // The successor tables N₀/N₁ per row (the branch directions).
        let len = rows.len();
        let mut next0 = vec![0usize; row_count];
        let mut next1 = vec![0usize; row_count];
        let mut has_jalr = false;
        for (k, inst) in rows.iter().enumerate() {
            let row = &inst.row;
            match row.kind {
                VirtualKind::System => {
                    // The halt freeze.
                    next0[k] = k;
                    next1[k] = k;
                }
                VirtualKind::Branch => {
                    next0[k] = (k + 1) % len;
                    let target = (k as i64 + (row.imm as i64) / 4).max(0) as usize;
                    next1[k] = target % len;
                }
                VirtualKind::Jump if (row.raw & 0x7f) == 0x67 => {
                    // JALR: the target is data-dependent — unsupported.
                    has_jalr = true;
                    next0[k] = (k + 1) % len;
                    next1[k] = (k + 1) % len;
                }
                VirtualKind::Jump => {
                    let target = (k as i64 + (row.imm as i64) / 4).max(0) as usize;
                    next0[k] = target % len;
                    next1[k] = target % len;
                }
                _ => {
                    next0[k] = (k + 1) % len;
                    next1[k] = (k + 1) % len;
                }
            }
        }
        for (k, inst) in rows.iter().enumerate() {
            let row = &inst.row;
            // The instruction bus: op(7) | rd(6) | rs1(6) | rs2(6) |
            // funct3(3) | imm(13), zero-padded to 64.
            let mut bus_word: u64 = 0;
            bus_word |= (row.raw & 0x7f) as u64;
            let rd6 = if row.rd == 0 { 32 } else { row.rd };
            bus_word |= ((rd6 as u64) & 0x3f) << 7;
            bus_word |= ((row.rs1 as u64) & 0x3f) << 13;
            bus_word |= ((row.rs2 as u64) & 0x3f) << 19;
            bus_word |= ((row.funct3 as u64) & 0x7) << 25;
            bus_word |= (row.imm & 0x1fff) << 28;
            for o in 0..64usize {
                bus[o | (k << 6)] = Gf128(((bus_word >> o) & 1) as u128);
            }
            // The routing selectors.
            match row.kind {
                VirtualKind::Alu if matches!(row.funct3, 1 | 5) => {
                    // An immediate shift: a static routing.
                    let amount = (row.imm & 0x3f) as usize;
                    let amount = amount.min((1usize << amount_bits) - 1);
                    let kind = if row.funct3 == 1 {
                        0
                    } else if (row.raw >> 25) == 0x20 {
                        2
                    } else {
                        1
                    };
                    if amount < (1usize << amount_bits) {
                        sel_lo[amount | (k << amount_bits)] = ONE;
                    }
                    sel_hi[kind | (k << 2)] = ONE;
                }
                VirtualKind::Store => {
                    let h = match row.funct3 {
                        0 => 0, // SB
                        1 => 1, // SH
                        2 => 2, // SW
                        _ => 3, // SD
                    };
                    sel_st[h | (k << 2)] = ONE;
                }
                _ => {}
            }
        }
        // The flat transition table.
        let a = a_rom;
        let mut tt = vec![ZERO; 1usize << (2 * a + 2)];
        for k in 0..row_count {
            for t in 0..2usize {
                for h in 0..2usize {
                    let target = if h == 1 {
                        k
                    } else if t == 1 {
                        next1[k]
                    } else {
                        next0[k]
                    };
                    tt[k | (target << a) | (t << (2 * a)) | (h << (2 * a + 1))] = ONE;
                }
            }
        }
        RomTables {
            a_rom,
            tt,
            has_jalr,
            bus,
            sel_lo,
            sel_hi,
            sel_st,
        }
    }
}
