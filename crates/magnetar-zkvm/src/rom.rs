//! The bytecode ROM's derived tables (paper §5.4/§5.5): the instruction
//! bus bits and the one-hot routing-selector columns the BinaryShout
//! selector checks authenticate.

use magnetar_cpu::expand::{row_class, VirtualInst, VirtualKind};
use magnetar_field::{Gf128, ONE, ZERO};

/// The (kind × stage) one-hot selector layout for the stage routing:
/// `h_ks = kind | (stage << 2)` — 4 kinds (SLL/SRL/SRA/pad) × 8 stages
/// = 32 entries per row (5 one-hot vars).
pub const SEL_KS_BITS: usize = 5;

/// The stage routing's kind for a register-shift row.
fn stage_kind(row: &magnetar_cpu::expand::RomRow) -> usize {
    if row.funct3 == 1 {
        0 // SLL
    } else if (row.raw >> 25) == 0x20 {
        2 // SRA
    } else {
        1 // SRL
    }
}

/// The ROM's public tables.
#[derive(Clone, Debug)]
pub struct RomTables {
    /// The ROM address width (rows padded to `2^a_rom`).
    pub a_rom: usize,
    /// The instruction bus bits: `bus[o | (k << 6)]` = bit `o` of row
    /// `k`'s bus `op‖rd‖rs1‖rs2‖funct3‖imm`.
    pub bus: Vec<Gf128>,
    /// The shift amount one-hot columns: `sel_lo[h | (k << amount_bits)]`
    /// (zero on non-shift rows).
    pub sel_lo: Vec<Gf128>,
    /// The shift kind one-hot columns: `sel_hi[h | (k << 2)]`.
    pub sel_hi: Vec<Gf128>,
    /// The store width one-hot columns: `sel_st[h | (k << 2)]`.
    pub sel_st: Vec<Gf128>,
    /// The decode-class one-hot columns: `class[c | (k << 5)]` — the
    /// class-selector check binds the committed per-cycle class to it.
    pub class: Vec<Gf128>,
    /// The (kind × stage) one-hot for the register-shift stage routing:
    /// `sel_ks[h_ks | (k << 5)]` with `h_ks = kind | (stage << 2)`
    /// (zero on non-stage rows).
    pub sel_ks: Vec<Gf128>,
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
        let mut class = vec![ZERO; (1usize << SEL_KS_BITS) * row_count];
        let mut sel_ks = vec![ZERO; (1usize << SEL_KS_BITS) * row_count];
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
                VirtualKind::ShiftStage { stage } => {
                    // The stage routing's (kind, stage) one-hot.
                    let h_ks = stage_kind(row) | (stage << 2);
                    sel_ks[h_ks | (k << SEL_KS_BITS)] = ONE;
                }
                _ => {}
            }
            // The decode class.
            class[row_class(row) | (k << SEL_KS_BITS)] = ONE;
        }
        RomTables {
            a_rom,
            bus,
            sel_lo,
            sel_hi,
            sel_st,
            class,
            sel_ks,
        }
    }
}
