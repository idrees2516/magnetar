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
    /// The shift amount one-hot columns: `sel_lo[h | (k << amount_bits)]`
    /// (zero on non-immediate-shift rows).
    pub sel_lo: Vec<Gf128>,
    /// The shift kind one-hot columns: `sel_hi[h | (k << 2)]` (live on
    /// immediate-shift **and** register-shift stage rows).
    pub sel_hi: Vec<Gf128>,
    /// The store width one-hot columns: `sel_st[h | (k << 2)]`.
    pub sel_st: Vec<Gf128>,
    /// The register-shift **stage one-hot** columns:
    /// `sel_stage[k' | (s << STAGE_BITS)]` — live on register-shift stage
    /// rows at the row's stage `s` (zero elsewhere). The stage-selector
    /// sum-check authenticates the routers' conditional amount selectors
    /// against this table through the fetch one-hot, exactly as the other
    /// selector checks do.
    pub sel_stage: Vec<Gf128>,
    /// The stage one-hot's width (8 slots; stages 0..=5 live).
    pub stage_bits: usize,
}

/// The register-shift stage count (must match `magnetar_cpu::SHIFT_STAGES`).
pub const STAGES: usize = 6;
/// The stage one-hot's index width (`2^STAGE_BITS = 8 ≥ STAGES` slots).
pub const STAGE_BITS: usize = 3;

impl RomTables {
    /// Build from the expanded program with a shift-amount width.
    pub fn build(rows: &[VirtualInst], amount_bits: usize) -> Self {
        let a_rom = rows.len().next_power_of_two().max(1).trailing_zeros() as usize;
        let row_count = 1usize << a_rom;
        let mut bus = vec![ZERO; (1usize << 6) * row_count];
        let mut sel_lo = vec![ZERO; (1usize << amount_bits) * row_count];
        let mut sel_hi = vec![ZERO; 4 * row_count];
        let mut sel_st = vec![ZERO; 4 * row_count];
        let mut sel_stage = vec![ZERO; (1usize << STAGE_BITS) * row_count];
        // The shift kind from the row's funct3/bit-30: 0 = SLL, 1 = SRL,
        // 2 = SRA.
        let kind_of = |row: &magnetar_cpu::expand::RomRow| -> usize {
            if row.funct3 == 1 {
                0
            } else if (row.raw >> 25) == 0x20 {
                2
            } else {
                1
            }
        };
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
                    let kind = kind_of(row);
                    if amount < (1usize << amount_bits) {
                        sel_lo[amount | (k << amount_bits)] = ONE;
                    }
                    sel_hi[kind | (k << 2)] = ONE;
                }
                VirtualKind::ShiftStage { stage } => {
                    // A register-shift stage: the **kind** is static (the
                    // ROM authenticates it like an immediate shift's);
                    // the **amount** is the conditional pair
                    // `{identity, 2^stage}` — the witness amount bit picks
                    // within it, and the stage one-hot below fixes which
                    // pair the row offers.
                    let kind = kind_of(row);
                    sel_hi[kind | (k << 2)] = ONE;
                    sel_stage[stage | (k << STAGE_BITS)] = ONE;
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
        RomTables {
            a_rom,
            bus,
            sel_lo,
            sel_hi,
            sel_st,
            sel_stage,
            stage_bits: STAGE_BITS,
        }
    }
}
