//! The **ERC-20 load test** — the paper's Table-6 workload shape
//! (`erc20-mixed-block`): balance transfers between two accounts, with
//! **every 16th transfer failing** the underflow check (a realistic mix
//! of loads, stores, checked subtraction, addition, and branches).
//!
//! The guest program (RV64I, one virtual cycle per instruction):
//!
//! ```text
//! x1 = N (the transfer count)   x2 = the amount
//! loop: x4 = ram[0]; x5 = ram[8]        // the two balances (64-bit)
//!       if (x1 & 15) == 15 → skip       // every 16th transfer fails
//!       if x4 < x2 (unsigned) → skip    // the ERC-20 underflow check
//!       x4 -= x2; x5 += x2
//!       ram[0] = x4; ram[8] = x5
//!       x1 -= 1; while x1 ≠ 0
//! halt with x10 = ram[0]
//! ```
//!
//! 15 cycles per executed transfer. The paper's `erc20-transfer` moves a
//! 256-bit amount (≈289 cycles/tx with its full instruction pipeline);
//! this implementation exercises the same statement shape —
//! load/checked-sub/store per transfer with 1-in-16 failures — at the
//! 64-bit balance width of this RV64I reference CPU.
//!
//! Run:
//!   `cargo run --release -p magnetar-cli --bin magnetar-bench-erc20 -- [LOG2_MAX]`

use magnetar_pcs::PcsParams;
use magnetar_zkvm::{keygen_with, proof_stats, prove, verify, PublicInput};
use std::time::Instant;

/// The instance-batching density knob (paper §5.1: cycles per SIMD
/// instance). `MAGNETAR_CPI=1|2|4|8…` sweeps it; the default is the
/// witness layout's production constant.
fn cpi_from_env() -> usize {
    std::env::var("MAGNETAR_CPI")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI)
}

fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
    ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
}

fn enc_s(imm: u32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
    (((imm >> 5) & 0x7f) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm & 0x1f) << 7)
        | 0x23
}

fn enc_b(imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
    let imm = imm as u32;
    (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3f) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xf) << 8)
        | (((imm >> 11) & 1) << 7)
        | 0x63
}

/// `sub rd, rs1, rs2`.
fn enc_sub(rs2: u32, rs1: u32, rd: u32) -> u32 {
    (0x20 << 25) | (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33
}

/// `add rd, rs1, rs2`.
fn enc_add(rs2: u32, rs1: u32, rd: u32) -> u32 {
    (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33
}

/// `lui rd, imm20`.
fn enc_lui(imm20: u32, rd: u32) -> u32 {
    ((imm20 & 0xfffff) << 12) | (rd << 7) | 0x37
}

/// The ERC-20 mixed-block transfer loop: `n` transfers between the two
/// balances at ram[0] and ram[1], every 16th failing the check. The
/// preamble loads `x1 = n` (LUI + ADDI when `n` exceeds the 12-bit
/// immediate); the branch offsets are computed from the actual pc
/// positions.
fn erc20_program(n: u32) -> Vec<u32> {
    let mut p: Vec<u32> = Vec::new();
    // The preamble: x1 = n (the transfer count).
    if n <= 2047 {
        p.push(enc_i(n, 0, 0, 1, 0x13));
    } else {
        let hi = ((n as i64 + 0x800) >> 12) as u32;
        let lo = n as i64 - ((hi as i64) << 12);
        p.push(enc_lui(hi, 1));
        p.push(enc_i(lo as u32, 1, 0, 1, 0x13));
    }
    p.push(enc_i(7, 0, 0, 2, 0x13)); // x2 = the transfer amount
    let loop_pc = p.len();
    p.push(enc_i(0, 0, 3, 4, 0x03)); // ld x4, 0(x0)
    p.push(enc_i(8, 0, 3, 5, 0x03)); // ld x5, 8(x0)
    p.push(enc_i(15, 1, 7, 6, 0x13)); // andi x6, x1, 15
    p.push(enc_i(15, 0, 0, 7, 0x13)); // addi x7, x0, 15
    let beq_pc = p.len();
    let skip_pc = beq_pc + 6; // the fail path joins after the two stores
    p.push(enc_b(((skip_pc - beq_pc) * 4) as i32, 7, 6, 0));
    let bltu_pc = p.len();
    p.push(enc_b(((skip_pc - bltu_pc) * 4) as i32, 2, 4, 6));
    p.push(enc_sub(2, 4, 4)); // x4 -= x2
    p.push(enc_add(2, 5, 5)); // x5 += x2
    p.push(enc_s(0, 4, 0, 3)); // sd x4, 0(x0)
    p.push(enc_s(8, 5, 0, 3)); // sd x5, 8(x0)
    debug_assert_eq!(p.len(), skip_pc);
    // skip:
    p.push(enc_i(-1i32 as u32, 1, 0, 1, 0x13)); // x1 -= 1
    let bne_pc = p.len();
    p.push(enc_b(
        ((loop_pc as i64 - bne_pc as i64) * 4) as i32,
        0,
        1,
        1,
    ));
    p.push(enc_i(0, 0, 3, 10, 0x03)); // ld x10, 0(x0)
    p.push(0x0000_0073); // ecall
    p
}

/// Cycles per executed transfer (the loop body).
const CYCLES_PER_TX: usize = 12;

fn main() {
    let max_log2: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let cpi = cpi_from_env();
    println!("ERC-20 mixed-block load test (the paper's Table-6 workload shape)");
    println!(
        "(64-bit balances, checked subtraction, every 16th transfer fails; {} cycles/tx, cpi={cpi})\n",
        CYCLES_PER_TX
    );
    // ---- The monolithic prover (the small scales). ----
    println!(
        "{:>7} {:>10} {:>10} {:>10} {:>12} {:>10} {:>12} {:>11}",
        "T", "tx", "prove(ms)", "verify(ms)", "cycles/s", "tx/s", "proof(KiB)", "RSS(MiB)"
    );
    for log2_t in [10usize, 11, 12].iter().filter(|t| **t <= max_log2) {
        let total = 1usize << log2_t;
        let n_tx = total / CYCLES_PER_TX;
        let prog = erc20_program(n_tx as u32);
        let input = vec![1_000_000u64, 0u64];
        let (pk, vk) = keygen_with(prog.clone(), total, 2, PcsParams::default(), cpi);
        let trace = magnetar_cpu::interp::execute(&pk.program, &input, total);
        let out = trace.final_state.output;
        // Sanity: successful transfers each move 7 from ram[0].
        let expect = 1_000_000u64 - 7 * (n_tx as u64 - n_tx as u64 / 16);
        assert_eq!(out, expect, "the from-balance after {n_tx} mixed transfers");
        let pub_in = PublicInput {
            input: input.clone(),
            output: out,
        };
        let t0 = Instant::now();
        let proof = prove(&pk, &pub_in);
        let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let ok = verify(&vk, &pub_in, &proof);
        let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
        assert!(ok.is_ok(), "erc20 at T=2^{log2_t} must verify");
        let stats = proof_stats(&pk, &proof);
        println!(
            "{:>7} {:>10} {:>10.0} {:>10.1} {:>12.0} {:>10.0} {:>12.1} {:>11.1}",
            format!("2^{log2_t}"),
            n_tx,
            prove_ms,
            verify_ms,
            total as f64 / (prove_ms / 1e3),
            n_tx as f64 / (prove_ms / 1e3),
            stats.bytes_total as f64 / 1024.0,
            magnetar_ivc::peak_rss_bytes() as f64 / 1048576.0
        );
    }
    // ---- The streaming prover (the large scales). ----
    println!("\nStreaming parallel prover (checkpoints -> rayon segments -> shared root):");
    println!(
        "{:>7} {:>4} {:>10} {:>10} {:>12} {:>10} {:>10} {:>9}",
        "T", "segs", "prove(ms)", "peakRSS", "cycles/s", "tx/s", "ckpt(ms)", "root(ms)"
    );
    for log2_t in [13usize, 14, 15, 16].iter().filter(|t| **t <= max_log2) {
        let total = 1usize << log2_t;
        let n_tx = total / CYCLES_PER_TX;
        let prog = erc20_program(n_tx as u32);
        let input = vec![1_000_000u64, 0u64];
        let trace = magnetar_cpu::interp::execute(&prog, &input, total);
        let out = trace.final_state.output;
        let expect = 1_000_000u64 - 7 * (n_tx as u64 - n_tx as u64 / 16);
        assert_eq!(out, expect, "the from-balance after {n_tx} mixed transfers");
        // Adaptive ladder: 2^10-cycle segments above 2^13 total.
        let segments = if total >= (1usize << 13) {
            total / (1usize << 10)
        } else {
            1
        }
        .clamp(1, 128)
        .next_power_of_two();
        let t0 = Instant::now();
        let (proof, stats) = magnetar_ivc::prove_ivc_streaming(
            &prog,
            &input,
            out,
            total,
            segments,
            &PcsParams::default(),
            cpi,
        );
        let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
        let ok = magnetar_ivc::verify_ivc(
            &prog,
            &input,
            out,
            total,
            &proof,
            &PcsParams::default(),
            cpi,
        );
        assert!(ok.is_ok(), "erc20 streaming at T=2^{log2_t} must verify");
        println!(
            "{:>7} {:>4} {:>10.0} {:>9.0}M {:>12.0} {:>10.0} {:>10.0} {:>9.0}",
            format!("2^{log2_t}"),
            segments,
            prove_ms,
            stats.peak_rss_bytes as f64 / 1048576.0,
            total as f64 / (prove_ms / 1e3),
            n_tx as f64 / (prove_ms / 1e3),
            stats.wall_checkpoints.as_secs_f64() * 1e3,
            stats.wall_root.as_secs_f64() * 1e3
        );
    }
}
