//! **The pooled ERC-20 load driver** — one machine's share of a
//! pool-proven workload: `N` independent batches of `2^LOG2_BATCH`
//! cycles each, proved and verified in-process back-to-back. The
//! aggregate pool run splits a large total (2^23, 2^24 cycles…) across
//! machines; this binary times one machine's share and reports the
//! per-batch rates, the machine's aggregate rate, and the total
//! verified transfer count — the pooled architecture's demonstration
//! shape (the single-machine envelope is bounded by the combined
//! commitment + the lockstep discharge; the pool scales the *total*
//! proven workload linearly in machines).
//!
//! Run:
//!   `magnetar-bench-erc20-pool <LOG2_BATCH> <BATCHES>`   (e.g. `13 13`)

use magnetar_pcs::PcsParams;
use magnetar_zkvm::{keygen_with, prove, verify, PublicInput};
use std::time::Instant;

fn cpi_from_env() -> usize {
    std::env::var("MAGNETAR_CPI")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI)
}

fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
    ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
}

fn enc_lui(imm: u32, rd: u32) -> u32 {
    (imm << 12) | (rd << 7) | 0x37
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

fn enc_sub(a: u32, b: u32, d: u32) -> u32 {
    (0x20 << 25) | (a << 20) | (b << 15) | (d << 7) | 0x33
}

fn enc_add(a: u32, b: u32, d: u32) -> u32 {
    (a << 20) | (b << 15) | (d << 7) | 0x33
}

/// The paper's `erc20-mixed-block` guest — the exact program the
/// single-run bench proves (12 cycles per executed transfer, every 16th
/// failing the underflow check).
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
    let mut args = std::env::args().skip(1);
    let log2_batch: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(13);
    let batches: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let cpi = cpi_from_env();
    let total = 1usize << log2_batch;
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
    println!(
        "pooled ERC-20 driver: {batches} batches of 2^{log2_batch} cycles \
         ({n_tx} tx each, cpi={cpi})"
    );
    println!(
        "{:>7} {:>10} {:>12} {:>12} {:>11}",
        "batch", "prove(ms)", "verify(ms)", "cycles/s", "tx/s"
    );
    let t_all = Instant::now();
    let mut proved_tx = 0usize;
    for b in 0..batches {
        let t0 = Instant::now();
        let proof = prove(&pk, &pub_in);
        let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let ok = verify(&vk, &pub_in, &proof);
        let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
        assert!(ok.is_ok(), "pool batch {b} must verify");
        proved_tx += n_tx;
        println!(
            "{:>7} {:>10.0} {:>12.1} {:>12.0} {:>11.0}",
            b,
            prove_ms,
            verify_ms,
            total as f64 / (prove_ms / 1e3),
            n_tx as f64 / (prove_ms / 1e3),
        );
    }
    let wall = t_all.elapsed().as_secs_f64();
    println!(
        "\nshare total: {batches} batches, {proved_tx} transfers PROVED AND VERIFIED \
         in {wall:.2}s — machine rate {:.0} tx/s, {:.0} cycles/s",
        proved_tx as f64 / wall,
        (batches * total) as f64 / wall
    );
}
