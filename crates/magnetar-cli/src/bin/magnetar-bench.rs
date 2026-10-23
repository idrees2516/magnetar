//! The end-to-end benchmark harness: the full Construction-3 protocol
//! (and the tree-IVC aggregation) at increasing trace lengths, with the
//! paper-comparable metrics — proving throughput (cycles/s), proof
//! size, and verifier time.
//!
//! Run: `cargo run --release -p magnetar-cli --bin magnetar-bench [-- T_MAX]`
//! (T_MAX caps the log2 trace length; default 12).

use magnetar_pcs::PcsParams;
use magnetar_zkvm::{keygen_with, proof_stats, prove, verify, PublicInput};
use std::time::Instant;

fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
    ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
}

fn enc_s(imm: u32, rs2: u32, rs1: u32, funct3: u32, op: u32) -> u32 {
    (((imm >> 5) & 0x7f) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm & 0x1f) << 7)
        | op
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

/// ALU chain in a countdown loop: the core arithmetization at a small
/// ROM footprint.
fn alu_program() -> Vec<u32> {
    vec![
        enc_i(32, 0, 0, 1, 0x13),                // addi x1, x0, 32   (pc 0)
        enc_i(-1i32 as u32, 1, 0, 1, 0x13),      // addi x1, x1, -1   (pc 1)
        enc_i(1, 0, 0, 2, 0x13),                 // addi x2, x0, 1    (pc 2)
        (2 << 20) | (1 << 15) | (3 << 7) | 0x33, // add x3, x1, x2 (pc 3)
        (3 << 20) | (1 << 15) | (4 << 7) | 0x33, // add x4, x1, x3 (pc 4)
        (4 << 20) | (1 << 15) | (5 << 7) | 0x33, // add x5, x1, x4 (pc 5)
        enc_b(-20, 0, 1, 1),                     // bne x1, x0, loop  (pc 6)
        enc_i(0, 5, 0, 10, 0x13),                // addi x10, x5, 0   (pc 7)
        0x0000_0073,                             // ecall             (pc 8)
    ]
}

/// Memory mix in a loop: a sub-word store then a load each iteration.
fn mem_program() -> Vec<u32> {
    vec![
        enc_i(32, 0, 0, 1, 0x13), // addi x1, x0, 32   (pc 0)
        enc_i(1, 0, 0, 4, 0x13),  // addi x4, x0, 1    (pc 1)
        enc_s(8, 4, 1, 0, 0x23),  // sb x4, 8(x1)      (pc 2)
        (8 << 20) | (1 << 15) | (3 << 12) | (5 << 7) | 0x03, // ld x5, 8(x1) (pc 3)
        enc_i(-1i32 as u32, 1, 0, 1, 0x13), // addi x1, x1, -1   (pc 4)
        enc_b(-16, 0, 1, 1),      // bne x1, x0, loop  (pc 5)
        enc_i(0, 5, 0, 10, 0x13), // addi x10, x5, 0   (pc 6)
        0x0000_0073,              // ecall             (pc 7)
    ]
}

/// Branchy: a tight taken/not-taken loop.
fn branch_program() -> Vec<u32> {
    vec![
        enc_i(32, 0, 0, 1, 0x13),           // addi x1, x0, 32  (pc 0)
        enc_i(1, 0, 0, 2, 0x13),            // addi x2, x0, 1   (pc 1)
        enc_i(-1i32 as u32, 1, 0, 1, 0x13), // addi x1, x1, -1  (pc 2)
        enc_b(-4, 0, 1, 1),                 // bne x1, x0, loop (pc 3)
        enc_i(0, 1, 0, 10, 0x13),           // addi x10, x1, 0  (pc 4)
        0x0000_0073,                        // ecall            (pc 5)
    ]
}

/// Shift mix in a loop (the router path).
fn shift_program() -> Vec<u32> {
    vec![
        enc_i(0x0def, 0, 0, 1, 0x13), // addi x1, x0, 0xdef (pc 0)
        enc_i(32, 0, 0, 3, 0x13),     // addi x3, x0, 32    (pc 1)
        (1 << 20) | (1 << 15) | (1 << 12) | (2 << 7) | 0x13, // slli x2, x1, 1 (pc 2)
        (2 << 20) | (2 << 15) | (5 << 12) | (1 << 7) | 0x13, // srli x1, x2, 2 (pc 3)
        enc_i(-1i32 as u32, 3, 0, 3, 0x13), // addi x3, x3, -1    (pc 4)
        enc_b(-12, 0, 3, 1),          // bne x3, x0, loop   (pc 5)
        enc_i(0, 1, 0, 10, 0x13),     // addi x10, x1, 0    (pc 6)
        0x0000_0073,                  // ecall              (pc 7)
    ]
}

struct Row {
    workload: &'static str,
    log2_t: usize,
    prove_ms: f64,
    verify_ms: f64,
    cycles_per_s: f64,
    proof_bytes: usize,
}

fn bench_one(workload: &'static str, program: &[u32], log2_t: usize) -> Row {
    let t = 1usize << log2_t;
    let (pk, vk) = keygen_with(program.to_vec(), t, 2, PcsParams::default());
    // Execute once to learn the output.
    let trace = magnetar_cpu::interp::execute(&pk.program, &[], t);
    let out = trace.final_state.output;
    let pub_in = PublicInput {
        input: vec![],
        output: out,
    };
    let t0 = Instant::now();
    let proof = prove(&pk, &pub_in);
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let ok = verify(&vk, &pub_in, &proof);
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
    assert!(ok.is_ok(), "{workload} at T={t} must verify");
    let stats = proof_stats(&pk, &proof);
    Row {
        workload,
        log2_t,
        prove_ms,
        verify_ms,
        cycles_per_s: t as f64 / (prove_ms / 1e3),
        proof_bytes: stats.bytes_total,
    }
}

fn main() {
    let max_log2: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    println!("Magnetar reference implementation — end-to-end benchmarks");
    println!(
        "(single-threaded Rust, test-scale parameters: RAM 2^8 words, AMOUNT_BITS=2, W=2048)\n"
    );
    let mut rows = Vec::new();
    for (name, prog) in [
        ("alu-chain", alu_program()),
        ("mem-mix", mem_program()),
        ("branchy", branch_program()),
        ("shift-mix", shift_program()),
    ] {
        for log2_t in [8usize, 10] {
            rows.push(bench_one(name, &prog, log2_t));
        }
    }
    // T = 2^11 exceeds the reference prover's memory envelope (the
    // dense joint materializations of the Twist/Shout provers); the
    // paper's streaming provers are the production path (docs/STATUS.md).

    println!(
        "{:<12} {:>7} {:>12} {:>12} {:>14} {:>12}",
        "workload", "T", "prove (ms)", "verify (ms)", "cycles/s", "proof (KiB)"
    );
    for r in &rows {
        println!(
            "{:<12} {:>7} {:>12.1} {:>12.1} {:>14.0} {:>12.1}",
            r.workload,
            format!("2^{}", r.log2_t),
            r.prove_ms,
            r.verify_ms,
            r.cycles_per_s,
            r.proof_bytes as f64 / 1024.0
        );
    }

    // The tree-IVC aggregation at T = 2^12, 4 segments: one shared
    // commitment and one PCS tail.
    if max_log2 >= 10 {
        println!("\nTree-IVC (T = 2^10 total, 4 segments of 2^8 sharing one commitment):");
        let prog = alu_program();
        let trace = magnetar_cpu::interp::execute(&prog, &[], 1024);
        let out = trace.final_state.output;
        let t0 = Instant::now();
        let ivc_proof = magnetar_ivc::prove_ivc(&prog, &[], out, 1024, 4, &PcsParams::default());
        let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let ok = magnetar_ivc::verify_ivc(&prog, &[], out, 1024, &ivc_proof, &PcsParams::default());
        let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
        assert!(ok.is_ok());
        // The aggregate proof body: the 4 inner arguments + ONE PCS tail.
        let body_bytes: usize = ivc_proof
            .inners
            .iter()
            .map(|inner| {
                let mut n = 0usize;
                let twist_n = |t: &magnetar_memory::TwistProof| -> usize {
                    let mut n = 0;
                    for r in &t.read_runs {
                        n += r.rounds.iter().map(|m| m.len()).sum::<usize>() + 3;
                    }
                    n += t.write_run.rounds.iter().map(|m| m.len()).sum::<usize>() + 3;
                    n
                };
                n += twist_n(&inner.reg_twist) + twist_n(&inner.ram_twist);
                n += inner
                    .reg_psi
                    .iter()
                    .map(|p| p.rounds.iter().map(|m| m.len()).sum::<usize>())
                    .sum::<usize>();
                n += inner
                    .ram_psi
                    .iter()
                    .map(|p| p.rounds.iter().map(|m| m.len()).sum::<usize>())
                    .sum::<usize>();
                n += inner
                    .shift_router
                    .rounds
                    .iter()
                    .map(|m| m.len())
                    .sum::<usize>();
                n += inner
                    .store_router
                    .rounds
                    .iter()
                    .map(|m| m.len())
                    .sum::<usize>();
                n += inner
                    .bus_shout
                    .phi_rounds
                    .iter()
                    .map(|m| m.len())
                    .sum::<usize>();
                n += inner
                    .cross_step
                    .s1_rounds
                    .iter()
                    .map(|m| m.len())
                    .sum::<usize>();
                n += inner
                    .cross_step
                    .s2_rounds
                    .iter()
                    .map(|m| m.len())
                    .sum::<usize>();
                n * 16
            })
            .sum::<usize>();
        let tail_bytes = ivc_proof
            .mosaic
            .batch_rounds
            .iter()
            .map(|m| m.len())
            .sum::<usize>()
            * 16;
        println!(
            "  prove {prove_ms:.0} ms ({:.0} cycles/s), verify {verify_ms:.1} ms",
            1024.0 / (prove_ms / 1e3)
        );
        println!(
            "  proof body {body_bytes} B + ONE shared PCS tail {tail_bytes} B — vs 4 separate full proofs' 4 tails"
        );
    }
}
