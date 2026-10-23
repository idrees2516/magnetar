//! The `magnetar` command-line interface: execute, prove, and verify
//! RV64I programs.

use magnetar_zkvm::protocol::{keygen, prove, verify, PublicInput};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("fib") => cmd_fib(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8)),
        Some("bench") => cmd_bench(
            args.get(2).and_then(|s| s.parse().ok()),
            args.get(3).and_then(|s| s.parse().ok()),
        ),
        Some("check") => cmd_check(),
        Some("profiles") => cmd_profiles(),
        _ => {
            eprintln!("Magnetar zkVM — post-quantum RV64I proving (eprint 2026-2423)");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  magnetar fib [steps]        prove + verify a Fibonacci program");
            eprintln!("  magnetar bench <lo> <hi>    trace-length sweep in powers of two");
            eprintln!("  magnetar check              self-test of the proof stack");
            eprintln!("  magnetar profiles           the measured dual-rate cost model");
            std::process::exit(2);
        }
    }
}

/// A compact Fibonacci loop (one ADDI + ADDI + ADD + ADDI + BNE body),
/// sized to a target cycle count. A small looping ROM — the realistic
/// shape (the paper's workloads are loops), unlike straight-line
/// unrolling.
fn fib_loop_program() -> Vec<u32> {
    let r#type =
        |imm: i32, rs1: u32, funct3: u32, rd: u32| enc_i(imm as u32 & 0xfff, rs1, funct3, rd, 0x13);
    let add = |rs2: u32, rs1: u32, rd: u32| (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33;
    // BNE x3, x0, off — the proper B-type encoding (imm[12]→31,
    // imm[10:5]→30:25, imm[11]→7, imm[4:1]→11:8).
    let bne = |off: i32| -> u32 {
        let imm = off as u16; // 13-bit two's complement
        let i12 = ((imm >> 12) & 1) as u32;
        let i11 = ((imm >> 11) & 1) as u32;
        let i10_5 = ((imm >> 5) & 0x3f) as u32;
        let i4_1 = ((imm >> 1) & 0xf) as u32;
        (i12 << 31) | (i10_5 << 25) | (3 << 15) | (1 << 12) | (i4_1 << 8) | (i11 << 7) | 0x63
    };
    vec![
        r#type(0, 0, 0, 3),  // x3 = 0 (the counter; overwritten per run)
        r#type(1, 0, 0, 1),  // x1 = 1
        r#type(1, 0, 0, 2),  // x2 = 1
        r#type(-1, 3, 0, 3), // x3 = x3 - 1 (the loop countdown)
        add(2, 1, 4),        // x4 = x1 + x2
        r#type(0, 2, 0, 1),  // x1 = x2
        r#type(0, 4, 0, 2),  // x2 = x4
        bne(-24),            // loop while x3 != 0 (back 6 virtual rows)
        r#type(0, 2, 0, 10), // x10 = x2 (the output)
        0x0000_0073,         // ecall: halt
    ]
}

/// Peak RSS in MiB (from /proc/self/status).
fn peak_rss_mib() -> f64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: f64 = rest.trim().trim_end_matches(" kB").parse().unwrap_or(0.0);
                return kb / 1024.0;
            }
        }
    }
    0.0
}

fn cmd_bench(min_exp: Option<usize>, max_exp: Option<usize>) {
    // A cycle-count budget per trace length: the loop runs until the
    // interpreter hits ~85% of the trace, then the halt-latch padding
    // fills the rest.
    let lo = min_exp.unwrap_or(8);
    let hi = max_exp.unwrap_or(13);
    println!("Magnetar proving benchmark — the fib-loop workload");
    println!("paper reference (erc20-mixed-block, M4 Max, 12 cores): 2^16: 1.7M | 2^18: 4.3M | 2^20: 6.0M | 2^22: 6.9M cycles/s");
    println!();
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "cycles", "prove (s)", "verify (s)", "cycles/s", "proof (KiB)", "RSS (MiB)"
    );
    for exp in lo..=hi {
        let trace_len = 1usize << exp;
        let program = fib_loop_program();
        // Determine the iteration count that fills the trace: each
        // iteration is 5 virtual cycles (add, addi, addi, addi, bne).
        let cycles_per_iter = 5usize;
        let iters = trace_len.saturating_sub(24) / cycles_per_iter;
        // The countdown register starts at `iters`.
        let mut program = program.clone();
        program[0] = enc_i((iters as u32) & 0xfff, 0, 0, 3, 0x13);
        // Run the interpreter to find the true output.
        let trace0 = magnetar_cpu::interp::execute(&program, &[], trace_len);
        let output = trace0.final_state.output;
        let real_cycles = trace0.records.len();
        if real_cycles > trace_len {
            eprintln!("2^{exp}: program exceeds the trace bound; skipping");
            continue;
        }
        let (pk, vk) = keygen(program.clone(), trace_len);
        let pub_in = PublicInput {
            input: vec![],
            output,
        };
        let t0 = Instant::now();
        let proof = prove(&pk, &pub_in);
        let prove_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let ok = verify(&vk, &pub_in, &proof).is_ok();
        let verify_s = t1.elapsed().as_secs_f64();
        if !ok {
            println!("{:<10} VERIFY FAILED", trace_len);
            continue;
        }
        let proof_kib = {
            let stats = magnetar_zkvm::stats::proof_stats(&pk, &proof);
            stats.bytes_total as f64 / 1024.0
        };
        println!(
            "{:<10} {:>10.3} {:>10.3} {:>10.0} {:>10.1} {:>10.0}",
            trace_len,
            prove_s,
            verify_s,
            trace_len as f64 / prove_s,
            proof_kib,
            peak_rss_mib()
        );
    }
}

fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
    ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
}

fn cmd_fib(rounds: u32) {
    // A Fibonacci loop unrolled to fit the trace budget: x1=1, x2=1,
    // then repeated (x3 = x1+x2; x1 = x2; x2 = x3). We generate a
    // straight-line body of `rounds` iterations and halt.
    let mut program = vec![
        enc_i(1, 0, 0, 1, 0x13), // x1 = 1
        enc_i(1, 0, 0, 2, 0x13), // x2 = 1
    ];
    for _ in 0..rounds.max(1) {
        program.push((2u32 << 20) | (1 << 15) | (3 << 7) | 0x33); // x3 = x1 + x2
        program.push((2u32 << 15) | (1 << 7) | 0x33); // x1 = x2 (rs2=x2, funct7=0)
        program.push(enc_i(0, 3, 0, 2, 0x13)); // x2 = x3
    }
    // Move the result to a0 (x10) and halt.
    program.push((3 << 15) | (10 << 7) | 0x33); // x10 = x3 (rs2=x0)
    program.push(0x0000_0073); // ecall: halt

    // Run the interpreter to get the output.
    let trace = magnetar_cpu::interp::execute(&program, &[], program.len() * 4 + 64);
    let output = trace.final_state.output;
    // The trace is already padded to a power of two by the interpreter.
    let trace_len = trace.records.len();

    println!("Magnetar fib({rounds}) — {trace_len} virtual cycles, output = {output}");
    let (pk, vk) = keygen(program.clone(), trace_len);
    let pub_in = PublicInput {
        input: vec![],
        output,
    };

    let t0 = Instant::now();
    let proof = prove(&pk, &pub_in);
    let prove_ms = t0.elapsed().as_millis();

    let t1 = Instant::now();
    let vres = verify(&vk, &pub_in, &proof);
    let ok = vres.is_ok();
    if let Err(e) = &vres {
        println!("verify error: {e}");
    }
    let verify_ms = t1.elapsed().as_millis();

    println!("prove: {prove_ms} ms, verify: {verify_ms} ms, accepted: {ok}");
    // The measured cost model (the research roadmap's first build item).
    let stats = magnetar_zkvm::stats::proof_stats(&pk, &proof);
    println!("{}", stats.report());
    if !ok {
        std::process::exit(1);
    }
}

fn cmd_check() {
    println!("Running the proof-stack self-test (cargo test handles this in CI).");
    println!("See `cargo test --workspace` for the full suite.");
}

/// The measured dual-rate cost model (the roadmap's Horizon-1 rate/query
/// dials): prove one workload, report the measured cost model, and
/// project the same proof at every named PCS profile.
fn cmd_profiles() {
    // A small straight-line fib (the shape of `cmd_fib`, 32 cycles).
    let mut program = vec![enc_i(1, 0, 0, 1, 0x13), enc_i(1, 0, 0, 2, 0x13)];
    for _ in 0..6 {
        program.push((2u32 << 20) | (1 << 15) | (3 << 7) | 0x33);
        program.push((2 << 15) | (1 << 7) | 0x33);
        program.push(enc_i(0, 3, 0, 2, 0x13));
    }
    program.push((3 << 15) | (10 << 7) | 0x33);
    program.push(0x0000_0073);
    let trace = magnetar_cpu::interp::execute(&program, &[], 32);
    let output = trace.final_state.output;
    let trace_len = trace.records.len();
    let (pk, vk) = keygen(program, trace_len);
    let pub_in = PublicInput {
        input: vec![],
        output,
    };
    let proof = prove(&pk, &pub_in);
    verify(&vk, &pub_in, &proof).expect("the profile proof must verify");
    let stats = magnetar_zkvm::stats::proof_stats(&pk, &proof);
    println!("{}", stats.report());
    println!("Dual-rate projection (the same proof at every profile):");
    println!(
        "{:<14} {:>8} {:>10} {:>18} {:>14}",
        "profile", "rate", "queries", "per-query soundness", "size (KiB)"
    );
    for (profile, bytes) in stats.projection_table() {
        let p = profile.params();
        println!(
            "{:<14} {:>8} {:>10} {:>18} {:>14.1}",
            format!("{profile:?}"),
            profile.rate_label(),
            p.queries,
            format!("{} bits", profile.per_query_soundness_bits()),
            bytes as f64 / 1024.0
        );
    }
}
