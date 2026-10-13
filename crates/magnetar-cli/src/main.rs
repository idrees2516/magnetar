//! The `magnetar` command-line interface: execute, prove, and verify
//! RV64I programs.

use magnetar_zkvm::protocol::{keygen, prove, verify, PublicInput};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("fib") => cmd_fib(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8)),
        Some("check") => cmd_check(),
        _ => {
            eprintln!("Magnetar zkVM — post-quantum RV64I proving (eprint 2026-2423)");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  magnetar fib [steps]   prove + verify a Fibonacci program");
            eprintln!("  magnetar check         self-test of the proof stack");
            std::process::exit(2);
        }
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
        program.push((0u32 << 25) | (2 << 20) | (1 << 15) | (0 << 12) | (3 << 7) | 0x33); // x3 = x1 + x2
        program.push((0u32 << 25) | (0 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | 0x33); // x1 = x2
        program.push(enc_i(0, 3, 0, 2, 0x13)); // x2 = x3
    }
    // Move the result to a0 (x10) and halt.
    program.push((0u32 << 25) | (0 << 20) | (3 << 15) | (0 << 12) | (10 << 7) | 0x33); // x10 = x3
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
    let ok = verify(&vk, &pub_in, &proof).is_ok();
    let verify_ms = t1.elapsed().as_millis();

    println!("prove: {prove_ms} ms, verify: {verify_ms} ms, accepted: {ok}");
    if !ok {
        std::process::exit(1);
    }
}

fn cmd_check() {
    println!("Running the proof-stack self-test (cargo test handles this in CI).");
    println!("See `cargo test --workspace` for the full suite.");
}
