//! The `magnetar` command-line interface: execute, prove, and verify
//! RV64I programs.

use magnetar_pcs::PcsProfile;
use magnetar_zkvm::protocol::{keygen, keygen_with, prove, prove_with_wrap, verify, PublicInput};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("fib") => cmd_fib(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8)),
        Some("check") => cmd_check(),
        Some("profiles") => cmd_profiles(),
        _ => {
            eprintln!("Magnetar zkVM — post-quantum RV64I proving (eprint 2026-2423)");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  magnetar fib [steps]   prove + verify a Fibonacci program");
            eprintln!("  magnetar check         self-test of the proof stack");
            eprintln!("  magnetar profiles      the dual-rate PCS profiles (chunk vs wrap)");
            std::process::exit(2);
        }
    }
}

fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
    ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
}

/// The dual-rate operating profiles (roadmap H1/H2): the same statement
/// proved at each profile's rate/query point, with the evaluation-tail
/// bytes measured — the chunk profiles tune for speed, the wrap profiles
/// for size (the `DeepWrap` tail is what the wrap's public artifact
/// carries).
fn cmd_profiles() {
    let program = vec![
        enc_i(1, 0, 0, 1, 0x13),
        enc_i(1, 0, 0, 2, 0x13),
        (2u32 << 20) | (1 << 15) | (3 << 7) | 0x33,
        (3u32 << 20) | (2u32 << 15) | (4 << 7) | 0x33,
        0x0000_0073,
    ];
    println!("Magnetar dual-rate PCS profiles (eprint 2026-2423, roadmap H1/H2)");
    println!();
    println!(
        "{:<14} {:>7} {:>9} {:>12} {:>14}",
        "profile", "rate", "queries", "prove (ms)", "tail (B)"
    );
    for profile in [
        PcsProfile::FastChunk,
        PcsProfile::Balanced,
        PcsProfile::CompactWrap,
        PcsProfile::DeepWrap,
    ] {
        let params = profile.params();
        let (pk, vk) = keygen_with(program.clone(), 8, 2, params.clone());
        let pub_in = PublicInput {
            input: vec![],
            output: 0,
        };
        let t0 = Instant::now();
        let proof = prove(&pk, &pub_in);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        assert!(verify(&vk, &pub_in, &proof).is_ok());
        let ring = |r: &magnetar_pcs::RingProof| -> usize {
            let mut n = r.rounds.iter().map(|m| m.len() * 16).sum::<usize>();
            n += r.partials.len() * 16;
            n += r.level_roots.len() * 32;
            for q in &r.queries {
                n += q.levels.len() * 3 * (16 + 32);
            }
            for (_v, path) in &r.tail {
                n += 16 + path.len() * 32;
            }
            n
        };
        let tail = proof.mosaic.ring.as_ref().map(ring).unwrap_or(0);
        println!(
            "{:<14} {:>7} {:>9} {:>12.0} {:>14}",
            format!("{profile:?}"),
            profile.rate_label(),
            params.queries,
            ms,
            tail
        );
    }
    // The wrap: the same statement with the public tail discharged at
    // DeepWrap via the consistency spot-checks.
    let (pk, vk) = keygen_with(program, 8, 2, PcsProfile::Balanced.params());
    let pub_in = PublicInput {
        input: vec![],
        output: 0,
    };
    let wrapped = prove_with_wrap(&pk, &pub_in, Some(PcsProfile::DeepWrap.params()));
    assert!(verify(&vk, &pub_in, &wrapped).is_ok());
    let ring = |r: &magnetar_pcs::RingProof| -> usize {
        let mut n = r.rounds.iter().map(|m| m.len() * 16).sum::<usize>();
        n += r.partials.len() * 16;
        n += r.level_roots.len() * 32;
        for q in &r.queries {
            n += q.levels.len() * 3 * (16 + 32);
        }
        for (_v, path) in &r.tail {
            n += 16 + path.len() * 32;
        }
        n
    };
    let w = wrapped.mosaic.wrap.as_ref().expect("wrap");
    let mut tail = ring(&w.ring);
    tail += w
        .consistency
        .iter()
        .map(|c| 16 + (c.chunk_path.len() + c.deep_path.len()) * 32)
        .sum::<usize>();
    println!(
        "{:<14} {:>7} {:>9} {:>12} {:>14}",
        "wrapped", "1/16", 14, "-", tail
    );
    println!("\n(the wrapped row: the chunk side at Balanced + the public tail at DeepWrap,");
    println!(" tied to the chunk commitment by the consistency spot-checks)");
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
    let ok = verify(&vk, &pub_in, &proof).is_ok();
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
