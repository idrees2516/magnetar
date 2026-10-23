//! The `magnetar` command-line interface: execute, prove, and verify
//! RV64I programs — standalone, or claim-aggregated across segments with
//! the dual-rate PCS schedule.

use magnetar_pcs::PcsParams;
use magnetar_zkvm::ivc::{prove_chunked, verify_chunked};
use magnetar_zkvm::protocol::{
    keygen, keygen_aggregate, keygen_chunked, prove, prove_multi, verify, verify_multi, PublicInput,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("fib") => cmd_fib(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8)),
        Some("agg") => cmd_agg(
            args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4),
            args.get(3).map(|s| s.as_str()),
        ),
        Some("bench") => cmd_bench(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512)),
        Some("chunked") => cmd_chunked(
            args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32),
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(96),
        ),
        Some("check") => cmd_check(),
        _ => {
            eprintln!("Magnetar zkVM — post-quantum RV64I proving (eprint 2026-2423)");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  magnetar fib [steps]        prove + verify a Fibonacci program");
            eprintln!("  magnetar agg [segments] [rate]  claim-aggregated proof of N segments");
            eprintln!(
                "                              (rate: 'chunk' = 1/2, 'wrap' = 1/4, default wrap)"
            );
            eprintln!("  magnetar check               self-test of the proof stack");
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
        program.push((2u32 << 20) | (1 << 15) | (3 << 7) | 0x33); // x3 = x1 + x2
        program.push((2u32 << 15) | (1 << 7) | 0x33); // x1 = x2 (rs2=x2, funct7=0)
        program.push(enc_i(0, 3, 0, 2, 0x13)); // x2 = x3
    }
    // Move the result to a0 (x10) and halt.
    program.push((3 << 15) | (10 << 7) | 0x33); // x10 = x3 (rs2=x0)
    program.push(0x0000_0073); // ecall: halt

    // Run the interpreter to get the output and the exact trace length
    // (padded to the proving bound).
    let trace = magnetar_cpu::interp::execute(&program, &[], program.len() * 4 + 64);
    let output = trace.final_state.output;
    let trace_len = trace
        .records
        .len()
        .next_power_of_two()
        .max(trace.records.len());

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

/// The claim-aggregated prover (roadmap Horizon 2): `segments` executions
/// of one program — the rollup/batch flavor — share ONE commitment and ONE
/// evaluation argument; the printed cost model shows the amortization
/// against the standalone counterfactual.
fn cmd_agg(segments: usize, rate: Option<&str>) {
    let segments = segments.max(1).next_power_of_two();
    let pcs = match rate {
        Some("chunk") => PcsParams::chunk(),
        _ => PcsParams::wrap(),
    };
    // The doubling program: read word k, add 7, store it back, and bind
    // the result to a0.
    let program = vec![
        (3 << 12) | (3 << 7) | 0x03,                // ld x3, 0(x0)
        enc_i(7, 0, 0, 5, 0x13),                    // addi x5, x0, 7
        (5u32 << 20) | (3 << 15) | (3 << 7) | 0x33, // add x3, x3, x5
        (3u32 << 20) | (3 << 12) | 0x23,            // sd x3, 0(x0) (rs1=x0, imm=0)
        (3 << 12) | (10 << 7) | 0x03,               // ld x10, 0(x0)
        0x0000_0073,                                // ecall
    ];
    let pub_ins: Vec<PublicInput> = (0..segments)
        .map(|s| PublicInput {
            input: vec![s as u64 + 1],
            output: s as u64 + 8,
        })
        .collect();
    println!(
        "Magnetar agg — {segments} segments, {} rate, trace {} cycles/segment",
        if pcs.k == 1 {
            "chunk (1/2)"
        } else {
            "wrap (1/4)"
        },
        8
    );
    let (pk, vk) = keygen_aggregate(program, 8, 2, pcs, segments);
    let t0 = Instant::now();
    let proof = prove_multi(&pk, &pub_ins);
    let prove_ms = t0.elapsed().as_millis();
    let t1 = Instant::now();
    let ok = verify_multi(&vk, &pub_ins, &proof).is_ok();
    let verify_ms = t1.elapsed().as_millis();
    println!("prove: {prove_ms} ms, verify: {verify_ms} ms, accepted: {ok}");
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

fn cmd_chunked(chunk_cycles: usize, total_cycles: usize) {
    // A straight-line Fibonacci-style body spanning several chunks: the
    // register file carries the accumulator across every boundary.
    let chunk_cycles = chunk_cycles.max(16).next_power_of_two();
    let mut program = vec![
        enc_i(1, 0, 0, 1, 0x13), // x1 = 1
        enc_i(1, 0, 0, 2, 0x13), // x2 = 1
    ];
    let iters = total_cycles.max(8) / 3;
    for _ in 0..iters {
        program.push((2u32 << 20) | (1 << 15) | (3 << 7) | 0x33); // x3 = x1 + x2
        program.push((2 << 15) | (1 << 7) | 0x33); // x1 = x2
        program.push(enc_i(0, 3, 0, 2, 0x13)); // x2 = x3
    }
    program.push((3 << 15) | (10 << 7) | 0x33); // x10 = x3
    program.push(0x0000_0073); // ecall

    // The interpreter determines the program's true output.
    let trace = magnetar_cpu::interp::execute(&program, &[], program.len() * 4 + 64);
    let output = trace.final_state.output;

    let (pk, vk) = keygen_chunked(program, chunk_cycles);
    let pub_in = PublicInput {
        input: vec![],
        output,
    };
    println!("Magnetar tree-IVC — chunk size {chunk_cycles} cycles, program output = {output}");
    let t0 = Instant::now();
    let (proof, public) = prove_chunked(&pk, &pub_in, 64);
    let prove_ms = t0.elapsed().as_millis();
    let k = proof.chunks.len();

    let t1 = Instant::now();
    let v = verify_chunked(&vk, &pub_in, &public, &proof);
    let verify_ms = t1.elapsed().as_millis();
    if let Err(e) = &v {
        eprintln!("verify error: {e}");
    }

    println!(
        "{k} chunks ({k} x {chunk_cycles} = {} cycles), prove: {prove_ms} ms, verify: {verify_ms} ms, accepted: {}",
        k * chunk_cycles,
        v.is_ok()
    );
    // The measured aggregation win (the roadmap H2 cost model).
    let stats = magnetar_zkvm::stats::aggregate_stats(&vk, &proof);
    println!("{}", stats.report());
    if !v.is_ok() {
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// The end-to-end benchmark (paper §6.1, Table 6's methodology, scaled to
// this host): trace-length scaling, the aggregation ledger, the tree-IVC
// chunk count, and the dual-rate schedule — all with exact wire sizes.
// ---------------------------------------------------------------------------

/// The workload: a straight-line Fibonacci body (register-file carried).
fn fib_program(iters: usize) -> Vec<u32> {
    let mut program = vec![enc_i(1, 0, 0, 1, 0x13), enc_i(1, 0, 0, 2, 0x13)];
    for _ in 0..iters.max(1) {
        program.push((2u32 << 20) | (1 << 15) | (3 << 7) | 0x33); // x3 = x1 + x2
        program.push((2 << 15) | (1 << 7) | 0x33); // x1 = x2
        program.push(enc_i(0, 3, 0, 2, 0x13)); // x2 = x3
    }
    program.push((3 << 15) | (10 << 7) | 0x33); // x10 = x3
    program.push(0x0000_0073); // ecall
    program
}

fn output_of(program: &[u32], trace_len: usize) -> u64 {
    magnetar_cpu::interp::execute_padded(program, &[], trace_len, trace_len)
        .final_state
        .output
}

fn cmd_bench(max_trace: usize) {
    println!("# Magnetar end-to-end benchmark");
    println!();
    println!("The paper's Table 6 methodology (trace scaling of prove/verify time,");
    println!("proof size, and memory), plus the roadmap H2 artifacts (claim");
    println!("aggregation, tree-IVC chunking, and the dual-rate schedule), with");
    println!("exact uncompressed wire sizes. Host: 2 vCPU / ~4 GiB; the paper's");
    println!("host is an M4 Max (12 perf. cores / 128 GiB), so traces are scaled.");
    println!();
    // ---- Trace-length scaling (monolithic, wrap rate).
    println!("## Trace-length scaling (one segment, rate 1/4)");
    println!();
    println!(
        "| Trace | Guest (us) | Prove (ms) | Cycles/s | Proof (KiB) | Verify (ms) | Peak RSS (MiB) |"
    );
    println!("|------:|-----------:|-----------:|---------:|------------:|------------:|----------------:|");
    for &trace in &[64usize, 128, 256] {
        if trace > max_trace {
            continue;
        }
        let program = fib_program(trace / 3);
        let tg = Instant::now();
        let out = output_of(&program, trace);
        let guest_us = tg.elapsed().as_micros();
        let pub_in = PublicInput {
            input: vec![],
            output: out,
        };
        let (pk, vk) = keygen(program, trace);
        let t0 = Instant::now();
        let proof = prove(&pk, &pub_in);
        let prove_ms = t0.elapsed().as_millis();
        let size = proof.size_bytes();
        let t1 = Instant::now();
        let ok = verify(&vk, &pub_in, &proof).is_ok();
        let verify_ms = t1.elapsed().as_millis();
        assert!(ok);
        let cps = if prove_ms > 0 {
            trace as f64 * 1000.0 / prove_ms as f64
        } else {
            0.0
        };
        let cps_s = if cps >= 1e6 {
            format!("{:.2}M", cps / 1e6)
        } else {
            format!("{:.0}", cps)
        };
        println!(
            "| 2^{} | {} | {} | {} | {:.1} | {} | {} |",
            trace.trailing_zeros(),
            guest_us,
            prove_ms,
            cps_s,
            size as f64 / 1024.0,
            verify_ms,
            magnetar_zkvm::wire::peak_rss_kib() / 1024,
        );
    }
    // ---- Claim aggregation: K segments share one commitment; compare
    // against K standalone proofs (the counterfactual).
    println!();
    println!("## Claim aggregation (K segments of 2^6 cycles, rate 1/4)");
    println!();
    println!("| K | K standalone (KiB) | Aggregated (KiB) | Saving | Prove (ms) | Verify (ms) |");
    println!("|--:|-------------------:|-----------------:|-------:|-----------:|------------:|");
    let seg_trace = 64usize;
    let program = fib_program(16);
    let out = output_of(&program, seg_trace);
    for &k in &[1usize, 2, 4] {
        let pub_ins: Vec<PublicInput> = (0..k)
            .map(|_| PublicInput {
                input: vec![],
                output: out,
            })
            .collect();
        // The standalone counterfactual: K separate single-segment proofs.
        let (spk, _svk) = keygen(program.clone(), seg_trace);
        let standalone: usize = (0..k)
            .map(|_| {
                let p = prove(&spk, &pub_ins[0]);
                p.size_bytes()
            })
            .sum();
        let (pk, vk) = keygen_aggregate(
            program.clone(),
            seg_trace,
            2,
            PcsParams::wrap(),
            k.max(1).next_power_of_two(),
        );
        let t0 = Instant::now();
        let proof = prove_multi(&pk, &pub_ins);
        let prove_ms = t0.elapsed().as_millis();
        let size = proof.size_bytes();
        let t1 = Instant::now();
        let ok = verify_multi(&vk, &pub_ins, &proof).is_ok();
        let verify_ms = t1.elapsed().as_millis();
        assert!(ok, "aggregated proof must verify");
        let saving = 1.0 - size as f64 / standalone as f64;
        println!(
            "| {k} | {:.1} | {:.1} | {:.0}% | {} | {} |",
            standalone as f64 / 1024.0,
            size as f64 / 1024.0,
            saving * 100.0,
            prove_ms,
            verify_ms,
        );
    }
    // ---- Tree-IVC: one execution across dynamic chunks.
    println!();
    println!("## Tree-IVC chunking (one execution, dynamic chunk count)");
    println!();
    println!("| Chunk cycles | Chunks | Total cycles | Proof (KiB) | Prove (ms) | Verify (ms) |");
    println!("|-------------:|-------:|-------------:|------------:|-----------:|------------:|");
    for &chunk in &[32usize, 64] {
        let program = fib_program(24);
        let out = output_of(&program, 256);
        let pub_in = PublicInput {
            input: vec![],
            output: out,
        };
        let (pk, vk) = keygen_chunked(program, chunk);
        let t0 = Instant::now();
        let (proof, public) = prove_chunked(&pk, &pub_in, 64);
        let prove_ms = t0.elapsed().as_millis();
        let k = proof.chunks.len();
        let size = proof.size_bytes();
        let t1 = Instant::now();
        let ok = verify_chunked(&vk, &pub_in, &public, &proof).is_ok();
        let verify_ms = t1.elapsed().as_millis();
        assert!(ok, "chunked proof must verify");
        println!(
            "| {chunk} | {k} | {} | {:.1} | {} | {} |",
            k * chunk,
            size as f64 / 1024.0,
            prove_ms,
            verify_ms,
        );
    }
    // ---- The dual-rate schedule.
    println!();
    println!("## Dual-rate schedule (4 segments of 2^6 cycles)");
    println!();
    println!("| Profile | Rate | Proof (KiB) | Verify (ms) |");
    println!("|---------|-----:|------------:|-------------:|");
    let profiles = [("chunk", PcsParams::chunk()), ("wrap", PcsParams::wrap())];
    for (name, pcs) in profiles.iter().map(|(n, p)| (*n, p.clone())) {
        let program = fib_program(16);
        let out = output_of(&program, 64);
        let pub_ins: Vec<PublicInput> = (0..4)
            .map(|_| PublicInput {
                input: vec![],
                output: out,
            })
            .collect();
        let rate_k = pcs.k;
        let (pk, vk) = keygen_aggregate(program, 64, 2, pcs, 4);
        let proof = prove_multi(&pk, &pub_ins);
        let size = proof.size_bytes();
        let t1 = Instant::now();
        let ok = verify_multi(&vk, &pub_ins, &proof).is_ok();
        let verify_ms = t1.elapsed().as_millis();
        assert!(ok);
        println!(
            "| {name} | 2^-{rate_k} | {:.1} | {} |",
            size as f64 / 1024.0,
            verify_ms,
        );
    }
}
