//! The streaming-prover benchmark: sequential (full-trace) vs streaming
//! (checkpointed, rayon-parallel) tree-IVC — wall time per phase and
//! peak RSS. `VmHWM` is a **lifetime** high-water mark, so for honest
//! memory numbers each mode runs in its own process (R10): pass `seq`
//! or `stream` as the mode argument.
//!
//! Run:
//!   `cargo run --release -p magnetar-cli --bin magnetar-bench-stream -- [LOG2_MAX] [seq|stream|both]`

use magnetar_pcs::PcsParams;
use std::time::Instant;

fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
    ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
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

fn alu_program() -> Vec<u32> {
    vec![
        enc_i(32, 0, 0, 1, 0x13),
        enc_i(-1i32 as u32, 1, 0, 1, 0x13),
        enc_i(1, 0, 0, 2, 0x13),
        (2 << 20) | (1 << 15) | (3 << 7) | 0x33,
        (3 << 20) | (1 << 15) | (4 << 7) | 0x33,
        (4 << 20) | (1 << 15) | (5 << 7) | 0x33,
        enc_b(-20, 0, 1, 1),
        enc_i(0, 5, 0, 10, 0x13),
        0x0000_0073,
    ]
}

/// The segment-count ceiling: the streaming root's combined commitment
/// still materializes `packed + codeword + tree` (≈3.3× the packed
/// bytes), so the segment count is capped by the host's RAM, not the
/// protocol. `MAGNETAR_MAX_SEGMENTS` overrides (default 16,384 = the
/// 2^24 ladder at 2^10 cycles/segment).
fn max_segments() -> usize {
    std::env::var("MAGNETAR_MAX_SEGMENTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16_384)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let max_log2: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let min_log2: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let mode = args.next().unwrap_or_else(|| "both".to_string());
    let run_seq = mode == "seq" || mode == "both";
    let run_stream = mode == "stream" || mode == "both";
    println!(
        "Streaming prover benchmark ({} rayon threads, mode {mode})",
        rayon::current_num_threads()
    );
    println!(
        "{:>7} {:>4} {:>9} {:>10} | {:>11} {:>7} | {:>9} {:>9} {:>9} {:>9}",
        "T",
        "segs",
        "prove (ms)",
        "peak RSS",
        "ckpt (ms)",
        "cycle/s",
        "segs (ms)",
        "root (ms)",
        "bound",
        "table B"
    );
    let prog = alu_program();
    for log2_t in (8usize..=24)
        .filter(|t| *t >= min_log2 && *t <= max_log2)
    {
        let total = 1usize << log2_t;
        let trace = magnetar_cpu::interp::execute(&prog, &[], total);
        let out = trace.final_state.output;
        // Adaptive segment ladder: segments of at least 2^8 cycles, and —
        // above 2^13 total — at most 2^13 cycles per segment (the
        // monolithic per-segment envelope of this container).
        let min_segs = total / (1usize << 10); // 2^10-cycle segments: bounded per-worker envelope at every scale
        let mut ladder: Vec<usize> = Vec::new();
        let mut s = 1usize;
        while s * 256 <= total {
            if s >= min_segs.max(1) {
                ladder.push(s);
            }
            s *= 2;
        }
        for segments in ladder {
            if total / segments < 2 {
                continue;
            }
            // The streaming envelope is bounded per worker; the segment
            // count grows with the trace (the paper's ladder runs to 2^22
            // with 4,096 segments — this one is uncapped).
            if segments > max_segments() {
                break;
            }
            // The monolithic (sequential) prover only fits the container
            // envelope up to 2^13; above that, streaming-only.
            let run_seq_here = run_seq && log2_t <= 13;
            if run_seq_here {
                let t0 = Instant::now();
                let seq = magnetar_ivc::prove_ivc(
                    &prog,
                    &[],
                    out,
                    total,
                    segments,
                    &PcsParams::default(),
                );
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                assert!(magnetar_ivc::verify_ivc(
                    &prog,
                    &[],
                    out,
                    total,
                    &seq,
                    &PcsParams::default(),
                    magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
                )
                .is_ok());
                println!(
                    "{:>7} {:>4} {:>9.0} {:>9.1}M | {:>11} {:>7} | {:>9} {:>9} {:>9} {:>9}",
                    format!("2^{log2_t}"),
                    segments,
                    ms,
                    magnetar_ivc::peak_rss_bytes() as f64 / 1048576.0,
                    "seq",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-"
                );
            }
            if run_stream {
                let (stream, stats) = magnetar_ivc::prove_ivc_streaming(
                    &prog,
                    &[],
                    out,
                    total,
                    segments,
                    &PcsParams::default(),
                    magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI,
                );
                assert!(magnetar_ivc::verify_ivc(
                    &prog,
                    &[],
                    out,
                    total,
                    &stream,
                    &PcsParams::default(),
                    magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
                )
                .is_ok());
                let bound = stats.packed_bytes
                    + stats.codeword_bytes
                    + stats.segment_table_bytes
                    + stats.checkpoint_bytes;
                println!(
                    "{:>7} {:>4} {:>9.0} {:>9.1}M | {:>11.0} {:>7.0} | {:>9.0} {:>9.0} {:>8.1}M {:>9}",
                    format!("2^{log2_t}"), segments,
                    stats.wall_total.as_secs_f64() * 1e3,
                    stats.peak_rss_bytes as f64 / 1048576.0,
                    stats.wall_checkpoints.as_secs_f64() * 1e3,
                    total as f64 / stats.wall_total.as_secs_f64(),
                    stats.wall_segments.as_secs_f64() * 1e3,
                    stats.wall_root.as_secs_f64() * 1e3,
                    bound as f64 / 1048576.0,
                    stats.segment_table_bytes / 1024,
                );
            }
        }
    }
}
