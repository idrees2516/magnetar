fn main() {
    let log2_t: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(11);
    let t = 1usize << log2_t;
    // reuse the bench's alu program via the CLI lib? — inline it:
    let enc_i = |imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32| {
        ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
    };
    let enc_b = |imm: i32, rs2: u32, rs1: u32, funct3: u32| {
        let imm = imm as u32;
        (((imm >> 12) & 1) << 31)
            | (((imm >> 5) & 0x3f) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | (((imm >> 1) & 0xf) << 8)
            | (((imm >> 11) & 1) << 7)
            | 0x63
    };
    let program = vec![
        enc_i(32, 0, 0, 1, 0x13),
        enc_i(-1i32 as u32, 1, 0, 1, 0x13),
        enc_i(1, 0, 0, 2, 0x13),
        (2 << 20) | (1 << 15) | (3 << 7) | 0x33,
        (3 << 20) | (1 << 15) | (4 << 7) | 0x33,
        (4 << 20) | (1 << 15) | (5 << 7) | 0x33,
        enc_b(-20, 0, 1, 1),
        enc_i(0, 5, 0, 10, 0x13),
        0x0000_0073,
    ];
    let (pk, vk) = magnetar_zkvm::keygen_with(
        program.clone(),
        t,
        2,
        magnetar_pcs::PcsParams::default(),
        magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI,
    );
    let trace = magnetar_cpu::interp::execute(&pk.program, &[], t);
    let out = trace.final_state.output;
    let pub_in = magnetar_zkvm::PublicInput {
        input: vec![],
        output: out,
    };
    let t0 = std::time::Instant::now();
    let proof = magnetar_zkvm::prove(&pk, &pub_in);
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    let ok = magnetar_zkvm::verify(&vk, &pub_in, &proof);
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
    let stats = magnetar_zkvm::proof_stats(&pk, &proof);
    println!("T=2^{log2_t}: prove {prove_ms:.0} ms ({:.0} cycles/s), verify {verify_ms:.1} ms (ok={:?}), proof {:.1} KiB",
        t as f64 / (prove_ms / 1e3), ok.is_ok(), stats.bytes_total as f64 / 1024.0);
}
