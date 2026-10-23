//! CPU baseline bench for the Phase-1 exit criterion
//! ("2x over CPU baseline on 1x H100").
//!
//! Run: `cargo run --release -p magnetar-bitsum --example bench_cpu -- 16`
//! (argument = number of variables m; default 16).

use magnetar_bitsum::{sumcheck, Transcript, F128};

struct SplitMix(u64);
impl magnetar_bitsum::Rng for SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x09E3_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn main() {
    let m: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(16);
    let n = 1usize << m;
    println!("magnetar-bitsum CPU baseline: multilinear sumcheck over GF(2^128)");
    println!(
        "m = {m} variables, n = {n} table entries ({} KiB witness)",
        n * 16 / 1024
    );

    let mut rng = SplitMix(0x5EED_0000 + m as u64);
    let evals: Vec<F128> = (0..n).map(|_| F128::random(&mut rng)).collect();

    // ---- warmup (table build) ----
    {
        let mut t = Transcript::new("warmup");
        let _ = sumcheck::prove(&evals[..32.min(n)], &mut t);
    }

    let t0 = std::time::Instant::now();
    let mut t = Transcript::new("bench");
    let out = sumcheck::prove(&evals, &mut t).expect("prove");
    let prove_t = t0.elapsed();

    let t1 = std::time::Instant::now();
    let mut tv = Transcript::new("bench");
    let oc = sumcheck::verify_with_final_eval(&out.proof, out.claimed_sum, out.final_eval, &mut tv)
        .expect("verify");
    let verify_t = t1.elapsed();
    assert_eq!(
        sumcheck::eval_mle(&evals, &oc.eval_point),
        oc.final_claim,
        "outer evaluation check"
    );

    println!(
        "prove:  {:>12.3?}  ({:>8.1} Melem/s)",
        prove_t,
        n as f64 / prove_t.as_secs_f64() / 1e6
    );
    println!("verify: {:>12.3?}", verify_t);
    println!(
        "proof:  {} round messages ({} bytes)",
        out.proof.messages.len(),
        out.proof.messages.len() * 32
    );
    println!();
    println!("Phase-1 exit criterion reference: 2x this proving rate on 1x H100");
    println!("via the cutile feature (see docs/architecture/bitsum.md for the GPU bring-up).");
}
