//! GPU bench for the Phase-1 kernel spec.
//!
//! Run (CUDA 13.x, sm_80+):
//! ```sh
//! cargo run --release -p magnetar-bitsum --features cutile --example bench_gpu -- 20
//! ```
//! (argument = number of variables m; default 20 = 2^20 entries).

use magnetar_bitsum::{Transcript, F128};

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
        .unwrap_or(20);
    let n = 1usize << m;
    println!("magnetar-bitsum GPU bench: bitsliced GF(2^128) multilinear sumcheck");
    println!("m = {m}, n = {n} entries, {} groups of 32", n / 32);

    let mut rng = SplitMix(0x5EED_0001 + m as u64);
    let evals: Vec<F128> = (0..n).map(|_| F128::random(&mut rng)).collect();

    // warm-up JIT on a small instance
    {
        let mut t = Transcript::new("warmup");
        let _ = magnetar_bitsum::gpu::prove(&evals[..2048.min(n)], &mut t);
    }

    let t0 = std::time::Instant::now();
    let mut t = Transcript::new("bench");
    let out = magnetar_bitsum::gpu::prove(&evals, &mut t).expect("gpu prove");
    let prove_t = t0.elapsed();

    // verify with the CPU verifier
    let t1 = std::time::Instant::now();
    let mut tv = Transcript::new("bench");
    let oc = magnetar_bitsum::sumcheck::verify_with_final_eval(
        &out.proof,
        out.claimed_sum,
        out.final_eval,
        &mut tv,
    )
    .expect("verify");
    let verify_t = t1.elapsed();
    assert_eq!(
        magnetar_bitsum::sumcheck::eval_mle(&evals, &oc.eval_point),
        oc.final_claim,
        "outer evaluation check"
    );

    let meps = n as f64 / prove_t.as_secs_f64() / 1e6;
    println!("gpu prove:  {:>12.3?}  ({:>8.1} Melem/s)", prove_t, meps);
    println!("verify:     {:>12.3?}", verify_t);
    println!();
    println!("Phase-1 exit criterion: >= 2x the CPU baseline");
    println!("  CPU baseline (magnetar-bitsum, this repo): 1.5 Melem/s");
    println!("  this GPU result:                          {meps:.1} Melem/s");
    println!(
        "  speedup:                                  {:.0}x",
        meps / 1.5
    );
    println!();
    println!("Modeled ceilings (docs/architecture/bitsum.md cost model):");
    println!("  HBM bound (3.35 TB/s, 48 B/elem-round):  ~62 Melem-round/s");
    println!("  tree ALU bound (~28k tile-ops/group):    ~25-39 Melem-round/s");
    println!("  -> a 2^m-entry instance costs ~2n elem-rounds total");
    let rounds_total = 2.0 * n as f64;
    println!(
        "     (= {} elem-rounds for this instance)",
        rounds_total as u64
    );
}
