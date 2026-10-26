//! A micro-benchmark of the small-value round window's one-pass state
//! build (the mask-major pass): isolates the Möbius-convolution/pack cost
//! from the rest of the prover so the per-suffix constant is measurable.

use magnetar_field::{Gf128, ONE};
use magnetar_sumcheck::engine::SumcheckProver;
use magnetar_sumcheck::public::EqFactor;
use magnetar_sumcheck::{BindOrder, RoundMsg, SumcheckSpec, Term};
use rand::RngExt;

const V: usize = 6;

fn main() {
    let ell = 17usize;
    let stride = 1usize << (ell - V);
    let mut rng = rand::rng();
    let m0: Vec<u64> = (0..stride).map(|_| rng.random::<u64>()).collect();
    let m1: Vec<u64> = (0..stride).map(|_| rng.random::<u64>()).collect();
    let m2: Vec<u64> = (0..stride).map(|_| rng.random::<u64>()).collect();
    let tau: Vec<Gf128> = (0..ell).map(|_| Gf128(rng.random::<u128>())).collect();

    let t0 = std::time::Instant::now();
    let iters = 8;
    let mut sink = Gf128(0);
    for _ in 0..iters {
        let spec = SumcheckSpec {
            num_vars: ell,
            dense: vec![Vec::new(), Vec::new(), Vec::new()],
            public: vec![Box::new(EqFactor::new(tau.clone()))],
            terms: vec![
                Term {
                    coeff: ONE,
                    dense: vec![0, 1],
                    public: vec![0],
                },
                Term {
                    coeff: ONE,
                    dense: vec![2],
                    public: vec![0],
                },
            ],
            claim: Gf128(0),
        };
        let mut pr = SumcheckProver::with_mask_window(
            &spec,
            BindOrder::HighBitsFirst { high: 8 },
            vec![m0.clone(), m1.clone(), m2.clone()],
        );
        let mut tr = magnetar_hash::Transcript::new("wb", &[b"x"]);
        for _ in 0..V {
            let msg: RoundMsg = pr.message();
            sink.0 ^= msg.0[0].0;
            let r = tr.challenge_field("r");
            pr.fold_msg(&msg, r);
        }
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "window pass (ell={ell} v={V} stride={stride}): {:.2} ms/iter, {:.3} µs/suffix, sink-bit={}",
        dt * 1e3,
        dt * 1e6 / stride as f64,
        sink.0 & 1
    );
    let pc: f64 = m0
        .iter()
        .chain(m1.iter())
        .map(|w| w.count_ones() as f64)
        .sum::<f64>()
        / (2.0 * stride as f64);
    println!("avg popcount of the raw truth words: {pc:.1}");
}
