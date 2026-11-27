//! The BEEP-layer benchmark: the monolithic ring switch with the DEEP
//! layer off, at the default, and at the ladder's parameters — proof
//! size, prove time, verify time.
//!
//! ```sh
//! cargo run --release -p magnetar-pcs --example deep_bench -- [n]
//! ```

use magnetar_field::{eq_table, EqBits, Gf128};
use magnetar_hash::Transcript;
use magnetar_pcs::ring::{prove_ring_switch, verify_ring_switch, RingProof};
use magnetar_pcs::{commit_table, DeepParams, PcsParams};

fn proof_size(p: &RingProof) -> usize {
    let mut sz = 0usize;
    sz += p.partials.len() * 16;
    for r in &p.rounds {
        sz += r.len() * 16;
    }
    sz += p.level_roots.len() * 32;
    sz += 16; // terminal
    for q in &p.queries {
        for lvl in &q.levels {
            sz += 3 * 16 + lvl.a_vals.len() * 16 + lvl.c_vals.len() * 16
                + (lvl.a_path.len() + lvl.c_path.len()) * 32;
        }
    }
    for (vals, path) in &p.tail {
        sz += vals.len() * 16 + path.len() * 32;
    }
    if let Some(dp) = p.deep.as_ref() {
        for b in &dp.bind_answers {
            sz += b.len() * 16;
        }
        sz += 8;
        for r in &dp.rounds {
            sz += r.len() * 2 * 16;
        }
        for a in &dp.answers {
            for b in a {
                sz += b.len() * 16;
            }
        }
        sz += dp.nonces.len() * 8;
    }
    sz + 8
}

fn run(n: usize, deep: DeepParams, label: &str) {
    use rand::RngExt;
    let mut rng = rand::rng();
    let bits: Vec<bool> = (0..(1 << n) * 128).map(|_| rng.random::<bool>()).collect();
    let bytes: Vec<u8> = bits.iter().map(|&b| b as u8).collect();
    let packed = magnetar_field::pack_table(&bytes);
    let r_a: Vec<Gf128> = (0..n).map(|_| Gf128(rng.random::<u128>())).collect();
    let r_b: [Gf128; 7] = std::array::from_fn(|_| Gf128(rng.random::<u128>()));
    let eq_a = EqBits::build(&r_a).values;
    let mut y = Gf128(0);
    for u in 0..(1 << n) {
        for b in 0..128usize {
            let b_bits: Vec<bool> = (0..7).map(|j| (b >> j) & 1 == 1).collect();
            let eb = eq_table(&r_b, &b_bits);
            y += magnetar_field::mul(
                eq_a[u],
                magnetar_field::mul(eb, Gf128(bits[u * 128 + b] as u128)),
            );
        }
    }
    let params = PcsParams {
        k: 2,
        queries: 10,
        deep,
        domain: format!("bench-ring-{label}"),
    };
    let (main_cw, main_tree, stmt) = commit_table(&params, &packed);
    let mut t = Transcript::new("ring", &[b"t"]);
    let t0 = std::time::Instant::now();
    let proof = prove_ring_switch(&params, &main_cw, &main_tree, &packed, &r_a, &r_b, y, &mut t);
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
    let sz = proof_size(&proof);
    let mut tv = Transcript::new("ring", &[b"t"]);
    let t1 = std::time::Instant::now();
    verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &proof, &mut tv)
        .expect("the honest proof must verify");
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
    println!(
        "{label:<26} n={n:<3} proof ≈ {sz:>8} B   prove {prove_ms:>8.1} ms   verify {verify_ms:>7.2} ms"
    );
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(14);
    println!("== The BEEP layer on the monolithic ring switch (k=2, q=10) ==\n");
    let (ladder, lines) = magnetar_pcs::deep_ladder(2, n, 128.0).expect("ladder");
    let line = &lines[0];
    println!(
        "ladder: m={} threshold=2^{:.1} fold-grind={} bits (sample {:.0} bits, post-grind fold {:.0} bits, ~{:.0} hashes/round)\n",
        ladder.samples,
        (line.threshold as f64).log2(),
        ladder.fold_grind.added_bits(),
        line.sample_bits,
        line.fold_error_bits,
        line.fold_grind_work
    );
    run(n, DeepParams::off(), "legacy (deep off)");
    run(n, DeepParams::default(), "deep default (m=3)");
    run(n, ladder, "deep ladder");
}
