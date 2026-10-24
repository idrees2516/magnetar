//! Profile the exact Virtual-Spartan outer sum-check shape at q = 2^10:
//! where do the rounds spend their time (message vs fold, bin vs fld)?

use magnetar_field::{Gf128, ONE, ZERO};
use magnetar_sumcheck::public::EqFactor;
use magnetar_sumcheck::{BindOrder, SumcheckProver, SumcheckSpec, Term};

fn rg(rng: &mut impl rand::Rng) -> Gf128 {
    Gf128(rand::RngExt::random::<u128>(rng))
}

#[test]
fn profile_outer_rounds() {
    let mut rng = rand::rng();
    let q = 1usize << 10; // instances
    let s = 10usize;
    let rows = 2048usize;
    let row_vars = 11usize;
    let joint_vars = s + row_vars;
    let n = q * rows;
    let t0 = std::time::Instant::now();
    let az: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
    let bz: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
    let cz: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
    println!("table build: {:?}", t0.elapsed());
    let tau: Vec<Gf128> = (0..joint_vars).map(|_| rg(&mut rng)).collect();
    let spec = SumcheckSpec {
        num_vars: joint_vars,
        dense: vec![az, bz, cz],
        public: vec![Box::new(EqFactor::new(tau))],
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
        claim: ZERO,
    };
    let order = BindOrder::HighBitsFirst { high: s };
    let mut p = SumcheckProver::with_dense_bits(&spec, order, vec![false; 3]);
    let mut msg_total = std::time::Duration::ZERO;
    let mut fold_total = std::time::Duration::ZERO;
    for i in 0..joint_vars {
        let t0 = std::time::Instant::now();
        let m = p.message();
        msg_total += t0.elapsed();
        let t1 = std::time::Instant::now();
        let r = Gf128(0x9E3779B97F4A7C150000_0000_0000_0001u128 >> i);
        p.fold_msg(&m, r);
        fold_total += t1.elapsed();
        let _ = i;
    }
    println!("message total: {:?}", msg_total);
    println!("fold total: {:?}", fold_total);
}

#[test]
fn profile_fused_msg3() {
    let mut rng = rand::rng();
    let n = 1usize << 21;
    let block = 1usize << 20;
    let f1: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
    let f2: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
    let f3: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
    let mut acc = [ZERO; 5];
    let n4 = block / 4;
    let t0 = std::time::Instant::now();
    let ok = magnetar_field::avx::msg3_batch(&mut acc, ONE, &f1, &f2, &f3, 0, block, n4);
    println!(
        "fused msg3 ({} chunks, {} positions): {:?} ok={}",
        n4,
        4 * n4,
        t0.elapsed(),
        ok
    );
}

#[test]
fn profile_mul4_raw() {
    let mut rng = rand::rng();
    let a: [Gf128; 4] = [(); 4].map(|_| rg(&mut rng));
    let b: [Gf128; 4] = [(); 4].map(|_| rg(&mut rng));
    let iters = 1_000_000u64;
    let t0 = std::time::Instant::now();
    let mut acc = [ZERO; 4];
    for i in 0..iters {
        let r = magnetar_field::avx::mul4(a, b);
        acc[0] = Gf128(acc[0].0 ^ r[(i & 3) as usize].0);
    }
    println!(
        "mul4 batch (1M calls = 4M muls): {:?} acc={:x}",
        t0.elapsed(),
        acc[0].0
    );
    // Scalar reference.
    let t1 = std::time::Instant::now();
    let mut acc2 = ZERO;
    for i in 0..iters * 4 {
        let r = magnetar_field::mul(a[0], b[0]);
        if i & 3 == 0 {
            acc2 = Gf128(acc2.0 ^ r.0);
        }
    }
    println!("scalar mul (4M muls): {:?} acc={:x}", t1.elapsed(), acc2.0);
}
