//! Cross-validation of the engine's small-value fast paths and the eq-slab
//! discipline against the generic engine: for bit-valued dense tables and
//! eq public factors, the round messages must be BIT-IDENTICAL.

use magnetar_field::{Gf128, ONE, ZERO};
use magnetar_sumcheck::public::{EqFactor, EqProdFactor};
use magnetar_sumcheck::{BindOrder, RoundMsg, SumcheckProver, SumcheckSpec, Term};

fn rg(rng: &mut impl rand::Rng) -> Gf128 {
    Gf128(rand::RngExt::random::<u128>(rng))
}

fn bit(rng: &mut impl rand::Rng) -> Gf128 {
    Gf128((rand::RngExt::random::<u128>(rng) & 1) as u128)
}

/// Drive a full sum-check and collect all messages.
fn run_collect(spec: &SumcheckSpec, order: BindOrder, bits: Vec<bool>) -> Vec<RoundMsg> {
    let mut p = SumcheckProver::with_dense_bits(spec, order, bits);
    let mut msgs = Vec::new();
    for _ in 0..spec.num_vars {
        let m = p.message();
        // Fold at a deterministic pseudo-challenge derived from the message
        // (transcript-free; any challenge works for bit-identity checking).
        let mut x = 0x9E3779B97F4A7C15u64;
        for &c in &m.0[..=1.min(m.0.len() - 1)] {
            x ^= c.0 as u64;
        }
        let r = Gf128((x as u128) << 64 | (x.rotate_left(17) as u128));
        p.fold_msg(&m, r);
        msgs.push(m);
    }
    msgs
}

#[test]
fn bit_paths_match_generic_messages() {
    let mut rng = rand::rng();
    for num_vars in [4usize, 7] {
        let n = 1usize << num_vars;
        // Random bit tables.
        let az: Vec<Gf128> = (0..n).map(|_| bit(&mut rng)).collect();
        let bz: Vec<Gf128> = (0..n).map(|_| bit(&mut rng)).collect();
        let cz: Vec<Gf128> = (0..n).map(|_| bit(&mut rng)).collect();
        let tau: Vec<Gf128> = (0..num_vars).map(|_| rg(&mut rng)).collect();
        let claim = {
            // Σ eq(tau, x)·(az·bz + cz) over the cube.
            let eq = magnetar_field::EqBits::build(&tau).values;
            let mut acc = ZERO;
            for i in 0..n {
                let prod = magnetar_field::mul(az[i], bz[i]) + cz[i];
                acc += magnetar_field::mul(eq[i], prod);
            }
            acc
        };
        let mk_spec = || SumcheckSpec {
            num_vars,
            dense: vec![az.clone(), bz.clone(), cz.clone()],
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
            claim,
        };
        for order in [
            BindOrder::Increasing,
            BindOrder::HighBitsFirst { high: 2 },
            BindOrder::HighBitsFirst { high: num_vars.min(3) },
        ] {
            let generic = run_collect(&mk_spec(), order, vec![false; 3]);
            let fast = run_collect(&mk_spec(), order, vec![true, true, true]);
            for (i, (g, f)) in generic.iter().zip(fast.iter()).enumerate() {
                assert_eq!(g.0, f.0, "num_vars={num_vars} round={i} order={order:?}");
            }
        }
    }
}

#[test]
fn slab_matches_generic_for_non_bit_tables() {
    // Eq-slab correctness independent of the bit paths: generic dense
    // tables with an EqFactor public — the slab must keep the round
    // identities intact (Σ_{j≥1} c_j = running), the verifier's check.
    let mut rng = rand::rng();
    for num_vars in [4usize, 6] {
        let n = 1usize << num_vars;
        let az: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
        let bz: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
        let tau: Vec<Gf128> = (0..num_vars).map(|_| rg(&mut rng)).collect();
        let eq = magnetar_field::EqBits::build(&tau).values;
        let mut claim = ZERO;
        for i in 0..n {
            claim += magnetar_field::mul(eq[i], magnetar_field::mul(az[i], bz[i]));
        }
        let spec = SumcheckSpec {
            num_vars,
            dense: vec![az, bz],
            public: vec![Box::new(EqFactor::new(tau))],
            terms: vec![Term {
                coeff: ONE,
                dense: vec![0, 1],
                public: vec![0],
            }],
            claim,
        };
        for order in [BindOrder::Increasing, BindOrder::HighBitsFirst { high: 3 }] {
            let msgs = run_collect(&spec, order, vec![false; 2]);
            // The invariant: every round's message must sum (Σ_{j≥1} c_j)
            // to the running claim. Emulate the verifier's round check.
            let mut running = claim;
            for (i, m) in msgs.iter().enumerate() {
                let mut s = ZERO;
                for &c in &m.0[1..] {
                    s += c;
                }
                assert_eq!(
                    s,
                    running,
                    "num_vars={num_vars} round={i} order={order:?}: round-sum check"
                );
                // fold running at the same challenge the driver used.
                let mut x = 0x9E3779B97F4A7C15u64;
                for &c in &m.0[..=1.min(m.0.len() - 1)] {
                    x ^= c.0 as u64;
                }
                let r = Gf128((x as u128) << 64 | (x.rotate_left(17) as u128));
                let mut mr = ZERO;
                let mut xp = Gf128(1);
                for &c in &m.0 {
                    mr += magnetar_field::mul(c, xp);
                    xp = magnetar_field::mul(xp, r);
                }
                running = mr;
            }
        }
    }
}

#[test]
fn eq_prod_slab_round_sums_hold() {
    // The EqProdFactor slab (the memory-checking shape: eq over disjoint
    // variable subsets, some variables idle).
    let mut rng = rand::rng();
    let num_vars = 6;
    let n = 1usize << num_vars;
    let az: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
    // parts: eq(p1, x0..x2) and eq(p2, x3..x4); x5 idle.
    let p1: Vec<Gf128> = (0..3).map(|_| rg(&mut rng)).collect();
    let p2: Vec<Gf128> = (0..2).map(|_| rg(&mut rng)).collect();
    let f = EqProdFactor::new(
        vec![
            (p1, vec![0, 1, 2]),
            (p2, vec![3, 4]),
        ],
        num_vars,
    );
    let table = magnetar_sumcheck::public::PublicFactor::materialize(&f);
    let mut claim = ZERO;
    for i in 0..n {
        claim += magnetar_field::mul(table[i], az[i]);
    }
    let spec = SumcheckSpec {
        num_vars,
        dense: vec![az],
        public: vec![Box::new(f)],
        terms: vec![Term {
            coeff: ONE,
            dense: vec![0],
            public: vec![0],
        }],
        claim,
    };
    for order in [BindOrder::Increasing, BindOrder::HighBitsFirst { high: 2 }] {
        let msgs = run_collect(&spec, order, vec![false; 1]);
        let mut running = claim;
        for (i, m) in msgs.iter().enumerate() {
            let mut s = ZERO;
            for &c in &m.0[1..] {
                s += c;
            }
            assert_eq!(s, running, "eqprod num_vars=6 round={i} order={order:?}");
            let mut x = 0x9E3779B97F4A7C15u64;
            for &c in &m.0[..=1.min(m.0.len() - 1)] {
                x ^= c.0 as u64;
            }
            let r = Gf128((x as u128) << 64 | (x.rotate_left(17) as u128));
            let mut mr = ZERO;
            let mut xp = Gf128(1);
            for &c in &m.0 {
                mr += magnetar_field::mul(c, xp);
                xp = magnetar_field::mul(xp, r);
            }
            running = mr;
        }
    }
}

#[test]
fn debug_sum_eq_single_item() {
    let mut rng = rand::rng();
    let nv = 4;
    let n = 1usize << nv;
    let p: Vec<Gf128> = (0..nv).map(|_| rg(&mut rng)).collect();
    let alpha = rg(&mut rng);
    let f = magnetar_sumcheck::public::SumEqFactor::new(vec![(alpha, p.clone())]);
    let table = magnetar_sumcheck::public::PublicFactor::materialize(&f);
    let eq = magnetar_field::EqBits::build(&p).values;
    for i in 0..n {
        let want = magnetar_field::mul(alpha, eq[i]);
        assert_eq!(table[i], want, "i={i}");
    }
}

#[test]
fn debug_sum_eq_small() {
    // 2 free coords only, nv=2: trace by hand.
    let p1 = Gf128(0b10); // coordinate 0 = 2 (free); coordinate 1 = 0 (pinned)
    let alpha = Gf128(1);
    let f = magnetar_sumcheck::public::SumEqFactor::new(vec![(alpha, vec![p1, Gf128(0)])]);
    let table = magnetar_sumcheck::public::PublicFactor::materialize(&f);
    let eq = magnetar_field::EqBits::build(&[p1, Gf128(0)]).values;
    for i in 0..4 {
        eprintln!("i={i} got={:x} want={:x}", table[i].0, eq[i].0);
    }
}

#[test]
fn debug_sum_eq_two_free() {
    let p1 = Gf128(0b10);
    let p2 = Gf128(0b11);
    let alpha = Gf128(1);
    let f = magnetar_sumcheck::public::SumEqFactor::new(vec![(alpha, vec![p1, p2])]);
    let table = magnetar_sumcheck::public::PublicFactor::materialize(&f);
    let eq = magnetar_field::EqBits::build(&[p1, p2]).values;
    for i in 0..4 {
        eprintln!("i={i} got={:x} want={:x}", table[i].0, eq[i].0);
    }
}

#[test]
fn profile_sumeq_materialize() {
    // The Mosaic-shape SumEq at T=2^10-like dimensions: ~22 claims, each
    // with ~16 free coordinates + Boolean suffix bits at ell=17.
    let mut rng = rand::rng();
    let ell = 17usize;
    let mut items = Vec::new();
    for i in 0..22usize {
        let mut p = vec![Gf128(0); ell];
        // 16 free coords + 1 pinned bit.
        for c in 0..16 {
            p[c] = rg(&mut rng);
        }
        p[16] = Gf128((i % 2) as u128);
        items.push((rg(&mut rng), p));
    }
    let f = magnetar_sumcheck::public::SumEqFactor::new(items);
    let t0 = std::time::Instant::now();
    let table = magnetar_sumcheck::public::PublicFactor::materialize(&f);
    println!("SumEq materialize (22 claims, 2^17 cube): {:?} nonzero={}",
        t0.elapsed(),
        table.iter().filter(|&&v| v != Gf128(0)).count());
}
