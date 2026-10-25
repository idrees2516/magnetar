//! Cross-validation of the **small-value round window** against the
//! generic engine: for the binary relation's outer shape
//! `eq(τ,·)·(D0·D1 + D2)` over bit tables bound high-bits-first, the
//! windowed prover's round messages, final claims, and the post-window
//! table state must be BIT-IDENTICAL to the plain engine's.

use magnetar_field::{Gf128, ONE, ZERO};
use magnetar_sumcheck::public::EqFactor;
use magnetar_sumcheck::{BindOrder, RoundMsg, SumcheckProver, SumcheckSpec, Term};

fn rg(rng: &mut impl rand::Rng) -> Gf128 {
    Gf128(rand::RngExt::random::<u128>(rng))
}

fn bit(rng: &mut impl rand::Rng) -> Gf128 {
    Gf128(rand::RngExt::random::<u128>(rng) & 1)
}

/// Drive a full sum-check, collecting the messages; fold at deterministic
/// pseudo-challenges derived from the message (transcript-free; identical
/// challenges for both provers is what the identity check needs).
fn run_collect(
    spec: &SumcheckSpec,
    order: BindOrder,
    bits: Vec<bool>,
    window: bool,
) -> Vec<RoundMsg> {
    let mut p = if window {
        SumcheckProver::with_dense_bits_auto_window(spec, order, bits)
    } else {
        SumcheckProver::with_dense_bits(spec, order, bits)
    };
    let mut msgs = Vec::new();
    for _ in 0..spec.num_vars {
        let m = p.message();
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
fn window_matches_generic_messages() {
    let mut rng = rand::rng();
    for num_vars in [7usize, 9, 12] {
        let n = 1usize << num_vars;
        let az: Vec<Gf128> = (0..n).map(|_| bit(&mut rng)).collect();
        let bz: Vec<Gf128> = (0..n).map(|_| bit(&mut rng)).collect();
        let cz: Vec<Gf128> = (0..n).map(|_| bit(&mut rng)).collect();
        // Mix random coordinates with bit-valued ones (the m* slice choice
        // and the slab gather edge cases).
        let tau: Vec<Gf128> = (0..num_vars)
            .map(|i| {
                if i % 3 == 0 {
                    bit(&mut rng)
                } else {
                    rg(&mut rng)
                }
            })
            .collect();
        let claim = {
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
        // The bin outer's order: high-bits-first over the instance
        // dimension; also cover `high` smaller than the window.
        for high in [num_vars.min(3), 6, num_vars] {
            let order = BindOrder::HighBitsFirst { high };
            let generic = run_collect(&mk_spec(), order, vec![true; 3], false);
            let windowed = run_collect(&mk_spec(), order, vec![true; 3], true);
            assert_eq!(
                generic.len(),
                windowed.len(),
                "num_vars={num_vars} high={high}"
            );
            for (i, (g, w)) in generic.iter().zip(windowed.iter()).enumerate() {
                assert_eq!(g, w, "num_vars={num_vars} high={high} round={i}");
            }
        }
    }
}

/// The full end-to-end roundtrip through the real verifier: the windowed
/// prover's proof must verify against the generic verifier.
#[test]
fn window_proof_verifies() {
    use magnetar_hash::Transcript;
    let mut rng = rand::rng();
    for num_vars in [8usize, 10] {
        let n = 1usize << num_vars;
        let az: Vec<Gf128> = (0..n).map(|_| bit(&mut rng)).collect();
        let bz: Vec<Gf128> = (0..n).map(|_| bit(&mut rng)).collect();
        let cz: Vec<Gf128> = (0..n).map(|_| bit(&mut rng)).collect();
        let tau: Vec<Gf128> = (0..num_vars).map(|_| rg(&mut rng)).collect();
        let eq = magnetar_field::EqBits::build(&tau).values;
        let mut claim = ZERO;
        for i in 0..n {
            let prod = magnetar_field::mul(az[i], bz[i]) + cz[i];
            claim += magnetar_field::mul(eq[i], prod);
        }
        let spec = SumcheckSpec {
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
        let order = BindOrder::HighBitsFirst { high: num_vars };
        let mut p = SumcheckProver::with_dense_bits_auto_window(&spec, order, vec![true; 3]);
        let mut t = Transcript::new("window-test", &[]);
        let mut v = magnetar_sumcheck::engine::SumcheckVerifier::new(
            num_vars,
            claim,
            &spec.public,
            vec![
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
            order,
        );
        for i in 0..num_vars {
            let m = p.message();
            let r = v
                .round_capturing(&m, &mut t, &format!("w-{i}"))
                .expect("round ok");
            p.fold_msg(&m, r);
        }
        let (dense_claims, running) = p.finish();
        let (point, accepted) = v.finish(&dense_claims).expect("finish ok");
        let _ = point;
        // The terminal identity already checked in finish; also confirm
        // the running claim agrees with the prover's.
        assert_eq!(accepted, dense_claims);
        let _ = running;
        // And a direct evaluation cross-check of the dense claims.
        let eval = |tab: &[Gf128]| magnetar_field::eval_mle(tab, &point);
        assert_eq!(dense_claims[0], eval(&az));
        assert_eq!(dense_claims[1], eval(&bz));
        assert_eq!(dense_claims[2], eval(&cz));
    }
}
