//! Debug: the exact fld-outer shape (HighBitsFirst{high:s}, claim ZERO,
//! 3 K-valued dense tables + eq public) through the real verifier.

use magnetar_field::{Gf128, ONE, ZERO};
use magnetar_sumcheck::public::{EqFactor, PublicFactor};
use magnetar_sumcheck::{BindOrder, SumcheckProver, SumcheckSpec, SumcheckVerifier, Term};

fn rg(rng: &mut impl rand::Rng) -> Gf128 {
    Gf128(rand::RngExt::random::<u128>(rng))
}

#[test]
fn fld_outer_shape_roundtrips() {
    let mut rng = rand::rng();
    for (s, row_vars) in [(3usize, 5usize), (4usize, 5), (3usize, 11)] {
        let num_vars = s + row_vars;
        let n = 1usize << num_vars;
        let rows = 1usize << row_vars;
        // az, bz, cz with az*bz = cz everywhere (honest).
        let az: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
        let bz: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
        let cz: Vec<Gf128> = az
            .iter()
            .zip(bz.iter())
            .map(|(&x, &y)| magnetar_field::mul(x, y))
            .collect();
        let tau: Vec<Gf128> = (0..num_vars).map(|_| rg(&mut rng)).collect();
        let eq = magnetar_field::EqBits::build(&tau).values;
        let mut claim = ZERO;
        for i in 0..n {
            let prod = magnetar_field::mul(az[i], bz[i]) + cz[i];
            claim += magnetar_field::mul(eq[i], prod);
        }
        // Honest: prod = 0 → claim = 0.
        assert_eq!(claim, ZERO);
        let spec = SumcheckSpec {
            num_vars,
            dense: vec![az, bz, cz],
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
        let order = BindOrder::HighBitsFirst { high: s };
        let mut p = SumcheckProver::with_dense_bits(&spec, order, vec![false; 3]);
        let pf: Vec<Box<dyn PublicFactor + Send + Sync>> = vec![Box::new(EqFactor::new(tau))];
        let mut v = SumcheckVerifier::new(num_vars, claim, &pf, spec.terms.clone(), order);
        for i in 0..num_vars {
            let m = p.message();
            let label = format!("round-{i}");
            let _ = &label;
            match v.round_external(&m, r_of(i)) {
                Ok(()) => {}
                Err(e) => {
                    panic!("s={s} row_vars={row_vars} round {i}: {e}");
                }
            }
            p.fold_msg(&m, r_of(i));
        }
        let (claims, _) = p.finish();
        let _ = claims;
        // Full verifier finish with the prover's dense claims.
        let claims_vec = claims.clone();
        let _ = v.finish(&claims_vec);
        let _ = rows;
        let _ = &rng;
    }
}

fn r_of(i: usize) -> Gf128 {
    // Deterministic distinct challenges.
    let x = (i as u128).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x1234_5678_9ABC_DEF0);
    Gf128(x | 1)
}
