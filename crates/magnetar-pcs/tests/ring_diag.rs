// Temporary diagnostic for the ring identity (to be removed).
use magnetar_field::{eq_coefficients, mul, phi_c, EqBits, Gf128, ZERO};
use magnetar_hash::Transcript;
use magnetar_pcs::ring::fold_partials;
use magnetar_pcs::PcsParams;

#[test]
fn ring_diag() {
    let mut rng = rand::rng();
    let n = 4;
    use rand::RngExt;
    let bits: Vec<bool> = (0..(1 << n) * 128).map(|_| rng.random::<bool>()).collect();
    let bytes: Vec<u8> = bits.iter().map(|&b| b as u8).collect();
    let packed = magnetar_field::pack_table(&bytes);
    let r_a: Vec<Gf128> = (0..n).map(|_| Gf128(rng.random::<u128>())).collect();
    let r_b: [Gf128; 7] = std::array::from_fn(|_| Gf128(rng.random::<u128>()));

    // (1) The partials and the recombination.
    let eq_z = EqBits::build(&r_a).values;
    let mut s = vec![ZERO; 128];
    for u in 0..(1 << n) {
        let mut w = packed[u].0;
        while w != 0 {
            let b = w.trailing_zeros() as usize;
            s[b] += eq_z[u];
            w &= w - 1;
        }
    }
    let c = eq_coefficients(&r_b);
    let mut y = ZERO;
    for u in 0..(1 << n) {
        for b in 0..128usize {
            if bits[u * 128 + b] {
                y += mul(eq_z[u], c[b]);
            }
        }
    }
    let mut lhs = ZERO;
    for b in 0..128 {
        lhs += mul(c[b], s[b]);
    }
    println!("recombination match={}", lhs == y);
    assert_eq!(lhs, y, "recombination");

    // (2) sigma vs the brute-force sum P(u)*A(u).
    let rho_vec: Vec<Gf128> = (0..7).map(|_| Gf128(rng.random::<u128>())).collect();
    let mut rho = [ZERO; 7];
    rho.copy_from_slice(&rho_vec);
    let rb = eq_coefficients(&rho);
    let sigma = fold_partials(&s, &rb);
    let mut brute = ZERO;
    for u in 0..(1 << n) {
        let e = eq_z[u];
        let a = phi_c(e, &rb);
        brute += mul(packed[u], a);
    }
    println!("sigma match={}", sigma == brute);
    assert_eq!(sigma, brute, "sigma == sum_u P(u) A(u)");
    let _ = PcsParams::default();
    let _ = Transcript::new("x", &[]);
}

use magnetar_sumcheck::public::PublicFactor as _;
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    BindOrder, SumcheckSpec, Term,
};

/// The engine roundtrip with dense = packed and public = the ring eq-ind.
#[test]
fn ring_engine_roundtrip() {
    let mut rng = rand::rng();
    let n = 5;
    use rand::RngExt;
    let bits: Vec<bool> = (0..(1 << n) * 128).map(|_| rng.random::<bool>()).collect();
    let bytes: Vec<u8> = bits.iter().map(|&b| b as u8).collect();
    let packed = magnetar_field::pack_table(&bytes);
    let r_a: Vec<Gf128> = (0..n).map(|_| Gf128(rng.random::<u128>())).collect();

    let eq_z = EqBits::build(&r_a).values;
    let mut s = vec![ZERO; 128];
    for u in 0..(1 << n) {
        let mut w = packed[u].0;
        while w != 0 {
            let b = w.trailing_zeros() as usize;
            s[b] += eq_z[u];
            w &= w - 1;
        }
    }
    let rho_vec: Vec<Gf128> = (0..7).map(|_| Gf128(rng.random::<u128>())).collect();
    let mut rho = [ZERO; 7];
    rho.copy_from_slice(&rho_vec);
    let rb = eq_coefficients(&rho);
    let sigma = fold_partials(&s, &rb);

    // The brute-force check that the A-table the engine will fold equals
    // phi_rb(eq_z) pointwise, and the sum equals sigma.
    let mut brute = ZERO;
    for u in 0..(1 << n) {
        brute += mul(packed[u], phi_c(eq_z[u], &rb));
    }
    assert_eq!(sigma, brute);

    // The engine roundtrip with a TableFactor built from the same A table.
    let a_table: Vec<Gf128> = eq_z.iter().map(|&e| phi_c(e, &rb)).collect();
    let spec = SumcheckSpec {
        num_vars: n,
        dense: vec![packed.clone()],
        public: vec![Box::new(magnetar_sumcheck::public::TableFactor::new(
            a_table,
        ))],
        terms: vec![Term {
            coeff: Gf128(1),
            dense: vec![0],
            public: vec![0],
        }],
        claim: sigma,
    };
    let mut prover = SumcheckProver::new(&spec, BindOrder::Increasing);
    let mut tp = magnetar_hash::Transcript::new("e", &[]);
    let mut rounds = Vec::new();
    let mut challenges = Vec::new();
    for _ in 0..n {
        let m = prover.message();
        rounds.push(m.0.clone());
        let r = prover.absorb_and_fold(&mut tp, "r");
        challenges.push(r);
    }
    let (claims, _) = prover.finish();
    let mut t = magnetar_hash::Transcript::new("e", &[]);
    let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(magnetar_pcs::ring::RingEqInd {
            z: r_a.clone(),
            rb,
        })];
    let mut verifier = SumcheckVerifier::new(
        n,
        sigma,
        &public,
        vec![Term {
            coeff: Gf128(1),
            dense: vec![0],
            public: vec![0],
        }],
        BindOrder::Increasing,
    );
    for m in &rounds {
        verifier
            .round_capturing(&magnetar_sumcheck::RoundMsg(m.clone()), &mut t, "r")
            .unwrap();
    }
    let res = verifier.finish(&[claims[0]]);
    println!("engine finish: {res:?}");
    assert!(res.is_ok(), "engine roundtrip must verify");
}

/// The eval_at identity: the MLE of the materialized A table at a random
/// point must equal phi_rb(eq(z, point)).
#[test]
fn ring_eval_at_identity() {
    let mut rng = rand::rng();
    let n = 6;
    use rand::RngExt;
    let z: Vec<Gf128> = (0..n).map(|_| Gf128(rng.random::<u128>())).collect();
    let rho_vec: Vec<Gf128> = (0..7).map(|_| Gf128(rng.random::<u128>())).collect();
    let mut rho = [ZERO; 7];
    rho.copy_from_slice(&rho_vec);
    let rb = eq_coefficients(&rho);

    let eq_z = EqBits::build(&z).values;
    let a_table: Vec<Gf128> = eq_z.iter().map(|&e| phi_c(e, &rb)).collect();

    // (a) On the Boolean cube: materialize() == tensor_eval.
    let ind = magnetar_pcs::ring::RingEqInd { z: z.clone(), rb };
    let table = ind.materialize();
    for (u, &a_u) in table.iter().enumerate() {
        let pt: Vec<Gf128> = (0..n).map(|i| Gf128(((u >> i) & 1) as u128)).collect();
        assert_eq!(
            ind.eval_at(&pt),
            a_u,
            "Boolean point {u}: tensor_eval must match the materialized table"
        );
    }
    // (b) Off the cube: the table's MLE must equal tensor_eval (the
    // multilinearity cross-check at kappa = 7).
    for _ in 0..8 {
        let p: Vec<Gf128> = (0..n).map(|_| Gf128(rng.random::<u128>())).collect();
        let mle = magnetar_field::eval_mle(&a_table, &p);
        let direct = ind.eval_at(&p);
        assert_eq!(mle, direct, "tensor_eval must be the MLE of the A table");
    }
    println!("eval_at identity OK (on and off the cube)");
}

/// The PhiTab fast path must be bit-identical to the per-bit phi_c.
#[test]
fn phi_tab_matches_phi_c() {
    let mut rng = rand::rng();
    use rand::RngExt;
    for _ in 0..64 {
        let rb: [Gf128; 128] = std::array::from_fn(|_| Gf128(rng.random::<u128>()));
        let tab = magnetar_field::PhiTab::build(&rb);
        for _ in 0..64 {
            let p = Gf128(rng.random::<u128>());
            assert_eq!(tab.apply(p), phi_c(p, &rb));
        }
    }
}
