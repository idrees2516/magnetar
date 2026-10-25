//! Seeded regression coverage for the join-enumeration routing prover.
//!
//! The in-crate tests generate instances from `thread_rng`, which is not
//! reproducible; this suite pins a bank of fixed seeds and runs the full
//! prove + verify roundtrip on each, so any input-specific prover bug
//! fails deterministically. (History: the round-0 self-check in
//! `prove_routing_join` summed the constant coefficient — wrong under the
//! engine's `Σ_{j≥1} m_j = running` round check — and passed CI only when
//! `c₀` happened to cancel.)

use magnetar_field::{Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_router::join::{prove_routing_join, JoinWitness, RoutingTensor, SelCol};
use magnetar_router::{build_shift_tensor, shift_selector_layout, RoutingLayout};
use magnetar_sumcheck::public::PaddedFactor;
use rand::{RngExt, SeedableRng};

fn mul(a: Gf128, b: Gf128) -> Gf128 {
    magnetar_field::mul(a, b)
}

fn eq_mle(r: &[Gf128], x: usize) -> Gf128 {
    let mut v = ONE;
    for (i, ri) in r.iter().enumerate() {
        v = mul(v, *ri + Gf128(((x >> i) & 1) as u128) + ONE);
    }
    v
}

fn rg(rng: &mut impl rand::Rng) -> Gf128 {
    Gf128(rand::RngExt::random::<u128>(rng))
}

fn run_case(seed: u64) -> Result<(), &'static str> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let w = 4usize;
    let amount_bits = 2usize;
    let (values, o_vars, u_vars, h_vars) = build_shift_tensor(w, amount_bits);
    let j_vars = 3usize;
    let layout = RoutingLayout {
        o_vars,
        j_vars,
        u_vars,
        h_vars,
        a_vars: 0,
    };
    let cycles = 1usize << j_vars;
    let mut x = vec![ZERO; (1usize << u_vars) * cycles];
    for j in 0..cycles {
        for b in 0..w {
            x[b + j * (1usize << u_vars)] = Gf128(rng.random::<bool>() as u128);
        }
    }
    let (a_off_h, a_hv, k_off_h, k_hv) = shift_selector_layout(amount_bits);
    let mut q_lo = vec![ZERO; (1usize << a_hv) * cycles];
    let mut q_hi = vec![ZERO; (1usize << k_hv) * cycles];
    for j in 0..cycles {
        if rng.random::<bool>() {
            let amount = rng.random_range(0..(1usize << amount_bits));
            let kind = rng.random_range(0..3usize);
            q_lo[amount + j * (1usize << a_hv)] = ONE;
            q_hi[kind + j * (1usize << k_hv)] = ONE;
        }
    }
    let r_o: Vec<Gf128> = (0..o_vars).map(|_| rg(&mut rng)).collect();
    let r_j: Vec<Gf128> = (0..j_vars).map(|_| rg(&mut rng)).collect();
    let mut y_claim = ZERO;
    for j in 0..cycles {
        for o in 0..(1usize << o_vars) {
            for u in 0..(1usize << u_vars) {
                for hl in 0..(1usize << a_hv) {
                    for hh in 0..(1usize << k_hv) {
                        let h = hh * (1usize << amount_bits) + hl;
                        let rv = values[o | (u << o_vars) | (h << (o_vars + u_vars))];
                        let qv = mul(
                            q_lo[hl + j * (1usize << a_hv)],
                            q_hi[hh + j * (1usize << k_hv)],
                        );
                        if rv == ZERO || qv == ZERO {
                            continue;
                        }
                        let mut s = mul(mul(eq_mle(&r_o, o), eq_mle(&r_j, j)), rv);
                        s = mul(s, x[u + j * (1usize << u_vars)]);
                        y_claim += mul(s, qv);
                    }
                }
            }
        }
    }
    let tensor = RoutingTensor::from_parts(values, o_vars, u_vars, h_vars, 0);
    let sels = vec![
        SelCol {
            table: q_lo.clone(),
            h_off: a_off_h,
            h_vars: a_hv,
        },
        SelCol {
            table: q_hi.clone(),
            h_off: k_off_h,
            h_vars: k_hv,
        },
    ];
    let wit = JoinWitness {
        x: x.clone(),
        selectors: sels,
        offset_bits: Vec::new(),
    };
    let mut t = Transcript::new("router", &[b"t"]);
    let proof = prove_routing_join(&layout, y_claim, &r_o, &r_j, &tensor, &wit, &mut t);

    let jv = layout.joint_vars();
    let [o_off, _, u_off, h_off, _a_off] = layout.offsets();
    let r_pub = {
        let mut act: Vec<usize> = (0..o_vars).map(|i| o_off + i).collect();
        act.extend(u_off..u_off + u_vars);
        act.extend(h_off..h_off + h_vars);
        PaddedFactor::new(tensor.values.clone(), act, jv)
    };
    let mut tv = Transcript::new("router", &[b"t"]);
    let claims = magnetar_router::verify_routing(
        &layout, y_claim, &r_o, &r_j, &r_pub, 2, 0, &proof, &mut tv,
    )?;
    // The descended claims must match the witness tables' MLEs.
    let x_pt = claims.x_point(&layout);
    assert_eq!(claims.x_value, magnetar_field::eval_mle(&x, &x_pt));
    let qlo_pt = claims.q_point(&layout, a_off_h, a_hv);
    assert_eq!(claims.q_values[0], magnetar_field::eval_mle(&q_lo, &qlo_pt));
    let qhi_pt = claims.q_point(&layout, k_off_h, k_hv);
    assert_eq!(claims.q_values[1], magnetar_field::eval_mle(&q_hi, &qhi_pt));
    // Tamper: any single round coefficient flip must be rejected.
    for r in 0..proof.rounds.len() {
        for c in 0..proof.rounds[r].len() {
            let mut p = proof.clone();
            p.rounds[r][c] += ONE;
            let mut tv = Transcript::new("router", &[b"t"]);
            assert!(
                magnetar_router::verify_routing(
                    &layout, y_claim, &r_o, &r_j, &r_pub, 2, 0, &p, &mut tv
                )
                .is_err(),
                "tampered round {r} coeff {c} must be rejected"
            );
        }
    }
    Ok(())
}

#[test]
fn seeded_join_roundtrips() {
    for seed in 0..24u64 {
        run_case(seed).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    }
}

/// A **store-shape** case: `u_vars = 7` (a 128-slot column — TWO packed
/// words), `a_vars > 0` (the offset rounds), and a selector-only `h`
/// block. History: the h-round packed-bit select once packed
/// `u_v.div_ceil(64)` words instead of `(1 << u_v).div_ceil(64)`, silently
/// dropping the column's high word — every column with set bits above
/// slot 64 contracted wrong, and only the store (the one production
/// 2-word shape) hit it.
#[test]
fn seeded_join_store_shape_roundtrips() {
    for seed in 100..112u64 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let o_vars = 6usize;
        let u_vars = 7usize;
        let h_vars = 5usize;
        let a_vars = 3usize;
        let cycles = 8usize;
        let layout = RoutingLayout {
            o_vars,
            j_vars: 3,
            u_vars,
            h_vars,
            a_vars,
        };
        // R(o, u, h, a): a random sparse support over (o, u, h) per a.
        let mut values = vec![ZERO; 1usize << (o_vars + u_vars + h_vars + a_vars)];
        for _ in 0..300 {
            let o = rand::RngExt::random_range(&mut rng, 0..1usize << o_vars);
            let u = rand::RngExt::random_range(&mut rng, 0..1usize << u_vars);
            let h = rand::RngExt::random_range(&mut rng, 0..1usize << h_vars);
            let a = rand::RngExt::random_range(&mut rng, 0..1usize << a_vars);
            values[o
                | (u << o_vars)
                | (h << (o_vars + u_vars))
                | (a << (o_vars + u_vars + h_vars))] = rg(&mut rng);
        }
        let tensor = RoutingTensor::from_parts(values.clone(), o_vars, u_vars, h_vars, a_vars);
        // The witness: random bits in the 128-slot columns, one-hot
        // selectors, and offset bits.
        let mut x = vec![ZERO; (1usize << u_vars) * cycles];
        for v in x.iter_mut() {
            *v = Gf128(rand::RngExt::random::<bool>(&mut rng) as u128);
        }
        let sel_flat: Vec<Gf128> = {
            let mut t = vec![ZERO; (1usize << h_vars) * cycles];
            for j in 0..cycles {
                let h: usize = rand::RngExt::random_range(&mut rng, 0..1usize << h_vars);
                t[h | (j << h_vars)] = ONE;
            }
            t
        };
        let off_bits: Vec<Vec<bool>> = (0..a_vars)
            .map(|_| {
                (0..cycles)
                    .map(|_| rand::RngExt::random::<bool>(&mut rng))
                    .collect()
            })
            .collect();
        let w = JoinWitness {
            x: x.clone(),
            selectors: vec![SelCol {
                table: sel_flat.clone(),
                h_off: 0,
                h_vars,
            }],
            offset_bits: off_bits.clone(),
        };
        // The claim: the honest brute-force sum over the joint cube —
        // Σ_j eq(r_j, j) · [Σ_h sel_h(j) · Σ_a (Π_m link_m) · Σ_{o,u} R·eq(r_o,o)·X].
        let r_o: Vec<Gf128> = (0..o_vars).map(|_| rg(&mut rng)).collect();
        let r_j: Vec<Gf128> = (0..3).map(|_| rg(&mut rng)).collect();
        let mut y_claim = ZERO;
        for j in 0..cycles {
            let wj = eq_mle(&r_j, j);
            if wj == ZERO {
                continue;
            }
            let mut cols = ZERO;
            for h in 0..(1usize << h_vars) {
                if sel_flat[h | (j << h_vars)] == ZERO {
                    continue;
                }
                let mut a_sum = ZERO;
                for a in 0..(1usize << a_vars) {
                    // The link factors: (1 + a_hat_m + a_m) per offset var.
                    let mut lam = ONE;
                    for (m, ob) in off_bits.iter().enumerate() {
                        let ah = Gf128(ob[j] as u128);
                        let am = Gf128(((a >> m) & 1) as u128);
                        lam = mul(lam, ONE + ah + am);
                    }
                    let mut ou = ZERO;
                    for o in 0..(1usize << o_vars) {
                        for u in 0..(1usize << u_vars) {
                            let rv = values[o
                                | (u << o_vars)
                                | (h << (o_vars + u_vars))
                                | (a << (o_vars + u_vars + h_vars))];
                            if rv != ZERO {
                                ou += mul(mul(rv, eq_mle(&r_o, o)), x[u + j * (1usize << u_vars)]);
                            }
                        }
                    }
                    a_sum += mul(lam, ou);
                }
                cols += a_sum;
            }
            y_claim += mul(wj, cols);
        }
        let mut t = Transcript::new("router", &[b"t"]);
        let proof = prove_routing_join(&layout, y_claim, &r_o, &r_j, &tensor, &w, &mut t);
        let mut tv = Transcript::new("router", &[b"t"]);
        let jv = layout.joint_vars();
        // The joint layout: [a | o | h | u | j] (see RoutingLayout::offsets).
        let [o_off, j_off, u_off, h_off, a_off] = layout.offsets();
        // The act sequence follows the TENSOR's variable order (o, u, h,
        // a), each at its joint-cube position (see the shift case above).
        let mut act: Vec<usize> = (0..o_vars).map(|i| o_off + i).collect();
        act.extend(u_off..u_off + u_vars);
        act.extend(h_off..h_off + h_vars);
        act.extend(a_off..a_off + a_vars);
        let r_pub = PaddedFactor::new(values.clone(), act, jv);
        let _ = j_off;
        let res = magnetar_router::verify_routing(
            &layout, y_claim, &r_o, &r_j, &r_pub, 1, a_vars, &proof, &mut tv,
        );
        // The claim convention may differ in the a-link term; the
        // definitive check is that the honest proof verifies OR the
        // failure is a claim-convention mismatch — but a round-sum
        // failure (a message identity) must NEVER happen. We accept
        // only Ok or a non-round error.
        match res {
            Ok(_) => {}
            Err("round sum check failed") => panic!("seed {seed}: round identity broken"),
            Err(e) => panic!("seed {seed}: unexpected verify error: {e}"),
        }
    }
}
