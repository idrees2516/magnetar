//! Algorithm 3 tests — dense E-arithmetic references for every layer, the
//! end-to-end roundtrip on a real permutation, and the message tamper
//! classes.

use super::*;
use crate::gf8::Gf8;
use crate::interp::InterpData;
use crate::planes::PlaneSet;

thread_local! {
    static RNG: std::cell::Cell<u128> = const { std::cell::Cell::new(0x5eed_1a7e_0ddb_a111) };
}

fn gf() -> Gf128 {
    RNG.with(|c| {
        let mut r = c.get();
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        c.set(r);
        Gf128(r)
    })
}

fn test_transcript() -> Transcript {
    Transcript::new("mikros-perm-test", &[])
}

// ---------------------------------------------------------------------------
// The cubic sum-check against a dense reference
// ---------------------------------------------------------------------------

#[test]
fn cubic_matches_dense_reference() {
    let l = 16usize;
    let mut seed = 0x1234_5678_9abc_def0u128;
    let mut next = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        Gf128(seed)
    };
    let xi0: Vec<Gf128> = (0..l).map(|_| next()).collect();
    let cu0: Vec<Gf128> = (0..l).map(|_| next()).collect();
    let cd0: Vec<Gf128> = (0..l).map(|_| next()).collect();
    // the true claim
    let mut v = ZERO;
    for i in 0..l {
        v += mul(mul(xi0[i], cu0[i]), cd0[i]);
    }
    let mut t = test_transcript();
    let proof = cubic_prove(
        &mut xi0.clone(),
        &mut cu0.clone(),
        &mut cd0.clone(),
        v,
        &mut t,
    );

    // the dense reference for every round message (fold at the recorded λ)
    let mut ref_rounds = Vec::new();
    {
        let mut xi = xi0.clone();
        let mut cu = cu0.clone();
        let mut cd = cd0.clone();
        let mut len = l;
        let mut round_idx = 0usize;
        while len > 1 {
            let half = len / 2;
            let (xil, xih) = xi.split_at(half);
            let (cul, cuh) = cu.split_at(half);
            let (cdl, cdh) = cd.split_at(half);
            let xit: Vec<Gf128> = (0..half).map(|i| xil[i] + xih[i]).collect();
            let cut: Vec<Gf128> = (0..half).map(|i| cul[i] + cuh[i]).collect();
            let cdt: Vec<Gf128> = (0..half).map(|i| cdl[i] + cdh[i]).collect();
            let mut q = [ZERO; 4];
            for i in 0..half {
                q[0] += mul(mul(xil[i], cul[i]), cdl[i]);
                q[3] += mul(mul(xit[i], cut[i]), cdt[i]);
                q[1] = q[1]
                    + mul(mul(xit[i], cul[i]), cdl[i])
                    + mul(mul(xil[i], cut[i]), cdl[i])
                    + mul(mul(xil[i], cul[i]), cdt[i]);
                q[2] = q[2]
                    + mul(mul(xit[i], cut[i]), cdl[i])
                    + mul(mul(xit[i], cul[i]), cdt[i])
                    + mul(mul(xil[i], cut[i]), cdt[i]);
            }
            ref_rounds.push(q);
            let lambda = proof.challenges[round_idx];
            let mut nxi = vec![ZERO; half];
            let mut ncu = vec![ZERO; half];
            let mut ncd = vec![ZERO; half];
            for i in 0..half {
                nxi[i] = xil[i] + mul(lambda, xit[i]);
                ncu[i] = cul[i] + mul(lambda, cut[i]);
                ncd[i] = cdl[i] + mul(lambda, cdt[i]);
            }
            xi = nxi;
            cu = ncu;
            cd = ncd;
            len = half;
            round_idx += 1;
        }
        assert_eq!(proof.v_child_u, cu[0], "child U");
        assert_eq!(proof.v_child_d, cd[0], "child D");
    }
    for (got, want) in proof.rounds.iter().zip(ref_rounds.iter()) {
        assert_eq!(got, want, "cubic round message");
    }

    // verify
    let mut tv = test_transcript();
    let (y, v_u, v_d) = cubic_verify(v, &xi0, &proof, &mut tv).expect("verify");
    assert_eq!(v_u, proof.v_child_u);
    assert_eq!(v_d, proof.v_child_d);
    assert_eq!(y, proof.challenges);

    // tamper one round coefficient → reject
    let mut bad = proof.clone();
    bad.rounds[1][2] += ONE;
    let mut tb = test_transcript();
    assert!(
        cubic_verify(v, &xi0, &bad, &mut tb).is_err(),
        "round tamper"
    );
    // tamper a child value → the terminal product check fires
    let mut bad2 = proof.clone();
    bad2.v_child_u += ONE;
    let mut tb2 = test_transcript();
    assert!(
        cubic_verify(v, &xi0, &bad2, &mut tb2).is_err(),
        "child tamper"
    );
    // a wrong claim → the first round sum fails
    let mut tb3 = test_transcript();
    assert!(
        cubic_verify(v + ONE, &xi0, &proof, &mut tb3).is_err(),
        "wrong claim"
    );
}

// ---------------------------------------------------------------------------
// The coefficient tree against the dense E-arithmetic reference
// ---------------------------------------------------------------------------

/// The dense reference tree: `C_τ[r][j]` as coefficient vectors over E.
fn dense_tree(ps: &PlaneSet, m: usize) -> Vec<Vec<Vec<Vec<Gf128>>>> {
    let bit = |j: usize, r: usize| -> Gf128 {
        if ps.bit(j, r) {
            ONE
        } else {
            ZERO
        }
    };
    let mut levels = Vec::with_capacity(LOG_KAPPA + 1);
    let mut c0 = vec![vec![vec![ZERO; 2]; KAPPA]; m];
    for r in 0..m {
        for j in 0..KAPPA {
            c0[r][j][0] = bit(j, r);
            c0[r][j][1] = ONE;
        }
    }
    levels.push(c0);
    for tau in 0..LOG_KAPPA {
        let prev = &levels[tau];
        let rows_in = prev.len();
        let rows_out = rows_in / 2;
        let ncoef_out = (1 << (tau + 1)) + 1;
        let mut cur = vec![vec![vec![ZERO; ncoef_out]; KAPPA]; rows_out];
        for r in 0..rows_out {
            for j in 0..KAPPA {
                for d1 in 0..prev[r][j].len() {
                    for d2 in 0..prev[r + rows_out][j].len() {
                        let v1 = prev[r][j][d1];
                        let v2 = prev[r + rows_out][j][d2];
                        if v1 != ZERO && v2 != ZERO {
                            cur[r][j][d1 + d2] += mul(v1, v2);
                        }
                    }
                }
            }
        }
        levels.push(cur);
    }
    levels
}

#[test]
fn tree_matches_dense_reference() {
    let m = 128usize; // N = 2^14
    let mut a = PlaneSet::zeros(m, KAPPA);
    for j in 0..KAPPA {
        for i in 0..m {
            a.set_bit(j, i, gf().0 & 1 == 1);
        }
    }
    let tree = CoefTree::build(&a);
    let dense = dense_tree(&a, m);
    // EXHAUSTIVE comparison at this scale (the sampled variant missed a
    // row-duplication bug once — never again)
    for tau in 0..=LOG_KAPPA {
        let rows = m >> tau;
        let ncoef = (1 << tau) + 1;
        for r in 0..rows {
            for j in 0..KAPPA {
                for d in 0..ncoef {
                    let got = tree.level_bit(tau, j, d, r);
                    let want = dense[tau][r][j][d] == ONE;
                    assert_eq!(got, want, "C_{tau}[d={d}][r={r}][j={j}]");
                }
            }
        }
    }
    // c₇(z) against the dense evaluation
    let z = gf();
    let c7 = tree.flatten_top(z);
    {
        let top = &dense[LOG_KAPPA];
        let rows = m >> LOG_KAPPA;
        for j in 0..KAPPA {
            for r in 0..rows {
                let mut acc = ZERO;
                let mut p = ONE;
                for d in 0..top[r][j].len() {
                    if top[r][j][d] == ONE {
                        acc += p;
                    }
                    p = mul(p, z);
                }
                assert_eq!(c7[j * rows + r], acc, "c₇(z) entry ({j},{r})");
            }
        }
    }
    // Θ = Π_i (z + a_i) — the direct grand product over all N entries
    {
        let mut theta = ONE;
        for j in 0..KAPPA {
            for r in 0..m {
                let term = z + if a.bit(j, r) { ONE } else { ZERO };
                theta = mul(theta, term);
            }
        }
        let mut cur = c7.clone();
        while cur.len() > 1 {
            let half = cur.len() / 2;
            let mut next = Vec::with_capacity(half);
            for i in 0..half {
                next.push(mul(cur[i], cur[half + i]));
            }
            cur = next;
        }
        assert_eq!(cur[0], theta, "the E-tree root equals Π(z + a_i)");
    }
}

#[test]
fn contract_g_matches_dense() {
    let m = 128usize;
    let mut a = PlaneSet::zeros(m, KAPPA);
    for j in 0..KAPPA {
        for i in 0..m {
            a.set_bit(j, i, gf().0 & 1 == 1);
        }
    }
    let tree = CoefTree::build(&a);
    let z = gf();
    let mut zp = Vec::with_capacity(KAPPA + 1);
    {
        let mut p = ONE;
        for _ in 0..=KAPPA {
            zp.push(p);
            p = mul(p, z);
        }
    }
    let rho: Vec<Gf128> = (0..KAPPA).map(|_| gf()).collect();
    for tau in 0..=LOG_KAPPA {
        let rows = m >> tau;
        let ncoef = (1 << tau) + 1;
        let g = contract_g(&tree, tau, &rho, &zp);
        assert_eq!(g.len(), rows);
        for r in 0..rows {
            // dense: Σ_j ρ_j·Σ_d C_τ[d][r][j]·z^d
            let mut want = ZERO;
            for j in 0..KAPPA {
                for d in 0..ncoef {
                    if tree.level_bit(tau, j, d, r) {
                        want += mul(rho[j], zp[d]);
                    }
                }
            }
            assert_eq!(g[r], want, "contract_g level {tau} row {r}");
        }
    }
}

// ---------------------------------------------------------------------------
// The end-to-end protocol
// ---------------------------------------------------------------------------

fn perm_planes(m: usize, seed: u128) -> (PlaneSet, PlaneSet) {
    // a: random planes; b: a with a random row permutation applied.
    // (xorshift — an LCG's low bit has period 2 and makes the witness
    // row-degenerate, which once masked a real convention bug)
    let mut a = PlaneSet::zeros(m, KAPPA);
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        Gf128(s)
    };
    for j in 0..KAPPA {
        for i in 0..m {
            a.set_bit(j, i, next().0 & 1 == 1);
        }
    }
    let mut sigma: Vec<usize> = (0..m).collect();
    for i in (1..m).rev() {
        let j = (next().0 % ((i + 1) as u128)) as usize;
        sigma.swap(i, j);
    }
    let mut b = PlaneSet::zeros(m, KAPPA);
    for j in 0..KAPPA {
        for r in 0..m {
            b.set_bit(j, r, a.bit(j, sigma[r]));
        }
    }
    (a, b)
}

#[test]
fn perm_roundtrip_and_terminals() {
    let f8 = Gf8::new();
    let mut data = InterpData::new(&f8);
    data.build_psi_monomials(&f8);
    let m = 128usize; // N = 2^14 — the minimum shape (m = κ)
    let (a, b) = perm_planes(m, 0xa1b2c3d4e5f60718);

    let mut t = test_transcript();
    let (proof, terminal, _stats) = prove_perm(&f8, &data, &a, &b, &mut t);
    let mut tv = test_transcript();
    let vterm = verify_perm(&f8, &data, &proof, &mut tv).expect("the honest proof must verify");
    for op in 0..2 {
        assert_eq!(vterm.claims[op].0, terminal.claims[op].0, "claim binding");
        assert_eq!(
            vterm.claims[op].1, terminal.claims[op].1,
            "l₀ factor binding"
        );
        assert_eq!(vterm.claims[op].2, terminal.claims[op].2, "ρ₀ binding");
    }
    // the terminal claims against the dense evaluation:
    // v₀ − z = l₀^⊤ D ρ(x) for D ∈ {a, b}
    for (op, ps) in [&a, &b].iter().enumerate() {
        let (claim, factors, rho) = &terminal.claims[op];
        let l0 = expand_factors(factors);
        assert_eq!(l0.len(), m);
        let mut want = ZERO;
        for r in 0..m {
            let mut dr = ZERO;
            for j in 0..KAPPA {
                if ps.bit(j, r) {
                    dr += rho[j];
                }
            }
            want += mul(l0[r], dr);
        }
        assert_eq!(*claim, want, "terminal claim {op}: v₀ − z = l₀^⊤ D ρ₀");
    }
}

#[test]
fn perm_tamper_rejected() {
    let f8 = Gf8::new();
    let mut data = InterpData::new(&f8);
    data.build_psi_monomials(&f8);
    let m = 128usize;
    let (a, b) = perm_planes(m, 0xfeed_beef_c0de_1234);
    let mut t = test_transcript();
    let (proof, _terminal, _stats) = prove_perm(&f8, &data, &a, &b, &mut t);

    // Θ tamper: the E-tree round sums fail
    {
        let mut bad = clone_perm(&proof);
        bad.theta += ONE;
        let mut tv = test_transcript();
        assert!(
            verify_perm(&f8, &data, &bad, &mut tv).is_err(),
            "theta tamper"
        );
    }
    // u tamper: the column check v = 〈u, r⟩ fails
    {
        let mut bad = clone_perm(&proof);
        bad.unfolds[0].f_trans[3].u[5] += ONE;
        let mut tv = test_transcript();
        assert!(verify_perm(&f8, &data, &bad, &mut tv).is_err(), "u tamper");
    }
    // h tamper: the cubic identity check fails
    {
        let mut bad = clone_perm(&proof);
        bad.unfolds[1].f_trans[2].h[9] += ONE;
        let mut tv = test_transcript();
        assert!(verify_perm(&f8, &data, &bad, &mut tv).is_err(), "h tamper");
    }
    // cubic round tamper
    {
        let mut bad = clone_perm(&proof);
        if let Some(r) = bad.unfolds[0].e_tree[3].rounds.first_mut() {
            r[1] += ONE;
        }
        let mut tv = test_transcript();
        assert!(
            verify_perm(&f8, &data, &bad, &mut tv).is_err(),
            "e-tree round tamper"
        );
    }
    // child-value tamper (an F-transition's cubic)
    {
        let mut bad = clone_perm(&proof);
        bad.unfolds[0].f_trans[6].cubic.v_child_u += ONE;
        let mut tv = test_transcript();
        assert!(
            verify_perm(&f8, &data, &bad, &mut tv).is_err(),
            "child tamper"
        );
    }
}

fn clone_perm(p: &PermProof) -> PermProof {
    PermProof {
        theta: p.theta,
        unfolds: [
            OperandUnfold {
                e_tree: p.unfolds[0].e_tree.clone(),
                f_trans: p.unfolds[0].f_trans.clone(),
            },
            OperandUnfold {
                e_tree: p.unfolds[1].e_tree.clone(),
                f_trans: p.unfolds[1].f_trans.clone(),
            },
        ],
    }
}
