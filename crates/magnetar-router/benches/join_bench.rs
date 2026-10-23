//! Timing: the join-enumeration prover vs the generic sparse prover on a
//! store-router instance at T = 2^12 (run with `cargo bench`-style timing
//! via `cargo test --release --bench join_bench -- --nocapture`).

use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_router::*;

#[test]
fn bench_routers() {
    let j_vars = 12usize;
    let total = 1usize << j_vars;
    let (values, o_vars, u_vars, h_vars, a_vars) = build_store_tensor(64);
    // Sparse witness: half the cycles are stores (the realistic ratio).
    let mut x = vec![ZERO; (1usize << u_vars) * total];
    let mut q = vec![ZERO; 4 * total];
    let mut off = vec![ZERO; 3 * total];
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for j in 0..total {
        for u in 0..(1usize << u_vars) {
            x[u + j * (1usize << u_vars)] = Gf128((next() & 1) as u128);
        }
        if j % 2 == 0 {
            let h = (next() % 4) as usize;
            q[h + j * 4] = ONE;
            let a = (next() % 8) as usize;
            off[j * 3] = Gf128((a & 1) as u128);
            off[1 + j * 3] = Gf128(((a >> 1) & 1) as u128);
            off[2 + j * 3] = Gf128(((a >> 2) & 1) as u128);
        }
    }
    let off_kmajor: Vec<Gf128> = (0..3 * total)
        .map(|i| off[(i % total) * 3 + i / total])
        .collect();
    let r_o: Vec<Gf128> = (0..o_vars)
        .map(|i| Gf128(0x1111_0000_0000_0001u64.wrapping_add(i as u64) as u128))
        .collect();
    let r_j: Vec<Gf128> = (0..j_vars)
        .map(|i| Gf128(0x2222_0000_0000_0001u64.wrapping_add(i as u64) as u128))
        .collect();
    let layout = RoutingLayout {
        o_vars,
        j_vars,
        u_vars,
        h_vars,
        a_vars,
    };
    let jv = layout.joint_vars();
    let [_o_off, _j_off, _u_off, _h_off, _a_off] = layout.offsets();
    let claim = ZERO; // For the sparse prover, the claim must be exact; the
                      // join prover debug-asserts round 0 sums to it. Compute the true claim
                      // cheaply: the join prover's own round-0 sum IS the claim — run the
                      // join first without the assert by using a release build (debug_assert
                      // off) and reading... simpler: compute via the join once with the
                      // assert bypassed by matching the running total.
                      // -> Use the sparse prover to compute the claim from a smaller path:
                      // actually compute the claim by summing over the (u, h, a, j) support.
    let mut cl = ZERO;
    let eqw = |r: &[Gf128], b: usize| -> Gf128 {
        let mut v = ONE;
        for (i, &ri) in r.iter().enumerate() {
            v = if (b >> i) & 1 == 1 {
                mul(v, ri)
            } else {
                mul(v, ONE + ri)
            };
        }
        v
    };
    for j in 0..total {
        let h = match (0..4).find(|&h| q[h + j * 4] == ONE) {
            Some(h) => h,
            None => continue,
        };
        let e_j = eqw(&r_j, j);
        for a in 0..8usize {
            let mut lk = ONE;
            for k in 0..3 {
                let ak = (a >> k) & 1;
                let ahat = off[k + j * 3];
                lk = mul(lk, ONE + Gf128(ak as u128) + ahat);
            }
            for o in 0..64usize {
                let u0 = o;
                let u1 = o + 64;
                let rv0 = values[o
                    | (u0 << o_vars)
                    | (h << (o_vars + u_vars))
                    | (a << (o_vars + u_vars + h_vars))];
                let rv1 = values[o
                    | (u1 << o_vars)
                    | (h << (o_vars + u_vars))
                    | (a << (o_vars + u_vars + h_vars))];
                if rv0 != ZERO {
                    cl += mul(
                        mul(mul(mul(eqw(&r_o, o), e_j), lk), rv0),
                        x[u0 + j * (1usize << u_vars)],
                    );
                }
                if rv1 != ZERO {
                    cl += mul(
                        mul(mul(mul(eqw(&r_o, o), e_j), lk), rv1),
                        x[u1 + j * (1usize << u_vars)],
                    );
                }
            }
        }
    }
    let _ = claim;
    // Join.
    let tensor = RoutingTensor::from_parts(values.clone(), o_vars, u_vars, h_vars, a_vars);
    let sels = [SelCol {
        table: &q,
        h_off: 0,
        h_vars,
    }];
    let wit = JoinWitness {
        x: &x,
        selectors: &sels,
        offset_bits: &off_kmajor,
    };
    let t0 = std::time::Instant::now();
    let mut t = Transcript::new("bench", &[b"t"]);
    let _join = prove_routing_join(&layout, cl, &r_o, &r_j, &tensor, &wit, &mut t);
    let t_join = t0.elapsed();
    println!("join  T={total}: {t_join:?}");
    // Sparse (skipped at large T — it does not finish in reasonable time;
    // measure at T = 2^8 instead and extrapolate).
    let j_vars_s = 8usize;
    let t_small = 1usize << j_vars_s;
    let xs = &x[..(1usize << u_vars) * t_small];
    let qs = &q[..4 * t_small];
    let _offs: Vec<Gf128> = (0..3 * t_small)
        .map(|i| off[(i % t_small) * 3 + i / t_small])
        .collect();
    let layout_s = RoutingLayout {
        o_vars,
        j_vars: j_vars_s,
        u_vars,
        h_vars,
        a_vars,
    };
    let jvs = layout_s.joint_vars();
    let [o_off, j_off, u_off, h_off, a_off] = layout_s.offsets();
    let cls = {
        let mut c = ZERO;
        for j in 0..t_small {
            let h = match (0..4).find(|&h| qs[h + j * 4] == ONE) {
                Some(h) => h,
                None => continue,
            };
            let e_j = eqw(&r_j[..j_vars_s], j);
            for a in 0..8usize {
                let mut lk = ONE;
                for k in 0..3 {
                    let ak = (a >> k) & 1;
                    let ahat = off[k + j * 3];
                    lk = mul(lk, ONE + Gf128(ak as u128) + ahat);
                }
                for o in 0..64usize {
                    for s in 0..2usize {
                        let u = o + 64 * s;
                        let rv = values[o
                            | (u << o_vars)
                            | (h << (o_vars + u_vars))
                            | (a << (o_vars + u_vars + h_vars))];
                        if rv != ZERO {
                            c += mul(
                                mul(mul(mul(eqw(&r_o, o), e_j), lk), rv),
                                xs[u + j * (1usize << u_vars)],
                            );
                        }
                    }
                }
            }
        }
        c
    };
    // Sparse factors in own spaces.
    let x_fac = PaddedDense::new(
        xs.to_vec(),
        {
            let mut a: Vec<usize> = (u_off..u_off + u_vars).collect();
            a.extend(j_off..j_off + j_vars_s);
            a
        },
        jvs,
    );
    let q_fac = PaddedDense::new(
        qs.to_vec(),
        {
            let mut a: Vec<usize> = (h_off..h_off + h_vars).collect();
            a.extend(j_off..j_off + j_vars_s);
            a
        },
        jvs,
    );
    let mut dense = vec![x_fac, q_fac];
    for k in 0..3 {
        let mut tb = vec![ZERO; 2 * t_small];
        for j in 0..t_small {
            let ahat = off[k + j * 3];
            tb[2 * j] = ONE + ahat;
            tb[2 * j + 1] = ahat;
        }
        let mut a = vec![a_off + k];
        a.extend(j_off..j_off + j_vars_s);
        dense.push(PaddedDense::new(tb, a, jvs));
    }
    let r_pub = PaddedTableFactor::new(
        values.clone(),
        {
            let mut act: Vec<usize> = (0..o_vars).map(|i| o_off + i).collect();
            act.extend(u_off..u_off + u_vars);
            act.extend(h_off..h_off + h_vars);
            act.extend(a_off..a_off + a_vars);
            act
        },
        jvs,
    );
    let r_js: Vec<Gf128> = r_j[..j_vars_s].to_vec();
    let t0 = std::time::Instant::now();
    let mut t = Transcript::new("bench", &[b"t"]);
    let _sparse = prove_routing(&layout_s, cls, &r_o, &r_js, &r_pub, &dense, 1, &mut t);
    let t_sparse = t0.elapsed();
    let est = (t_sparse.as_nanos() as f64 * (total as f64 / t_small as f64)
        / t_join.as_nanos() as f64) as u64;
    println!("sparse T={t_small}: {t_sparse:?}  (extrapolated ~{est}x slower at T={total})");
    let _ = (o_off, jv, o_vars, layout.joint_vars());
}
