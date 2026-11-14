//! Profile the Virtual-Spartan outer paths at the exact erc20 shape
//! (q = 2^12 instances, rows = 2^9, joint = 21, window v = 6):
//! where does `vs-win-build` go — the eq materialize vs the pass — and
//! what the fld-outer (generic engine, full eq table) costs per round.
//! Pure timing; correctness is cross-validated by the suite's tests.

use magnetar_cpu::WITNESS_LAYOUT;
use magnetar_field::{Gf128, ONE, ZERO};
use magnetar_spartan::virtual_spartan::DualRelation;
use magnetar_sumcheck::public::EqFactor;
use magnetar_sumcheck::{BindOrder, PublicFactor, SumcheckProver, SumcheckSpec, Term};

fn rg(rng: &mut impl rand::Rng) -> Gf128 {
    Gf128(rand::RngExt::random::<u128>(rng))
}

#[test]
fn profile_window_erc20_shape() {
    let mut rng = rand::rng();
    // The erc20 shape at T = 2^12, cpi = 1.
    let q: usize = 1 << 12;
    let s: usize = 12;
    let rows: usize = 512;
    let row_vars: usize = 9;
    let joint_vars: usize = s + row_vars; // 21 (bin)
    let m: usize = WITNESS_LAYOUT::M; // 4096 columns
    let w_cols = m / 2;
    let w128 = w_cols.div_ceil(128);

    // A realistic packed witness: q instances x w128 u128 words.
    let w_packed: Vec<u128> = (0..q * w128)
        .map(|i| rand::RngExt::random::<u128>(&mut rng) ^ (i as u128))
        .collect();
    // Word-valued public x columns (2 per instance).
    let x_cols: Vec<Gf128> = (0..q * 2).map(|_| rg(&mut rng)).collect();

    // The real circuit matrices (cpi = 1).
    let spec = magnetar_cpu::build_circuit();
    let row_vars_fld = spec.a_fld.rows.trailing_zeros() as usize;
    let joint_vars_fld = s + row_vars_fld;
    println!("fld geometry: rows_fld={} joint_vars_fld={}", spec.a_fld.rows, joint_vars_fld);
    let rel = DualRelation {
        m,
        a_bin: spec.a_bin.clone(),
        b_bin: spec.b_bin.clone(),
        c_bin: spec.c_bin.clone(),
        a_fld: spec.a_fld.clone(),
        b_fld: spec.b_fld.clone(),
        c_fld: spec.c_fld.clone(),
    };
    println!(
        "circuit: rows={} (bin nnz {} / fld nnz {}), m={}",
        rel.a_bin.rows,
        rel.a_bin.entries.len() + rel.b_bin.entries.len() + rel.c_bin.entries.len(),
        rel.a_fld.entries.len() + rel.b_fld.entries.len() + rel.c_fld.entries.len(),
        m
    );

    // ---------- 1. vs-apply: the mask build (bin) ----------
    let v = magnetar_sumcheck::window_probe(s).min(6);
    let t0 = std::time::Instant::now();
    let m0 = rel.a_bin.apply_simd_bit_masks(&w_packed, q, w_cols, s, v);
    let m1 = rel.b_bin.apply_simd_bit_masks(&w_packed, q, w_cols, s, v);
    let m2 = rel.c_bin.apply_simd_bit_masks(&w_packed, q, w_cols, s, v);
    println!("mask build (3 tables): {:?}", t0.elapsed());
    let (m0, m1, m2) = (m0.unwrap(), m1.unwrap(), m2.unwrap());
    let stride = m0.len();
    let dens = |mm: &[u64]| {
        mm.iter().map(|w| w.count_ones() as f64).sum::<f64>() / (mm.len() as f64 * 64.0)
    };
    println!(
        "stride={} densities: d0={:.3} d1={:.3} d2={:.3}",
        stride,
        dens(&m0),
        dens(&m1),
        dens(&m2)
    );

    // ---------- 2. eq tables: full vs suffix ----------
    let tau: Vec<Gf128> = (0..joint_vars).map(|_| rg(&mut rng)).collect();
    let t0 = std::time::Instant::now();
    let eq_full = EqFactor::new(tau.clone()).materialize();
    let t_eq_full = t0.elapsed();
    println!(
        "eq FULL (2^{joint_vars} = {} entries, {} MiB): {:?}",
        eq_full.len(),
        eq_full.len() * 16 / 1024 / 1024,
        t_eq_full
    );
    let t0 = std::time::Instant::now();
    let eq_suffix = magnetar_field::EqBits::build(&tau[..joint_vars - v]).values;
    let t_eq_suffix = t0.elapsed();
    println!(
        "eq SUFFIX (2^{} = {} entries, {} KiB): {:?}",
        joint_vars - v,
        eq_suffix.len(),
        eq_suffix.len() * 16 / 1024,
        t_eq_suffix
    );

    // ---------- 3. the window construction (current = full eq) ----------
    let spec_w = SumcheckSpec {
        num_vars: joint_vars,
        dense: vec![Vec::new(), Vec::new(), Vec::new()],
        public: vec![Box::new(EqFactor::new(tau.clone()))],
        terms: vec![
            Term { coeff: ONE, dense: vec![0, 1], public: vec![0] },
            Term { coeff: ONE, dense: vec![2], public: vec![0] },
        ],
        claim: ZERO,
    };
    let order = BindOrder::HighBitsFirst { high: s };
    let mut best = std::time::Duration::from_secs(100);
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let p = SumcheckProver::with_mask_window(
            &spec_w,
            order,
            vec![m0.clone(), m1.clone(), m2.clone()],
        );
        let el = t0.elapsed();
        if el < best {
            best = el;
        }
        drop(p);
    }
    println!(
        "with_mask_window construction (materialize + pass), best of 3: {:?}  => pass alone ≈ {:?}",
        best,
        best - t_eq_full
    );

    // Rounds through the whole protocol (warm path).
    let mut p = SumcheckProver::with_mask_window(&spec_w, order, vec![m0, m1, m2]);
    let mut msg_t = std::time::Duration::ZERO;
    let mut fold_t = std::time::Duration::ZERO;
    let mut tr = magnetar_hash::Transcript::new("prof", &[]);
    for _ in 0..joint_vars {
        let t0 = std::time::Instant::now();
        let mmsg = p.message();
        msg_t += t0.elapsed();
        let t1 = std::time::Instant::now();
        let r = tr.challenge_field("r");
        p.fold_msg(&mmsg, r);
        fold_t += t1.elapsed();
    }
    println!(
        "bin outer total rounds: message {:?} + fold {:?} (incl. the 6 window rounds)",
        msg_t, fold_t
    );

    // ---------- 4. the fld outer (generic engine, full eq fold) ----------
    let t0 = std::time::Instant::now();
    let az_fld = rel.a_fld.apply_simd_packed(&w_packed, &x_cols, q, w_cols).0;
    let bz_fld = rel.b_fld.apply_simd_packed(&w_packed, &x_cols, q, w_cols).0;
    let cz_fld = rel.c_fld.apply_simd_packed(&w_packed, &x_cols, q, w_cols).0;
    println!("fld tables build (3 x 2^{joint_vars}): {:?}", t0.elapsed());

    let tau_fld: Vec<Gf128> = (0..joint_vars_fld).map(|_| rg(&mut rng)).collect();
    let spec_f = SumcheckSpec {
        num_vars: joint_vars_fld,
        dense: vec![az_fld, bz_fld, cz_fld],
        public: vec![Box::new(EqFactor::new(tau_fld.clone()))],
        terms: vec![
            Term { coeff: ONE, dense: vec![0, 1], public: vec![0] },
            Term { coeff: ONE, dense: vec![2], public: vec![0] },
        ],
        claim: ZERO,
    };
    let t0 = std::time::Instant::now();
    let mut pf = SumcheckProver::new(&spec_f, order);
    println!(
        "fld outer construction (eq materialize over 2^{joint_vars_fld}): {:?}",
        t0.elapsed()
    );
    let mut msg_t = std::time::Duration::ZERO;
    let mut fold_t = std::time::Duration::ZERO;
    let mut tr = magnetar_hash::Transcript::new("prof", &[]);
    for i in 0..joint_vars_fld {
        let t0 = std::time::Instant::now();
        let mmsg = pf.message();
        msg_t += t0.elapsed();
        let t1 = std::time::Instant::now();
        let r = tr.challenge_field("r");
        pf.fold_msg(&mmsg, r);
        fold_t += t1.elapsed();
        if i == 0 {
            println!("fld round 0 (largest cube) message: {:?}", t0.elapsed());
        }
    }
    println!(
        "fld outer total rounds: message {:?} + fold {:?}",
        msg_t, fold_t
    );
    let _ = ZERO;
}
