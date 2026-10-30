//! The Mikros §6 benchmark suite (paper Tables 3–4), on this stack's fields.
//!
//! * `mikros-bench hadamard --log2-n 30` — the binary Hadamard-product proof
//!   of length `2^30` (Table 3): single-threaded PIOP prover, the verifier,
//!   the proof size, and peak RSS, with the per-phase cost anatomy.
//! * `mikros-bench hashes [--batch 256] [--which all|sha256|keccak|blake3]`
//!   — end-to-end packed R1CS batches of hash computations (Table 4):
//!   SHA-256 compression, Keccak-f[1600], BLAKE3 compression. The batch is
//!   ν = ceil(batch/128) packed Algorithm-6 proofs of κ = 128 instances.
//! * `mikros-bench perm --log2-n 20` — the packed permutation argument
//!   (§4.3/Algorithm 3, the Table-5 "excl.-perm." complement) with its
//!   per-phase cost anatomy.
//! * `mikros-bench plonk --log2-n 16 [--seed 42]` — the single-instance
//!   PLONK SNARK (Algorithm 5, Table 5): a simulated binary circuit, the
//!   full gate + copy + grouped-opening stack with its cost anatomy.
//!
//! All runs are single-threaded, matching the paper's protocol.

use std::time::Instant;

use magnetar_field::Gf128;
use magnetar_field::{mul, ZERO};
use magnetar_hash::Transcript;
use magnetar_mikros::batch::{prove_batch, setup_batch, verify_batch, BatchProof};
use magnetar_mikros::circuits::{
    blake3_compress, keccak_f1600, sha256_compression, CircuitBuilder,
};
use magnetar_mikros::gf8::Gf8;
use magnetar_mikros::hadamard::{prove_hadamard_timed, verify_hadamard};
use magnetar_mikros::interp::InterpData;
use magnetar_mikros::perm::{prove_perm, verify_perm};
use magnetar_mikros::planes::{PlaneSet, RowWeightTensor};
use magnetar_mikros::spartan::{prove_spartan, verify_spartan, SpartanProof};
use magnetar_pcs::eval::{PcsParams, PcsProof};

fn peak_rss_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .unwrap_or(0);
            return kb * 1024;
        }
    }
    0
}

fn fmt_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KiB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

/// A fast xorshift for witness generation (benchmark data only).
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn gf(&mut self) -> Gf128 {
        Gf128((self.next() as u128) | ((self.next() as u128) << 64))
    }
}

/// A deterministic per-instance input word.
fn inst_word(inst: usize, salt: usize) -> u32 {
    let mut z = (inst as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (salt as u64).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    (z ^ (z >> 31)) as u32
}

// ---------------------------------------------------------------------------
// Table 3 — the binary Hadamard product
// ---------------------------------------------------------------------------

fn bench_hadamard(log2_n: usize) {
    assert!(log2_n >= 14, "m = 2^n/128 must be at least 64");
    assert!(
        log2_n <= 30,
        "the benchmark ladder tops at the paper's 2^30"
    );
    let n = 1usize << log2_n;
    let m = n / 128;
    let log_m = log2_n - 7;
    println!(
        "== Mikros Table 3: binary Hadamard product, N = 2^{log2_n} (κ = 128, m = 2^{log_m}) =="
    );

    let f8 = Gf8::new();
    let mut data = InterpData::new(&f8);
    data.build_psi_monomials(&f8);
    println!("interp tables built");

    // Witness planes: κ column-planes of m bits (random).
    let t_gen = Instant::now();
    let mut rng = XorShift(0x1234_5678_9abc_def0);
    let mut a = PlaneSet::zeros(m, 128);
    let mut b = PlaneSet::zeros(m, 128);
    for j in 0..128 {
        let pa = a.plane_mut(j);
        let pb = b.plane_mut(j);
        for w in pa.iter_mut() {
            *w = rng.next();
        }
        for w in pb.iter_mut() {
            *w = rng.next();
        }
    }
    // ξ = ts(x) for a random x ∈ E^N: l = the high log_m factors,
    // r = the low 7 factors (the packing coordinates).
    let xs: Vec<Gf128> = (0..log2_n).map(|_| rng.gf()).collect();
    let l_factors: Vec<(Gf128, Gf128)> = xs[..log_m]
        .iter()
        .map(|&xk| (Gf128(1) + xk, xk)) // ts factors: (1−x, x) → char 2: (1+x, x)
        .collect();
    let r_factors: Vec<(Gf128, Gf128)> =
        xs[log_m..].iter().map(|&xk| (Gf128(1) + xk, xk)).collect();
    let r: Vec<Gf128> = {
        let mut v = vec![Gf128(1)];
        for &(f0, f1) in &r_factors {
            let half = v.len();
            let mut nv = vec![ZERO; half * 2];
            for i in 0..half {
                nv[i] = mul(v[i], f0);
                nv[half + i] = mul(v[i], f1);
            }
            v = nv;
        }
        v
    };
    let l = RowWeightTensor::new(l_factors.clone(), true);
    println!(
        "witness + weights generated in {:.0} ms",
        t_gen.elapsed().as_secs_f64() * 1e3
    );

    // The honest claim c = <d, r> via the bitsliced d-fold (the prover's own
    // step-1; charged to the claim setup, not the proof).
    let t_c = Instant::now();
    let mut d_setup = Vec::with_capacity(128);
    {
        let mut prod = PlaneSet::zeros(m, 128);
        for j in 0..128 {
            prod.and_into(j, &a, &b);
        }
        for j in 0..128 {
            let pj = prod.plane(j);
            let mut bits = 0u128;
            for e in 0..128 {
                bits |= (PlaneSet::parity_dot(pj, l.plane(e)) as u128) << e;
            }
            d_setup.push(Gf128(bits));
        }
    }
    let mut c = ZERO;
    for j in 0..128 {
        c += mul(d_setup[j], r[j]);
    }
    println!(
        "claim c computed in {:.0} ms (the d-fold route)",
        t_c.elapsed().as_secs_f64() * 1e3
    );
    drop(d_setup);

    // Prove (single thread).
    let mut t = Transcript::new("mikros-bench-hadamard", &[&n.to_le_bytes()]);
    let (stats, proof, _) = prove_hadamard_timed(&f8, &data, &a, &b, &l, &r, c, &mut t);
    println!("prover:      {:.2} ms", stats.total_ms);
    println!(
        "  anatomy:   d-fold {:.0} | enc {:.0} | spread {:.0} | bucket {:.0} | fold {:.0} | h {:.1} | a,b {:.0} | fact1 {:.0} (ms)",
        stats.d_fold_ms,
        stats.encodings_ms,
        stats.spreads_ms,
        stats.bucket_ms,
        stats.fold_ms,
        stats.h_ms,
        stats.ab_ms,
        stats.fact1_ms,
    );
    println!(
        "  bits/s:    {:.0}/s over N = {} bits (single thread)",
        n as f64 / (stats.total_ms / 1e3),
        n
    );

    // Verify.
    let tv0 = Instant::now();
    let mut tv = Transcript::new("mikros-bench-hadamard", &[&n.to_le_bytes()]);
    let term = verify_hadamard(&f8, &data, &r, c, &proof, &mut tv).expect("verify");
    let verify_us = tv0.elapsed().as_secs_f64() * 1e6;
    println!(
        "verifier:    {:.0} µs  ({} fact-1 challenges)",
        verify_us,
        term.y.len()
    );

    // Proof size.
    let size = proof.d.len() * 16 + proof.h.len() * 16 + proof.fact1.rounds.len() * 3 * 16 + 2 * 16;
    println!("proof size:  {size} B  (paper Table 3: 4,888 B)");
    println!(
        "peak RSS:    {} (paper: 0.89 GiB)",
        fmt_bytes(peak_rss_bytes())
    );
    println!("paper ref:   684.48 ms prover / 34 µs verifier on a Xeon Gold 6342 (AVX-512 + GFNI)");
}

// ---------------------------------------------------------------------------
// Table 4 — end-to-end packed hash batches (Algorithm 6)
// ---------------------------------------------------------------------------

/// The per-hash bench harness: build the circuit, replay κ witnesses,
/// and prove ν = ceil(batch/128) packed blocks.
fn bench_one_hash(name: &str, batch: usize) {
    use magnetar_mikros::r1cs::BinaryR1cs;

    // The circuit (constraints) built once from a reference instance; the
    // witnesses for each instance come from a replay with fresh inputs.
    let build_cs = || -> BinaryR1cs {
        let mut b = CircuitBuilder::new();
        match name {
            "sha256" => {
                let state: [u32; 8] = [
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                    0x1f83d9ab, 0x5be0cd19,
                ];
                let block: [u32; 16] = [0; 16];
                sha256_compression(&mut b, state, block);
            }
            "keccak" => {
                let mut lanes = [[[0u32; 64]; 5]; 5];
                for lane_row in lanes.iter_mut() {
                    for lane in lane_row.iter_mut() {
                        *lane = b.io_word64(0);
                    }
                }
                keccak_f1600(&mut b, &lanes);
            }
            "blake3" => {
                blake3_compress(&mut b, [0u32; 8], [0u32; 16], 0, 64, 0);
            }
            _ => unreachable!(),
        }
        b.build()
    };
    let make_witness = |inst: usize| -> Vec<bool> {
        let mut rb = CircuitBuilder::replay();
        match name {
            "sha256" => {
                let state: [u32; 8] = [
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                    0x1f83d9ab, 0x5be0cd19,
                ];
                let block: [u32; 16] = core::array::from_fn(|i| inst_word(inst, 0x100 + i));
                sha256_compression(&mut rb, state, block);
            }
            "keccak" => {
                let mut lanes = [[[0u32; 64]; 5]; 5];
                for (y, lane_row) in lanes.iter_mut().enumerate() {
                    for (x, lane) in lane_row.iter_mut().enumerate() {
                        let salt = y * 5 + x;
                        *lane =
                            rb.io_word64((salt as u64) | ((inst_word(inst, salt) as u64) << 32));
                    }
                }
                keccak_f1600(&mut rb, &lanes);
            }
            "blake3" => {
                let cv: [u32; 8] = core::array::from_fn(|i| inst_word(inst, 0x200 + i));
                let block: [u32; 16] = core::array::from_fn(|i| inst_word(inst, 0x300 + i));
                blake3_compress(&mut rb, cv, block, inst as u64, 64, 0);
            }
            _ => unreachable!(),
        }
        rb.w
    };

    let t_cs = Instant::now();
    let cs = build_cs();
    let build_ms = t_cs.elapsed().as_secs_f64() * 1e3;
    let n = cs.s_l.n;
    println!(
        "  circuit: {} rows | N = 2^{} | io bits {} | built in {:.0} ms",
        cs.n_constraints,
        n.trailing_zeros(),
        cs.io_positions.len(),
        build_ms
    );

    let f8 = Gf8::new();
    let mut data = InterpData::new(&f8);
    data.build_psi_monomials(&f8);
    let params = PcsParams::default();

    let blocks = batch.div_ceil(128);
    let mut prove_ms = 0.0f64;
    let mut verify_ms = 0.0f64;
    let mut setup_ms = 0.0f64;
    let mut proof_bytes = 0usize;
    // BinarySpartan (the production SNARK path) on the same instances.
    let mut sp_prove_ms = 0.0f64;
    let mut sp_verify_ms = 0.0f64;
    let mut sp_proof_bytes = 0usize;
    for block in 0..blocks {
        // κ witnesses by replay
        let witnesses: Vec<Vec<bool>> = (0..128).map(|j| make_witness(block * 128 + j)).collect();
        for w in &witnesses {
            assert!(cs.satisfied_by(w), "instance must satisfy the circuit");
        }
        let packed_io = magnetar_mikros::batch::packed_io_words(&cs, &witnesses);

        let ts = Instant::now();
        let setup = setup_batch(&cs, &witnesses, &params);
        setup_ms += ts.elapsed().as_secs_f64() * 1e3;

        let tag = format!("mikros-bench-{name}");
        let mut t = Transcript::new(&tag, &[&(block as u64).to_le_bytes()]);
        let tp = Instant::now();
        let proof = prove_batch(&f8, &data, &cs, &setup, &params, &packed_io, &mut t);
        prove_ms += tp.elapsed().as_secs_f64() * 1e3;

        let mut tv = Transcript::new(&tag, &[&(block as u64).to_le_bytes()]);
        let tv0 = Instant::now();
        verify_batch(&f8, &data, &cs, &params, &packed_io, &proof, &mut tv)
            .unwrap_or_else(|e| panic!("verify failed: {e}"));
        verify_ms += tv0.elapsed().as_secs_f64() * 1e3;

        proof_bytes += batch_proof_size(&proof);

        // BinarySpartan: the outer (instance + row phases) + the inner +
        // the openings, on the SAME packed block and commitment.
        let mut ts2 = Transcript::new(&format!("{tag}-spartan"), &[&(block as u64).to_le_bytes()]);
        let tsp = Instant::now();
        let sp = prove_spartan(&cs, &setup, &params, &packed_io, &mut ts2);
        sp_prove_ms += tsp.elapsed().as_secs_f64() * 1e3;
        let mut tvs = Transcript::new(&format!("{tag}-spartan"), &[&(block as u64).to_le_bytes()]);
        let tvs0 = Instant::now();
        verify_spartan(&cs, &params, &packed_io, &sp, &mut tvs)
            .unwrap_or_else(|e| panic!("spartan verify failed: {e}"));
        sp_verify_ms += tvs0.elapsed().as_secs_f64() * 1e3;
        sp_proof_bytes += spartan_proof_size(&sp);
    }
    println!(
        "  batch:    {} instances = {} packed block(s) of κ = 128",
        blocks * 128,
        blocks
    );
    println!("  -- Algorithm 6 (the Table-4 benchmark setting: public wiring)");
    println!(
        "  prover:   {:.2} ms total ({:.2} ms/instance) [commit+preproc {:.0} ms]",
        prove_ms + setup_ms,
        (prove_ms + setup_ms) / (blocks * 128) as f64,
        setup_ms
    );
    println!("  verifier: {:.2} ms total", verify_ms);
    println!(
        "  proof:   {} ({:.1} KiB)",
        proof_bytes,
        proof_bytes as f64 / 1024.0
    );
    println!("  -- BinarySpartan (the production path: outer + inner, packed openings)");
    println!(
        "  prover:   {:.2} ms total ({:.2} ms/instance)",
        sp_prove_ms,
        sp_prove_ms / (blocks * 128) as f64
    );
    println!("  verifier: {:.2} ms total", sp_verify_ms);
    println!(
        "  proof:   {} ({:.1} KiB)",
        sp_proof_bytes,
        sp_proof_bytes as f64 / 1024.0
    );
    println!("  peak RSS: {}", fmt_bytes(peak_rss_bytes()));
    let paper = match name {
        "sha256" => "79.54 ms prover / 37.47 ms verifier / 367.71 KiB proof / 606.79 MiB RSS",
        "keccak" => "100.15 ms / 19.30 ms / 380.31 KiB / 55.53 MiB",
        "blake3" => "56.13 ms / 35.74 ms / 355.64 KiB / 632.25 MiB",
        _ => "",
    };
    println!("  paper ref (Xeon Gold 6342, GFNI): {paper}");
}

fn spartan_proof_size(p: &SpartanProof) -> usize {
    let f1 = |f: &magnetar_mikros::fact1::Fact1Proof| f.rounds.len() * 3 * 16 + 32;
    let mut total = 16 // io_claim
        + f1(&p.fact1_io)
        + p.outer_inst.len() * 4 * 16
        + p.outer_row.iter().map(|m| m.len() * 16).sum::<usize>()
        + 3 * 16 // terminals
        + f1(&p.inner)
        + p.w_claim.0.len() * 16 + 16; // the witness claim (point, value)
                                       // the packed opening (Algorithm 4)
    total += p.po.v_msg.len() * 16 + f1(&p.po.fact1) + 16 + pcs_size(&p.po.pcs);
    // the io claim's plain opening
    total += pcs_size(&p.pcs_io);
    total
}

fn batch_proof_size(p: &BatchProof) -> usize {
    let f1 = |f: &magnetar_mikros::fact1::Fact1Proof| f.rounds.len() * 3 * 16 + 32;
    let had = p.had.d.len() * 16 + p.had.h.len() * 16 + f1(&p.had.fact1);
    let mut total = 16 // io_claim
        + f1(&p.fact1_io)
        + 16 // v_o
        + had
        + p.v_msgs.iter().map(|v| v.len() * 16).sum::<usize>()
        + f1(&p.fact1_lin)
        + 3 * 16 // pub_vals
        + p.witness_claims.iter().map(|(pt, _)| pt.len() * 16 + 16).sum::<usize>();
    for pcs in &p.pcs_proofs {
        total += pcs_size(pcs);
    }
    total
}

fn pcs_size(p: &PcsProof) -> usize {
    let mut total = p.rounds.iter().map(|r| r.len() * 16).sum::<usize>();
    total += p.level_roots.len() * 32;
    total += 16; // terminal_claim
    for q in &p.queries {
        for lvl in &q.levels {
            // a/b/c scalars + the two leaf-group reveals + the two paths.
            total += 16 * 3
                + (lvl.a_vals.len() + lvl.c_vals.len()) * 16
                + (lvl.a_path.len() + lvl.c_path.len()) * 32;
        }
    }
    total += p
        .tail
        .iter()
        .map(|(vals, path)| vals.len() * 16 + path.len() * 32)
        .sum::<usize>();
    total
}

fn bench_hashes(which: &str, batch: usize) {
    println!("== Mikros Table 4: end-to-end packed hash batches (Algorithm 6) ==");
    let names: Vec<&str> = match which {
        "all" => vec!["sha256", "keccak", "blake3"],
        other => vec![other],
    };
    for name in names {
        println!("-- {name} --");
        bench_one_hash(name, batch);
    }
}

// ---------------------------------------------------------------------------
// The permutation argument (Algorithm 3) anatomy
// ---------------------------------------------------------------------------

fn bench_perm(log2_n: usize) {
    assert!(log2_n >= 20, "m = 2^n/128 ≥ κ requires N ≥ 2^20");
    assert!(
        log2_n <= 24,
        "the container ladder tops at 2^24 for this run"
    );
    let n = 1usize << log2_n;
    let m = n / 128;
    println!(
        "== Mikros §4.3: packed permutation proof, N = 2^{log2_n} (κ = 128, m = 2^{}) ==",
        m.trailing_zeros()
    );

    let f8 = Gf8::new();
    let mut data = InterpData::new(&f8);
    data.build_psi_monomials(&f8);

    // a: random planes; b: a with a random row permutation.
    let t_gen = Instant::now();
    let mut rng = XorShift(0xfeed_beef_c0de_d00d);
    let mut a = PlaneSet::zeros(m, 128);
    for j in 0..128 {
        for w in a.plane_mut(j).iter_mut() {
            *w = rng.next();
        }
    }
    let mut sigma: Vec<usize> = (0..m).collect();
    for i in (1..m).rev() {
        let j = (rng.next() as usize) % (i + 1);
        sigma.swap(i, j);
    }
    let mut b = PlaneSet::zeros(m, 128);
    for (r, &src_row) in sigma.iter().enumerate() {
        for j in 0..128 {
            if a.bit(j, src_row) {
                b.set_bit(j, r, true);
            }
        }
    }
    println!(
        "witness + σ generated in {:.0} ms",
        t_gen.elapsed().as_secs_f64() * 1e3
    );

    let mut t = Transcript::new("mikros-bench-perm", &[&n.to_le_bytes()]);
    let (proof, terminal, stats) = prove_perm(&f8, &data, &a, &b, &mut t);
    println!("prover:      {:.2} ms", stats.total_ms);
    println!(
        "  anatomy:   tree {:.0} | e-vecs {:.0} | e-tree {:.0} | u {:.0} | η {:.0} | g {:.0} | cubic {:.0} (ms, both operands)",
        stats.tree_build_ms,
        stats.e_vec_ms,
        stats.e_tree_ms,
        stats.u_ms,
        stats.eta_ms,
        stats.g_ms,
        stats.f_cubic_ms,
    );
    println!(
        "  bits/s:    {:.0}/s over N = {} bits (single thread)",
        n as f64 / (stats.total_ms / 1e3),
        n
    );

    let tv0 = Instant::now();
    let mut tv = Transcript::new("mikros-bench-perm", &[&n.to_le_bytes()]);
    let vterm = verify_perm(&f8, &data, &proof, &mut tv).expect("verify");
    let verify_ms = tv0.elapsed().as_secs_f64() * 1e3;
    println!("verifier:    {:.2} ms", verify_ms);
    assert_eq!(vterm.claims[0].0, terminal.claims[0].0, "terminal binding");

    // Proof size: Θ + two operands' unfolds.
    let mut size = 16usize;
    for unfold in &proof.unfolds {
        for cubic in &unfold.e_tree {
            size += cubic.rounds.len() * 4 * 16 + 2 * 16;
        }
        for tr in &unfold.f_trans {
            size += tr.u.len() * 16 + tr.h.len() * 16 + tr.cubic.rounds.len() * 4 * 16 + 2 * 16;
        }
    }
    println!(
        "proof size:  {size} B ({:.1} KiB) — the perm argument's share of the Table-5 shape",
        size as f64 / 1024.0
    );
    println!("peak RSS:    {}", fmt_bytes(peak_rss_bytes()));
    println!(
        "paper ref:   Table 5 single-instance (N=2^25, κ_pos=16): 86.56 s total, 6.72 s excl. perm"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("hadamard");
    let mut opts = std::collections::HashMap::new();
    let mut i = 2;
    while i + 1 < args.len() {
        opts.insert(args[i].clone(), args[i + 1].clone());
        i += 2;
    }
    match cmd {
        "hadamard" => {
            let log2_n = opts
                .get("--log2-n")
                .map(|v| v.parse().unwrap())
                .unwrap_or(24);
            bench_hadamard(log2_n);
        }
        "hashes" => {
            let batch = opts
                .get("--batch")
                .map(|v| v.parse().unwrap())
                .unwrap_or(256);
            let which = opts.get("--which").map(String::as_str).unwrap_or("all");
            bench_hashes(which, batch);
        }
        "perm" => {
            let log2_n = opts
                .get("--log2-n")
                .map(|v| v.parse().unwrap())
                .unwrap_or(20);
            bench_perm(log2_n);
        }
        "plonk" => {
            let log2_n = opts
                .get("--log2-n")
                .map(|v| v.parse().unwrap())
                .unwrap_or(16);
            let seed = opts.get("--seed").map(|v| v.parse().unwrap()).unwrap_or(42);
            bench_plonk(log2_n, seed);
        }
        "erc20" => {
            let log2_n = opts
                .get("--log2-n")
                .map(|v| v.parse().unwrap())
                .unwrap_or(16);
            let seed = opts.get("--seed").map(|v| v.parse().unwrap()).unwrap_or(42);
            bench_erc20(log2_n, seed);
        }
        other => {
            eprintln!(
                "unknown command {other:?}; usage: mikros-bench hadamard --log2-n 30 | hashes --batch 256 | perm --log2-n 20"
            );
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Algorithm 5 — the single-instance PLONK SNARK (Table 5)
// ---------------------------------------------------------------------------

fn bench_plonk(log2_n: usize, seed: u64) {
    assert!(log2_n >= 14, "Theorem 1: N ≥ κ² = 2^14");
    assert!(log2_n <= 22, "the container-bounded ladder tops at 2^22");
    let n = 1usize << log2_n;
    println!(
        "== Mikros Algorithm 5: single-instance PLONK, N = 2^{log2_n} (κ = 128, m = {}, κ_pos = 4, d = 32) ==",
        n / 128
    );
    let t0 = Instant::now();
    let (circuit, witness) = magnetar_mikros::plonk::simulated_circuit(log2_n, seed);
    println!(
        "circuit built in {:.0} ms ({} gates, n_io = {}, L = {})",
        t0.elapsed().as_secs_f64() * 1e3,
        circuit.n,
        circuit.n_io,
        circuit.l_len()
    );
    // sanity: the witness satisfies the circuit
    assert!(circuit.gates_hold(
        &witness.w[0..n],
        &witness.w[n..2 * n],
        &witness.w[2 * n..3 * n]
    ));
    assert!(circuit.copies_hold(&witness.w));
    let d_l = circuit.d_l();
    let io: Vec<bool> = witness.w[3 * n..3 * n + circuit.n_io].to_vec();

    let t1 = Instant::now();
    let params = PcsParams::default();
    let setup = magnetar_mikros::plonk::setup_plonk(&circuit, &witness, params);
    println!(
        "setup (pack + commit) in {:.0} ms: witness 3m = {}, public m_pub = {} words (d_L = {d_l})",
        t1.elapsed().as_secs_f64() * 1e3,
        3 * n / 128,
        (5 + 3 * d_l) * n / 128,
    );

    let mut t = Transcript::new("mikros-plonk-bench", &[]);
    let (proof, stats) = magnetar_mikros::plonk::prove_plonk(&circuit, &witness, &setup, &mut t);
    println!(
        "prover:      {:.1} ms total (gate {:.1} | hadamard {:.1} | copy {:.1} | openings {:.1})",
        stats.total_ms, stats.gate_ms, stats.hadamard_ms, stats.copy_ms, stats.open_ms
    );
    println!(
        "  gates/s:   {:.0} over N = {n} (single thread)",
        n as f64 / (stats.total_ms / 1e3)
    );

    let mut tv = Transcript::new("mikros-plonk-bench", &[]);
    let tv0 = Instant::now();
    let ok = magnetar_mikros::plonk::verify_plonk(&circuit, &setup, &io, &proof, &mut tv).is_ok();
    let verify_ms = tv0.elapsed().as_secs_f64() * 1e3;
    println!("verifier:    {verify_ms:.1} ms  (accepted: {ok})");

    // the proof size (every Gf128 is 16 bytes; the PCS share at the
    // FastChunk profile, cf. the Table-4 accounting)
    let mut proof_bytes = 0usize;
    proof_bytes += proof.c_claims.len() * 16;
    for hp in proof
        .gate_proofs
        .iter()
        .chain(std::iter::once(&proof.nested))
    {
        proof_bytes += hp.d.len() * 16 + hp.h.len() * 16;
        proof_bytes += hp.fact1.rounds.len() * 3 * 16 + 2 * 16;
    }
    for unfold in proof.copy.unfolds.iter() {
        for cub in unfold.e_tree.iter() {
            proof_bytes += cub.rounds.len() * 4 * 16 + 2 * 16;
        }
        for ft in unfold.f_trans.iter() {
            proof_bytes += ft.u.len() * 16 + ft.h.len() * 16;
            proof_bytes += ft.cubic.rounds.len() * 4 * 16 + 2 * 16;
        }
    }
    proof_bytes += proof.wire_values.len() * 3 * 16;
    proof_bytes += proof.digit_values.len() * 3 * 16;
    for g in [&proof.wit_open, &proof.pub_open] {
        proof_bytes += g.v_msgs.len() * 128 * 16;
        proof_bytes += g.fact1.rounds.len() * 3 * 16 + 2 * 16;
        proof_bytes += 4096; // the FastChunk operating point (cf. Table 4)
    }
    println!(
        "proof size:  {} (approx., the PCS share at the FastChunk profile)",
        fmt_bytes(proof_bytes as u64)
    );
    println!("peak RSS:    {}", {
        let kb = peak_rss_bytes() / 1024;
        format!("{kb} KiB")
    });
    println!(
        "paper ref:   Table 5 (N = 2^25, Xeon Gold 6342): 86.56 s prover / 24.43 ms verifier / 1,169 KiB — Mikros excl. perm. 6.72 s (1.51× vs Flock)"
    );
    assert!(ok, "the proof must verify");
}

// ---------------------------------------------------------------------------
// The ERC-20 transfer benchmark (Algorithm 5 on the state-transition core)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn rdtsc() -> u64 {
    #[allow(unsafe_code)]
    // SAFETY: _rdtsc is side-effect-free on x86-64 (invariant TSC on every
    // Xeon since Nehalem; the calibration below re-derives the rate).
    unsafe {
        std::arch::x86_64::_rdtsc()
    }
}
#[cfg(not(target_arch = "x86_64"))]
fn rdtsc() -> u64 {
    0
}

/// Calibrate the TSC rate against the wall clock (a 200 ms spin).
fn tsc_hz() -> f64 {
    let warm = rdtsc();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let _ = warm;
    let t0 = Instant::now();
    let c0 = rdtsc();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let c1 = rdtsc();
    let t1 = Instant::now();
    let secs = t1.duration_since(t0).as_secs_f64().max(1e-9);
    ((c1.saturating_sub(c0)) as f64 / secs).max(1.0)
}

fn bench_erc20(log2_n: usize, seed: u64) {
    use magnetar_mikros::plonk::{
        erc20_circuit, setup_plonk, verify_plonk, ERC20_ROWS_PER_TRANSFER,
    };
    assert!(log2_n >= 14, "Theorem 1: N ≥ κ² = 2^14");
    assert!(log2_n <= 22, "the container-bounded ladder tops at 2^22");
    let n = 1usize << log2_n;
    let transfers = (n / ERC20_ROWS_PER_TRANSFER).max(1);
    println!(
        "== Mikros Algorithm 5: ERC-20 transfer batch, N = 2^{log2_n} ({transfers} transfers, {} rows/transfer + 257 io bits) ==",
        ERC20_ROWS_PER_TRANSFER
    );
    let t0 = Instant::now();
    let (circuit, witness, placed) = erc20_circuit(transfers, n, seed);
    println!(
        "circuit built in {:.0} ms ({} gates, {} placed transfers, n_io = {}, L = {})",
        t0.elapsed().as_secs_f64() * 1e3,
        circuit.n,
        placed,
        circuit.n_io,
        circuit.l_len()
    );
    assert!(circuit.gates_hold(
        &witness.w[0..circuit.n],
        &witness.w[circuit.n..2 * circuit.n],
        &witness.w[2 * circuit.n..3 * circuit.n]
    ));
    assert!(circuit.copies_hold(&witness.w));
    let d_l = circuit.d_l();
    let io: Vec<bool> = witness.w[3 * circuit.n..3 * circuit.n + circuit.n_io].to_vec();

    let t1 = Instant::now();
    let params = PcsParams::default();
    let setup = setup_plonk(&circuit, &witness, params);
    println!(
        "setup (pack + commit) in {:.0} ms: d_L = {d_l}",
        t1.elapsed().as_secs_f64() * 1e3
    );

    let hz = tsc_hz();
    let mut t = Transcript::new("mikros-plonk-bench", &[]);
    let c0 = rdtsc();
    let (proof, stats) = magnetar_mikros::plonk::prove_plonk(&circuit, &witness, &setup, &mut t);
    let c1 = rdtsc();
    let prove_secs = stats.total_ms / 1e3;
    let gps = circuit.n as f64 / (stats.total_ms / 1000.0);
    let cycles = c1.saturating_sub(c0);
    let mut tv = Transcript::new("mikros-plonk-bench", &[]);
    let v0 = Instant::now();
    let ok = verify_plonk(&circuit, &setup, &io, &proof, &mut tv).is_ok();
    let v_ms = v0.elapsed().as_secs_f64() * 1e3;
    // the proof size (the plonk table's accounting)
    let mut proof_bytes = 0usize;
    proof_bytes += proof.c_claims.len() * 16;
    for hp in proof
        .gate_proofs
        .iter()
        .chain(std::iter::once(&proof.nested))
    {
        proof_bytes += hp.d.len() * 16 + hp.h.len() * 16;
        proof_bytes += hp.fact1.rounds.len() * 3 * 16 + 2 * 16;
    }
    for unfold in proof.copy.unfolds.iter() {
        for cub in unfold.e_tree.iter() {
            proof_bytes += cub.rounds.len() * 4 * 16 + 2 * 16;
        }
        for ft in unfold.f_trans.iter() {
            proof_bytes += ft.u.len() * 16 + ft.h.len() * 16;
            proof_bytes += ft.cubic.rounds.len() * 4 * 16 + 2 * 16;
        }
    }
    proof_bytes += proof.wire_values.len() * 3 * 16;
    proof_bytes += proof.digit_values.len() * 3 * 16;
    for g in [&proof.wit_open, &proof.pub_open] {
        proof_bytes += g.v_msgs.len() * 128 * 16;
        proof_bytes += g.fact1.rounds.len() * 3 * 16 + 2 * 16;
        proof_bytes += 4096; // the FastChunk operating point (cf. Table 4)
    }

    println!(
        "prover:      {:.1} ms total (gate {:.0} | hadamard {:.0} | copy {:.0} | open {:.0})",
        stats.total_ms, stats.gate_ms, stats.hadamard_ms, stats.copy_ms, stats.open_ms
    );
    println!(
        "  gates/s:   {gps:.0} over N = {} (single thread); {:.3} ms/transfer",
        circuit.n,
        stats.total_ms / transfers as f64
    );
    let tps = transfers as f64 / prove_secs;
    println!("  transfers/s (prover):   {tps:.1}");
    println!(
        "  TSC:      {cycles} cycles ({:.2} GHz calibrated) → {:.0} cycles/s during prove | {:.1} cycles/gate | {:.0} cycles/transfer",
        hz / 1e9,
        if prove_secs > 0.0 { cycles as f64 / prove_secs } else { 0.0 },
        cycles as f64 / circuit.n as f64,
        cycles as f64 / transfers as f64
    );
    println!(
        "verifier:    {:.1} ms (accepted: {ok}); proof size: {} (approx., FastChunk PCS share)",
        v_ms,
        fmt_bytes(proof_bytes as u64)
    );
    println!(
        "peak RSS:    {:.0} MiB",
        peak_rss_bytes() as f64 / (1 << 20) as f64
    );
    // ---- the Ethereum real-time verdict (the 12 s slot budget)
    println!("---- Ethereum real-time analysis (single core, measured rate) ----");
    // mainnet planning figures: ~30M gas target per 12 s slot; a standard
    // ERC-20 transfer ≈ 50K gas (warm ~24K, fresh ~51K) → ~600 transfers/block
    const SLOT_SECS: f64 = 12.0;
    const GAS_LIMIT: f64 = 30.0e6;
    const GAS_PER_TRANSFER: f64 = 50.0e3;
    let block_tx = (GAS_LIMIT / GAS_PER_TRANSFER) as usize; // 600
    let block_gates = block_tx * ERC20_ROWS_PER_TRANSFER; // the state-transition core
    let block_secs = block_gates as f64 / gps;
    println!(
        "  full-block ERC-20 load: {block_tx} transfers ≈ {block_gates} gates (the state-transition core) → {block_secs:.2} s at {gps:.0} gates/s",
    );
    println!(
        "  verdict:  {} — headroom {:.2}× vs the {SLOT_SECS:.0} s slot",
        if block_secs <= SLOT_SECS {
            "REAL-TIME"
        } else {
            "BEHIND"
        },
        SLOT_SECS / block_secs.max(1e-9)
    );
    // the extended stacks (the honest scope note): keccak storage proofs
    // and ECDSA are not in this circuit — see docs/BENCHMARKS.md.
    println!(
        "  (state-transition core only: no keccak storage proofs, no ECDSA — the extended-stack analysis is in docs/BENCHMARKS.md)"
    );
}
