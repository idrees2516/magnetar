//! Ring switching (paper §5.2 / [13]): reduce a virtual evaluation claim on
//! a **bit-valued** vector to claims on its **packed** `K`-valued
//! representation, discharged against the main codeword commitment.
//!
//! ## The commitment-free construction (Diamond–Posen, ePrint 2024/504 §4–5)
//!
//! The bit table `w: {0,1}^n × {0,1}^7 → B` is packed as
//! `P(u) = Σ_b β_b w(u, b)` (polynomial basis: bit `b` of `P(u)` *is*
//! `w(u, b)`). A claim `ŵ(r_A, r_B) = y` discharges as:
//!
//! 1. **The partial evaluations**. The prover sends the 128 values
//!    `s_b = ŵ_b(r_A) = Σ_u eq(r_A,u)·w(u,b)` — one pass over the packed
//!    table. The verifier checks the recombination
//!    `y = Σ_b eq(r_B,b)·s_b`, which pins the claim to the `s_b`.
//! 2. **The row batch**. The verifier samples `ρ ← K^7` and derives
//!    `σ = fold(s, eq(ρ,·))` — the bit-transpose pairing
//!    `Σ_i eq(ρ,i)·b_i` with `b_i = Σ_j bit_i(s_j)·β_j`.
//! 3. **One sum-check over the word cube** with the **packed table itself**
//!    as the dense factor and the **transparent** factor
//!    `A(u) = φ_{eq(ρ,·)}(eq(r_A, u))`:
//!    `Σ_u P(u)·A(u) = σ`.
//!    The identity is exact: both sides are the `β_j`-packed
//!    `eq(ρ)`-projections of the per-plane partial sums, so the honest
//!    proof passes by algebra. Every round binds a fresh word challenge
//!    `r̃`, ending at the product claim `P̂(r̃)·A(r̃)`.
//! 4. **The main codeword's own fold chain at the same challenges** —
//!    levels `1..=n` fresh, level `0` **the commitment itself** — with
//!    correlated path queries and the tail. No second encode, no separate
//!    `C`-codeword, no anchors: the chain *is* the committed argument.
//!
//! ## Soundness
//!
//! The `s_b` are pinned by two independent checks. The recombination ties
//! `y` to them; the row batch ties them to the committed table: completing
//! the sum-check and the chain certifies
//! `Σ_u P(u)·A(u) = fold(s, eq(ρ,·))`, and since the committed table's own
//! partials `s*` satisfy the same identity with `A` (the correctness
//! identity), any deviation `s ≠ s*` must satisfy
//! `Σ_i eq(ρ,i)·E_i = 0` for the bit-transpose error `E ≠ 0` — a nonzero
//! multilinear polynomial in `ρ` that vanishes with probability at most
//! `7/2^128` (Schwartz–Zippel). With `s` pinned, the recombination pins
//! `y = ŵ(r_A, r_B)` for the *committed* table, and the sum-check plus
//! the fold chain carry the packed claim `P̂(r̃)` with the standard
//! FRI-style proximity analysis of the main PCS (crate docs).

use crate::eval::{level_domain, LevelOpen, PathQuery, PcsParams, PcsStatement};
use magnetar_field::{eq_coefficients, mul, EqBits, Gf128, PhiTab, ONE, ZERO};
use magnetar_hash::{MerkleTree, Transcript, DIGEST};
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    BindOrder, SumcheckSpec, Term,
};
use rayon::prelude::*;

/// A ring-switch proof for one bit-vector claim: the DP24 reduction plus
/// the main codeword's fold chain.
#[derive(Clone, Debug)]
pub struct RingProof {
    /// The 128 per-plane partial evaluations `s_b = ŵ_b(r_A)`.
    pub partials: Vec<Gf128>,
    /// Sum-check round messages over the word cube (degree 2, 3
    /// coefficients per round).
    pub rounds: Vec<Vec<Gf128>>,
    /// Fold-level roots of the main codeword's chain, levels `1..=n`
    /// (level 0 is the commitment itself).
    pub level_roots: Vec<[u8; DIGEST]>,
    /// The claimed `P̂(r̃)` (equals the tail entry `D_n(0)`).
    pub terminal_claim: Gf128,
    /// Correlated path queries on the main codeword's chain.
    pub queries: Vec<PathQuery>,
    /// The tail `D_n` (`2^k` values) with paths.
    pub tail: Vec<(Gf128, Vec<[u8; DIGEST]>)>,
}

/// The transparent row-batched eq-indicator
/// `A(u) = φ_{eq(ρ,·)}(eq(z, u))` — multilinear in `u` (each coordinate of
/// a product of per-variable-linear elements is multilinear over the
/// Boolean cube), verifier-evaluable in `O(n)` without any table.
#[derive(Clone, Debug)]
pub struct RingEqInd {
    /// The word point `z` (= `r_A`).
    pub z: Vec<Gf128>,
    /// The row-batch coefficients `eq(ρ, ·)`.
    pub rb: [Gf128; 128],
}

impl RingEqInd {
    /// The DP24 tensor-algebra evaluation — the multilinear extension of
    /// the materialized Boolean table. (The *formula*
    /// `φ_rb(eq(z, ·))` agrees with the table on the Boolean cube but is
    /// **not** multilinear — the coordinate functionals are only
    /// F₂-linear — so it may NOT be used off the cube. The tensor
    /// recursion below is the correct extension: linear in each `q_i` by
    /// construction, and agreeing with the table on the cube.)
    fn tensor_eval(&self, point: &[Gf128]) -> Gf128 {
        // T = from_vertical(ONE): a 128-component tensor state t with
        // t[0] = 1. Each word coordinate applies
        //   T ← T + z_i·T (vertical scale) + H(q_i)·T (horizontal scale),
        // where the horizontal multiplication by q_i sends component k to
        // Σ_j [β_j·q_i]_k · t_j (the transpose of the multiplication
        // matrix, β_j = x^j).
        let mut t = vec![ZERO; 128];
        t[0] = ONE;
        for (&z_i, &q_i) in self.z.iter().zip(point.iter()) {
            let mut new_t = t.clone();
            for j in 0..128usize {
                let tj = t[j];
                if tj == ZERO {
                    continue;
                }
                new_t[j] += mul(tj, z_i);
                let h = mul(Gf128(1u128 << j), q_i);
                let mut bits = h.0;
                while bits != 0 {
                    let k = bits.trailing_zeros() as usize;
                    new_t[k] += tj;
                    bits &= bits - 1;
                }
            }
            t = new_t;
        }
        // fold_vertical: the transpose-pairing with the row-batch
        // coefficients (the same fold as the partials').
        fold_partials(&t, &self.rb)
    }
}

impl magnetar_sumcheck::public::PublicFactor for RingEqInd {
    fn num_vars(&self) -> usize {
        self.z.len()
    }

    fn materialize(&self) -> Vec<Gf128> {
        let eq = EqBits::build(&self.z).values;
        let tab = PhiTab::build(&self.rb);
        eq.par_iter().map(|&e| tab.apply(e)).collect()
    }

    fn eval_at(&self, point: &[Gf128]) -> Gf128 {
        self.tensor_eval(point)
    }
}

/// The bit-transpose fold `σ = Σ_i rb_i·b_i` with
/// `b_i = Σ_j bit_i(s_j)·β_j` — the row-batch pairing of the partials.
#[inline]
pub fn fold_partials(s: &[Gf128], rb: &[Gf128; 128]) -> Gf128 {
    let mut acc = ZERO;
    for i in 0..128 {
        let mut bits = 0u128;
        for (j, &s_j) in s.iter().enumerate() {
            if (s_j.0 >> i) & 1 == 1 {
                bits |= 1u128 << j;
            }
        }
        acc += mul(rb[i], Gf128(bits));
    }
    acc
}

/// The 128 per-plane partial evaluations
/// `s_b = Σ_u eq(r_A, u)·bit_b(P(u))` — one popcount-driven pass over the
/// packed table with the eq table (parallel over chunks).
fn partial_evals(eq_z: &[Gf128], packed: &[Gf128]) -> Vec<Gf128> {
    debug_assert_eq!(eq_z.len(), packed.len());
    packed
        .par_iter()
        .zip(eq_z.par_iter())
        .fold(
            || [ZERO; 128],
            |mut acc, (&p, &e)| {
                let mut bits = p.0;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    acc[b] += e;
                    bits &= bits - 1;
                }
                acc
            },
        )
        .reduce(
            || [ZERO; 128],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
        .to_vec()
}

/// Prove the bit-claim `ŵ(r_A, r_B) = y`, where the bit vector is given by
/// its packed table `P` — the **main committed table**. `main_codeword` /
/// `main_tree` belong to the main commitment; the fold chain reuses the
/// codeword, so there is no second encode and no separate level-0 tree.
#[allow(clippy::too_many_arguments)]
pub fn prove_ring_switch(
    params: &PcsParams,
    main_codeword: &crate::code::Codeword,
    main_tree: &MerkleTree,
    packed: &[Gf128],
    r_a: &[Gf128],
    r_b: &[Gf128; 7],
    y: Gf128,
    t: &mut Transcript,
) -> RingProof {
    let n = packed.len().trailing_zeros() as usize;
    debug_assert_eq!(r_a.len(), n);
    // ---- 1. The partial evaluations s_b (one packed pass).
    let eq_z = EqBits::build(r_a).values;
    let partials = partial_evals(&eq_z, packed);
    drop(eq_z);
    // The honest identity: the partials recombine to the claim (the
    // verifier checks this; asserting it catches prover-side wiring bugs).
    {
        let c = eq_coefficients(r_b);
        let mut lhs = ZERO;
        for (b, &s_b) in partials.iter().enumerate() {
            lhs += mul(c[b], s_b);
        }
        debug_assert_eq!(
            lhs, y,
            "ring: the honest partials must recombine to the claim"
        );
    }
    t.absorb_field("ring-partials", &partials);
    // ---- 2. The row batch: sample rho, derive sigma.
    let rho_vec = t.challenge_field_vec("ring-rho", 7);
    let mut rho = [ZERO; 7];
    rho.copy_from_slice(&rho_vec);
    let rb = eq_coefficients(&rho);
    let sigma = fold_partials(&partials, &rb);
    // ---- 3. One sum-check over the word cube: dense = P, public = A.
    let ind = RingEqInd {
        z: r_a.to_vec(),
        rb,
    };
    let spec = SumcheckSpec {
        num_vars: n,
        dense: vec![packed.to_vec()],
        public: vec![Box::new(ind)],
        terms: vec![Term {
            coeff: Gf128(1),
            dense: vec![0],
            public: vec![0],
        }],
        claim: sigma,
    };
    let mut prover = SumcheckProver::new(&spec, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    for _ in 0..n {
        let msg = prover.message();
        rounds.push(msg.0.clone());
        challenges.push(prover.absorb_and_fold(t, "ring-eval"));
    }
    let (claims, _) = prover.finish();
    let terminal_claim = claims[0];
    t.absorb_field("ring-terminal", &[terminal_claim]);
    // ---- 4. The main codeword's own fold chain at the same challenges
    //     (levels 1..=n fresh; level 0 is the commitment).
    let mut cw = main_codeword.clone();
    let mut trees = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n + 1);
    values.push(cw.values.clone());
    for (i, &rc) in challenges.iter().enumerate() {
        cw = cw.fold(0, rc);
        trees.push(MerkleTree::build(
            &level_domain(&params.domain, i + 1),
            &cw.values,
        ));
        values.push(cw.values.clone());
    }
    let level_roots: Vec<[u8; DIGEST]> = trees.iter().map(|tr| tr.root()).collect();
    for (i, root) in level_roots.iter().enumerate() {
        t.absorb_digest(&format!("ring-root-{}", i + 1), root);
    }
    // Correlated queries (level 0 opens against the commitment's tree).
    let base_leaves = main_tree.n_leaves() as u64;
    let mut queries = Vec::with_capacity(params.queries);
    for qi in 0..params.queries {
        let pos = t.challenge_index(&format!("ring-query-{qi}"), base_leaves) as usize;
        queries.push(build_query(main_tree, &trees, &values, pos));
    }
    // Tail.
    let tail_tree = trees.last().expect("n >= 1 for a real witness");
    let tail_vals = values.last().unwrap();
    let tail: Vec<(Gf128, Vec<[u8; DIGEST]>)> = (0..tail_tree.n_leaves())
        .map(|i| {
            let (_, path) = tail_tree.open(i);
            (tail_vals[i], path)
        })
        .collect();
    RingProof {
        partials,
        rounds,
        level_roots,
        terminal_claim,
        queries,
        tail,
    }
}

fn build_query(
    main_tree: &MerkleTree,
    trees: &[MerkleTree],
    values: &[Vec<Gf128>],
    pos: usize,
) -> PathQuery {
    let n = trees.len();
    let mut levels = Vec::with_capacity(n);
    let mut idx = pos;
    for i in 0..n {
        let a_idx = idx & !1usize;
        let b_idx = idx | 1usize;
        let next_idx = idx >> 1;
        let (a_path, b_path, c_path) = if i == 0 {
            let (_, ap) = main_tree.open(a_idx);
            let (_, bp) = main_tree.open(b_idx);
            let (_, cp) = trees[0].open(next_idx);
            (ap, bp, cp)
        } else {
            let (_, ap) = trees[i - 1].open(a_idx);
            let (_, bp) = trees[i - 1].open(b_idx);
            let (_, cp) = trees[i].open(next_idx);
            (ap, bp, cp)
        };
        levels.push(LevelOpen {
            a: values[i][a_idx],
            b: values[i][b_idx],
            a_path,
            b_path,
            c: values[i + 1][next_idx],
            c_path,
        });
        idx = next_idx;
    }
    PathQuery { levels }
}

/// Verify a ring-switch proof for the claim `ŵ(r_A, r_B) = y` against the
/// main commitment `main_stmt`.
pub fn verify_ring_switch(
    main_stmt: &PcsStatement,
    params: &PcsParams,
    r_a: &[Gf128],
    r_b: &[Gf128; 7],
    y: Gf128,
    proof: &RingProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let n = main_stmt.n;
    if r_a.len() != n {
        return Err("ring: point length mismatch");
    }
    if proof.partials.len() != 128 {
        return Err("ring: partial count mismatch");
    }
    if proof.rounds.len() != n || proof.level_roots.len() != n {
        return Err("ring: proof shape mismatch");
    }
    if proof.queries.len() != params.queries {
        return Err("ring: query count mismatch");
    }
    // ---- 1. The recombination: y == Σ_b eq(r_B, b)·s_b.
    let c = eq_coefficients(r_b);
    let mut lhs = ZERO;
    for (b, &s_b) in proof.partials.iter().enumerate() {
        lhs += mul(c[b], s_b);
    }
    if lhs != y {
        return Err("ring: partial recombination mismatch");
    }
    t.absorb_field("ring-partials", &proof.partials);
    // ---- 2. The row batch: rho, sigma (verifier-derived).
    let rho_vec = t.challenge_field_vec("ring-rho", 7);
    let mut rho = [ZERO; 7];
    rho.copy_from_slice(&rho_vec);
    let rb = eq_coefficients(&rho);
    let sigma = fold_partials(&proof.partials, &rb);
    // ---- 3. Replay the sum-check (captures the fold challenges).
    let ind = RingEqInd {
        z: r_a.to_vec(),
        rb,
    };
    let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(ind)];
    let terms = vec![Term {
        coeff: Gf128(1),
        dense: vec![0],
        public: vec![0],
    }];
    let mut verifier = SumcheckVerifier::new(n, sigma, &public, terms, BindOrder::Increasing);
    let mut challenges = Vec::with_capacity(n);
    for msg in &proof.rounds {
        let round_msg = magnetar_sumcheck::RoundMsg(msg.clone());
        challenges.push(verifier.round_capturing(&round_msg, t, "ring-eval")?);
    }
    verifier.finish(&[proof.terminal_claim])?;
    t.absorb_field("ring-terminal", &[proof.terminal_claim]);
    // ---- 4. Level roots.
    for (i, root) in proof.level_roots.iter().enumerate() {
        t.absorb_digest(&format!("ring-root-{}", i + 1), root);
    }
    let leaves_at = |lvl: usize| -> u64 { 1u64 << (n - lvl + params.k) };
    // ---- 5. Correlated queries against the main commitment's chain.
    for (qi, q) in proof.queries.iter().enumerate() {
        let pos = t.challenge_index(&format!("ring-query-{qi}"), leaves_at(0)) as usize;
        if q.levels.len() != n {
            return Err("ring: query level count mismatch");
        }
        let mut idx = pos;
        for (i, lvl) in q.levels.iter().enumerate() {
            let a_idx = idx & !1usize;
            let b_idx = idx | 1usize;
            let next_idx = idx >> 1;
            let dom_i = level_domain(&main_stmt.domain, i);
            let expected_root = if i == 0 {
                main_stmt.root
            } else {
                proof.level_roots[i - 1]
            };
            let leaf = MerkleTree::leaf_digest_of(&dom_i, leaves_at(i) as usize, a_idx, lvl.a);
            let ra = MerkleTree::verify(&dom_i, leaves_at(i) as usize, a_idx, leaf, &lvl.a_path)
                .map_err(|_| "ring: query path a invalid")?;
            if ra != expected_root {
                return Err("ring: query root a mismatch");
            }
            let leaf = MerkleTree::leaf_digest_of(&dom_i, leaves_at(i) as usize, b_idx, lvl.b);
            let rb_ = MerkleTree::verify(&dom_i, leaves_at(i) as usize, b_idx, leaf, &lvl.b_path)
                .map_err(|_| "ring: query path b invalid")?;
            if rb_ != expected_root {
                return Err("ring: query root b mismatch");
            }
            let dom_next = level_domain(&main_stmt.domain, i + 1);
            let leaf =
                MerkleTree::leaf_digest_of(&dom_next, leaves_at(i + 1) as usize, next_idx, lvl.c);
            let rc = MerkleTree::verify(
                &dom_next,
                leaves_at(i + 1) as usize,
                next_idx,
                leaf,
                &lvl.c_path,
            )
            .map_err(|_| "ring: query path c invalid")?;
            if rc != proof.level_roots[i] {
                return Err("ring: query root c mismatch");
            }
            // Fold relation: (1+r)·a + r·b = c.
            let folded = lvl.a + mul(challenges[i], lvl.a + lvl.b);
            if folded != lvl.c {
                return Err("ring: fold relation failed");
            }
            idx = next_idx;
        }
    }
    // ---- 6. Tail.
    if proof.tail.len() != (1usize << params.k) {
        return Err("ring: tail length mismatch");
    }
    let tail_dom = level_domain(&main_stmt.domain, n);
    for (i, (val, path)) in proof.tail.iter().enumerate() {
        let leaf = MerkleTree::leaf_digest_of(&tail_dom, 1usize << params.k, i, *val);
        let root = MerkleTree::verify(&tail_dom, 1usize << params.k, i, leaf, path)
            .map_err(|_| "ring: tail path invalid")?;
        if root != proof.level_roots[n - 1] {
            return Err("ring: tail root mismatch");
        }
    }
    if proof.tail[0].0 != proof.terminal_claim {
        return Err("ring: tail terminal entry mismatch");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rand::RngExt::random::<u128>(rng))
    }

    fn pack_random_bits(rng: &mut impl rand::Rng, rows: usize) -> (Vec<bool>, Vec<Gf128>) {
        use rand::RngExt;
        let bits: Vec<bool> = (0..rows * 128).map(|_| rng.random::<bool>()).collect();
        let bytes: Vec<u8> = bits.iter().map(|&b| b as u8).collect();
        let packed = magnetar_field::pack_table(&bytes);
        (bits, packed)
    }

    #[test]
    fn ring_roundtrip_and_tamper() {
        use magnetar_field::eq_table;

        let mut rng = rand::rng();
        let params = PcsParams {
            k: 2,
            queries: 10,
            domain: "test-ring".into(),
        };
        let n = 4;
        let (bits, packed) = pack_random_bits(&mut rng, 1 << n);
        // Random claim point.
        let r_a: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
        let r_b: [Gf128; 7] = std::array::from_fn(|_| rg(&mut rng));
        // Compute w-hat(r) = Σ_{u,b} eq(r_a,u) eq(r_b,b) w(u,b).
        let eq_a = magnetar_field::EqBits::build(&r_a).values;
        let mut y = Gf128(0);
        for u in 0..(1 << n) {
            for b in 0..128usize {
                let b_bits: Vec<bool> = (0..7).map(|j| (b >> j) & 1 == 1).collect();
                let eb = eq_table(&r_b, &b_bits);
                y += magnetar_field::mul(
                    eq_a[u],
                    magnetar_field::mul(eb, Gf128(bits[u * 128 + b] as u128)),
                );
            }
        }
        // Commit the packed table.
        let (main_cw, main_tree, stmt) = crate::eval::commit_table(&params, &packed);
        let mut t = Transcript::new("ring", &[b"t"]);
        let proof = prove_ring_switch(
            &params, &main_cw, &main_tree, &packed, &r_a, &r_b, y, &mut t,
        );
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &proof, &mut tv).is_ok());
        // Wrong y must fail.
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(
            verify_ring_switch(&stmt, &params, &r_a, &r_b, y + Gf128(1), &proof, &mut tv).is_err()
        );
        // Tampered terminal.
        let mut p = proof.clone();
        p.terminal_claim += Gf128(1);
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &p, &mut tv).is_err());
        // Tampered partial (breaks the recombination).
        let mut p = proof.clone();
        p.partials[0] += Gf128(1);
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &p, &mut tv).is_err());
        // Tampered query fold value.
        let mut p = proof.clone();
        p.queries[0].levels[0].a += Gf128(1);
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &p, &mut tv).is_err());
        // Tampered level root.
        let mut p = proof.clone();
        p.level_roots[0][0] ^= 0xFF;
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &p, &mut tv).is_err());
        // Tampered round message.
        let mut p = proof.clone();
        p.rounds[0][1] += Gf128(1);
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &p, &mut tv).is_err());
        // Tampered tail entry.
        let mut p = proof.clone();
        p.tail[1].0 += Gf128(1);
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &p, &mut tv).is_err());
    }

    /// The core DP24 identity, checked end-to-end through the real
    /// prover: the sum-check over `P × A` must close at exactly
    /// `fold(s, eq(ρ,·))`, and the chain's tail must be `P̂(r̃)`.
    #[test]
    fn ring_honest_identity() {
        use crate::eval::commit_table;
        let mut rng = rand::rng();
        let params = PcsParams {
            k: 1,
            queries: 4,
            domain: "test-ring-id".into(),
        };
        let n = 5;
        let (_bits, packed) = pack_random_bits(&mut rng, 1 << n);
        let r_a: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
        let r_b: [Gf128; 7] = std::array::from_fn(|_| rg(&mut rng));
        // y via the dense definition.
        let eq_a = magnetar_field::EqBits::build(&r_a).values;
        let c = eq_coefficients(&r_b);
        let mut y = ZERO;
        for u in 0..(1 << n) {
            let mut cw = ZERO;
            for b in 0..128 {
                if (packed[u].0 >> b) & 1 == 1 {
                    cw += c[b];
                }
            }
            y += mul(eq_a[u], cw);
        }
        let (main_cw, main_tree, stmt) = commit_table(&params, &packed);
        let mut t = Transcript::new("ring-id", &[b"t"]);
        let proof = prove_ring_switch(
            &params, &main_cw, &main_tree, &packed, &r_a, &r_b, y, &mut t,
        );
        // The partials must equal the per-plane MLE evaluations.
        for b in 0..128usize {
            let mut want = ZERO;
            for u in 0..(1 << n) {
                if (packed[u].0 >> b) & 1 == 1 {
                    want += eq_a[u];
                }
            }
            assert_eq!(proof.partials[b], want, "partial {b}");
        }
        // The honest identity holds end-to-end: the verifier replays the
        // rounds (including the terminal identity with the tensor-eval
        // public factor) and checks the fold chain.
        let mut tv = Transcript::new("ring-id", &[b"t"]);
        assert!(
            verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &proof, &mut tv).is_ok(),
            "honest proof must verify"
        );
    }
}
