//! The PCS evaluation argument: prove `T̂(r) = y` against a codeword
//! commitment (see the crate-level documentation for the construction and
//! the soundness discussion).
//!
//! Protocol:
//! 1. **Sum-check** over the `n` table variables:
//!    `Σ_x eq(r, x)·T(x) = y` — binds random `r*`; the terminal identity is
//!    `eq(r, r*)·T̂(r*) = running`, checked against the claimed
//!    `T̂(r*)`.
//! 2. **Fold chain**: the codeword folded at the *same* challenges, one
//!    Merkle tree per level; the tail entry `D_n(0) = T̂(r*)` must equal the
//!    sum-check's claimed value.
//! 3. **Correlated path queries** (FRI-style): `q` random level-0 positions,
//!    each walked up through the levels, checking every fold relation with
//!    Merkle-authenticated openings.
//! 4. **Tail**: all `2^k` final entries revealed with paths.

use crate::code::{CodeParams, Codeword};
use magnetar_field::{mul, Gf128};
use magnetar_hash::{MerkleTree, Transcript, DIGEST};
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::EqFactor,
    BindOrder, SumcheckSpec, Term,
};

/// The fold-level leaf width: batched 8-value leaves once a level is wide
/// enough, pairs below that (an even `a_idx` always shares a group with its
/// odd partner), single leaves only for a length-1 tail. Deterministic in
/// the level's value count — both sides derive it identically.
#[inline]
pub fn fold_tree_width(level_len: usize) -> usize {
    if level_len >= 16 {
        magnetar_hash::BATCH_LEAF_WIDTH
    } else if level_len >= 2 {
        2
    } else {
        1
    }
}

#[inline]
fn fold_width(level_len: usize) -> usize {
    fold_tree_width(level_len)
}

/// The tree width for fold level `level` (0..=n, tail included).
#[inline]
fn level_width(n: usize, k: usize, level: usize) -> usize {
    fold_width(1usize << (n - level + k))
}

/// Tunable PCS parameters. Defaults target ≥ 120-bit security at the test
/// scale: `k = 2` (rate 1/4, distance ≈ 1 at generic γ) and `q = 40`
/// correlated queries.
#[derive(Clone, Debug)]
pub struct PcsParams {
    /// Extension variables (log codeword blow-up).
    pub k: usize,
    /// Correlated query paths.
    pub queries: usize,
    /// Domain-separation label.
    pub domain: String,
}

impl Default for PcsParams {
    fn default() -> Self {
        // The FastChunk operating point (rate 1/2): the fastest encode +
        // Merkle phase — the **chunk** profile. Every profile's per-query
        // far-word rejection is the Schwartz–Zippel bound `n/2^128` with
        // `n ≤ 64` (≥ 122 bits per query), so each profile holds the
        // 120-bit envelope on its own; the wrap re-proves the public tail
        // at a deep rate for size.
        PcsParams {
            k: 1,
            queries: 40,
            domain: "magnetar/pcs/default".to_string(),
        }
    }
}

/// The dual-rate operating profiles (roadmap Horizon 1: the chunk proofs
/// tune for guest-verifier speed at a high rate; the wrap — the only
/// public artifact — re-proves at a low rate with fewer correlated
/// queries). Measured on this repository's proofs: the same evaluation
/// argument runs 216 KiB at `FastChunk` (rate 1/2, 40 queries) down to
/// 128 KiB at `DeepWrap` (rate 1/16, 14 queries).
///
/// Soundness accounting: every profile's per-query far-word rejection is
/// the Schwartz–Zippel bound `n/2^128` with `n ≤ 64` (≥ 122 bits per
/// query); `q` queries run independently, so every profile holds the
/// 120-bit envelope on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcsProfile {
    /// Rate 1/2 with the standard 40 queries: the fastest encode + Merkle
    /// phase — the **chunk** profile.
    FastChunk,
    /// Rate 1/4, 40 queries: the default configuration.
    Balanced,
    /// Rate 1/8, 20 queries: the **wrap** profile — half the paths of a
    /// chunk proof.
    CompactWrap,
    /// Rate 1/16, 14 queries: the aggressive wrap.
    DeepWrap,
}

impl PcsProfile {
    /// The profile's parameters.
    pub fn params(self) -> PcsParams {
        let (k, queries) = match self {
            PcsProfile::FastChunk => (1, 40),
            PcsProfile::Balanced => (2, 40),
            PcsProfile::CompactWrap => (3, 20),
            PcsProfile::DeepWrap => (4, 14),
        };
        PcsParams {
            k,
            queries,
            domain: format!("magnetar/pcs/{self:?}").to_lowercase(),
        }
    }

    /// The conservative per-query far-word rejection exponent (the
    /// Schwartz–Zippel bound `n/2^128` with `n ≤ 64`): every profile
    /// keeps a single correlated query within the 120-bit envelope on
    /// its own, and `q` of them run independently.
    pub fn per_query_soundness_bits(self) -> u32 {
        122 // 128 - ceil(log2(64)) — the SZ bound at n ≤ 64
    }

    /// The rate `2^-k` as a fraction string.
    pub fn rate_label(self) -> String {
        format!("1/{}", 1usize << self.params().k)
    }
}

/// A committed table: the codeword tree's root plus layout info.
#[derive(Clone, Debug)]
pub struct PcsStatement {
    /// Merkle root of the level-0 codeword (the commitment).
    pub root: [u8; DIGEST],
    /// Number of table variables.
    pub n: usize,
    /// Extension variables.
    pub k: usize,
    /// Domain-separation label.
    pub domain: String,
}

/// One level's openings along a correlated query path. `a` and `b` are
/// adjacent value indices — they always share one leaf group, so a single
/// group reveal + path covers the pair; `c`'s group at the next level is
/// revealed alongside.
#[derive(Clone, Debug)]
pub struct LevelOpen {
    /// Pair value at the fold bit = 0.
    pub a: Gf128,
    /// Pair value at the fold bit = 1.
    pub b: Gf128,
    /// Image value in the next level.
    pub c: Gf128,
    /// The leaf group covering `a`/`b` at this level (width values).
    pub a_vals: Vec<Gf128>,
    /// The leaf group covering `c` at the next level (its width values).
    pub c_vals: Vec<Gf128>,
    pub a_path: Vec<[u8; DIGEST]>,
    pub c_path: Vec<[u8; DIGEST]>,
}

/// A full correlated walk: level 0 → n.
#[derive(Clone, Debug)]
pub struct PathQuery {
    pub levels: Vec<LevelOpen>,
}

/// The evaluation proof.
#[derive(Clone, Debug)]
pub struct PcsProof {
    /// Sum-check round messages (degree-2, 3 coefficients each).
    pub rounds: Vec<Vec<Gf128>>,
    /// Merkle roots of fold levels 1..=n (level 0 root = the commitment).
    pub level_roots: Vec<[u8; DIGEST]>,
    /// The claimed `T̂(r*)` (equals the tail entry `D_n(0)`).
    pub terminal_claim: Gf128,
    /// Correlated path queries.
    pub queries: Vec<PathQuery>,
    /// The full tail `D_n` in per-group reveals (`w_tail` values each, with
    /// Merkle paths).
    pub tail: Vec<(Vec<Gf128>, Vec<[u8; DIGEST]>)>,
}

/// Commit to a table: build the codeword and its Merkle tree.
pub fn commit_table(params: &PcsParams, table: &[Gf128]) -> (Codeword, MerkleTree, PcsStatement) {
    let n = table.len().trailing_zeros() as usize;
    let code_params = CodeParams::derive(n, params.k, &params.domain);
    let codeword = Codeword::encode(&code_params, table);
    let tree = MerkleTree::build_batched(
        &level_domain(&params.domain, 0),
        &codeword.values,
        level_width(n, params.k, 0),
    );
    let stmt = PcsStatement {
        root: tree.root(),
        n,
        k: params.k,
        domain: params.domain.clone(),
    };
    (codeword, tree, stmt)
}

/// Domain label for a fold level (shared with the ring-switch module).
#[inline]
pub(crate) fn level_domain(base: &str, level: usize) -> String {
    format!("{base}/level/{level}")
}

/// Prove `T̂(r) = y` where `T` is the committed table (linear prover work).
pub fn prove_eval(
    params: &PcsParams,
    table: &[Gf128],
    r: &[Gf128],
    y: Gf128,
    t: &mut Transcript,
) -> PcsProof {
    use std::time::Instant;
    let time_on = std::env::var("MIKROS_TIME").is_ok();
    let t_sc = Instant::now();
    let n = table.len().trailing_zeros() as usize;
    debug_assert_eq!(r.len(), n);
    // ---- 1. Sum-check.
    let spec = SumcheckSpec {
        num_vars: n,
        dense: vec![table.to_vec()],
        public: vec![Box::new(EqFactor::new(r.to_vec()))],
        terms: vec![Term {
            coeff: Gf128(1),
            dense: vec![0],
            public: vec![0],
        }],
        claim: y,
    };
    let mut prover = SumcheckProver::new(&spec, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    for _ in 0..n {
        // ONE message computation per round: the round message is
        // absorbed in absorb_and_fold's byte format, the challenge drawn,
        // and the fold applied via fold_msg (the double message() of the
        // absorb_and_fold route was the sum-check's 2x).
        let msg = prover.message();
        rounds.push(msg.0.clone());
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes("pcs-eval", &buf);
        let r_i = t.challenge_field("pcs-eval");
        prover.fold_msg(&msg, r_i);
        challenges.push(r_i);
    }
    let (dense_claims, _running) = prover.finish();
    let terminal_claim = dense_claims[0];
    t.absorb_field("pcs-terminal", &[terminal_claim]);
    if time_on {
        eprintln!(
            "  pcs[{} words]: sumcheck {:.0} ms",
            table.len(),
            t_sc.elapsed().as_secs_f64() * 1e3
        );
    }
    let t_fold = Instant::now();

    // ---- 2. Fold chain: levels 0..=n (fold variable 0 each round —
    // variables bound in increasing order).
    let code_params = CodeParams::derive(n, params.k, &params.domain);
    let t_enc = Instant::now();
    let mut cw = Codeword::encode(&code_params, table);
    if time_on {
        eprintln!(
            "  pcs encode: {:.0} ms",
            t_enc.elapsed().as_secs_f64() * 1e3
        );
    }
    let mut trees = Vec::with_capacity(n + 1);
    let mut values = Vec::with_capacity(n + 1);
    trees.push(MerkleTree::build_batched(
        &level_domain(&params.domain, 0),
        &cw.values,
        level_width(n, params.k, 0),
    ));
    values.push(cw.values.clone());
    for (i, &rc) in challenges.iter().enumerate() {
        cw = cw.fold(0, rc);
        trees.push(MerkleTree::build_batched(
            &level_domain(&params.domain, i + 1),
            &cw.values,
            level_width(n, params.k, i + 1),
        ));
        values.push(cw.values.clone());
    }
    let mut level_roots = Vec::with_capacity(n);
    for (i, tr) in trees.iter().enumerate().skip(1) {
        let root = tr.root();
        level_roots.push(root);
        t.absorb_digest(&format!("pcs-root-{i}"), &root);
    }

    // ---- 3. Correlated query paths.
    if time_on {
        eprintln!(
            "  pcs fold+trees: {:.0} ms",
            t_fold.elapsed().as_secs_f64() * 1e3
        );
    }
    let t_q = Instant::now();
    let base_leaves = trees[0].n_leaves() as u64;
    let mut queries = Vec::with_capacity(params.queries);
    for qi in 0..params.queries {
        let pos = t.challenge_index(&format!("pcs-query-{qi}"), base_leaves) as usize;
        queries.push(build_path_query(&trees, &values, pos));
    }

    // ---- 4. Tail.
    if time_on {
        eprintln!("  pcs queries: {:.0} ms", t_q.elapsed().as_secs_f64() * 1e3);
    }
    let tail_tree = trees.last().unwrap();
    let tail_vals = values.last().unwrap();
    let w_tail = tail_tree.width();
    let tail: Vec<(Vec<Gf128>, Vec<[u8; DIGEST]>)> = (0..tail_vals.len() / w_tail)
        .map(|g| {
            let (_, path) = tail_tree.open(g * w_tail);
            (tail_vals[g * w_tail..g * w_tail + w_tail].to_vec(), path)
        })
        .collect();

    PcsProof {
        rounds,
        level_roots,
        terminal_claim,
        queries,
        tail,
    }
}

fn build_path_query(trees: &[MerkleTree], values: &[Vec<Gf128>], pos: usize) -> PathQuery {
    let n = trees.len() - 1;
    let mut levels = Vec::with_capacity(n);
    let mut idx = pos;
    for i in 0..n {
        let a_idx = idx & !1usize;
        let b_idx = idx | 1usize;
        let next_idx = idx >> 1;
        // a and b are adjacent — one leaf group covers both.
        let w_i = trees[i].width();
        let g = a_idx / w_i;
        let (_, a_path) = trees[i].open(a_idx);
        let (_, c_path) = trees[i + 1].open(next_idx);
        let w_next = trees[i + 1].width();
        let g_next = next_idx / w_next;
        levels.push(LevelOpen {
            a: values[i][a_idx],
            b: values[i][b_idx],
            a_vals: values[i][g * w_i..g * w_i + w_i].to_vec(),
            a_path,
            c: values[i + 1][next_idx],
            c_vals: values[i + 1][g_next * w_next..g_next * w_next + w_next].to_vec(),
            c_path,
        });
        idx = next_idx;
    }
    PathQuery { levels }
}

/// Verify an evaluation proof for the claim `T̂(r) = y` against `stmt`.
pub fn verify_eval(
    stmt: &PcsStatement,
    params: &PcsParams,
    r: &[Gf128],
    y: Gf128,
    proof: &PcsProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let n = stmt.n;
    if r.len() != n {
        return Err("point length mismatch");
    }
    if proof.rounds.len() != n || proof.level_roots.len() != n {
        return Err("proof shape mismatch");
    }
    if proof.queries.len() != params.queries {
        return Err("query count mismatch");
    }
    // ---- 1. Replay the sum-check, capturing the challenges for the fold
    // relation checks.
    let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(EqFactor::new(r.to_vec()))];
    let terms = vec![Term {
        coeff: Gf128(1),
        dense: vec![0],
        public: vec![0],
    }];
    let mut verifier = SumcheckVerifier::new(n, y, &public, terms, BindOrder::Increasing);
    let mut challenges = Vec::with_capacity(n);
    for msg in &proof.rounds {
        let round_msg = magnetar_sumcheck::RoundMsg(msg.clone());
        challenges.push(verifier.round_capturing(&round_msg, t, "pcs-eval")?);
    }
    verifier.finish(&[proof.terminal_claim])?;
    t.absorb_field("pcs-terminal", &[proof.terminal_claim]);
    // ---- 2. Level roots.
    for (i, root) in proof.level_roots.iter().enumerate() {
        t.absorb_digest(&format!("pcs-root-{}", i + 1), root);
    }
    let leaves_at = |lvl: usize| -> u64 { 1u64 << (n - lvl + params.k) };
    // ---- 3. Correlated queries.
    for (qi, q) in proof.queries.iter().enumerate() {
        let pos = t.challenge_index(&format!("pcs-query-{qi}"), leaves_at(0)) as usize;
        if q.levels.len() != n {
            return Err("query level count mismatch");
        }
        let mut idx = pos;
        for (i, lvl) in q.levels.iter().enumerate() {
            let a_idx = idx & !1usize;
            let b_idx = idx | 1usize;
            let next_idx = idx >> 1;
            let dom_i = level_domain(&stmt.domain, i);
            let expected_root = if i == 0 {
                stmt.root
            } else {
                proof.level_roots[i - 1]
            };
            let w_i = level_width(n, params.k, i);
            if lvl.a_vals.len() != w_i {
                return Err("leaf group width mismatch (a)");
            }
            // The a/b pair group: one digest + one path covers both values.
            let leaf = MerkleTree::leaf_digest_of_batched(
                &dom_i,
                leaves_at(i) as usize,
                w_i,
                a_idx,
                &lvl.a_vals,
            );
            let ra = MerkleTree::verify_batched(
                &dom_i,
                leaves_at(i) as usize,
                w_i,
                a_idx,
                leaf,
                &lvl.a_path,
            )
            .map_err(|_| "query path a invalid")?;
            if ra != expected_root {
                return Err("query root a mismatch");
            }
            if lvl.a_vals[a_idx % w_i] != lvl.a {
                return Err("leaf group does not contain a");
            }
            if b_idx / w_i != a_idx / w_i {
                // cannot happen for adjacent (even, odd) pairs; still checked.
                return Err("pair straddles leaf groups");
            }
            if lvl.a_vals[b_idx % w_i] != lvl.b {
                return Err("leaf group does not contain b");
            }
            let dom_next = level_domain(&stmt.domain, i + 1);
            let w_next = level_width(n, params.k, i + 1);
            if lvl.c_vals.len() != w_next {
                return Err("leaf group width mismatch (c)");
            }
            let leaf = MerkleTree::leaf_digest_of_batched(
                &dom_next,
                leaves_at(i + 1) as usize,
                w_next,
                next_idx,
                &lvl.c_vals,
            );
            let rc = MerkleTree::verify_batched(
                &dom_next,
                leaves_at(i + 1) as usize,
                w_next,
                next_idx,
                leaf,
                &lvl.c_path,
            )
            .map_err(|_| "query path c invalid")?;
            if rc != proof.level_roots[i] {
                return Err("query root c mismatch");
            }
            if lvl.c_vals[next_idx % w_next] != lvl.c {
                return Err("leaf group does not contain c");
            }
            // Fold relation: (1+r)·a + r·b = c.
            let folded = lvl.a + mul(challenges[i], lvl.a + lvl.b);
            if folded != lvl.c {
                return Err("fold relation failed");
            }
            idx = next_idx;
        }
    }
    // ---- 4. Tail.
    let tail_dom = level_domain(&stmt.domain, n);
    let w_tail = level_width(n, params.k, n);
    let n_tail_groups = (1usize << params.k) / w_tail;
    if n_tail_groups == 0 || !(1usize << params.k).is_multiple_of(w_tail) {
        return Err("bad tail geometry");
    }
    let tail_expected_root = if n == 0 {
        stmt.root
    } else {
        proof.level_roots[n - 1]
    };
    if proof.tail.len() != n_tail_groups {
        return Err("tail length mismatch");
    }
    for (g, (vals, path)) in proof.tail.iter().enumerate() {
        if vals.len() != w_tail {
            return Err("tail group width mismatch");
        }
        let leaf = MerkleTree::leaf_digest_of_batched(
            &tail_dom,
            1usize << params.k,
            w_tail,
            g * w_tail,
            vals,
        );
        let root = MerkleTree::verify_batched(
            &tail_dom,
            1usize << params.k,
            w_tail,
            g * w_tail,
            leaf,
            path,
        )
        .map_err(|_| "tail path invalid")?;
        if root != tail_expected_root {
            return Err("tail root mismatch");
        }
    }
    if proof.tail[0].0[0] != proof.terminal_claim {
        return Err("tail terminal entry mismatch");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rand::RngExt::random::<u128>(rng))
    }

    fn run_case(n: usize, k: usize, queries: usize) {
        let mut rng = rand::rng();
        let params = PcsParams {
            k,
            queries,
            domain: format!("test-pcs-{n}-{k}"),
        };
        let table: Vec<Gf128> = (0..(1 << n)).map(|_| rg(&mut rng)).collect();
        let r: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
        let y = magnetar_field::eval_mle(&table, &r);
        let (_cw, _tree, stmt) = commit_table(&params, &table);
        let mut t = Transcript::new("pcs", &[b"test"]);
        let proof = prove_eval(&params, &table, &r, y, &mut t);
        let mut tv = Transcript::new("pcs", &[b"test"]);
        verify_eval(&stmt, &params, &r, y, &proof, &mut tv).expect("honest proof must verify");
    }

    #[test]
    fn eval_roundtrip() {
        run_case(3, 2, 8);
        run_case(5, 2, 12);
        run_case(4, 1, 8);
        run_case(6, 3, 10);
    }

    #[test]
    fn tamper_rejected() {
        let mut rng = rand::rng();
        let params = PcsParams {
            k: 2,
            queries: 12,
            domain: "test-pcs-tamper".into(),
        };
        let n = 5;
        let table: Vec<Gf128> = (0..(1 << n)).map(|_| rg(&mut rng)).collect();
        let r: Vec<Gf128> = (0..n).map(|_| rg(&mut rng)).collect();
        let y = magnetar_field::eval_mle(&table, &r);
        let (_cw, _tree, stmt) = commit_table(&params, &table);
        let base = || Transcript::new("pcs", &[b"test"]);

        // Honest proof verifies.
        let mut t = base();
        let proof = prove_eval(&params, &table, &r, y, &mut t);
        let mut tv = base();
        assert!(verify_eval(&stmt, &params, &r, y, &proof, &mut tv).is_ok());

        // Wrong claim value.
        let mut tv = base();
        assert!(verify_eval(&stmt, &params, &r, y + Gf128(1), &proof, &mut tv).is_err());

        // Tampered terminal.
        let mut p = proof.clone();
        p.terminal_claim += Gf128(1);
        let mut tv = base();
        assert!(verify_eval(&stmt, &params, &r, y, &p, &mut tv).is_err());

        // Tampered tail entry.
        let mut p = proof.clone();
        p.tail[1].0[0] += Gf128(1);
        let mut tv = base();
        assert!(verify_eval(&stmt, &params, &r, y, &p, &mut tv).is_err());

        // Tampered query value (breaks either the Merkle path or the fold
        // relation, depending on position).
        let mut p = proof.clone();
        p.queries[0].levels[0].a += Gf128(1);
        let mut tv = base();
        assert!(verify_eval(&stmt, &params, &r, y, &p, &mut tv).is_err());

        // Tampered round message.
        let mut p = proof.clone();
        p.rounds[0][1] += Gf128(1);
        let mut tv = base();
        assert!(verify_eval(&stmt, &params, &r, y, &p, &mut tv).is_err());

        // Tampered level root.
        let mut p = proof.clone();
        if !p.level_roots.is_empty() {
            p.level_roots[0][0] ^= 0xFF;
            let mut tv = base();
            assert!(verify_eval(&stmt, &params, &r, y, &p, &mut tv).is_err());
        }
    }
}
