//! Ring switching (paper §5.2 / [13]): reduce a virtual evaluation claim on
//! a **bit-valued** vector to claims on its **packed** `K`-valued
//! representation, discharged against the main codeword commitment.
//!
//! ## Construction (and why it is sound)
//!
//! The bit table `w: {0,1}^n × {0,1}^7 → B` is packed as
//! `P(u) = Σ_b β_b w(u, b)` (polynomial basis: the bits of `P(u)` *are*
//! `w(u, ·)`). For a claim `ŵ(r_A, r_B) = y`:
//!
//! 1. `c := (eq(r_B, b))_b` — 128 verifier-computable coefficients.
//! 2. The **c-folded table** `C(u) := φ_c(P(u)) = Σ_b c_b w(u, b)`, so
//!    `ŵ(r_A, r_B) = Σ_u eq(r_A, u)·C(u)`.
//! 3. **Sum-check** over the `n` block variables, binding `r*`; terminal
//!    claim `Ĉ(r*) = v''`.
//! 4. **Discharge with a fold chain on C's codeword** (same challenges):
//!    the tail entry `D_n(0) = Ĉ(r*)` is checked against `v''`; the chain
//!    is authenticated by correlated path queries; and the level-0 cube
//!    slice is anchored to the *main* commitment with exact `φ_c` checks:
//!    at `q` random `x`, the verifier opens the main codeword's `(x, 0)`
//!    entry (the packed leaf `P(x)`), reads its 128 bits, and checks
//!    `C(x) = φ_c(P(x))` — exact bit extraction, no free witnesses.
//!
//! The anchors defeat the "T-reveal slack" attack (sending folded-slice
//! values as free witnesses has a 126-dimensional cheat space over `K`);
//! every accepted value here is a fold of Merkle-authenticated data or an
//! exact bit extraction. The fold-chain soundness follows the same
//! FRI-style proximity analysis as the main PCS (see crate docs).

use crate::code::{CodeParams, Codeword};
use crate::eval::{level_domain, LevelOpen, PathQuery, PcsParams, PcsStatement};
use magnetar_field::{eq_coefficients, mul, phi_c, Gf128};
use magnetar_hash::{MerkleTree, Transcript, DIGEST};
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::EqFactor,
    BindOrder, SumcheckSpec, Term,
};

/// A ring-switch proof for one bit-vector claim.
#[derive(Clone, Debug)]
pub struct RingProof {
    /// Sum-check round messages (degree 2, 3 coefficients per round).
    pub rounds: Vec<Vec<Gf128>>,
    /// Fold-level roots of the C-codeword chain, levels 0..=n (level 0 is
    /// the C-codeword's own root, pinned by the φ-anchors).
    pub level_roots: Vec<[u8; DIGEST]>,
    /// The claimed `Ĉ(r*)` (= the tail entry `D_n(0)`).
    pub terminal_claim: Gf128,
    /// Correlated path queries on the C-chain.
    pub queries: Vec<PathQuery>,
    /// The tail `D_n` (`2^k` values) with paths.
    pub tail: Vec<(Gf128, Vec<[u8; DIGEST]>)>,
    /// Level-0 anchors `(x, packed_leaf, main_path, c_value, c_path)`: the
    /// exact `φ_c` checks tying `C`'s cube slice to the main commitment.
    pub anchors: Vec<Anchor>,
}

/// One φ-anchor opening.
#[derive(Clone, Debug)]
pub struct Anchor {
    /// Cube position `x` (the table index).
    pub x: usize,
    /// The main codeword's `(x, 0)` entry: the packed leaf `P(x)`.
    pub packed_leaf: Gf128,
    pub main_path: Vec<[u8; DIGEST]>,
    /// The C-codeword's `(x, 0)` entry.
    pub c_value: Gf128,
    pub c_path: Vec<[u8; DIGEST]>,
}

/// Prove the bit-claim `ŵ(r_A, r_B) = y`, where the bit vector is given by
/// its packed table `P` (the main committed table), `r_A` are the block
/// variables and `r_B` the 7 packing (bit) variables. `main_codeword` /
/// `main_tree` belong to the main commitment.
#[allow(clippy::too_many_arguments)]
pub fn prove_ring_switch(
    params: &PcsParams,
    main_codeword: &Codeword,
    main_tree: &MerkleTree,
    packed: &[Gf128],
    r_a: &[Gf128],
    r_b: &[Gf128; 7],
    y: Gf128,
    t: &mut Transcript,
) -> RingProof {
    let n = packed.len().trailing_zeros() as usize;
    debug_assert_eq!(r_a.len(), n);
    // 1–2. c coefficients and the c-folded table.
    let c = eq_coefficients(r_b);
    let c_table: Vec<Gf128> = packed.iter().map(|&p| phi_c(p, &c)).collect();
    // 3. Sum-check over the block variables.
    let spec = SumcheckSpec {
        num_vars: n,
        dense: vec![c_table.clone()],
        public: vec![Box::new(EqFactor::new(r_a.to_vec()))],
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
        let msg = prover.message();
        rounds.push(msg.0.clone());
        challenges.push(prover.absorb_and_fold(t, "ring-eval"));
    }
    let (claims, _) = prover.finish();
    let terminal_claim = claims[0];
    t.absorb_field("ring-terminal", &[terminal_claim]);

    // 4. The C-codeword chain at the same challenges.
    let code_params = CodeParams::derive(n, params.k, &params.domain);
    let mut cw = Codeword::encode(&code_params, &c_table);
    let c_domain = format!("{}/ring", params.domain);
    let mut trees = Vec::with_capacity(n + 1);
    let mut values = Vec::with_capacity(n + 1);
    trees.push(MerkleTree::build(&level_domain(&c_domain, 0), &cw.values));
    values.push(cw.values.clone());
    for (i, &rc) in challenges.iter().enumerate() {
        cw = cw.fold(0, rc);
        trees.push(MerkleTree::build(
            &level_domain(&c_domain, i + 1),
            &cw.values,
        ));
        values.push(cw.values.clone());
    }
    let level_roots: Vec<[u8; DIGEST]> = trees.iter().map(|tr| tr.root()).collect();
    for (i, root) in level_roots.iter().enumerate() {
        t.absorb_digest(&format!("ring-root-{i}"), root);
    }

    // Correlated queries.
    let base_leaves = trees[0].n_leaves() as u64;
    let mut queries = Vec::with_capacity(params.queries);
    for qi in 0..params.queries {
        let pos = t.challenge_index(&format!("ring-query-{qi}"), base_leaves) as usize;
        queries.push(build_query(&trees, &values, pos));
    }

    // Tail.
    let tail_tree = trees.last().unwrap();
    let tail_vals = values.last().unwrap();
    let tail: Vec<(Gf128, Vec<[u8; DIGEST]>)> = (0..tail_tree.n_leaves())
        .map(|i| {
            let (_, path) = tail_tree.open(i);
            (tail_vals[i], path)
        })
        .collect();

    // Anchors: q random cube positions.
    let mut anchors = Vec::with_capacity(params.queries);
    for ai in 0..params.queries {
        let x = t.challenge_index(&format!("ring-anchor-{ai}"), 1u64 << n) as usize;
        let (_, main_path) = main_tree.open(x);
        let packed_leaf = main_codeword.values[x];
        let (_, c_path) = trees[0].open(x);
        anchors.push(Anchor {
            x,
            packed_leaf,
            main_path,
            c_value: values[0][x],
            c_path,
        });
    }

    RingProof {
        rounds,
        level_roots,
        terminal_claim,
        queries,
        tail,
        anchors,
    }
}

fn build_query(trees: &[MerkleTree], values: &[Vec<Gf128>], pos: usize) -> PathQuery {
    let n = trees.len() - 1;
    let mut levels = Vec::with_capacity(n);
    let mut idx = pos;
    for i in 0..n {
        let a_idx = idx & !1usize;
        let b_idx = idx | 1usize;
        let next_idx = idx >> 1;
        let (_, a_path) = trees[i].open(a_idx);
        let (_, b_path) = trees[i].open(b_idx);
        let (_, c_path) = trees[i + 1].open(next_idx);
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
    if proof.rounds.len() != n || proof.level_roots.len() != n + 1 {
        return Err("ring: proof shape mismatch");
    }
    if proof.queries.len() != params.queries || proof.anchors.len() != params.queries {
        return Err("ring: query/anchor count mismatch");
    }
    let c = eq_coefficients(r_b);
    // ---- 1. Sum-check replay (captures the fold challenges).
    let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(EqFactor::new(r_a.to_vec()))];
    let terms = vec![Term {
        coeff: Gf128(1),
        dense: vec![0],
        public: vec![0],
    }];
    let mut verifier = SumcheckVerifier::new(n, y, &public, terms, BindOrder::Increasing);
    let mut challenges = Vec::with_capacity(n);
    for msg in &proof.rounds {
        let round_msg = magnetar_sumcheck::RoundMsg(msg.clone());
        challenges.push(verifier.round_capturing(&round_msg, t, "ring-eval")?);
    }
    verifier.finish(&[proof.terminal_claim])?;
    t.absorb_field("ring-terminal", &[proof.terminal_claim]);
    for (i, root) in proof.level_roots.iter().enumerate() {
        t.absorb_digest(&format!("ring-root-{i}"), root);
    }
    let c_domain = format!("{}/ring", params.domain);
    let leaves_at = |lvl: usize| -> u64 { 1u64 << (n - lvl + params.k) };
    // ---- 2. Correlated queries.
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
            let dom_i = level_domain(&c_domain, i);
            let leaf = MerkleTree::leaf_digest_of(&dom_i, leaves_at(i) as usize, a_idx, lvl.a);
            let ra = MerkleTree::verify(&dom_i, leaves_at(i) as usize, a_idx, leaf, &lvl.a_path)
                .map_err(|_| "ring: query path a invalid")?;
            if ra != proof.level_roots[i] {
                return Err("ring: query root a mismatch");
            }
            let leaf = MerkleTree::leaf_digest_of(&dom_i, leaves_at(i) as usize, b_idx, lvl.b);
            let rb = MerkleTree::verify(&dom_i, leaves_at(i) as usize, b_idx, leaf, &lvl.b_path)
                .map_err(|_| "ring: query path b invalid")?;
            if rb != proof.level_roots[i] {
                return Err("ring: query root b mismatch");
            }
            let dom_next = level_domain(&c_domain, i + 1);
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
            if rc != proof.level_roots[i + 1] {
                return Err("ring: query root c mismatch");
            }
            let folded = lvl.a + mul(challenges[i], lvl.a + lvl.b);
            if folded != lvl.c {
                return Err("ring: fold relation failed");
            }
            idx = next_idx;
        }
    }
    // ---- 3. Tail.
    if proof.tail.len() != (1usize << params.k) {
        return Err("ring: tail length mismatch");
    }
    let tail_dom = level_domain(&c_domain, n);
    for (i, (val, path)) in proof.tail.iter().enumerate() {
        let leaf = MerkleTree::leaf_digest_of(&tail_dom, 1usize << params.k, i, *val);
        let root = MerkleTree::verify(&tail_dom, 1usize << params.k, i, leaf, path)
            .map_err(|_| "ring: tail path invalid")?;
        if root != proof.level_roots[n] {
            return Err("ring: tail root mismatch");
        }
    }
    if proof.tail[0].0 != proof.terminal_claim {
        return Err("ring: tail terminal entry mismatch");
    }
    // ---- 4. Anchors: φ_c ties the C-chain's cube slice to the main
    // commitment.
    let main_leaves = 1u64 << (n + params.k);
    for anchor in &proof.anchors {
        if anchor.x >= (1usize << n) {
            return Err("ring: anchor index out of range");
        }
        // Main side: level-0 main codeword at (x, 0) — leaf index x.
        let main_leaf = MerkleTree::leaf_digest_of(
            &level_domain(&main_stmt.domain, 0),
            main_leaves as usize,
            anchor.x,
            anchor.packed_leaf,
        );
        let main_root = MerkleTree::verify(
            &level_domain(&main_stmt.domain, 0),
            main_leaves as usize,
            anchor.x,
            main_leaf,
            &anchor.main_path,
        )
        .map_err(|_| "ring: anchor main path invalid")?;
        if main_root != main_stmt.root {
            return Err("ring: anchor main root mismatch");
        }
        // Exact φ_c check.
        if phi_c(anchor.packed_leaf, &c) != anchor.c_value {
            return Err("ring: anchor phi_c mismatch");
        }
        // C side: the C-chain's level-0 entry at x.
        let c_leaf = MerkleTree::leaf_digest_of(
            &level_domain(&c_domain, 0),
            leaves_at(0) as usize,
            anchor.x,
            anchor.c_value,
        );
        let c_root = MerkleTree::verify(
            &level_domain(&c_domain, 0),
            leaves_at(0) as usize,
            anchor.x,
            c_leaf,
            &anchor.c_path,
        )
        .map_err(|_| "ring: anchor c path invalid")?;
        if c_root != proof.level_roots[0] {
            return Err("ring: anchor c root mismatch");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rng.gen::<u128>())
    }

    fn pack_random_bits(rng: &mut impl rand::Rng, rows: usize) -> (Vec<bool>, Vec<Gf128>) {
        let bits: Vec<bool> = (0..rows * 128).map(|_| rng.gen::<bool>()).collect();
        let bytes: Vec<u8> = bits.iter().map(|&b| b as u8).collect();
        let packed = magnetar_field::pack_table(&bytes);
        (bits, packed)
    }

    #[test]
    fn ring_roundtrip_and_tamper() {
        use magnetar_field::eq_table;
        use rand::Rng;
        let mut rng = rand::thread_rng();
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
                y = y + magnetar_field::mul(
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
        p.terminal_claim = p.terminal_claim + Gf128(1);
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &p, &mut tv).is_err());
        // Tampered anchor (phi_c mismatch).
        let mut p = proof.clone();
        p.anchors[0].c_value = p.anchors[0].c_value + Gf128(1);
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &p, &mut tv).is_err());
        // Tampered query fold value.
        let mut p = proof.clone();
        p.queries[0].levels[0].a = p.queries[0].levels[0].a + Gf128(1);
        let mut tv = Transcript::new("ring", &[b"t"]);
        assert!(verify_ring_switch(&stmt, &params, &r_a, &r_b, y, &p, &mut tv).is_err());
    }
}
