//! MosaicPCS (paper §5.2): pack everything the prover commits into one
//! multilinear polynomial over the hash-based PCS, so each proof carries
//! one commitment and one evaluation argument.
//!
//! Layout: the bit-witness `G: {0,1}^ℓ → B` (the witness matrix in
//! row-major layout: columns low, cycles high) is packed over the **top 7**
//! index bits: `P_G(u) = Σ_b β_b·G(u, b)`, `u ∈ {0,1}^{ℓ−7}`. The
//! commitment is the Merkle root of the affine-substitution codeword of
//! `P_G`; the `(x, ε=0)` slice is exactly the packed table.
//!
//! Discharge: one **batching sum-check** over the ℓ variables binding the
//! 7 packing variables first (`HighBitsFirst`), whose terminal
//! `Ĝ(r*) = v''` is then reduced by **ring switching** — the c\*-folded
//! table's own fold chain, anchored to the main commitment with exact
//! `φ_{c*}` bit-extraction checks at random `(x, 0)` positions.

use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::MerkleTree;
use magnetar_hash::Transcript;
use magnetar_pcs::code::{CodeParams, Codeword};
use magnetar_pcs::{
    ring::{prove_ring_switch, verify_ring_switch, RingProof},
    PcsParams, PcsStatement,
};
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::SumEqFactor,
    BindOrder, SumcheckSpec, Term,
};

/// A Mosaic commitment: the packed codeword, its tree, and the statement.
#[derive(Clone, Debug)]
pub struct MosaicCommitment {
    pub packed: Vec<Gf128>,
    pub codeword: Codeword,
    pub tree: MerkleTree,
    pub stmt: PcsStatement,
    /// ℓ (total bit-witness variables).
    pub ell: usize,
}

/// Commit the bit-witness matrix (row-major: columns low, cycles high;
/// total bits = `2^ℓ`).
pub fn commit_mosaic(params: &PcsParams, bits: &[bool]) -> MosaicCommitment {
    let ell = bits.len().trailing_zeros() as usize;
    assert!(ell >= 7, "witness must have at least 7 variables");
    // Pack over the top 7 bits: P(u) = Σ_b β_b G(u, b) with u the low
    // ℓ−7 bits.
    let n = ell - 7;
    let mut packed = vec![ZERO; 1usize << n];
    for u in 0..(1usize << n) {
        let mut v: u128 = 0;
        for b in 0..128usize {
            // The full index: u + (b << n).
            if bits[u | (b << n)] {
                v |= 1u128 << b;
            }
        }
        packed[u] = Gf128(v);
    }
    let code_params = CodeParams::derive(n, params.k, &params.domain);
    let codeword = Codeword::encode(&code_params, &packed);
    let tree = MerkleTree::build(&format!("{}/level/0", params.domain), &codeword.values);
    let stmt = PcsStatement {
        root: tree.root(),
        n,
        k: params.k,
        domain: params.domain.clone(),
    };
    MosaicCommitment {
        packed,
        codeword,
        tree,
        stmt,
        ell,
    }
}

/// The Mosaic discharge proof: the batching sum-check + the ring switch.
#[derive(Clone, Debug)]
pub struct MosaicProof {
    /// Batching sum-check rounds (degree 2).
    pub batch_rounds: Vec<Vec<Gf128>>,
    /// The terminal claim `Ĝ(r*) = v''`.
    pub terminal: Gf128,
    /// The ring-switch discharge.
    pub ring: RingProof,
}

/// Prove the Mosaic discharge for a set of claims on the bit-witness.
///
/// Each claim is `(point, value)` with `Ĝ(point) = value` (points in
/// original ℓ-coordinate order, little-endian).
pub fn prove_mosaic(
    params: &PcsParams,
    cm: &MosaicCommitment,
    claims: &[(Vec<Gf128>, Gf128)],
    t: &mut Transcript,
) -> MosaicProof {
    let ell = cm.ell;
    // α-batching of the claims.
    let alphas = t.challenge_field_vec("mosaic-alpha", claims.len());
    let mut lhs = ZERO;
    let mut items = Vec::with_capacity(claims.len());
    for ((pt, v), &a) in claims.iter().zip(alphas.iter()) {
        lhs = lhs + mul(a, *v);
        items.push((a, pt.clone()));
    }
    // Batching sum-check: Σ_x [Σ_i α_i eq(p_i, x)]·G(x) = lhs over the ℓ
    // variables, binding the top 7 first.
    let g_table: Vec<Gf128> = {
        // The G bit table as K values over the full ℓ-cube.
        // Reconstructed from the packed table.
        let mut g = vec![ZERO; 1usize << ell];
        for u in 0..(1usize << (ell - 7)) {
            let bits = magnetar_field::unpack_bits(cm.packed[u]);
            for b in 0..128usize {
                g[u | (b << (ell - 7))] = Gf128(bits[b] as u128);
            }
        }
        g
    };
    let spec = SumcheckSpec {
        num_vars: ell,
        dense: vec![g_table],
        public: vec![Box::new(SumEqFactor::new(items))],
        terms: vec![Term {
            coeff: ONE,
            dense: vec![0],
            public: vec![0],
        }],
        claim: lhs,
    };
    let order = BindOrder::HighBitsFirst { high: 7 };
    let mut prover = SumcheckProver::new(&spec, order);
    let mut batch_rounds = Vec::with_capacity(ell);
    for _ in 0..ell {
        let msg = prover.message();
        batch_rounds.push(msg.0.clone());
        prover.absorb_and_fold(t, "mosaic-batch");
    }
    let challenges = prover.challenges();
    let (dense_claims, _) = prover.finish();
    let terminal = dense_claims[0];
    t.absorb_field("mosaic-terminal", &[terminal]);
    let r_star = order.point(ell, &challenges);
    // Split: r_block (low ℓ−7), r_bit (top 7, in variable order 7..ℓ).
    let r_block: Vec<Gf128> = r_star[..ell - 7].to_vec();
    let r_bit_arr: [Gf128; 7] = {
        let mut a = [ZERO; 7];
        for j in 0..7 {
            a[j] = r_star[ell - 7 + j];
        }
        a
    };
    // Ring switch: the claim Ĝ(r*) = terminal with the packing over the
    // top 7 bits.
    let ring = prove_ring_switch(
        params,
        &cm.codeword,
        &cm.tree,
        &cm.packed,
        &r_block,
        &r_bit_arr,
        terminal,
        t,
    );
    MosaicProof {
        batch_rounds,
        terminal,
        ring,
    }
}

/// Verify the Mosaic discharge.
pub fn verify_mosaic(
    params: &PcsParams,
    cm_stmt: &PcsStatement,
    ell: usize,
    claims: &[(Vec<Gf128>, Gf128)],
    proof: &MosaicProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let alphas = t.challenge_field_vec("mosaic-alpha", claims.len());
    let mut lhs = ZERO;
    let mut items = Vec::with_capacity(claims.len());
    for ((pt, v), &a) in claims.iter().zip(alphas.iter()) {
        lhs = lhs + mul(a, *v);
        items.push((a, pt.clone()));
    }
    if proof.batch_rounds.len() != ell {
        return Err("mosaic: round count mismatch");
    }
    let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(SumEqFactor::new(items))];
    let terms = vec![Term {
        coeff: ONE,
        dense: vec![0],
        public: vec![0],
    }];
    let order = BindOrder::HighBitsFirst { high: 7 };
    let mut v = SumcheckVerifier::new(ell, lhs, &public, terms, order);
    let mut challenges = Vec::with_capacity(ell);
    for msg in &proof.batch_rounds {
        challenges.push(v.round_capturing(
            &magnetar_sumcheck::RoundMsg(msg.clone()),
            t,
            "mosaic-batch",
        )?);
    }
    v.finish(&[proof.terminal])
        .map_err(|_| "mosaic: batching terminal failed")?;
    t.absorb_field("mosaic-terminal", &[proof.terminal]);
    let r_star = order.point(ell, &challenges);
    let r_block: Vec<Gf128> = r_star[..ell - 7].to_vec();
    let r_bit_arr: [Gf128; 7] = {
        let mut a = [ZERO; 7];
        for j in 0..7 {
            a[j] = r_star[ell - 7 + j];
        }
        a
    };
    verify_ring_switch(
        cm_stmt,
        params,
        &r_block,
        &r_bit_arr,
        proof.terminal,
        &proof.ring,
        t,
    )
    .map_err(|e| match e {
        e => e,
    })
}
