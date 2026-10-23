//! MosaicPCS (paper §5.2): pack everything the prover commits into one
//! multilinear polynomial over the hash-based PCS, so each proof carries
//! one commitment and one evaluation argument.
//!
//! **Component layout**: every committed polynomial — the direct witness
//! segments and the auxiliary polynomials of the memory-checking arguments
//! (the one-hot address components and the increments) — is a bit-valued
//! `p_i: {0,1}^{ℓ_i} → B` placed on a disjoint, **aligned subcube** of one
//! larger cube `{0,1}^ℓ` (declared order, offsets aligned to the
//! component's size, exactly as `SegmentLayout` does for the witness).
//! `G` agrees with `p_i` on the i-th subcube and is zero outside the union.
//!
//! **Lift** (Equation 4): a claim `p̂_i(r) = v` becomes `Ĝ(r ∥ s_i) = v` by
//! appending the fixed Boolean suffix `s_i` — no adapter sum-check.
//!
//! **Discharge**: one batching sum-check over the `ℓ` variables (binding
//! the 7 packing variables first via `HighBitsFirst`), whose terminal is
//! reduced by **ring switching** — the c\*-folded table's own fold chain,
//! anchored to the main commitment with exact `φ_{c*}` bit-extraction
//! checks at random `(x, ε=0)` positions.

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

/// One committed component's placement.
#[derive(Clone, Debug)]
pub struct ComponentSpec {
    pub name: String,
    /// The component's variable count (its table has `2^vars` bits).
    pub vars: usize,
}

/// The Mosaic component layout: declared order, aligned offsets.
#[derive(Clone, Debug)]
pub struct MosaicLayout {
    pub components: Vec<ComponentSpec>,
    /// Total cube variables `ℓ`.
    pub ell: usize,
    /// `offsets[i]` = the component's start bit-index in the cube.
    pub offsets: Vec<usize>,
}

impl MosaicLayout {
    pub fn build(components: Vec<ComponentSpec>) -> Self {
        assert!(!components.is_empty());
        let mut offsets = Vec::with_capacity(components.len());
        let mut cursor = 0usize;
        for c in &components {
            let align = 1usize << c.vars;
            let aligned = cursor.div_ceil(align) * align;
            offsets.push(aligned);
            cursor = aligned + align;
        }
        let ell = (cursor.next_power_of_two().trailing_zeros() as usize).max(1);
        MosaicLayout {
            components,
            ell,
            offsets,
        }
    }

    /// The component's local variable count.
    pub fn vars(&self, i: usize) -> usize {
        self.components[i].vars
    }

    /// The fixed Boolean suffix `s_i` (the high `ℓ − ℓ_i` coordinates of
    /// the component's subcube).
    pub fn suffix(&self, i: usize) -> Vec<bool> {
        let lv = self.vars(i);
        let off = self.offsets[i];
        let n = self.ell - lv;
        let s = off >> lv;
        (0..n).map(|b| (s >> b) & 1 == 1).collect()
    }

    /// The index of a component by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.components.iter().position(|c| c.name == name)
    }
}

/// A claim on one component: `p̂_i(point) = value` with the point in the
/// component's local (little-endian) coordinates.
#[derive(Clone, Debug)]
pub struct MosaicClaim {
    pub component: usize,
    pub point: Vec<Gf128>,
    pub value: Gf128,
}

impl MosaicClaim {
    pub fn new(component: usize, point: Vec<Gf128>, value: Gf128) -> Self {
        MosaicClaim {
            component,
            point,
            value,
        }
    }

    /// Lift to the global point: local ∥ Boolean suffix.
    pub fn lifted_point(&self, layout: &MosaicLayout) -> Vec<Gf128> {
        let suffix = layout.suffix(self.component);
        let mut pt = self.point.clone();
        for b in suffix {
            pt.push(Gf128(b as u128));
        }
        pt
    }
}

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

/// Assemble the G bit table from the per-component bit tables (zero
/// outside the union of the subcubes).
fn assemble_g(layout: &MosaicLayout, tables: &[Vec<bool>]) -> Vec<bool> {
    let mut g = vec![false; 1usize << layout.ell];
    for (i, t) in tables.iter().enumerate() {
        let lv = layout.vars(i);
        let off = layout.offsets[i];
        debug_assert_eq!(t.len(), 1usize << lv);
        for (j, &b) in t.iter().enumerate() {
            g[off + j] = b;
        }
    }
    g
}

/// Commit the Mosaic: `tables[i]` is component `i`'s bit table.
pub fn commit_mosaic(
    params: &PcsParams,
    layout: &MosaicLayout,
    tables: &[Vec<bool>],
) -> MosaicCommitment {
    assert_eq!(tables.len(), layout.components.len());
    let ell = layout.ell;
    assert!(ell >= 7, "witness must have at least 7 variables");
    let bits = assemble_g(layout, tables);
    // Pack over the top 7 bits: P(u) = Σ_b β_b G(u, b) with u the low
    // ℓ−7 bits.
    let n = ell - 7;
    let mut packed = vec![ZERO; 1usize << n];
    for u in 0..(1usize << n) {
        let mut v: u128 = 0;
        for b in 0..128usize {
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
    /// The ZK round-mask offset (`magnetar_sumcheck::zk`), present when
    /// the proving key runs the zero-knowledge profile: the batching
    /// sum-check's messages are blinded per round and this one revealed
    /// value restores the terminal identity.
    pub zk_offset: Option<Gf128>,
}

/// Prove the Mosaic discharge for a set of component claims. When `zk`
/// is set the batching sum-check's round messages are zero-knowledge
/// masked (see `magnetar_sumcheck::zk`): each message is blinded with a
/// fresh random polynomial subject to the running constraint, so the
/// committed witness bits do not leak through the transcript.
pub fn prove_mosaic(
    params: &PcsParams,
    layout: &MosaicLayout,
    cm: &MosaicCommitment,
    claims: &[MosaicClaim],
    zk: bool,
    t: &mut Transcript,
) -> MosaicProof {
    let ell = cm.ell;
    // α-batching of the (lifted) claims.
    let alphas = t.challenge_field_vec("mosaic-alpha", claims.len());
    let mut lhs = ZERO;
    let mut items = Vec::with_capacity(claims.len());
    for (c, &a) in claims.iter().zip(alphas.iter()) {
        lhs += mul(a, c.value);
        items.push((a, c.lifted_point(layout)));
    }
    // Batching sum-check: Σ_x [Σ_i α_i eq(p_i, x)]·G(x) = lhs over the ℓ
    // variables, binding the top 7 first.
    let g_table: Vec<Gf128> = {
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
    let mut prover = SumcheckProver::new_owned(spec, order);
    let mut masker = magnetar_sumcheck::zk::ZkMasker::new();
    let mut batch_rounds = Vec::with_capacity(ell);
    for _ in 0..ell {
        let mut msg = prover.message_cached();
        if zk {
            masker.mask(&mut msg);
        }
        batch_rounds.push(msg.0.clone());
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes("mosaic-batch", &buf);
        let r = t.challenge_field("mosaic-batch");
        if zk {
            masker.commit(r);
        }
        prover.fold(r);
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
        zk_offset: zk.then(|| masker.offset()),
    }
}

/// Verify the Mosaic discharge.
pub fn verify_mosaic(
    params: &PcsParams,
    cm_stmt: &PcsStatement,
    ell: usize,
    layout: &MosaicLayout,
    claims: &[MosaicClaim],
    proof: &MosaicProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let zk = proof.zk_offset.is_some();
    let alphas = t.challenge_field_vec("mosaic-alpha", claims.len());
    let mut lhs = ZERO;
    let mut items = Vec::with_capacity(claims.len());
    for (c, &a) in claims.iter().zip(alphas.iter()) {
        lhs += mul(a, c.value);
        items.push((a, c.lifted_point(layout)));
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
    match (zk, proof.zk_offset) {
        (true, Some(off)) => {
            v.finish_offset(&[proof.terminal], off)
                .map_err(|_| "mosaic: batching terminal failed")?;
        }
        (false, None) => {
            v.finish(&[proof.terminal])
                .map_err(|_| "mosaic: batching terminal failed")?;
        }
        _ => return Err("mosaic: zk offset presence mismatch"),
    }
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
}
