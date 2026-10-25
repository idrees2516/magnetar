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

use magnetar_field::{Gf128, ZERO};
use magnetar_hash::MerkleTree;
use magnetar_hash::Transcript;
use magnetar_pcs::code::{CodeParams, Codeword};
use magnetar_pcs::{
    ring::{prove_ring_switch, verify_ring_switch, RingProof},
    PcsParams, PcsStatement,
};

use rayon::prelude::*;

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
/// outside the union of the subcubes). Test-only since the streaming
/// pack ([`pack_tables`]) replaced the union materialization.
#[cfg(test)]
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

/// One rectangular placement of a component inside the packed leaf
/// vector: a contiguous leaf range written into one bit-plane `p`. Under
/// the **split packing convention** the g-index's bits decompose as
/// `[high | plane (7) | low (split)]` and map to
/// `leaf = [high | low]`, `plane` — the leaf vector is indexed
/// `[instance | low]` when the layout stacks instances (the streaming
/// root's per-segment word ranges stay contiguous).
#[derive(Clone, Copy, Debug)]
struct PackPiece {
    table: usize,
    plane: usize,
    leaf_lo: usize,
    leaf_hi: usize,
    /// The table index at `leaf_lo`.
    j_lo: usize,
}

/// The pack geometry under the split convention. `split` = the low-bit
/// count of the leaf index below the 7 plane bits (the monolithic path:
/// `layout.ell - 7`; the block-aligned streaming layout: the per-segment
/// `ell_seg - 7`).
fn pack_pieces_split(
    layout: &MosaicLayout,
    first: usize,
    tables: &[Vec<bool>],
    split: usize,
) -> Vec<PackPiece> {
    let mut pieces = Vec::with_capacity(tables.len() + 128);
    for (i, t) in tables.iter().enumerate() {
        let off = layout.offsets[first + i];
        let size = 1usize << layout.vars(first + i);
        let vars = layout.vars(first + i);
        debug_assert_eq!(t.len(), size);
        // The offset is aligned to the component's size, so the g-range's
        // bits above `vars` are fixed. Components never span a
        // `split + 7` boundary (vars < split + 7 always holds: the largest
        // component is far smaller than the per-instance union).
        debug_assert!(vars <= split + 7);
        if vars <= split {
            // One plane; a single contiguous leaf range.
            let plane = (off >> split) & 127;
            let leaf_lo = ((off >> (split + 7)) << split) | (off & ((1usize << split) - 1));
            pieces.push(PackPiece {
                table: i,
                plane,
                leaf_lo,
                leaf_hi: leaf_lo + size - 1,
                j_lo: 0,
            });
        } else {
            // The component spans `vars - split` planes; each covers the
            // full low range of `2^split` leaves.
            let p_lo = (off >> split) & 127;
            let p_count = 1usize << (vars - split);
            let leaf_base = (off >> (split + 7)) << split; // the low bits start at 0
            for (k, p) in (p_lo..p_lo + p_count).enumerate() {
                pieces.push(PackPiece {
                    table: i,
                    plane: p,
                    leaf_lo: leaf_base,
                    leaf_hi: leaf_base + (1usize << split) - 1,
                    j_lo: k << split,
                });
            }
        }
    }
    pieces
}

/// Pack the component bit tables straight into the leaf vector — the
/// streaming commitment path (docs/PERFORMANCE.md R5/R7): no `2^ell`-bit
/// boolean union is materialized, and the leaves pack in parallel with
/// disjoint writes. Bit-identical to the `assemble_g` + pack loop.
pub fn pack_tables(layout: &MosaicLayout, tables: &[Vec<bool>], packed: &mut [Gf128]) {
    pack_tables_slice(layout, 0, tables, packed);
}

/// Pack `tables` — the components `[first, first + tables.len())` of the
/// (possibly combined) `layout` — into the leaf vector by **OR-ing** the
/// components' planes into the existing leaf words. Disjoint component
/// ranges make sequential slice packs compose into the full pack: this
/// is the per-segment streaming commitment entry (the leaf words overlap
/// across segments — different planes of the same word — so the caller
/// serializes slice packs).
pub fn pack_tables_slice(
    layout: &MosaicLayout,
    first: usize,
    tables: &[Vec<bool>],
    packed: &mut [Gf128],
) {
    pack_tables_slice_split(layout, layout.ell - 7, first, tables, packed)
}

/// The split-convention pack: `split` bits below the 7 plane bits (the
/// block-aligned streaming layout packs per-segment word ranges).
pub fn pack_tables_slice_split(
    layout: &MosaicLayout,
    split: usize,
    first: usize,
    tables: &[Vec<bool>],
    packed: &mut [Gf128],
) {
    let n = layout.ell - 7;
    assert_eq!(packed.len(), 1usize << n);
    let pieces = pack_pieces_split(layout, first, tables, split);
    pack_by_pieces(&pieces, tables, packed);
}

/// The parallel piece write: OR the pieces' planes into the leaves.
fn pack_by_pieces(pieces: &[PackPiece], tables: &[Vec<bool>], packed: &mut [Gf128]) {
    // R2: below the threshold one chunk (sequential); above, multiple
    // tasks per core, never smaller than the threshold.
    let chunk = if packed.len() < magnetar_field::PAR_THRESHOLD {
        packed.len().max(1)
    } else {
        (packed.len() / (rayon::current_num_threads().max(1) * 4))
            .clamp(magnetar_field::PAR_THRESHOLD, packed.len())
    };
    packed
        .par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, out)| {
            let base = ci * chunk;
            for (rel, leaf) in out.iter_mut().enumerate() {
                let u = base + rel;
                let mut v: u128 = 0;
                for pc in pieces.iter() {
                    if u >= pc.leaf_lo && u <= pc.leaf_hi {
                        let j = pc.j_lo + (u - pc.leaf_lo);
                        if tables[pc.table][j] {
                            v |= 1u128 << pc.plane;
                        }
                    }
                }
                *leaf = Gf128(leaf.0 | v);
            }
        });
}

/// Commit the Mosaic: `tables[i]` is component `i`'s bit table.
pub fn commit_mosaic(
    params: &PcsParams,
    layout: &MosaicLayout,
    tables: &[Vec<bool>],
) -> MosaicCommitment {
    commit_mosaic_split(params, layout, layout.ell - 7, tables)
}

/// Commit under the split packing convention (the block-aligned streaming
/// layout: `split` = the per-segment `ell_seg − 7`).
pub fn commit_mosaic_split(
    params: &PcsParams,
    layout: &MosaicLayout,
    split: usize,
    tables: &[Vec<bool>],
) -> MosaicCommitment {
    assert_eq!(tables.len(), layout.components.len());
    let ell = layout.ell;
    assert!(ell >= 7, "witness must have at least 7 variables");
    // P packs the 7 plane bits into the word bits: P(word) = Σ_b β_b
    // G(word, b) — the boolean union is never materialized.
    let n = ell - 7;
    let mut packed = vec![ZERO; 1usize << n];
    {
        let pieces = pack_pieces_split(layout, 0, tables, split);
        pack_by_pieces(&pieces, tables, &mut packed);
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

/// The Mosaic discharge proof: the hierarchical batching sum-checks (the
/// per-instance local stages + the segment stage) + the ring switch.
#[derive(Clone, Debug)]
pub struct MosaicProof {
    /// The per-instance local rounds (degree 2, 3 coefficients each) —
    /// masked when ZK is enabled (single-instance proofs only).
    pub local_rounds: Vec<Vec<Vec<Gf128>>>,
    /// The per-instance terminals `Ĝ_s(r_local) = v_s`.
    pub terminals: Vec<Gf128>,
    /// The segment-stage rounds (`log2(segments)`; empty for one
    /// instance).
    pub seg_rounds: Vec<Vec<Gf128>>,
    /// The terminal claim `Ĝ(r*) = v''`.
    pub terminal: Gf128,
    /// The ZK finish offset (ZERO for plain proofs): the residual of the
    /// per-round mask polynomials, revealed so the verifier's terminal
    /// identity closes. Uniform on its own — it hides the honest
    /// terminal's correlation with the witness.
    pub zk_offset: Gf128,
    /// The ring-switch discharge at the chunk rate (plain proofs).
    pub ring: Option<RingProof>,
    /// The dual-rate wrap (roadmap H2): when present, the deep-rate
    /// discharge REPLACES the chunk-rate ring switch — the public
    /// artifact's evaluation tail carries the deep rate's 14 queries.
    pub wrap: Option<crate::wrap::WrapProof>,
}

/// Prove the Mosaic discharge for a set of component claims (the
/// monolithic path: ONE instance).
pub fn prove_mosaic(
    params: &PcsParams,
    layout: &MosaicLayout,
    cm: &MosaicCommitment,
    claims: &[MosaicClaim],
    zk: bool,
    t: &mut Transcript,
) -> MosaicProof {
    prove_mosaic_opt(params, layout, cm, claims, zk, None, t)
}

/// Prove the Mosaic discharge, optionally **wrapped** at a deep rate
/// (roadmap H2's dual-rate wrap): the chunk side keeps its fast profile;
/// the public evaluation tail is discharged at the low rate.
#[allow(clippy::too_many_arguments)]
pub fn prove_mosaic_opt(
    params: &PcsParams,
    layout: &MosaicLayout,
    cm: &MosaicCommitment,
    claims: &[MosaicClaim],
    zk: bool,
    deep: Option<&PcsParams>,
    t: &mut Transcript,
) -> MosaicProof {
    prove_mosaic_hier(params, layout, 1, cm, claims, zk, deep, t)
}

/// Prove the **hierarchical** Mosaic discharge: `segments` instances
/// (tree-IVC segments sharing one block-aligned commitment) discharged in
/// lockstep plus a `log2(segments)`-round segment stage. See
/// [`crate::discharge`] for the design and the memory bound.
#[allow(clippy::too_many_arguments)]
pub fn prove_mosaic_hier(
    params: &PcsParams,
    per: &MosaicLayout,
    segments: usize,
    cm: &MosaicCommitment,
    claims: &[MosaicClaim],
    zk: bool,
    deep: Option<&PcsParams>,
    t: &mut Transcript,
) -> MosaicProof {
    // α-batching of the (combined-layout) claims.
    let alphas = t.challenge_field_vec("mosaic-alpha", claims.len());
    let (body, r_star) =
        crate::discharge::prove_hier_body(per, segments, &cm.packed, claims, &alphas, zk, t);
    let terminal = body.terminal;
    t.absorb_field("mosaic-terminal", &[terminal]);
    t.absorb_field("mosaic-zk-offset", &[body.zk_offset]);
    // The split packing: the word dimension is [local u coords] ∥
    // [segment coords], the 7 word bits are the LOCAL plane coords.
    let ell_s = per.ell;
    let mut r_block: Vec<Gf128> = r_star[..ell_s - 7].to_vec();
    r_block.extend_from_slice(&r_star[ell_s..]);
    let mut r_bit_arr = [ZERO; 7];
    for j in 0..7 {
        r_bit_arr[j] = r_star[ell_s - 7 + j];
    }
    let _pt_ring = crate::protocol::PhaseTimer::new("7c-ring-switch");
    let (ring, wrap) = match deep {
        Some(deep_params) => {
            let w = crate::wrap::prove_wrap(
                deep_params,
                &cm.tree,
                &cm.stmt.domain,
                &cm.packed,
                &r_block,
                &r_bit_arr,
                terminal,
                t,
            );
            (None, Some(w))
        }
        None => (
            Some(prove_ring_switch(
                params,
                &cm.codeword,
                &cm.tree,
                &cm.packed,
                &r_block,
                &r_bit_arr,
                terminal,
                t,
            )),
            None,
        ),
    };
    MosaicProof {
        local_rounds: body.local_rounds,
        terminals: body.terminals,
        seg_rounds: body.seg_rounds,
        terminal,
        zk_offset: body.zk_offset,
        ring,
        wrap,
    }
}

/// Verify the Mosaic discharge (the monolithic path: ONE instance).
pub fn verify_mosaic(
    params: &PcsParams,
    cm_stmt: &PcsStatement,
    ell: usize,
    layout: &MosaicLayout,
    claims: &[MosaicClaim],
    proof: &MosaicProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    verify_mosaic_hier(params, cm_stmt, ell, layout, 1, claims, proof, t)
}

/// Verify the **hierarchical** Mosaic discharge.
#[allow(clippy::too_many_arguments)]
pub fn verify_mosaic_hier(
    params: &PcsParams,
    cm_stmt: &PcsStatement,
    ell_combined: usize,
    per: &MosaicLayout,
    segments: usize,
    claims: &[MosaicClaim],
    proof: &MosaicProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let ell_c = per.ell + segments.trailing_zeros() as usize;
    if ell_c != ell_combined {
        return Err("mosaic: combined cube mismatch");
    }
    let alphas = t.challenge_field_vec("mosaic-alpha", claims.len());
    let body = crate::discharge::HierBody {
        local_rounds: proof.local_rounds.clone(),
        terminals: proof.terminals.clone(),
        seg_rounds: proof.seg_rounds.clone(),
        terminal: proof.terminal,
        zk_offset: proof.zk_offset,
    };
    let r_star = crate::discharge::verify_hier_body(per, segments, claims, &alphas, &body, t)?;
    t.absorb_field("mosaic-terminal", &[proof.terminal]);
    t.absorb_field("mosaic-zk-offset", &[proof.zk_offset]);
    // The split packing (see prove_mosaic_hier).
    let ell_s = per.ell;
    let mut r_block: Vec<Gf128> = r_star[..ell_s - 7].to_vec();
    r_block.extend_from_slice(&r_star[ell_s..]);
    let mut r_bit_arr = [ZERO; 7];
    for j in 0..7 {
        r_bit_arr[j] = r_star[ell_s - 7 + j];
    }
    match (&proof.ring, &proof.wrap) {
        (Some(ring), None) => verify_ring_switch(
            cm_stmt,
            params,
            &r_block,
            &r_bit_arr,
            proof.terminal,
            ring,
            t,
        ),
        (None, Some(wrap)) => {
            crate::wrap::verify_wrap(cm_stmt, &r_block, &r_bit_arr, proof.terminal, wrap, t)
        }
        _ => Err("mosaic: exactly one of ring/wrap must be present"),
    }
}

#[cfg(test)]
mod pack_tests {
    use super::*;

    /// The streaming pack must be bit-identical to the (former)
    /// assemble-then-pack path, on a layout whose components span several
    /// planes and misaligned boundaries.
    #[test]
    fn pack_tables_matches_assemble_and_pack() {
        let mut rng = rand::rng();
        use rand::RngExt;
        // A layout with mixed sizes: big components (multi-plane),
        // small ones, and a trailing hole (planes beyond the union stay
        // zero).
        let comps: Vec<(usize, usize)> = vec![
            (10, 6), // 2^16 bits
            (10, 6),
            (10, 4),
            (10, 3),
            (10, 8),
            (10, 6),
        ];
        let layout = MosaicLayout::build(
            comps
                .iter()
                .map(|(n, v)| ComponentSpec {
                    name: format!("c{n}-{v}"),
                    vars: v + n,
                })
                .collect(),
        );
        let tables: Vec<Vec<bool>> = comps
            .iter()
            .map(|(n, v)| {
                (0..(1usize << (n + v)))
                    .map(|_| rng.random::<bool>())
                    .collect()
            })
            .collect();
        // Reference: assemble + pack.
        let bits = assemble_g(&layout, &tables);
        let n = layout.ell - 7;
        let mut want = vec![ZERO; 1usize << n];
        for u in 0..(1usize << n) {
            let mut v: u128 = 0;
            for b in 0..128usize {
                if bits[u | (b << n)] {
                    v |= 1u128 << b;
                }
            }
            want[u] = Gf128(v);
        }
        let mut got = vec![ZERO; 1usize << n];
        pack_tables(&layout, &tables, &mut got);
        assert_eq!(want, got);
    }
}
