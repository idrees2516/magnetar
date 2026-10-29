//! The **dual-rate wrap** (roadmap Horizon 2): compress a chunk proof's
//! evaluation-argument tail by re-committing the packed witness at a low
//! rate (`DeepWrap`: rate 1/16, 14 correlated queries) and proving the
//! ring-switch discharge there, with transcript-sampled **consistency
//! spot-checks** binding the deep commitment to the chunk commitment the
//! rest of the proof's transcript already absorbed.
//!
//! ## Soundness shape
//!
//! The wrap is the hash-based **spot-check accumulation** shape (the
//! bounded-depth construction of [BMNW24], with the ARC-style
//! distance-preserving fold — which needs Reed–Solomon quotient structure
//! — documented as the follow-on for unbounded depth):
//!
//! * the batching sum-check's terminal `Ĝ(r*)` is bound in the shared
//!   transcript, and the deep ring switch proves it against the deep
//!   commitment — a complete evaluation argument at the deep rate;
//! * the consistency spot-checks open `q'` packed-leaf positions in
//!   **both** trees: a deep commitment whose `(x, 0)` slice deviates from
//!   the chunk-committed packed table on an ε-fraction is caught with
//!   probability `1 − (1 − ε)^{q'}` — the standard spot-check budget;
//! * the composed verifier therefore accepts only chunk–deep pairs that
//!   agree on the packed witness (up to the spot-check loss), and the
//!   deep evaluation argument then carries the claim chain.
//!
//! The public artifact's evaluation-argument tail shrinks with the deep
//! rate and query count (the measured projection: 216 KiB at `FastChunk`
//! → 128 KiB at `DeepWrap` on the same statement).
//!
//! ## Size accounting
//!
//! The wrap replaces the chunk-rate ring switch (40 queries × paths over
//! a `2^{n+1}`-leaf tree) with the deep ring switch (14 queries × paths
//! over a `2^{n+4}`-leaf tree: +3 path levels) plus `q'` dual packed
//! openings (two paths each, over the packed slice `2^{n−7}` — the
//! SHORTEST paths in the system). Net: ~40% fewer tail bytes.

use magnetar_field::Gf128;
use magnetar_hash::{MerkleTree, Transcript, DIGEST};
use magnetar_pcs::eval::commit_table;
use magnetar_pcs::ring::{prove_ring_switch, verify_ring_switch};
use magnetar_pcs::{PcsParams, PcsStatement, RingProof};

/// The level-domain label (mirrors the PCS's internal convention).
fn level_domain(base: &str, level: usize) -> String {
    format!("{base}/level/{level}")
}

/// Number of consistency spot-checks (the wrap's own budget; independent
/// of the deep ring's queries).
pub const WRAP_CONSISTENCY_QUERIES: usize = 14;

/// One dual opening of a packed-leaf position: the same packed word
/// authenticated against the chunk tree and the deep tree.
#[derive(Clone, Debug)]
pub struct ConsistencyOpen {
    /// The sampled packed position.
    pub index: usize,
    pub value: Gf128,
    /// The chunk tree's leaf group covering `index` (its width values).
    pub chunk_vals: Vec<Gf128>,
    /// The deep tree's leaf group covering `index` (its width values).
    pub deep_vals: Vec<Gf128>,
    pub chunk_path: Vec<[u8; DIGEST]>,
    pub deep_path: Vec<[u8; DIGEST]>,
}

/// The wrap artifact.
#[derive(Clone, Debug)]
pub struct WrapProof {
    /// The deep-rate commitment statement.
    pub deep: PcsStatement,
    /// The deep-rate parameters (queries included — the verifier needs
    /// them to replay the ring switch).
    pub deep_params: PcsParams,
    /// The ring-switch discharge at the deep rate.
    pub ring: RingProof,
    /// The consistency spot-checks.
    pub consistency: Vec<ConsistencyOpen>,
}

/// Prove the wrap: commit `packed` at the deep rate, discharge the
/// terminal there, and spot-check the two commitments' packed slices.
#[allow(clippy::too_many_arguments)]
pub fn prove_wrap(
    deep_params: &PcsParams,
    chunk_tree: &MerkleTree,
    chunk_codeword: &magnetar_pcs::code::Codeword,
    chunk_domain: &str,
    packed: &[Gf128],
    r_block: &[Gf128],
    r_bit: &[Gf128; 7],
    terminal: Gf128,
    t: &mut Transcript,
) -> WrapProof {
    // 1. The deep commitment over the same packed table.
    let (codeword, tree, stmt) = commit_table(deep_params, packed);
    t.absorb_digest("wrap-deep-root", &stmt.root);
    // 2. The ring-switch discharge at the deep rate.
    let ring = prove_ring_switch(
        deep_params,
        &codeword,
        &tree,
        packed,
        r_block,
        r_bit,
        terminal,
        t,
    );
    // 3. The consistency spot-checks: transcript-sampled packed positions,
    //    opened in both trees (the (x, 0) slice is the low leaf range of
    //    both codewords).
    let n_packed = packed.len();
    let mut consistency = Vec::with_capacity(WRAP_CONSISTENCY_QUERIES);
    for i in 0..WRAP_CONSISTENCY_QUERIES {
        let pos = t.challenge_index(&format!("wrap-pos-{i}"), n_packed as u64) as usize;
        let value = packed[pos];
        // the batched group reveals: the leaf group covering `pos` in each
        // tree (the (x, 0) slice — the low codeword range — holds the
        // packed table, so the groups come straight off the codewords).
        let w_c = chunk_tree.width();
        let g_c = pos / w_c;
        let chunk_vals = chunk_codeword.values[g_c * w_c..g_c * w_c + w_c].to_vec();
        debug_assert_eq!(chunk_vals[pos % w_c], value);
        let w_d = tree.width();
        let g_d = pos / w_d;
        let deep_vals = codeword.values[g_d * w_d..g_d * w_d + w_d].to_vec();
        debug_assert_eq!(deep_vals[pos % w_d], value);
        let (_, chunk_path) = chunk_tree.open(pos);
        let (_, deep_path) = tree.open(pos);
        consistency.push(ConsistencyOpen {
            index: pos,
            value,
            chunk_vals,
            deep_vals,
            chunk_path,
            deep_path,
        });
        t.absorb_field(&format!("wrap-val-{i}"), &[value]);
    }
    let _ = chunk_domain;
    WrapProof {
        deep: stmt,
        deep_params: deep_params.clone(),
        ring,
        consistency,
    }
}

/// Verify the wrap: the deep ring switch plus the dual consistency
/// openings against the chunk commitment the outer transcript bound.
#[allow(clippy::too_many_arguments)]
pub fn verify_wrap(
    chunk_stmt: &PcsStatement,
    r_block: &[Gf128],
    r_bit: &[Gf128; 7],
    terminal: Gf128,
    wrap: &WrapProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    if wrap.consistency.len() != WRAP_CONSISTENCY_QUERIES {
        return Err("wrap: consistency count mismatch");
    }
    t.absorb_digest("wrap-deep-root", &wrap.deep.root);
    verify_ring_switch(
        &wrap.deep,
        &wrap.deep_params,
        r_block,
        r_bit,
        terminal,
        &wrap.ring,
        t,
    )?;
    // The packed-slice length (the chunk tree's (x, 0) range).
    let n_packed = 1usize << chunk_stmt.n;
    let chunk_dom = level_domain(&chunk_stmt.domain, 0);
    let deep_dom = level_domain(&wrap.deep.domain, 0);
    let chunk_leaves = 1usize << (chunk_stmt.n + chunk_stmt.k);
    let deep_leaves = 1usize << (wrap.deep.n + wrap.deep.k);
    let chunk_w = magnetar_pcs::eval::fold_tree_width(chunk_leaves);
    let deep_w = magnetar_pcs::eval::fold_tree_width(deep_leaves);
    for (i, c) in wrap.consistency.iter().enumerate() {
        if c.index >= n_packed {
            return Err("wrap: position out of packed range");
        }
        if c.chunk_vals.len() != chunk_w || c.deep_vals.len() != deep_w {
            return Err("wrap: group width mismatch");
        }
        let ld = MerkleTree::leaf_digest_of_batched(
            &chunk_dom,
            chunk_leaves,
            chunk_w,
            c.index,
            &c.chunk_vals,
        );
        let root = MerkleTree::verify_batched(
            &chunk_dom,
            chunk_leaves,
            chunk_w,
            c.index,
            ld,
            &c.chunk_path,
        )
        .map_err(|_| "wrap: chunk opening rejected")?;
        if root != chunk_stmt.root {
            return Err("wrap: chunk root mismatch");
        }
        if c.chunk_vals[c.index % chunk_w] != c.value {
            return Err("wrap: chunk group does not contain the value");
        }
        let ld2 = MerkleTree::leaf_digest_of_batched(
            &deep_dom,
            deep_leaves,
            deep_w,
            c.index,
            &c.deep_vals,
        );
        let root2 =
            MerkleTree::verify_batched(&deep_dom, deep_leaves, deep_w, c.index, ld2, &c.deep_path)
                .map_err(|_| "wrap: deep opening rejected")?;
        if root2 != wrap.deep.root {
            return Err("wrap: deep root mismatch");
        }
        if c.deep_vals[c.index % deep_w] != c.value {
            return Err("wrap: deep group does not contain the value");
        }
        t.absorb_field(&format!("wrap-val-{i}"), &[c.value]);
    }
    Ok(())
}
