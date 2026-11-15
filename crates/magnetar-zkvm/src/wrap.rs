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

/// The batched consistency openings (the Canopy-class correlated-opening
/// compression): all `q'` transcript-sampled packed positions opened in
/// BOTH trees through the **union copath** — one sibling digest per
/// distinct root-path node (the shared top levels counted once) instead
/// of a full path per query. The verifier recomputes the copath
/// positions from the indices, so the proof carries digests only; the
/// binding is the standard Merkle root reconstruction (each deviation of
/// a revealed group from its commitment breaks the root exactly as an
/// individual opening would).
#[derive(Clone, Debug)]
pub struct ConsistencyBatch {
    /// The sampled packed positions (the transcript challenges, in
    /// sampling order).
    pub indices: Vec<usize>,
    /// The packed values at those positions.
    pub values: Vec<Gf128>,
    /// The chunk tree's covering groups (ascending), `width_c` values
    /// each — the (x, 0) slice straight off the chunk codeword.
    pub chunk_groups: Vec<Vec<Gf128>>,
    /// The deep tree's covering groups (ascending), `width_d` values each.
    pub deep_groups: Vec<Vec<Gf128>>,
    /// The chunk-side union copath, canonical order (digests only).
    pub chunk_copath: Vec<[u8; DIGEST]>,
    /// The deep-side union copath, canonical order.
    pub deep_copath: Vec<[u8; DIGEST]>,
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
    /// The consistency spot-checks (batched multi-open).
    pub consistency: ConsistencyBatch,
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
    // 3. The consistency spot-checks (batched): the SAME transcript
    //    schedule (wrap-pos-i challenges, wrap-val-i absorptions), the
    //    positions opened through the union copath in both trees — the
    //    (x, 0) slice is the low leaf range of both codewords, so the
    //    covering groups come straight off the codewords.
    let n_packed = packed.len();
    let mut indices = Vec::with_capacity(WRAP_CONSISTENCY_QUERIES);
    let mut values = Vec::with_capacity(WRAP_CONSISTENCY_QUERIES);
    for i in 0..WRAP_CONSISTENCY_QUERIES {
        let pos = t.challenge_index(&format!("wrap-pos-{i}"), n_packed as u64) as usize;
        indices.push(pos);
        values.push(packed[pos]);
        t.absorb_field(&format!("wrap-val-{i}"), &[packed[pos]]);
    }
    let chunk_multi = chunk_tree.open_multi(&indices);
    let deep_multi = tree.open_multi(&indices);
    let w_c = chunk_tree.width();
    let w_d = tree.width();
    let chunk_groups: Vec<Vec<Gf128>> = chunk_multi
        .groups
        .iter()
        .map(|&g| chunk_codeword.values[g * w_c..g * w_c + w_c].to_vec())
        .collect();
    let deep_groups: Vec<Vec<Gf128>> = deep_multi
        .groups
        .iter()
        .map(|&g| codeword.values[g * w_d..g * w_d + w_d].to_vec())
        .collect();
    debug_assert!(
        indices.iter().zip(values.iter()).all(|(&p, &v)| {
            let cg: Vec<usize> = chunk_multi.groups.clone();
            let dg: Vec<usize> = deep_multi.groups.clone();
            let rc = cg.binary_search(&(p / w_c)).ok().expect("covered");
            let rd = dg.binary_search(&(p / w_d)).ok().expect("covered");
            chunk_groups[rc][p % w_c] == v && deep_groups[rd][p % w_d] == v
        }),
        "the revealed groups contain the sampled values"
    );
    let consistency = ConsistencyBatch {
        indices,
        values,
        chunk_groups,
        deep_groups,
        chunk_copath: chunk_multi.siblings,
        deep_copath: deep_multi.siblings,
    };
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
    let c = &wrap.consistency;
    if c.indices.len() != WRAP_CONSISTENCY_QUERIES {
        return Err("wrap: consistency count mismatch");
    }
    for &i in &c.indices {
        if i >= n_packed {
            return Err("wrap: position out of packed range");
        }
    }
    // The chunk side: the union copath reconstructs the chunk root.
    let root = MerkleTree::verify_multi_batched(
        &chunk_dom,
        chunk_leaves,
        chunk_w,
        &c.indices,
        &c.chunk_groups,
        &c.chunk_copath,
    )
    .map_err(|_| "wrap: chunk multi-open rejected")?;
    if root != chunk_stmt.root {
        return Err("wrap: chunk root mismatch");
    }
    // The deep side.
    let root2 = MerkleTree::verify_multi_batched(
        &deep_dom,
        deep_leaves,
        deep_w,
        &c.indices,
        &c.deep_groups,
        &c.deep_copath,
    )
    .map_err(|_| "wrap: deep multi-open rejected")?;
    if root2 != wrap.deep.root {
        return Err("wrap: deep root mismatch");
    }
    // Every sampled value must sit at its position in BOTH revealed
    // groups (the sampled-position binding). The group lists are in
    // ascending index order, so each value's covering group is found by
    // its rank (binary search), not by its raw index.
    let rank_of = |groups: &[usize], g: usize| -> Option<usize> {
        groups.binary_search(&g).ok()
    };
    {
        let mut gc: Vec<usize> = c.indices.iter().map(|&i| i / chunk_w).collect();
        gc.sort_unstable();
        gc.dedup();
        let mut gd: Vec<usize> = c.indices.iter().map(|&i| i / deep_w).collect();
        gd.sort_unstable();
        gd.dedup();
        for (&i, &v) in c.indices.iter().zip(c.values.iter()) {
            let rg = rank_of(&gc, i / chunk_w).expect("group covered");
            if c.chunk_groups[rg][i % chunk_w] != v {
                return Err("wrap: chunk group does not contain the value");
            }
            let rd = rank_of(&gd, i / deep_w).expect("group covered");
            if c.deep_groups[rd][i % deep_w] != v {
                return Err("wrap: deep group does not contain the value");
            }
        }
    }
    for (i, &v) in c.values.iter().enumerate() {
        t.absorb_field(&format!("wrap-val-{i}"), &[v]);
    }
    Ok(())
}
