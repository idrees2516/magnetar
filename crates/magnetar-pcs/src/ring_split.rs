//! The **split ring switch** — the streaming root's per-segment
//! commitment trees (the single-machine 2^21+ memory lever).
//!
//! ## Why
//!
//! The monolithic ring switch commits ONE combined codeword
//! (`2^{n_total+k}` values) and keeps every fold level's tree and
//! values resident while the chain runs — `O(2^{n_total})` memory at
//! the root (the packed vector, the codeword, the c-table, the chain
//! values, and `n+1` full trees; ≈ 45 GiB projected at 2^21).
//!
//! ## The construction
//!
//! The combined cube is the product `(segment bits σ) ∥ (local cube)`,
//! and the codeword uses **segment-local** substitution vectors
//! (`CodeParams::derive(n_s, k, ..)` — the γ's span the local variables
//! only): at a Boolean segment index the combined codeword's entry is
//! exactly that segment's local codeword entry, so the combined codeword
//! IS the `[ε][s][x_loc]` concatenation of per-segment codewords — the
//! direct sum of the per-segment codes, same relative distance.
//!
//! Every fold level `i ≤ n_s` therefore decomposes into **cells**
//! `(s, ε)` of `2^{n_s−i}` contiguous values — each cell hashes to ONE
//! digest, and the level's root is a tiny **root layer** tree over the
//! `2^{σ+k}` cell digests. The chain is built **streaming, one segment
//! at a time** (rayon across segments): each worker derives its
//! `c_s = φ_c(packed_s)`, encodes both components, folds `n_s` times,
//! and keeps only its column of cell digests plus its `2^k` tail
//! values. Peak memory drops from `O(2^{n_total})` to
//! `O(one segment's chain)` + the digest columns.
//!
//! Above `n_s` the folds pair across segments; those levels are
//! `2^{σ−j+k}`-sized shared trees built over the collected tails.
//!
//! **The merged leaf**: each cell's tree commits **pairs**
//! `(P-entry, C-entry)` — the witness codeword and the c-folded
//! codeword in ONE Merkle tree. The anchors open ONE path per position
//! and check `φ_c(P) = C` exactly (the monolithic ring's two-path
//! anchor shape collapsed into one), and the correlated queries check
//! the fold relation on both components. The Fiat–Shamir order: the
//! hierarchical body's challenges (which determine `r_bit`) come
//! **before** the chain roots are absorbed, so the merged commitment
//! is bound before any query or anchor position is drawn.
//!
//! ## Soundness shape
//!
//! Identical to the monolithic ring switch: the batched sum-check pins
//! the terminal to `Ĉ(ρ)`; the fold chain + full tail pin the committed
//! codeword's evaluation; the anchors tie the C component to the P
//! component (the witness commitment) with the same `q`-position
//! coverage over the combined packed range. A query's walk stays
//! within one segment below level `n_s` (the fold pairs never straddle
//! a cell), and the per-position catch probability
//! `q / 2^{n_total+k−i}` is unchanged — every per-segment cell digest
//! is inside a transcript-bound root layer before the queries are
//! drawn.

use crate::code::{CodeParams, Codeword};
use magnetar_field::{eq_coefficients, mul, phi_c, Gf128};
use magnetar_hash::{leaf_digest_of_pair, leaf_digest_of_raw, MerkleTree, Transcript, DIGEST};
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::EqFactor,
    BindOrder, RoundMsg, SumcheckSpec, Term,
};
use rayon::prelude::*;

/// A one-shot phase timer (the crate-local `PcTimer` shape, shared with
/// `ring.rs` via a public constructor so this module can use it too).
struct STimer {
    enabled: bool,
    label: &'static str,
    start: std::time::Instant,
}
impl STimer {
    fn new(label: &'static str) -> Self {
        STimer {
            enabled: std::env::var_os("MAGNETAR_TIME").is_some(),
            label,
            start: std::time::Instant::now(),
        }
    }
}
impl Drop for STimer {
    fn drop(&mut self) {
        if self.enabled {
            eprintln!(
                "[time] {:<24} {:>9.1} ms",
                self.label,
                self.start.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

fn split_rss(tag: &str) {
    if std::env::var_os("MAGNETAR_RSS").is_some() {
        #[cfg(target_os = "linux")]
        {
            let r = std::fs::read_to_string("/proc/self/statm")
                .ok()
                .and_then(|s| {
                    s.split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse::<u64>().ok())
                })
                .map(|p| p * 4096)
                .unwrap_or(0);
            eprintln!("[rss] {:<24} {:>8.0} MiB", tag, r as f64 / 1048576.0);
        }
    }
}

/// The split-chain geometry: `n_s` local block variables, `sigma`
/// segment bits, `k` extension variables.
#[derive(Clone, Copy, Debug)]
pub struct SplitGeometry {
    pub n_s: usize,
    pub sigma: usize,
    pub k: usize,
}

impl SplitGeometry {
    pub fn new(n_s: usize, sigma: usize, k: usize) -> Self {
        SplitGeometry { n_s, sigma, k }
    }
    /// Total block variables (the combined cube's word dimension).
    pub fn n_total(&self) -> usize {
        self.n_s + self.sigma
    }
    /// The cell (subtree) leaf count at `level`. For `level ≤ n_s` the
    /// cells are the `(s, ε)` blocks; above `n_s` the single "cell" is
    /// the whole shared level.
    pub fn cell_size(&self, level: usize) -> usize {
        if level <= self.n_s {
            1usize << (self.n_s - level)
        } else {
            1usize << (self.n_total() - level + self.k)
        }
    }
    /// The root-layer leaf count at `level` (`1` for the shared levels
    /// above `n_s`: no layer, the cell tree IS the level tree).
    pub fn layer_size(&self, level: usize) -> usize {
        if level <= self.n_s {
            1usize << (self.sigma + self.k)
        } else {
            1
        }
    }
    /// Decompose a global level-`level` index into
    /// `(epsilon, segment, cell_local)`.
    pub fn decompose(&self, level: usize, idx: usize) -> (usize, usize, usize) {
        if level <= self.n_s {
            let rem_bits = self.n_s - level;
            let eps = idx >> (self.n_total() - level);
            let s = (idx >> rem_bits) & ((1usize << self.sigma) - 1);
            let local = idx & ((1usize << rem_bits) - 1);
            (eps, s, local)
        } else {
            (0, 0, idx)
        }
    }
    /// The layer-tree leaf index for a global level-`level` index.
    pub fn layer_index(&self, level: usize, idx: usize) -> usize {
        if level <= self.n_s {
            let (eps, s, _) = self.decompose(level, idx);
            (eps << self.sigma) | s
        } else {
            0
        }
    }
    /// The cell-tree domain at `level`.
    pub fn cell_domain(&self, base: &str, level: usize) -> String {
        if level <= self.n_s {
            format!("{base}/split/level/{level}/cell")
        } else {
            format!("{base}/split/level/{level}")
        }
    }
    /// The root-layer domain at `level` (unused above `n_s`).
    pub fn layer_domain(&self, base: &str, level: usize) -> String {
        format!("{base}/split/level/{level}/layer")
    }
}

/// One opening of a merged `(P, C)` pair value with its two-part path
/// (the within-cell path, then the root-layer path; the layer part is
/// empty for the shared levels above `n_s`).
#[derive(Clone, Debug)]
pub struct SplitOpen {
    pub value: (Gf128, Gf128),
    pub cell_path: Vec<[u8; DIGEST]>,
    pub layer_path: Vec<[u8; DIGEST]>,
}

/// One level of a correlated query walk: `(a, b)` at level `i` and `c`
/// at level `i+1`.
#[derive(Clone, Debug)]
pub struct SplitLevelOpen {
    pub a: SplitOpen,
    pub b: SplitOpen,
    pub c: SplitOpen,
}

/// One correlated query: the walk from level 0 to the tail.
#[derive(Clone, Debug)]
pub struct SplitQuery {
    pub levels: Vec<SplitLevelOpen>,
}

/// One φ-anchor: the merged tree's `(x, 0)` pair with its two-part path.
#[derive(Clone, Debug)]
pub struct SplitAnchor {
    pub x: usize,
    pub pair: (Gf128, Gf128),
    pub cell_path: Vec<[u8; DIGEST]>,
    pub layer_path: Vec<[u8; DIGEST]>,
}

/// A merged `(P, C)` field pair with a sibling path.
pub type PairPath = ((Gf128, Gf128), Vec<[u8; DIGEST]>);

/// The split ring-switch proof (the streaming root's evaluation tail).
#[derive(Clone, Debug)]
pub struct SplitRingProof {
    /// The combined sum-check's round messages (degree 2).
    pub rounds: Vec<Vec<Gf128>>,
    /// The claimed `Ĉ(ρ)` (the tail entry at ε = 0).
    pub terminal_claim: Gf128,
    /// The per-level roots (`n_total + 1`; the layers below `n_s`, the
    /// shared trees above).
    pub level_roots: Vec<[u8; DIGEST]>,
    /// The correlated path queries.
    pub queries: Vec<SplitQuery>,
    /// The final tail: `2^k` merged pairs with paths in the last shared
    /// tree.
    pub tail: Vec<PairPath>,
    /// The φ anchors tying the C component to the P component.
    pub anchors: Vec<SplitAnchor>,
}

/// One segment's derived chain: the per-level cell digests (its column
/// of each root layer) and its `2^k` tail pairs.
struct SegChain {
    /// `cells[level * 2^k + eps]` = the `(s, ε)` cell digest at `level`
    /// (levels `0..=n_s`).
    cells: Vec<[u8; DIGEST]>,
    /// The level-`n_s` values: `2^k` merged pairs.
    tail: Vec<(Gf128, Gf128)>,
}

/// Fold one component of a pair vector at bit 0 with challenge `r`.
fn fold_pairs(v: &[(Gf128, Gf128)], r: Gf128) -> Vec<(Gf128, Gf128)> {
    let mut out = Vec::with_capacity(v.len() / 2);
    let mut i = 0usize;
    while i < v.len() {
        let a = v[i];
        let b = v[i + 1];
        out.push((a.0 + mul(r, a.0 + b.0), a.1 + mul(r, a.1 + b.1)));
        i += 2;
    }
    out
}

/// Fold a value vector at bit 0 in place:
/// `v'[j] = v[2j] + r·(v[2j] + v[2j+1])` (both components' codewords
/// share the fold challenges).
fn fold_values_in_place(v: &mut Vec<Gf128>, r: Gf128) {
    let n = v.len() / 2;
    for j in 0..n {
        let a = v[2 * j];
        let b = v[2 * j + 1];
        v[j] = a + mul(r, a + b);
    }
    v.truncate(n);
}

/// Derive one segment's chain (cell digests + the tail) — the streaming
/// worker's path: the two component codewords fold IN PLACE, and each
/// `(level, ε)` cell tree is built for its digest and dropped — the
/// transient is `O(one segment's codeword + one cell tree)`.
fn derive_segment_chain(
    geo: &SplitGeometry,
    base: &str,
    slice: &[Gf128],
    c: &[Gf128; 128],
    rho: &[Gf128],
) -> SegChain {
    let n_s = geo.n_s;
    let k = geo.k;
    let c_s: Vec<Gf128> = slice.iter().map(|&p| phi_c(p, c)).collect();
    let local_params = CodeParams::derive(n_s, k, base);
    let mut pv = Codeword::encode(&local_params, slice).values;
    let mut cv = Codeword::encode(&local_params, &c_s).values;
    drop(c_s);
    let mut cells: Vec<[u8; DIGEST]> = Vec::with_capacity((n_s + 1) << k);
    for i in 0..=n_s {
        let cell_size = 1usize << (n_s - i);
        for eps in 0..(1usize << k) {
            let lo = eps * cell_size;
            let pairs: Vec<(Gf128, Gf128)> =
                (0..cell_size).map(|j| (pv[lo + j], cv[lo + j])).collect();
            let tree = MerkleTree::build_pairs(&geo.cell_domain(base, i), &pairs);
            cells.push(tree.root());
        }
        if i < n_s {
            fold_values_in_place(&mut pv, rho[i]);
            fold_values_in_place(&mut cv, rho[i]);
        }
    }
    let tail: Vec<(Gf128, Gf128)> = (0..(1usize << k)).map(|eps| (pv[eps], cv[eps])).collect();
    SegChain { cells, tail }
}

/// One in-segment opening request: the cell `(level, ε)` and the local
/// index within the cell.
#[derive(Clone, Copy, Debug)]
struct OpenReq {
    level: usize,
    eps: usize,
    local: usize,
}

/// Where an answered opening is routed: the query's `a`/`b`/`c` side at
/// a walk level, or an anchor slot.
#[derive(Clone, Copy, Debug)]
enum Tag {
    Qa(usize, usize),
    Qb(usize, usize),
    Qc(usize, usize),
    Anchor(usize),
}

/// One routed opening request: the cell request, its root-layer sibling
/// path, and the output-slot tag.
type RoutedReq = (OpenReq, Vec<[u8; DIGEST]>, Tag);

/// One wave's parallel answer: the segment, its answered openings (in
/// request order), and the request list (for the tag routing).
type WaveAnswer<'a> = (usize, Vec<Option<SplitOpen>>, &'a Vec<RoutedReq>);

/// Answer all of one segment's in-segment opening requests with
/// `O(one level's values + one cell tree)` memory: the two component
/// codewords fold in place level by level, and each `(level, ε)` cell
/// tree is built only while requests need it (then dropped). This is
/// the opening phase's memory bound — the kept-tree variant held every
/// digest level of every cell (measured ~8× the codeword per segment).
/// `layer_paths[idx]` is the request's root-layer sibling path (the
/// caller owns the layer trees).
fn open_segment_positions(
    geo: &SplitGeometry,
    base: &str,
    slice: &[Gf128],
    c: &[Gf128; 128],
    rho: &[Gf128],
    reqs: &[OpenReq],
    layer_paths: &[Vec<[u8; DIGEST]>],
) -> Vec<Option<SplitOpen>> {
    let n_s = geo.n_s;
    let k = geo.k;
    debug_assert_eq!(reqs.len(), layer_paths.len());
    debug_assert!(reqs.iter().all(|r| r.level < n_s));
    let mut out: Vec<Option<SplitOpen>> = vec![None; reqs.len()];
    let max_level = reqs.iter().map(|r| r.level).max().unwrap_or(0);
    let c_s: Vec<Gf128> = slice.iter().map(|&p| phi_c(p, c)).collect();
    let local_params = CodeParams::derive(n_s, k, base);
    let mut pv = Codeword::encode(&local_params, slice).values;
    let mut cv = Codeword::encode(&local_params, &c_s).values;
    drop(c_s);
    for i in 0..=max_level {
        let cell_size = 1usize << (n_s - i);
        let mut by_eps: Vec<Vec<usize>> = vec![Vec::new(); 1 << k];
        for (idx, r) in reqs.iter().enumerate() {
            if r.level == i {
                by_eps[r.eps].push(idx);
            }
        }
        for eps in 0..(1usize << k) {
            if by_eps[eps].is_empty() {
                continue;
            }
            let lo = eps * cell_size;
            let pairs: Vec<(Gf128, Gf128)> =
                (0..cell_size).map(|j| (pv[lo + j], cv[lo + j])).collect();
            let tree = MerkleTree::build_pairs(&geo.cell_domain(base, i), &pairs);
            for &idx in &by_eps[eps] {
                let r = reqs[idx];
                let (_, cell_path) = tree.open(r.local);
                out[idx] = Some(SplitOpen {
                    value: pairs[r.local],
                    cell_path,
                    layer_path: layer_paths[idx].clone(),
                });
            }
        }
        if i < max_level {
            fold_values_in_place(&mut pv, rho[i]);
            fold_values_in_place(&mut cv, rho[i]);
        }
    }
    out
}

/// The shared levels' materialization: level `n_s + j`'s values (the
/// folded tails) — small (`2^{σ+k−j}` pairs).
fn shared_levels(
    geo: &SplitGeometry,
    chains: &[SegChain],
    rho: &[Gf128],
) -> Vec<Vec<(Gf128, Gf128)>> {
    let (sigma, k, n_s) = (geo.sigma, geo.k, geo.n_s);
    let segments = 1usize << sigma;
    let mut cur: Vec<(Gf128, Gf128)> = (0..(1usize << (sigma + k)))
        .map(|idx| {
            let eps = idx >> sigma;
            let s = idx & (segments - 1);
            chains[s].tail[eps]
        })
        .collect();
    // cur holds level `n_s`'s values; each fold produces the next.
    let mut out = Vec::with_capacity(sigma + 1);
    out.push(cur.clone());
    for j in 0..sigma {
        cur = fold_pairs(&cur, rho[n_s + j]);
        out.push(cur.clone());
    }
    out
}

/// Prove the split ring switch.
///
/// `packed` is the combined packed vector in **segment-major** order
/// (`[s | x_loc]`, `2^{n_total}` words — the streaming layout).
#[allow(clippy::too_many_arguments)]
pub fn prove_ring_switch_split(
    params: &crate::eval::PcsParams,
    geo: SplitGeometry,
    packed: &[Gf128],
    r_a: &[Gf128],
    r_b: &[Gf128; 7],
    y: Gf128,
    t: &mut Transcript,
) -> SplitRingProof {
    let n_total = geo.n_total();
    let sigma = geo.sigma;
    let segments = 1usize << sigma;
    let n_s = geo.n_s;
    debug_assert_eq!(packed.len(), 1 << n_total);
    debug_assert_eq!(r_a.len(), n_total);
    let base = format!("{}/ring", params.domain);

    // 1–2. c coefficients and the combined c-folded table.
    let c = eq_coefficients(r_b);
    let _pt_cf = STimer::new("split-cfold");
    let c_table: Vec<Gf128> = packed.iter().map(|&p| phi_c(p, &c)).collect();
    drop(_pt_cf);

    // 3. The sum-check over the combined block variables — the same
    //    engine path as the monolithic ring (the dense table folds in
    //    place; the chains re-derive the per-segment c-tables).
    let _pt_sc = STimer::new("split-sumcheck");
    let spec = SumcheckSpec {
        num_vars: n_total,
        dense: vec![c_table],
        public: vec![Box::new(EqFactor::new(r_a.to_vec()))],
        terms: vec![Term {
            coeff: Gf128(1),
            dense: vec![0],
            public: vec![0],
        }],
        claim: y,
    };
    let mut prover = SumcheckProver::new(&spec, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(n_total);
    let mut challenges = Vec::with_capacity(n_total);
    for _ in 0..n_total {
        let msg = prover.message();
        rounds.push(msg.0.clone());
        challenges.push(prover.absorb_and_fold(t, "ring-eval"));
    }
    let (claims, _) = prover.finish();
    drop(_pt_sc);
    let terminal_claim = claims[0];
    t.absorb_field("ring-terminal", &[terminal_claim]);

    // 4. The per-segment chains (rayon; each worker holds one segment's
    //    two local codewords at a time — the O(segment) envelope).
    let _pt_chain = STimer::new("split-chains");
    split_rss("chains-start");
    let chains: Vec<SegChain> = (0..segments)
        .into_par_iter()
        .map(|s| {
            let slice = &packed[s << n_s..(s + 1) << n_s];
            derive_segment_chain(&geo, &base, slice, &c, &challenges)
        })
        .collect();
    split_rss("chains-done");
    drop(_pt_chain);

    // The root layers + the shared upper levels.
    let _pt_layers = STimer::new("split-layers");
    let mut level_roots: Vec<[u8; DIGEST]> = Vec::with_capacity(n_total + 1);
    let mut layer_trees: Vec<MerkleTree> = Vec::with_capacity(n_s + 1);
    for i in 0..=n_s {
        let mut leaves = vec![[0u8; DIGEST]; 1usize << (sigma + params.k)];
        for (s, ch) in chains.iter().enumerate() {
            for eps in 0..(1usize << params.k) {
                leaves[(eps << sigma) | s] = ch.cells[i * (1 << params.k) + eps];
            }
        }
        let tree = MerkleTree::build_from_digests(&geo.layer_domain(&base, i), &leaves);
        level_roots.push(tree.root());
        layer_trees.push(tree);
    }
    let shared = shared_levels(&geo, &chains, &challenges);
    for j in 0..sigma.saturating_sub(1) {
        let level = n_s + 1 + j;
        let tree = MerkleTree::build_pairs(&geo.cell_domain(&base, level), &shared[j + 1]);
        level_roots.push(tree.root());
    }
    let tail_tree = MerkleTree::build_pairs(&geo.cell_domain(&base, n_total), &shared[sigma]);
    level_roots.push(tail_tree.root());
    split_rss("layers-done");
    drop(_pt_layers);
    for (i, root) in level_roots.iter().enumerate() {
        t.absorb_digest(&format!("ring-root-{i}"), root);
    }

    // 5. Draw ALL opening positions up front (the transcript's draw
    //    order is queries, then anchors — nothing between them consumes
    //    a challenge), then answer the openings **in waves of kept
    //    chains**: each touched segment's chain is re-derived ONCE, but
    //    at most `wave` segments' kept trees are resident at a time —
    //    the opening phase's memory stays `O(wave × one segment's
    //    chain)` regardless of the segment count (the 2^21+ envelope).
    let _pt_q = STimer::new("split-openings");
    let base_leaves = 1u64 << (n_total + params.k);
    let positions: Vec<usize> = (0..params.queries)
        .map(|qi| t.challenge_index(&format!("ring-query-{qi}"), base_leaves) as usize)
        .collect();
    let anchor_xs: Vec<usize> = (0..params.queries)
        .map(|ai| t.challenge_index(&format!("ring-anchor-{ai}"), 1u64 << n_total) as usize)
        .collect();
    // The requests below are collected per segment directly from the
    // positions and anchors — no separate touched-segment pass needed.
    // The shared levels' trees (small) for the upper openings: levels
    // `n_s+1 ..= n_total` (level `n_s` itself opens through its root
    // LAYER — its cells are single values — so `layer_trees[n_s]`
    // serves it).
    let shared_trees: Vec<MerkleTree> = (1..=sigma)
        .map(|j| {
            let level = n_s + j;
            MerkleTree::build_pairs(&geo.cell_domain(&base, level), &shared[j])
        })
        .collect();
    // The wave size (segments derived concurrently for openings):
    // `MAGNETAR_OPEN_WAVE` overrides; each derivation holds one
    // segment's two codewords plus one cell tree, so the wave bounds
    // the opening phase's memory independent of the segment count.
    let wave: usize = std::env::var("MAGNETAR_OPEN_WAVE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| rayon::current_num_threads().max(1));
    // The per-query opening slots: `[qi][level]` for the a/b/c sides
    // (the c slot at level n_total-1 is the tail's ε, handled below).
    let nq = positions.len();
    let mut qa: Vec<Vec<Option<SplitOpen>>> = vec![vec![None; n_total]; nq];
    let mut qb: Vec<Vec<Option<SplitOpen>>> = vec![vec![None; n_total]; nq];
    let mut qc: Vec<Vec<Option<SplitOpen>>> = vec![vec![None; n_total]; nq];
    let mut anchors: Vec<Option<SplitAnchor>> = vec![None; anchor_xs.len()];
    // The upper-level opens (level ≥ n_s) need only the small global
    // trees — fill them now.
    let upper_open = |level: usize, at: usize| -> SplitOpen {
        if level == n_s {
            let value = shared[0][at];
            let (_, lpath) = layer_trees[n_s].open(geo.layer_index(level, at));
            SplitOpen {
                value,
                cell_path: Vec::new(),
                layer_path: lpath,
            }
        } else {
            let j = level - n_s;
            let (_, path) = shared_trees[j - 1].open(at);
            SplitOpen {
                value: shared[j][at],
                cell_path: path,
                layer_path: Vec::new(),
            }
        }
    };
    for (qi, &pos) in positions.iter().enumerate() {
        let mut idx = pos;
        for i in 0..n_total {
            let a_idx = idx & !1usize;
            let b_idx = idx | 1usize;
            let next_idx = idx >> 1;
            if i >= n_s {
                qa[qi][i] = Some(upper_open(i, a_idx));
                qb[qi][i] = Some(upper_open(i, b_idx));
            }
            if i + 1 >= n_s {
                qc[qi][i] = Some(upper_open(i + 1, next_idx));
            }
            idx = next_idx;
        }
    }
    // The per-segment in-segment requests (levels < n_s), collected
    // once: `(req, layer_path, tag)` where the tag routes the answer.
    let mut per_seg: std::collections::HashMap<usize, Vec<RoutedReq>> =
        std::collections::HashMap::new();
    {
        let mut push_req = |level: usize, at: usize, tag: Tag| {
            if level >= n_s {
                return; // the upper levels were already answered
            }
            let (eps, s, local) = geo.decompose(level, at);
            let li = geo.layer_index(level, at);
            let (_, lpath) = layer_trees[level].open(li);
            per_seg
                .entry(s)
                .or_default()
                .push((OpenReq { level, eps, local }, lpath, tag));
        };
        for (qi, &pos) in positions.iter().enumerate() {
            let mut idx = pos;
            for i in 0..n_total {
                let a_idx = idx & !1usize;
                let b_idx = idx | 1usize;
                let next_idx = idx >> 1;
                if i < n_s {
                    push_req(i, a_idx, Tag::Qa(qi, i));
                    push_req(i, b_idx, Tag::Qb(qi, i));
                }
                if i + 1 < n_s {
                    push_req(i + 1, next_idx, Tag::Qc(qi, i));
                }
                idx = next_idx;
            }
        }
        for (ai, &x) in anchor_xs.iter().enumerate() {
            push_req(0, x, Tag::Anchor(ai));
        }
    }
    let mut seg_list: Vec<(usize, Vec<RoutedReq>)> = per_seg.into_iter().collect();
    seg_list.sort_by_key(|(s, _)| *s);
    for chunk in seg_list.chunks(wave.max(1)) {
        // The wave's derivations run in parallel (one segment's two
        // codewords + one cell tree each), then route sequentially.
        let results: Vec<WaveAnswer> = chunk
            .par_iter()
            .map(|(s, list)| {
                let reqs: Vec<OpenReq> = list.iter().map(|(r, _, _)| *r).collect();
                let layer_paths: Vec<Vec<[u8; DIGEST]>> =
                    list.iter().map(|(_, l, _)| l.clone()).collect();
                let slice = &packed[*s << n_s..(*s + 1) << n_s];
                (
                    *s,
                    open_segment_positions(
                        &geo,
                        &base,
                        slice,
                        &c,
                        &challenges,
                        &reqs,
                        &layer_paths,
                    ),
                    list,
                )
            })
            .collect();
        for (_s, opened, list) in results {
            for (idx, (_, _, tag)) in list.iter().enumerate() {
                let open = opened[idx].clone().expect("answered");
                match tag {
                    Tag::Qa(qi, i) => qa[*qi][*i] = Some(open),
                    Tag::Qb(qi, i) => qb[*qi][*i] = Some(open),
                    Tag::Qc(qi, i) => qc[*qi][*i] = Some(open),
                    Tag::Anchor(ai) => {
                        let x = anchor_xs[*ai];
                        anchors[*ai] = Some(SplitAnchor {
                            x,
                            pair: open.value,
                            cell_path: open.cell_path,
                            layer_path: open.layer_path,
                        });
                    }
                }
            }
        }
        // The wave's derivations die here — the next wave's memory is
        // bounded by its own.
    }
    split_rss("openings-done");
    let queries: Vec<SplitQuery> = (0..nq)
        .map(|qi| {
            let mut levels = Vec::with_capacity(n_total);
            for i in 0..n_total {
                levels.push(SplitLevelOpen {
                    a: qa[qi][i].take().expect("query a answered"),
                    b: qb[qi][i].take().expect("query b answered"),
                    c: qc[qi][i].take().expect("query c answered"),
                });
            }
            SplitQuery { levels }
        })
        .collect();
    let anchors: Vec<SplitAnchor> = anchors
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .expect("every anchor's segment was derived");
    drop(_pt_q);

    // 6. The tail.
    let _pt_tail = STimer::new("split-tail");
    let tail: Vec<PairPath> = (0..tail_tree.n_leaves())
        .map(|i| {
            let (_, path) = tail_tree.open(i);
            (shared[sigma][i], path)
        })
        .collect();
    drop(_pt_tail);

    SplitRingProof {
        rounds,
        terminal_claim,
        level_roots,
        queries,
        tail,
        anchors,
    }
}

/// Verify the split ring switch.
#[allow(clippy::too_many_arguments)]
pub fn verify_ring_switch_split(
    params: &crate::eval::PcsParams,
    geo: SplitGeometry,
    r_a: &[Gf128],
    r_b: &[Gf128; 7],
    y: Gf128,
    proof: &SplitRingProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let n_total = geo.n_total();
    let n_s = geo.n_s;
    if r_a.len() != n_total {
        return Err("split ring: point length mismatch");
    }
    if proof.rounds.len() != n_total || proof.level_roots.len() != n_total + 1 {
        return Err("split ring: proof shape mismatch");
    }
    if proof.queries.len() != params.queries || proof.anchors.len() != params.queries {
        return Err("split ring: query/anchor count mismatch");
    }
    let c = eq_coefficients(r_b);
    // ---- 1. The sum-check replay (captures the fold challenges).
    let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(EqFactor::new(r_a.to_vec()))];
    let terms = vec![Term {
        coeff: Gf128(1),
        dense: vec![0],
        public: vec![0],
    }];
    let mut verifier = SumcheckVerifier::new(n_total, y, &public, terms, BindOrder::Increasing);
    let mut challenges = Vec::with_capacity(n_total);
    for msg in &proof.rounds {
        let round_msg = RoundMsg(msg.clone());
        challenges.push(verifier.round_capturing(&round_msg, t, "ring-eval")?);
    }
    verifier.finish(&[proof.terminal_claim])?;
    t.absorb_field("ring-terminal", &[proof.terminal_claim]);
    for (i, root) in proof.level_roots.iter().enumerate() {
        t.absorb_digest(&format!("ring-root-{i}"), root);
    }
    let base = format!("{}/ring", params.domain);
    // Verify one opening against the level geometry.
    let check_open = |level: usize,
                      at: usize,
                      open: &SplitOpen,
                      want_root: &[u8; DIGEST]|
     -> Result<(), &'static str> {
        let (_, _seg, local) = geo.decompose(level, at);
        let cell_size = geo.cell_size(level);
        let cell_dom = geo.cell_domain(&base, level);
        let leaf = leaf_digest_of_pair(&cell_dom, cell_size, local, open.value);
        let cell_digest = MerkleTree::verify(&cell_dom, cell_size, local, leaf, &open.cell_path)
            .map_err(|_| "split ring: cell path invalid")?;
        let layer_size = geo.layer_size(level);
        if layer_size == 1 {
            if &cell_digest != want_root {
                return Err("split ring: level root mismatch (shared)");
            }
            return Ok(());
        }
        let li = geo.layer_index(level, at);
        let ldom = geo.layer_domain(&base, level);
        let lleaf = leaf_digest_of_raw(&ldom, layer_size, li, cell_digest);
        let root = MerkleTree::verify(&ldom, layer_size, li, lleaf, &open.layer_path)
            .map_err(|_| "split ring: layer path invalid")?;
        if &root != want_root {
            return Err("split ring: level root mismatch");
        }
        Ok(())
    };
    // ---- 2. The correlated queries.
    let base_leaves = 1u64 << (n_total + params.k);
    for (qi, q) in proof.queries.iter().enumerate() {
        let pos = t.challenge_index(&format!("ring-query-{qi}"), base_leaves) as usize;
        if q.levels.len() != n_total {
            return Err("split ring: query level count mismatch");
        }
        let mut idx = pos;
        for (i, lvl) in q.levels.iter().enumerate() {
            let a_idx = idx & !1usize;
            let b_idx = idx | 1usize;
            let next_idx = idx >> 1;
            check_open(i, a_idx, &lvl.a, &proof.level_roots[i])?;
            check_open(i, b_idx, &lvl.b, &proof.level_roots[i])?;
            check_open(i + 1, next_idx, &lvl.c, &proof.level_roots[i + 1])?;
            let r = challenges[i];
            if lvl.c.value.0 != lvl.a.value.0 + mul(r, lvl.a.value.0 + lvl.b.value.0) {
                return Err("split ring: P fold relation failed");
            }
            if lvl.c.value.1 != lvl.a.value.1 + mul(r, lvl.a.value.1 + lvl.b.value.1) {
                return Err("split ring: C fold relation failed");
            }
            idx = next_idx;
        }
    }
    // ---- 3. The tail.
    if proof.tail.len() != (1usize << params.k) {
        return Err("split ring: tail length mismatch");
    }
    let tail_dom = geo.cell_domain(&base, n_total);
    for (i, ((pv, cv), path)) in proof.tail.iter().enumerate() {
        let leaf = leaf_digest_of_pair(&tail_dom, 1usize << params.k, i, (*pv, *cv));
        let root = MerkleTree::verify(&tail_dom, 1usize << params.k, i, leaf, path)
            .map_err(|_| "split ring: tail path invalid")?;
        if root != proof.level_roots[n_total] {
            return Err("split ring: tail root mismatch");
        }
    }
    if proof.tail[0].0 .1 != proof.terminal_claim {
        return Err("split ring: tail terminal entry mismatch");
    }
    // ---- 4. The anchors: one merged path + the exact φ_c check.
    for anchor in &proof.anchors {
        if anchor.x >= (1usize << n_total) {
            return Err("split ring: anchor index out of range");
        }
        let s = anchor.x >> n_s;
        let local = anchor.x & ((1usize << n_s) - 1);
        let cell_dom = geo.cell_domain(&base, 0);
        let leaf = leaf_digest_of_pair(&cell_dom, 1usize << n_s, local, anchor.pair);
        let cell_digest =
            MerkleTree::verify(&cell_dom, 1usize << n_s, local, leaf, &anchor.cell_path)
                .map_err(|_| "split ring: anchor cell path invalid")?;
        let ldom = geo.layer_domain(&base, 0);
        let layer_size = 1usize << (geo.sigma + params.k);
        let lleaf = leaf_digest_of_raw(&ldom, layer_size, s, cell_digest);
        let root = MerkleTree::verify(&ldom, layer_size, s, lleaf, &anchor.layer_path)
            .map_err(|_| "split ring: anchor layer path invalid")?;
        if root != proof.level_roots[0] {
            return Err("split ring: anchor root mismatch");
        }
        // The exact φ_c tie between the two components of the merged leaf.
        if phi_c(anchor.pair.0, &c) != anchor.pair.1 {
            return Err("split ring: anchor phi_c mismatch");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        use rand::RngExt;
        Gf128(rng.random::<u128>())
    }

    /// The split ring at several geometries: prove + verify + the
    /// standard tamper classes.
    #[test]
    fn split_ring_roundtrip_and_tamper() {
        let mut rng = rand::rng();
        for (n_s, sigma, k) in [(4usize, 1usize, 1usize), (5, 2, 1), (4, 3, 2), (6, 1, 1)] {
            let geo = SplitGeometry::new(n_s, sigma, k);
            let params = crate::eval::PcsParams {
                k,
                queries: 10,
                domain: format!("test-split-{n_s}-{sigma}-{k}"),
            };
            let n_total = geo.n_total();
            let packed: Vec<Gf128> = (0..(1 << n_total)).map(|_| rg(&mut rng)).collect();
            let r_a: Vec<Gf128> = (0..n_total).map(|_| rg(&mut rng)).collect();
            let r_b: [Gf128; 7] = std::array::from_fn(|_| rg(&mut rng));
            // y = Σ eq(r_a,x)·C(x) with C = φ_c(packed).
            let c = eq_coefficients(&r_b);
            let eq_a = magnetar_field::EqBits::build(&r_a).values;
            let mut y = Gf128(0);
            for (u, &p) in packed.iter().enumerate() {
                y += magnetar_field::mul(eq_a[u], phi_c(p, &c));
            }
            let mut t = Transcript::new("split-ring", &[b"t"]);
            let proof = prove_ring_switch_split(&params, geo, &packed, &r_a, &r_b, y, &mut t);
            let mut tv = Transcript::new("split-ring", &[b"t"]);
            let vres = verify_ring_switch_split(&params, geo, &r_a, &r_b, y, &proof, &mut tv);
            assert!(
                vres.is_ok(),
                "n_s={n_s} sigma={sigma} k={k}: {:?}",
                vres.err()
            );
            // Wrong y.
            let mut tv = Transcript::new("split-ring", &[b"t"]);
            assert!(verify_ring_switch_split(
                &params,
                geo,
                &r_a,
                &r_b,
                y + Gf128(1),
                &proof,
                &mut tv
            )
            .is_err());
            // Tampered terminal.
            let mut p = proof.clone();
            p.terminal_claim += Gf128(1);
            let mut tv = Transcript::new("split-ring", &[b"t"]);
            assert!(verify_ring_switch_split(&params, geo, &r_a, &r_b, y, &p, &mut tv).is_err());
            // Tampered anchor pair (phi_c mismatch).
            let mut p = proof.clone();
            p.anchors[0].pair.1 += Gf128(1);
            let mut tv = Transcript::new("split-ring", &[b"t"]);
            assert!(verify_ring_switch_split(&params, geo, &r_a, &r_b, y, &p, &mut tv).is_err());
            // Tampered query value (fold relation).
            let mut p = proof.clone();
            p.queries[0].levels[0].a.value.0 += Gf128(1);
            let mut tv = Transcript::new("split-ring", &[b"t"]);
            assert!(verify_ring_switch_split(&params, geo, &r_a, &r_b, y, &p, &mut tv).is_err());
            // Tampered level root.
            let mut p = proof.clone();
            p.level_roots[0] = [7u8; DIGEST];
            let mut tv = Transcript::new("split-ring", &[b"t"]);
            assert!(verify_ring_switch_split(&params, geo, &r_a, &r_b, y, &p, &mut tv).is_err());
        }
    }
}
