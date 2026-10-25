//! The **hierarchical Mosaic discharge** — the streaming root's bounded
//! memory path (ePrint 2025/611's hierarchical claim-batching shape,
//! with the 2026/587 round window applied per instance).
//!
//! ## The problem at 2^15+
//!
//! The root's batching sum-check runs over the **combined** cube
//! `2^{ell_seg + log2(segments)}` with two materialized factors: the G
//! bit table (`2^ell · 16 B` — one field element per committed **bit**)
//! and the `Σ α_i·eq(p_i, ·)` batching table (the same size again). At
//! 2^16 cycles that is 2×1 GiB before a single round, plus the
//! `Σ_i 2^{free_i}` scatter and `O(ell)` full-cube round passes — the
//! measured 2^15–2^16 streaming gate.
//!
//! ## The hierarchy
//!
//! The combined cube is the product `(segment bits) ∥ (local cube)` (the
//! block-aligned layout) and `G` is zero outside each segment's block,
//! so the batched sum **factors per segment** — cross-segment eq terms
//! vanish on segment `s`'s block:
//!
//! `Σ_x SumEq(x)·G(x) = Σ_s [ Σ_{x_l} LocalSumEq_s(x_l)·G_s(x_l) ]`.
//!
//! The discharge therefore runs as:
//!
//! 1. **Per-instance local stages in lockstep** (rayon-parallel): each
//!    instance proves its batching sum-check over its `ell_seg` variables
//!    against its **packed slice** — G's window bits are read straight
//!    from the commitment's 128-bit words (the window covers the top
//!    plane bits, so all `2^v` window values of a suffix live in ONE
//!    word) — and the per-item suffix `eq` tables are the only
//!    materialization, `O(items · 2^{ell_seg − v})` field elements. All
//!    instances share the per-round challenge (every instance's message
//!    is absorbed each round), ending at per-instance evaluation claims
//!    `Ĝ_s(r_local) = v_s` at a **common point**.
//! 2. **The segment stage**: a `log2(segments)`-round sum-check that
//!    eq-batches the `v_s` column into ONE claim
//!    `Ĝ_combined(r_seg ∥ r_local) = v''` (the dense factor is the
//!    `2^σ`-entry column).
//! 3. The ring switch (or the deep wrap) at the combined point.
//!
//! Every instance's state stays separate throughout — no cross-segment
//! "prefix adaptation" pass ever touches the `2^{ell_seg}` tables, which
//! is what keeps the root's peak memory at `O(one instance)`.

use crate::mosaic::{MosaicClaim, MosaicLayout};
use magnetar_field::{mul, EqBits, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_sumcheck::engine::{SumcheckProver, SumcheckVerifier};
use magnetar_sumcheck::public::{EqFactor, SumEqFactor, TableFactor};
use magnetar_sumcheck::{BindOrder, RoundMsg, SumcheckSpec, Term};
use rayon::prelude::*;

/// One instance's windowed local batching sum-check over its local cube.
///
/// The instance is a tree-IVC segment (the streaming root) or the whole
/// proof (the monolithic path, `segments = 1`).
/// One item's suffix classification: the point's coordinates split into
/// the **fixed Boolean** base and the **free** positions, with the free
/// WORD-index coordinates (cube bits `< ell − 7`) separated from the free
/// non-window PLANE coordinates (bits `[ell − 7, ell − v)`). The
/// word-index part of every claim's support is (nearly) the full word
/// range — all components' own coordinates are the LOW cube bits and
/// their aligned offsets live in the high (plane) bits — so the
/// eq-weighted fold over the free word coordinates is ONE packed-table
/// pass per **distinct free-word point**, shared by every claim whose
/// word coordinates coincide (the witness-segment claims share
/// `r_low`-prefix ‖ `r_inst`; the memory claims share `r_k ‖ r_cycle`).
pub(crate) struct Support {
    /// The fixed Boolean suffix bits (word part low, plane part high).
    base: usize,
    /// The free word-index positions and the point's values there.
    word_pos: Vec<usize>,
    word_pts: Vec<Gf128>,
    /// The free non-window plane positions (cube bits in
    /// `[ell − 7, ell − v)`) and the point's values there.
    plane_pos: Vec<usize>,
    plane_pts: Vec<Gf128>,
}

impl Support {
    /// Classify the full suffix point `pt` (coords `[0, ell − v)`,
    /// little-endian cube positions). `word_bits = ell − 7`,
    /// `plane_hi = ell − v`.
    pub(crate) fn build(pt: &[Gf128], word_bits: usize, plane_hi: usize) -> Self {
        let mut base = 0usize;
        let mut word_pos = Vec::new();
        let mut word_pts = Vec::new();
        let mut plane_pos = Vec::new();
        let mut plane_pts = Vec::new();
        for (b, &val) in pt.iter().enumerate() {
            if val == ZERO {
                // fixed at 0
            } else if val == ONE {
                base |= 1usize << b;
            } else if b < word_bits {
                word_pos.push(b);
                word_pts.push(val);
            } else {
                debug_assert!(b < plane_hi, "window coordinate in the suffix classifier");
                // The plane coordinate at cube bit b is PATTERN bit
                // (b − word_bits) in the packed word's bit index.
                plane_pos.push(b - word_bits);
                plane_pts.push(val);
            }
        }
        Support {
            base,
            word_pos,
            word_pts,
            plane_pos,
            plane_pts,
        }
    }

    /// The word-index part of the base.
    #[inline]
    pub(crate) fn base_word(&self, word_bits: usize) -> usize {
        self.base & ((1usize << word_bits) - 1)
    }

    /// The free-plane part of the base (already positioned at the plane
    /// cube bits).
    #[inline]
    pub(crate) fn base_plane(&self, word_bits: usize) -> usize {
        self.base >> word_bits
    }

    /// Call `f(i, word)` for every free word pattern: `i` indexes
    /// [`Self::word_eq`] and `word` is the concrete word index.
    #[allow(dead_code)] // kept for the window's future word-enumeration passes
    pub(crate) fn enumerate_words(&self, word_bits: usize, mut f: impl FnMut(usize, usize)) {
        let base_word = self.base_word(word_bits);
        if self.word_pos.is_empty() {
            f(0, base_word);
            return;
        }
        for i in 0..(1usize << self.word_pos.len()) {
            let mut word = base_word;
            for (j, &pos) in self.word_pos.iter().enumerate() {
                if (i >> j) & 1 == 1 {
                    word |= 1usize << pos;
                }
            }
            f(i, word);
        }
    }
}

/// One instance's windowed local batching sum-check over its local cube.
pub(crate) struct LocalDischarge<'a> {
    ell: usize,
    /// The window size: covers the top `v` local variables (all within
    /// the 7 packing planes, so G's window sub-cube is one packed word).
    v: usize,
    /// `2^(ell - v)`: the suffix count.
    stride: usize,
    /// The commitment's packed vector (borrowed).
    packed: &'a [Gf128],
    /// The instance's first word index.
    word_base: usize,
    /// The word index mask `2^(ell-7) - 1`.
    word_mask: usize,
    /// The per-item eq-weighted, past-folded monomial accumulators
    /// (`[item][2^v]`; G is multilinear over the window).
    q: Vec<Vec<Gf128>>,
    /// Per item: `Π_{i<round} eq(τ_{c,w,i}, r_i)`.
    c_eq: Vec<Gf128>,
    /// Per item: the window coordinates' τ in ROUND order
    /// (`tau_w[c][j] = pt_c[ell-1-j]`).
    tau_w: Vec<Vec<Gf128>>,
    /// Per item: the coefficient `α_c`.
    alphas: Vec<Gf128>,
    /// Per item: the suffix classification (fixed base, free word and
    /// plane coordinates).
    supports: Vec<Support>,
    /// The distinct free-word groups: (base_word, free word positions,
    /// the shared eq table over the free word coordinates).
    groups: Vec<(usize, Vec<usize>, Vec<Gf128>)>,
    /// Each item's group index.
    group_of: Vec<usize>,
    /// Window rounds completed.
    round: usize,
    /// The per-pattern MLE weights `eq(r_{<round}, m)` (G's contraction).
    eq_r: Vec<Gf128>,
    /// The running claim (updated to `m_j(r_j)` per round).
    pub running: Gf128,
    /// The post-window tail engine (set at the window's end).
    tail: Option<SumcheckProver>,
    /// All challenges so far, in round order.
    challenges: Vec<Gf128>,
}

impl<'a> LocalDischarge<'a> {
    /// Build the instance's state: the **group-fold** over the free word
    /// coordinates (one packed-table pass per distinct free-word point,
    /// shared by every claim whose word coordinates coincide), then the
    /// tiny per-claim plane folds and the window monomial butterfly.
    /// Replaces the per-suffix 64-plane extraction + scatter of the
    /// naive pass (measured 12–30× on the streaming scales) and the
    /// `2^{ell-v}`-per-item suffix tables.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        layout: &MosaicLayout,
        packed: &'a [Gf128],
        word_base: usize,
        items: &[(Gf128, Vec<Gf128>)],
        claim: Gf128,
    ) -> Self {
        let ell = layout.ell;
        debug_assert!(ell >= 8, "the Mosaic packing needs ell >= 7");
        let v = 6.min(ell - 7).max(1).min(ell - 1);
        let stride = 1usize << (ell - v);
        let word_mask = (1usize << (ell - 7)) - 1;
        let word_bits = ell - 7;
        let plane_hi = ell - v;
        let n_items = items.len();
        debug_assert!((1..=6).contains(&v));
        debug_assert!(n_items >= 1);
        let tau_w: Vec<Vec<Gf128>> = items
            .iter()
            .map(|(_, pt)| (0..v).map(|j| pt[ell - 1 - j]).collect())
            .collect();
        let alphas: Vec<Gf128> = items.iter().map(|(a, _)| *a).collect();
        // Classify every claim's suffix point.
        let _pt_cls = crate::protocol::PhaseTimer::new("ld-classify");
        let supports: Vec<Support> = items
            .iter()
            .map(|(_, pt)| Support::build(&pt[..plane_hi], word_bits, plane_hi))
            .collect();
        drop(_pt_cls);
        // Group the claims by (base word, free word positions, their
        // point values): the fold output is shared within a group.
        let _pt_grp = crate::protocol::PhaseTimer::new("ld-groups");
        let mut group_of: Vec<usize> = vec![usize::MAX; n_items];
        let mut groups: Vec<(usize, Vec<usize>, Vec<Gf128>)> = Vec::new(); // (base_word, word_pos, word_pts)
        for (c, sup) in supports.iter().enumerate() {
            let bw = sup.base_word(word_bits);
            let mut found = None;
            for (gi, g) in groups.iter().enumerate() {
                if g.0 == bw
                    && g.1 == sup.word_pos
                    && g.2.iter().zip(sup.word_pts.iter()).all(|(a, b)| a == b)
                {
                    found = Some(gi);
                    break;
                }
            }
            match found {
                Some(gi) => group_of[c] = gi,
                None => {
                    groups.push((bw, sup.word_pos.clone(), sup.word_pts.clone()));
                    group_of[c] = groups.len() - 1;
                }
            }
        }
        drop(_pt_grp);
        if std::env::var("MAGNETAR_TIME").is_ok() {
            let total_word: usize = groups.iter().map(|g| 1usize << g.1.len()).sum();
            eprintln!(
                "[time] ld-shape: ell={ell} stride={stride} items={n_items} groups={} word-vol={total_word}",
                groups.len()
            );
        }
        // Build each group's free-word eq table once (stored in the
        // group tuple for the fold AND the later bound_se walk).
        for g in groups.iter_mut() {
            let eq = if g.2.is_empty() {
                vec![ONE]
            } else {
                EqBits::build(&g.2).values
            };
            g.2 = eq;
        }
        // The group folds: out1 = the eq-weighted fold of the packed
        // words over the group's free word coordinates, a 128-lane
        // field table (one lane per plane pattern).
        let _pt_fold = crate::protocol::PhaseTimer::new("ld-word-fold");
        let folds: Vec<[Gf128; 128]> = groups
            .par_iter()
            .map(|(bw, wpos, weq)| {
                let mut out = [ZERO; 128];
                let npat = 1usize << wpos.len();
                // The free word positions are almost always the identity
                // range [0, fw) (the components' own coordinates are the
                // low cube bits) — take the branch-free fast path.
                let identity = wpos.iter().enumerate().all(|(j, &p)| p == j);
                for i in 0..npat {
                    let e = weq[i];
                    if e == ZERO {
                        continue;
                    }
                    let word = if identity {
                        bw | i
                    } else {
                        let mut w = *bw;
                        for (j, &pos) in wpos.iter().enumerate() {
                            if (i >> j) & 1 == 1 {
                                w |= 1usize << pos;
                            }
                        }
                        w
                    };
                    // The word's set bits scatter into the plane lanes.
                    let mut bits = packed[word_base + word].0;
                    while bits != 0 {
                        let pl = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        out[pl] += e;
                    }
                }
                out
            })
            .collect();
        drop(_pt_fold);
        // Per claim: the plane folds + the window monomial butterfly.
        let _pt_local = crate::protocol::PhaseTimer::new("ld-local-fold");
        let q: Vec<Vec<Gf128>> = supports
            .iter()
            .enumerate()
            .map(|(c, sup)| {
                let out1 = &folds[group_of[c]];
                // eq-fold over the free non-window plane coordinates
                // (their cube bits sit at the pattern's low positions —
                // plane coord ell-7 is pattern bit 0, and so on up to
                // ell-v at pattern bit 7-v-1).
                let n_plane = 1usize << sup.plane_pts.len();
                let plane_eq = if sup.plane_pts.is_empty() {
                    vec![ONE]
                } else {
                    EqBits::build(&sup.plane_pts).values
                };
                // Fold each free plane coordinate: the coord at cube bit
                // (word_bits + orig_t) is pattern bit orig_t. `bind_var`
                // folds AND HALVES the table, so the k-th bind works on
                // position (orig_t − k) and the surviving lanes' bits
                // shift down accordingly.
                let mut tab: Vec<Gf128> = out1.to_vec();
                for (k, &orig_t) in sup.plane_pos.iter().enumerate() {
                    magnetar_field::bind_var(&mut tab, orig_t - k, sup.plane_pts[k]);
                }
                let _ = plane_eq;
                let _ = n_plane;
                let np = sup.plane_pos.len();
                // Gather the window-pattern lanes: the surviving table is
                // `2^{7-np}` lanes indexed by the (fixed-plane ‖ window)
                // pattern with the free bits removed. The fixed base
                // bits remap down by the number of free bits below them;
                // the window block starts at `7 - v - np`.
                let base_plane = sup.base_plane(word_bits); // the fixed plane bits
                let win_lo = 7 - v - np;
                let mut fixed = 0usize;
                for q in 0..(7 - v) {
                    if (base_plane >> q) & 1 == 1 {
                        let below = sup.plane_pos.iter().filter(|&&t| t < q).count();
                        fixed |= 1usize << (q - below);
                    }
                }
                let mut out2 = vec![ZERO; 1usize << v];
                for w in 0..(1usize << v) {
                    let pat = fixed | (w << win_lo);
                    out2[w] = tab[pat];
                }
                // The Möbius butterfly over the v window bits — field
                // values: `out2[e] ← Σ_{m ⊆ e} raw[m]` (the high side
                // accumulates the low, exactly the bit-mask swirl's
                // semantics lifted to field entries).
                let n = 1usize << v;
                for j in 0..v {
                    let bit = 1usize << j;
                    for base in (0..n).step_by(bit * 2) {
                        for o in 0..bit {
                            let lo = out2[base + o];
                            out2[base + bit + o] += lo;
                        }
                    }
                }
                out2
            })
            .collect();
        drop(_pt_local);
        LocalDischarge {
            ell,
            v,
            stride,
            packed,
            word_base,
            word_mask,
            q,
            c_eq: vec![ONE; n_items],
            tau_w,
            alphas,
            supports,
            groups,
            group_of,
            round: 0,
            eq_r: vec![ONE; 1usize << v],
            running: claim,
            tail: None,
            challenges: Vec::with_capacity(ell),
        }
    }

    /// The current round's message (degree 2 → 3 coefficients).
    pub(crate) fn message(&self) -> RoundMsg {
        if self.round < self.v {
            self.window_message()
        } else {
            self.tail.as_ref().expect("tail active").message()
        }
    }

    /// The window rounds' message from the monomial state.
    fn window_message(&self) -> RoundMsg {
        let j = self.round;
        let coord = self.v - 1 - j;
        let mut c0 = ZERO;
        let mut c1 = ZERO;
        let mut c2 = ZERO;
        for (item, q) in self.q.iter().enumerate() {
            let mut a0 = ZERO;
            let mut a1 = ZERO;
            for (m, &val) in q.iter().enumerate() {
                if val == ZERO {
                    continue;
                }
                // The future coordinates (t < coord) each contribute their
                // item-specific τ when the monomial covers them
                // (`Σ_b eq(τ,b)·b^e = τ` for `e ≥ 1`, `1` for `e = 0`).
                let mut fw = ONE;
                for t in 0..coord {
                    if (m >> t) & 1 == 1 {
                        fw = mul(fw, self.tau_w[item][self.v - 1 - t]);
                    }
                }
                let w = mul(val, fw);
                if (m >> coord) & 1 == 1 {
                    a1 += w;
                } else {
                    a0 += w;
                }
            }
            // m(X) += α·C·((1+τ)+X)·(A0 + A1·X)
            let c = mul(self.alphas[item], self.c_eq[item]);
            let tau = self.tau_w[item][j];
            let t0 = ONE + tau;
            c0 += mul(c, mul(t0, a0));
            c1 += mul(c, a0) + mul(c, mul(t0, a1));
            c2 += mul(c, a1);
        }
        RoundMsg(vec![c0, c1, c2])
    }

    /// Bind the round's challenge (the message was already computed and
    /// absorbed by the driver).
    pub(crate) fn bind_msg(&mut self, msg: &RoundMsg, r: Gf128) {
        let mut mr = ZERO;
        let mut xp = Gf128(1);
        for &c in &msg.0 {
            mr += mul(c, xp);
            xp = mul(xp, r);
        }
        self.running = mr;
        self.challenges.push(r);
        if self.round < self.v {
            let j = self.round;
            let coord = self.v - 1 - j;
            for (item, q) in self.q.iter_mut().enumerate() {
                for m in 0..q.len() {
                    if (m >> coord) & 1 == 1 {
                        let val = q[m];
                        q[m] = ZERO;
                        q[m ^ (1 << coord)] += mul(val, r);
                    }
                }
                self.c_eq[item] = mul(self.c_eq[item], ONE + self.tau_w[item][j] + r);
            }
            for (m, e) in self.eq_r.iter_mut().enumerate() {
                let mb = Gf128(((m >> coord) & 1) as u128);
                *e = mul(*e, ONE + r + mb);
            }
            self.round += 1;
            if self.round == self.v {
                self.finish_window();
            }
        } else {
            let tail = self.tail.as_mut().expect("tail active");
            tail.fold_msg(msg, r);
            self.round += 1;
        }
    }

    /// The window's fast-forward: materialize the bound G table from the
    /// packed words (one select per (pattern, suffix)), the bound SumEq
    /// table as the `α·C`-weighted sum of the suffix tables, and hand the
    /// remaining `ell − v` rounds to the generic engine.
    fn finish_window(&mut self) {
        let _pt_fw = crate::protocol::PhaseTimer::new("ld-finish-window");
        let plane_lo_bits = 7 - self.v;
        let stride = self.stride;
        let v = self.v;
        let word_base = self.word_base;
        let word_mask = self.word_mask;
        let packed = self.packed;
        let eq_r = &self.eq_r;
        let ell = self.ell;
        let mut bound_g = vec![ZERO; stride];
        let chunk = if stride >= 2 * magnetar_field::PAR_THRESHOLD {
            (stride / (rayon::current_num_threads().max(1) * 4))
                .clamp(magnetar_field::PAR_THRESHOLD, stride)
        } else {
            stride
        };
        let _pt_bg = crate::protocol::PhaseTimer::new("fw-bound-g");
        bound_g
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, o)| {
                let base = ci * chunk;
                for (rel, out) in o.iter_mut().enumerate() {
                    let x = base + rel;
                    let u = x & word_mask;
                    let b_lo = x >> (ell - 7);
                    let word = packed[word_base + u].0;
                    let mut acc = ZERO;
                    for m in 0..(1usize << v) {
                        let w = eq_r[m];
                        if w == ZERO {
                            continue;
                        }
                        if (word >> (b_lo + m * (1usize << plane_lo_bits))) & 1 == 1 {
                            acc += w;
                        }
                    }
                    *out = acc;
                }
            });
        // The bound SumEq: Σ_c α_c·C_c·eq_c(x) over the suffix cube —
        // the group outer product: within a group every member shares
        // the free-word eq table, so the word dimension is walked once
        // per group and each member contributes only its (tiny) plane
        // support: at most `2^{7-v}` patterns per claim per word.
        drop(_pt_bg);
        let _pt_bs = crate::protocol::PhaseTimer::new("fw-bound-se");
        let word_bits = self.ell - 7;
        let mut bound_se = vec![ZERO; stride];
        let groups = &self.groups;
        let group_of = &self.group_of;
        let supports = &self.supports;
        let alphas = &self.alphas;
        let c_eq = &self.c_eq;
        // Parallel over groups (each group's scatter is disjoint in the
        // words it touches? no — groups share the word range; keep the
        // scatter sequential per group but the groups rayon-parallel via
        // per-thread buffers is overkill — sequential here, the volume is
        // `Σ_groups 2^{|Fw|} · Σ_members 2^{np}`).
        for (gi, (base_word, wpos, weq)) in groups.iter().enumerate() {
            let members: Vec<usize> = (0..supports.len()).filter(|&c| group_of[c] == gi).collect();
            // Precompute each member's plane contribution:
            // (scalar α_c·C_c, plane support (pattern → weight)).
            let plane_supp: Vec<(Gf128, Vec<(usize, Gf128)>)> = members
                .iter()
                .map(|&c| {
                    let s = mul(alphas[c], c_eq[c]);
                    let sup = &supports[c];
                    let np = sup.plane_pts.len();
                    let peq = if np == 0 {
                        vec![ONE]
                    } else {
                        EqBits::build(&sup.plane_pts).values
                    };
                    let base_plane = sup.base_plane(word_bits);
                    let mut pats = Vec::with_capacity(1usize << np);
                    for (j, &e) in peq.iter().enumerate() {
                        if e == ZERO {
                            continue;
                        }
                        let mut p = base_plane;
                        for (t, &pos) in sup.plane_pos.iter().enumerate() {
                            if (j >> t) & 1 == 1 {
                                p |= 1usize << pos;
                            }
                        }
                        pats.push((p, mul(s, e)));
                    }
                    (s, pats)
                })
                .collect();
            let _ = &plane_supp;
            // The word walk (identity fast path — see the fold).
            let npat = 1usize << wpos.len();
            let identity = wpos.iter().enumerate().all(|(j, &p)| p == j);
            // Split across threads by plane blocks then word ranges: the
            // suffix cube factors as (word ‖ plane), so `bound_se`'s plane
            // `p` occupies the contiguous block `[p·2^wb, (p+1)·2^wb)` and
            // each block is walked independently (the group's eq table
            // and the members' plane weights are read-only).
            let n_planes = 1usize << (7 - v);
            let wb = word_bits;
            for p in 0..n_planes {
                // The members contributing to this plane (their pattern
                // list contains p with weight pe).
                let contrib: Vec<(usize, Gf128)> = plane_supp
                    .iter()
                    .enumerate()
                    .filter_map(|(m, (_, pats))| {
                        pats.iter()
                            .find(|&&(pp, _)| pp == p)
                            .map(|&(_, pe)| (m, pe))
                    })
                    .collect();
                if contrib.is_empty() {
                    continue;
                }
                let blk = 1usize << wb;
                let base_idx = p * blk;
                let (lo, hi) = (base_idx, base_idx + blk);
                let slice = &mut bound_se[lo..hi];
                let bw = *base_word;
                let weq_c = weq;
                let fw = wpos.len();
                let supp_mask = (1usize << fw) - 1;
                if identity {
                    // The common production shape: the free word
                    // positions are [0, fw). A plane-block offset is in
                    // the support iff its HIGH bits match the base's —
                    // the eq table indexes the low fw bits.
                    slice
                        .par_iter_mut()
                        .enumerate()
                        .chunks(1usize << 12)
                        .for_each(|cells| {
                            for (off, cell) in cells {
                                if (off >> fw) != (bw >> fw) {
                                    continue;
                                }
                                let i = off & supp_mask;
                                let e = weq_c[i];
                                if e == ZERO {
                                    continue;
                                }
                                for (_m, pe) in contrib.iter() {
                                    *cell += mul(*pe, e);
                                }
                            }
                        });
                } else {
                    // The general walk (rare: a component offset reaches
                    // into the word bits, or non-identity positions).
                    let wpos_c = wpos.clone();
                    for i in 0..npat {
                        let e = weq_c[i];
                        if e == ZERO {
                            continue;
                        }
                        let word = if identity {
                            bw | i
                        } else {
                            let mut w = bw;
                            for (j, &pos) in wpos_c.iter().enumerate() {
                                if (i >> j) & 1 == 1 {
                                    w |= 1usize << pos;
                                }
                            }
                            w
                        };
                        if word >= blk {
                            continue; // not this plane block's word range? word < 2^wb always
                        }
                        for (_m, pe) in contrib.iter() {
                            slice[word] += mul(*pe, e);
                        }
                    }
                }
            }
        }
        drop(_pt_fw);
        let _pt_tail = crate::protocol::PhaseTimer::new("ld-tail-rounds");
        drop(_pt_bs);
        let _pt_spec = crate::protocol::PhaseTimer::new("fw-tail-spec");
        let claim = self.running;
        let spec = SumcheckSpec {
            num_vars: ell - v,
            dense: vec![bound_g],
            public: vec![Box::new(TableFactor::new(bound_se))],
            terms: vec![Term {
                coeff: ONE,
                dense: vec![0],
                public: vec![0],
            }],
            claim,
        };
        let order = BindOrder::HighBitsFirst { high: 7 - v };
        self.tail = Some(SumcheckProver::new(&spec, order));
    }

    /// After all `ell` rounds: the instance's terminal claim
    /// `Ĝ_s(r_local) = v_s` and the local point (original coordinates).
    pub(crate) fn finish(self) -> (Gf128, Vec<Gf128>) {
        debug_assert_eq!(self.round, self.ell);
        let tail = self.tail.expect("tail consumed all rounds");
        let tail_ch = tail.challenges();
        let (claims, _) = tail.finish();
        let v_s = claims[0];
        // The local point in original coordinates. The full round
        // sequence matches BindOrder::HighBitsFirst{high: 7} exactly:
        // the window rounds bind ell-1 … ell-v; the tail
        // (HighBitsFirst{7-v} over the remaining ell-v variables) binds
        // ell-v-1 … ell-7 next (its high phase), then 0, 1, 2, ….
        let n_win = self.v;
        let n_tail = self.ell - self.v;
        debug_assert_eq!(tail_ch.len(), n_tail);
        let mut point = vec![ZERO; self.ell];
        for j in 0..n_win {
            point[self.ell - 1 - j] = self.challenges[j];
        }
        let high_tail = 7 - self.v;
        for i in 0..n_tail {
            let orig = if i < high_tail {
                self.ell - self.v - 1 - i
            } else {
                i - high_tail
            };
            point[orig] = tail_ch[i];
        }
        (v_s, point)
    }
}

/// One instance's claim sum and lifted items.
pub(crate) type InstanceItems = Vec<(Gf128, Vec<(Gf128, Vec<Gf128>)>)>;

/// Group the claims by instance and lift them into the per-instance
/// layout. A claim's instance is its component index's high bits
/// (`component / n_comp`) — several claims may share a component (the
/// boundary, cross-step, and Ψ arguments produce multiple points on one
/// component), so the grouping follows the claim's own segment, never
/// its position. Returns one `(claim_sum, items)` pair per instance.
pub(crate) fn instance_items(
    per: &MosaicLayout,
    segments: usize,
    claims: &[MosaicClaim],
    alphas: &[Gf128],
) -> InstanceItems {
    let n_comp = per.components.len();
    debug_assert!(claims.iter().all(|c| c.component / n_comp < segments));
    let mut sums = vec![ZERO; segments];
    let mut items: Vec<Vec<(Gf128, Vec<Gf128>)>> = (0..segments)
        .map(|_| Vec::with_capacity(claims.len() / segments + 8))
        .collect();
    for (claim, &alpha) in claims.iter().zip(alphas.iter()) {
        let s = claim.component / n_comp;
        sums[s] += mul(alpha, claim.value);
        let local = MosaicClaim {
            component: claim.component % n_comp,
            point: claim.point.clone(),
            value: claim.value,
        };
        items[s].push((alpha, local.lifted_point(per)));
    }
    sums.into_iter().zip(items).collect()
}

/// The hierarchical Mosaic proof body (everything but the ring switch).
#[derive(Clone, Debug)]
pub struct HierBody {
    /// The per-instance local rounds, each a 3-coefficient message
    /// (`[instance][round][3]`).
    pub local_rounds: Vec<Vec<Vec<Gf128>>>,
    /// The per-instance terminals `Ĝ_s(r_local) = v_s` (bound into the
    /// transcript before the segment stage samples its point).
    pub terminals: Vec<Gf128>,
    /// The segment-stage rounds (`log2(segments)` of them; empty for a
    /// single instance).
    pub seg_rounds: Vec<Vec<Gf128>>,
    /// The terminal `Ĝ_combined(r*) = v''`.
    pub terminal: Gf128,
    /// The ZK finish offset (ZERO for plain proofs; ZK requires a single
    /// instance).
    pub zk_offset: Gf128,
}

/// Prove the hierarchical discharge body: the lockstep per-instance local
/// stages + the segment stage. Returns the body, the combined point
/// `r*`, and the terminal.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_hier_body(
    per: &MosaicLayout,
    segments: usize,
    packed: &[Gf128],
    claims: &[MosaicClaim],
    alphas: &[Gf128],
    zk: bool,
    t: &mut Transcript,
) -> (HierBody, Vec<Gf128>) {
    let sigma = segments.trailing_zeros() as usize;
    let ell_s = per.ell;
    let n = per.ell - 7; // the per-instance word count log2
    assert!(
        !zk || segments == 1,
        "ZK masking of the hierarchical discharge requires a single instance"
    );
    let inst = instance_items(per, segments, claims, alphas);
    // The per-instance discharges (the pass runs inside `new`) —
    // rayon-parallel across the instances (the construction pass is
    // pure per-instance work: suffix eq tables + packed reads).
    let mut dis: Vec<LocalDischarge> = inst
        .par_iter()
        .enumerate()
        .map(|(s, (claim, items))| LocalDischarge::new(per, packed, s << n, items, *claim))
        .collect();
    let mut masker = magnetar_sumcheck::zk::ZkMasker::new();
    let mut local_rounds: Vec<Vec<Vec<Gf128>>> = vec![Vec::with_capacity(ell_s); segments];
    // ---- The lockstep local rounds. ----
    // The per-instance messages compute rayon-parallel (order-preserving
    // collect; every instance absorbs the SAME shared challenge after).
    for _j in 0..ell_s {
        let msgs: Vec<RoundMsg> = dis.par_iter().map(|d| d.message()).collect();
        let mut transmitted: Vec<Vec<Gf128>> = Vec::with_capacity(segments);
        if zk {
            for m in &msgs {
                let mut masked = m.clone();
                masker.mask(&mut masked);
                transmitted.push(masked.0);
            }
        } else {
            for m in &msgs {
                transmitted.push(m.0.clone());
            }
        }
        for (s, tr) in transmitted.iter().enumerate() {
            local_rounds[s].push(tr.clone());
        }
        let mut buf = Vec::with_capacity(segments * 3 * 16);
        for tr in &transmitted {
            for c in tr {
                buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
            }
        }
        t.absorb_bytes("mosaic-batch", &buf);
        let r = t.challenge_field("mosaic-batch");
        if zk {
            masker.commit(r);
        }
        dis.par_iter_mut()
            .zip(msgs.par_iter())
            .for_each(|(d, m)| d.bind_msg(m, r));
    }
    let mut terminals = Vec::with_capacity(segments);
    let mut r_local = Vec::new();
    for d in dis {
        let (v_s, point) = d.finish();
        terminals.push(v_s);
        r_local = point; // the same for every instance
    }
    // Bind the per-instance terminals before the segment stage samples
    // its point.
    {
        let mut buf = Vec::with_capacity(segments * 16);
        for &v in &terminals {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(v));
        }
        t.absorb_bytes("mosaic-local-terminal", &buf);
    }
    // ---- The segment stage (σ rounds over the v_s column). ----
    let mut seg_rounds = Vec::new();
    let mut terminal = terminals[0];
    let mut r_seg: Vec<Gf128> = Vec::new();
    if sigma > 0 {
        let e = t.challenge_field_vec("mosaic-seg-pt", sigma);
        let mut claim = ZERO;
        for (s, &v) in terminals.iter().enumerate() {
            let mut w = ONE;
            for (k, &ek) in e.iter().enumerate() {
                let bit = Gf128(((s >> k) & 1) as u128);
                w = mul(w, ONE + ek + bit);
            }
            claim += mul(w, v);
        }
        // The dense factor: the v_s column's MLE over the segment bits.
        let col: Vec<Gf128> = (0..segments).map(|s| terminals[s]).collect();
        let spec = SumcheckSpec {
            num_vars: sigma,
            dense: vec![col],
            public: vec![Box::new(EqFactor::new(e))],
            terms: vec![Term {
                coeff: ONE,
                dense: vec![0],
                public: vec![0],
            }],
            claim,
        };
        let mut p = SumcheckProver::new(&spec, BindOrder::Increasing);
        for _ in 0..sigma {
            let m = p.message();
            if zk {
                let mut masked = m.clone();
                masker.mask(&mut masked);
                seg_rounds.push(masked.0.clone());
                absorb_msg(t, &masked, "mosaic-seg");
            } else {
                seg_rounds.push(m.0.clone());
                absorb_msg(t, &m, "mosaic-seg");
            }
            let r = t.challenge_field("mosaic-seg");
            if zk {
                masker.commit(r);
            }
            p.fold_msg(&m, r);
        }
        r_seg = p.challenges();
        let (dc, _) = p.finish();
        terminal = dc[0];
    }
    let zk_offset = if zk { masker.offset() } else { ZERO };
    // The combined point: [segment σ coords] ∥ [local ell_s coords].
    let mut r_star = r_local;
    r_star.extend(r_seg);
    (
        HierBody {
            local_rounds,
            terminals,
            seg_rounds,
            terminal,
            zk_offset,
        },
        r_star,
    )
}

/// Absorb a single message under `label`.
fn absorb_msg(t: &mut Transcript, m: &RoundMsg, label: &str) {
    let mut buf = Vec::with_capacity(m.0.len() * 16);
    for c in &m.0 {
        buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
    }
    t.absorb_bytes(label, &buf);
}

/// Verify the hierarchical discharge body. On success returns the
/// combined point `r*` (for the ring switch / wrap).
pub(crate) fn verify_hier_body(
    per: &MosaicLayout,
    segments: usize,
    claims: &[MosaicClaim],
    alphas: &[Gf128],
    body: &HierBody,
    t: &mut Transcript,
) -> Result<Vec<Gf128>, &'static str> {
    let sigma = segments.trailing_zeros() as usize;
    let ell_s = per.ell;
    if body.local_rounds.len() != segments {
        return Err("mosaic: instance count mismatch");
    }
    for rounds in &body.local_rounds {
        if rounds.len() != ell_s {
            return Err("mosaic: local round count mismatch");
        }
    }
    if body.seg_rounds.len() != sigma {
        return Err("mosaic: segment round count mismatch");
    }
    if body.terminals.len() != segments {
        return Err("mosaic: terminal count mismatch");
    }
    let inst = instance_items(per, segments, claims, alphas);
    // The per-instance public factors (the instance's SumEq over the
    // LOCAL lifted points) and the verifiers borrowing them.
    let publics: Vec<Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>>> = inst
        .iter()
        .map(|(_claim, items)| {
            vec![Box::new(SumEqFactor::new(items.clone()))
                as Box<
                    dyn magnetar_sumcheck::public::PublicFactor + Send + Sync,
                >]
        })
        .collect();
    let mut verifiers: Vec<SumcheckVerifier> = inst
        .iter()
        .zip(publics.iter())
        .map(|((claim, _items), public)| {
            SumcheckVerifier::new(
                ell_s,
                *claim,
                public,
                vec![Term {
                    coeff: ONE,
                    dense: vec![0],
                    public: vec![0],
                }],
                BindOrder::HighBitsFirst { high: 7 },
            )
        })
        .collect();
    // The lockstep local rounds.
    for j in 0..ell_s {
        let mut msgs = Vec::with_capacity(segments);
        for s in 0..segments {
            msgs.push(RoundMsg(body.local_rounds[s][j].clone()));
        }
        let mut buf = Vec::with_capacity(segments * 3 * 16);
        for m in &msgs {
            for c in &m.0 {
                buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
            }
        }
        t.absorb_bytes("mosaic-batch", &buf);
        let r = t.challenge_field("mosaic-batch");
        for (v, m) in verifiers.iter_mut().zip(msgs.iter()) {
            v.round_external(m, r)
                .map_err(|_| "mosaic: local round check failed")?;
        }
    }
    // The per-instance terminals: transmitted, checked against each
    // instance's terminal identity, and bound into the transcript.
    {
        let mut buf = Vec::with_capacity(segments * 16);
        for &v in &body.terminals {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(v));
        }
        t.absorb_bytes("mosaic-local-terminal", &buf);
    }
    let mut terminals = Vec::with_capacity(segments);
    let mut r_local: Vec<Gf128> = Vec::new();
    for (v, &v_s) in verifiers.iter().zip(body.terminals.iter()) {
        let (point, _) = v
            .finish_offset(&[v_s], body.zk_offset)
            .map_err(|_| "mosaic: local terminal failed")?;
        terminals.push(v_s);
        r_local = point;
    }
    // The segment stage.
    let mut terminal = terminals[0];
    let mut r_seg: Vec<Gf128> = Vec::new();
    if sigma > 0 {
        let e = t.challenge_field_vec("mosaic-seg-pt", sigma);
        let mut claim = ZERO;
        for (s, &v) in terminals.iter().enumerate() {
            let mut w = ONE;
            for (k, &ek) in e.iter().enumerate() {
                let bit = Gf128(((s >> k) & 1) as u128);
                w = mul(w, ONE + ek + bit);
            }
            claim += mul(w, v);
        }
        let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
            vec![Box::new(EqFactor::new(e))];
        let mut v = SumcheckVerifier::new(
            sigma,
            claim,
            &public,
            vec![Term {
                coeff: ONE,
                dense: vec![0],
                public: vec![0],
            }],
            BindOrder::Increasing,
        );
        for msg in &body.seg_rounds {
            let m = RoundMsg(msg.clone());
            absorb_msg(t, &m, "mosaic-seg");
            let r = t.challenge_field("mosaic-seg");
            v.round_external(&m, r)
                .map_err(|_| "mosaic: segment round check failed")?;
        }
        let v_s_claim = body.terminal;
        let (point, _) = v
            .finish_offset(&[v_s_claim], ZERO)
            .map_err(|_| "mosaic: segment terminal failed")?;
        terminal = v_s_claim;
        r_seg = point;
    }
    if body.terminal != terminal {
        return Err("mosaic: terminal mismatch");
    }
    let mut r_star = r_local;
    r_star.extend(r_seg);
    Ok(r_star)
}

#[cfg(test)]
mod tests {
    /// The alternating SWAR masks for the in-place Möbius butterfly over
    /// a `u64` truth table (bit `t` = coordinate `t`, little-endian) —
    /// the tests' reference Möbius against the production machinery.
    #[inline]
    pub(super) fn swirl_mask(j: usize) -> u64 {
        match j {
            0 => 0xAAAA_AAAA_AAAA_AAAA,
            1 => 0xCCCC_CCCC_CCCC_CCCC,
            2 => 0xF0F0_F0F0_F0F0_F0F0,
            3 => 0xFF00_FF00_FF00_FF00,
            4 => 0xFFFF_0000_FFFF_0000,
            _ => 0xFFFF_FFFF_0000_0000,
        }
    }

    use super::*;

    /// The LocalDischarge's messages must be byte-identical to the
    /// generic engine's on the same per-instance statement, and the full
    /// hierarchical body must verify — the multi-segment regression.
    #[test]
    fn local_discharge_matches_generic_engine() {
        use crate::mosaic::{commit_mosaic_split, ComponentSpec, MosaicLayout};
        use magnetar_pcs::PcsParams;
        let mut rng = rand::rng();
        // A small per-instance layout: ell_s = 12 (union 2^12 + 2^10).
        let per = MosaicLayout::build(vec![
            ComponentSpec {
                name: "big".into(),
                vars: 12,
            },
            ComponentSpec {
                name: "small".into(),
                vars: 10,
            },
        ]);
        let segments = 2usize;
        let combined = magnetar_zkvm_test::block_align(&per, segments);
        // Random bit tables per instance.
        let tables: Vec<Vec<Vec<bool>>> = (0..segments)
            .map(|_| {
                vec![
                    (0..(1usize << 12))
                        .map(|_| rand::RngExt::random::<bool>(&mut rng))
                        .collect(),
                    (0..(1usize << 10))
                        .map(|_| rand::RngExt::random::<bool>(&mut rng))
                        .collect(),
                ]
            })
            .collect();
        let mut rg = || Gf128(rand::RngExt::random::<u128>(&mut rng));
        let all_tables: Vec<Vec<bool>> = tables.concat();
        let cm = commit_mosaic_split(&PcsParams::default(), &combined, per.ell - 7, &all_tables);
        // Claims: random points per (instance, component), values = the
        // MLE of the table at the point.
        let mut claims = Vec::new();
        for (s, seg_tables) in tables.iter().enumerate() {
            for (c, tab) in seg_tables.iter().enumerate() {
                let point: Vec<Gf128> = (0..per.vars(c)).map(|_| rg()).collect();
                let value = {
                    let tab_g: Vec<Gf128> = tab.iter().map(|&b| Gf128(b as u128)).collect();
                    magnetar_field::eval_mle(&tab_g, &point)
                };
                claims.push(crate::mosaic::MosaicClaim {
                    component: s * per.components.len() + c,
                    point,
                    value,
                });
            }
        }
        let alphas: Vec<Gf128> = claims.iter().map(|_| rg()).collect();
        // The per-instance generic cross-check: build instance 1's g_table
        // and SumEq spec directly.
        let inst_items = instance_items(&per, segments, &claims, &alphas);
        for (s, (claim, items)) in inst_items.iter().enumerate() {
            // The g_table over the local cube: the union of the instance's
            // components (zero in gaps).
            let n_local = 1usize << per.ell;
            let mut g = vec![Gf128(0); n_local];
            for (c, tab) in tables[s].iter().enumerate() {
                for (j, &b) in tab.iter().enumerate() {
                    g[per.offsets[c] + j] = Gf128(b as u128);
                }
            }
            let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
                vec![Box::new(SumEqFactor::new(items.clone()))];
            let spec = SumcheckSpec {
                num_vars: per.ell,
                dense: vec![g],
                public,
                terms: vec![Term {
                    coeff: ONE,
                    dense: vec![0],
                    public: vec![0],
                }],
                claim: *claim,
            };
            let order = BindOrder::HighBitsFirst { high: 7 };
            let mut gen = SumcheckProver::new(&spec, order);
            let mut loc = LocalDischarge::new(&per, &cm.packed, s << (per.ell - 7), items, *claim);
            let mut t = magnetar_hash::Transcript::new("xcheck", &[]);
            for j in 0..per.ell {
                let gm = gen.message();
                let lm = loc.message();
                assert_eq!(gm.0, lm.0, "instance {s} round {j}");
                let mut buf = Vec::new();
                for c in &lm.0 {
                    buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
                }
                t.absorb_bytes("r", &buf);
                let r = t.challenge_field("r");
                gen.fold_msg(&gm, r);
                loc.bind_msg(&lm, r);
            }
        }
        // The full body must verify.
        let mut t = magnetar_hash::Transcript::new("body", &[]);
        let (body, r_star) =
            prove_hier_body(&per, segments, &cm.packed, &claims, &alphas, false, &mut t);
        let mut t2 = magnetar_hash::Transcript::new("body", &[]);
        let got = verify_hier_body(&per, segments, &claims, &alphas, &body, &mut t2)
            .expect("body verifies");
        assert_eq!(got, r_star);
    }

    mod magnetar_zkvm_test {
        pub fn block_align(
            per: &crate::mosaic::MosaicLayout,
            segments: usize,
        ) -> crate::mosaic::MosaicLayout {
            let sigma = segments.trailing_zeros() as usize;
            let ell = per.ell + sigma;
            let mut components = Vec::new();
            let mut offsets = Vec::new();
            for s in 0..segments {
                let base = s << per.ell;
                for (i, c) in per.components.iter().enumerate() {
                    components.push(crate::mosaic::ComponentSpec {
                        name: format!("s{s}-{}", c.name),
                        vars: c.vars,
                    });
                    offsets.push(base + per.offsets[i]);
                }
            }
            crate::mosaic::MosaicLayout {
                components,
                ell,
                offsets,
            }
        }
    }

    /// The SWAR Möbius mask must equal the direct XOR-fold.
    #[test]
    fn swirl_mobius_matches_direct() {
        let mut rng = rand::rng();
        for v in [1usize, 3, 6] {
            let n = 1usize << v;
            let bits: Vec<bool> = (0..n)
                .map(|_| rand::RngExt::random::<bool>(&mut rng))
                .collect();
            let mut m = 0u64;
            for t in 0..n {
                if bits[t] {
                    m |= 1u64 << t;
                }
            }
            for j in 0..v {
                m ^= (m << (1 << j)) & swirl_mask(j);
            }
            for t in 0..n {
                let mut want = false;
                for s in 0..=t {
                    if t & s == s && bits[s] {
                        want = !want;
                    }
                }
                assert_eq!((m >> t) & 1, want as u64, "v={v} t={t}");
            }
        }
    }
}

#[cfg(test)]
mod fold_debug_tests {
    use super::tests::swirl_mask;
    use magnetar_field::ZERO;

    use super::*;

    /// Brute-force the per-item window monomial state `q[c][m] =
    /// Σ_{x_suf} eq(pt_suf, x)·Ĝ_m(x)` and compare against the
    /// LocalDischarge fold path on a small random instance.
    #[test]
    fn fold_q_matches_bruteforce() {
        use crate::mosaic::{ComponentSpec, MosaicLayout};
        let mut rng = rand::rng();
        let per = MosaicLayout::build(vec![
            ComponentSpec {
                name: "big".into(),
                vars: 12,
            },
            ComponentSpec {
                name: "small".into(),
                vars: 10,
            },
        ]);
        let ell = per.ell;
        let v = 6.min(ell - 7).max(1).min(ell - 1);
        let plane_hi = ell - v;
        let word_bits = ell - 7;
        // One packed "instance": random bits over the full cube.
        let n_bits = 1usize << ell;
        let bits: Vec<bool> = (0..n_bits)
            .map(|_| rand::RngExt::random::<bool>(&mut rng))
            .collect();
        let mut packed = vec![Gf128(0); 1usize << word_bits];
        for x in 0..n_bits {
            if bits[x] {
                let w = x & ((1usize << word_bits) - 1);
                let p = x >> word_bits;
                packed[w].0 |= 1u128 << p;
            }
        }
        // One claim on "big": a fully random point.
        let point: Vec<Gf128> = (0..12)
            .map(|_| Gf128(rand::RngExt::random::<u128>(&mut rng)))
            .collect();
        let claim = crate::mosaic::MosaicClaim {
            component: 0,
            point: point.clone(),
            value: ZERO,
        };
        let alphas = vec![Gf128(3)];
        let inst = instance_items(&per, 1, &[claim], &alphas);
        let (_, items) = &inst[0];
        let claim_val = ZERO;
        let ld = LocalDischarge::new(&per, &packed, 0, items, claim_val);
        let pt = &items[0].1;
        // ---- brute force ----
        let suffix_eq = EqBits::build(&pt[..plane_hi]).values;
        let stride = 1usize << plane_hi;
        let plane_lo_bits = 7 - v;
        let mut brute = vec![ZERO; 1usize << v];
        for x in 0..stride {
            let e = suffix_eq[x];
            if e == ZERO {
                continue;
            }
            let u = x & ((1usize << word_bits) - 1);
            let b_lo = x >> word_bits;
            let word = packed[u].0;
            let mut mask = 0u64;
            for m in 0..(1usize << v) {
                if (word >> (b_lo + m * (1usize << plane_lo_bits))) & 1 == 1 {
                    mask |= 1u64 << m;
                }
            }
            for j in 0..v {
                mask ^= (mask << (1 << j)) & swirl_mask(j);
            }
            let mut s = mask;
            while s != 0 {
                let m = s.trailing_zeros() as usize;
                s &= s - 1;
                brute[m] += e;
            }
        }
        for m in 0..(1usize << v) {
            assert_eq!(
                ld.q[0][m], brute[m],
                "monomial {m}: fold {:?} vs brute {:?} (ell={ell} v={v})",
                ld.q[0][m], brute[m]
            );
        }
    }
}
