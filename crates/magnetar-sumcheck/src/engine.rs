//! The sum-check prover and verifier (paper §3.2).
//!
//! Round structure (characteristic two):
//! 1. Prover computes the round polynomial `m_i(X)` (degree ≤ `d`) as
//!    coefficients and sends them.
//! 2. Verifier checks `Σ_{j≥1} m_{i,j} = running` (the constant coefficient
//!    cancels under `m(0) + m(1)`), then samples `r_i ← K`.
//! 3. Both fold every factor table at `r_i`.
//!
//! After `m` rounds the verifier holds the terminal identity
//! `Σ_t coeff_t · Π public_j(r*) · Π dense_j(r*) = running_final`, where the
//! dense evaluations are the prover's claims — these become the output
//! evaluation claims of the reduction (paper Definition 5).

use crate::{BindOrder, EqSlabInfo, RoundMsg, SumcheckSpec};
use magnetar_field::{
    inv, mul, mul_bit, par_bind_var, par_bind_var_bit, par_gather_bit_at, Gf128, ONE, ZERO,
};
use magnetar_hash::Transcript;
use rayon::prelude::*;

/// Minimum remaining-cube size before the round message is computed with
/// chunked parallelism (R2). Field arithmetic reduces order-independently
/// (R1), so the parallel message is bit-identical to the sequential one.
const PAR_MSG_THRESHOLD: usize = 1 << 13;

/// Per-public-factor slab classification for every joint variable,
/// precomputed at construction (the factors are dropped after
/// materialization; the classification is all the fold needs).
type SlabProfile = Vec<EqSlabInfo>;

/// Prover-side state: the folding tables plus the running claim.
pub struct SumcheckProver {
    spec_terms: Vec<crate::Term>,
    dense: Vec<Vec<Gf128>>,
    public_tables: Vec<Vec<Gf128>>,
    /// Which dense tables are currently **bit-valued** (`{0, 1}` — the
    /// small-value setting): their pair products are AND/XOR and their
    /// folds are selects (no CLMUL). Cleared by the first fold.
    dense_bits: Vec<bool>,
    /// The eq-slab profile per public factor (`None` = generic fold).
    slab_profile: Vec<Option<SlabProfile>>,
    /// The **small-value round window** (the sum-check papers' round
    /// batching): while active (rounds `0..v`), the messages come from the
    /// monomial state and the tables stay un-folded. See `window.rs`.
    window: Option<crate::window::WindowState>,
    /// The **packed window masks** (the mask-major window mode): when
    /// present, the three bit tables live as one `u64` window
    /// truth-table word per suffix, and the window's fast-forward
    /// contracts these instead of `dense`.
    dense_masks: Option<Vec<Vec<u64>>>,
    coeffs: Vec<Gf128>,
    running: Gf128,
    order: BindOrder,
    num_vars: usize,
    round: usize,
    challenges: Vec<Gf128>,
}

/// Per-term, per-round message-path classification (the small-value fast
/// paths plus the fused generic kernels).
#[derive(Clone, Copy, PartialEq)]
enum Path {
    Generic,
    Bit1,
    BitDensePub,
    Bit2Dense,
    Bit2DensePub,
}

impl SumcheckProver {
    pub fn new(spec: &SumcheckSpec, order: BindOrder) -> Self {
        Self::with_dense_bits(spec, order, vec![false; spec.dense.len()])
    }

    /// Construct with explicit **bit-valued dense-table flags** (the
    /// small-value fast paths of the sum-check papers: a bit-valued factor
    /// makes the round-message products AND/XOR selects and the fold a
    /// branchless select). The caller guarantees the flagged tables only
    /// contain `0` or `1` (e.g. a binary-matrix application over a bit
    /// witness); the engine clears a flag as soon as its table is folded
    /// with a challenge (the result is field-valued).
    pub fn with_dense_bits(spec: &SumcheckSpec, order: BindOrder, bits: Vec<bool>) -> Self {
        Self::with_dense_bits_opt_window(spec, order, bits, false)
    }

    /// Construct with the **small-value round window** enabled when the
    /// spec matches the windowable shape (the binary relation's outer
    /// sum-check `eq·(D0·D1 + D2)` over bit tables, bound
    /// high-bits-first): the first `min(v, high)` rounds' messages are
    /// computed in one pass over the un-folded tables (587 §5.1/C.5).
    /// Messages stay bit-identical to the generic engine's; non-matching
    /// specs fall back to the plain engine.
    pub fn with_dense_bits_auto_window(
        spec: &SumcheckSpec,
        order: BindOrder,
        bits: Vec<bool>,
    ) -> Self {
        Self::with_dense_bits_opt_window(spec, order, bits, true)
    }

    /// Construct in the **mask-major window mode**: the three bit tables
    /// arrive packed (one `u64` window truth-table word per suffix, the
    /// `apply_simd_bit_masks` layout) instead of as `2^ell` field-entry
    /// dense tables. The spec's `dense` vectors may be empty; the window
    /// state builds from the masks, and the fast-forward contracts the
    /// masks into the post-window dense tables. Requires the windowable
    /// shape (`eq·(D0·D1 + D2)`, high-bits-first) — panics otherwise.
    pub fn with_mask_window(
        spec: &SumcheckSpec,
        order: BindOrder,
        masks: Vec<Vec<u64>>,
    ) -> Self {
        spec.validate().expect("invalid sumcheck spec");
        assert_eq!(masks.len(), 3, "the mask window needs exactly 3 tables");
        assert!(
            matches!(order, BindOrder::HighBitsFirst { .. }),
            "the mask window binds high-bits-first"
        );
        let bits = vec![true; 3];
        let public_tables: Vec<Vec<Gf128>> = spec.public.iter().map(|f| f.materialize()).collect();
        let slab_profile: Vec<Option<SlabProfile>> = spec
            .public
            .iter()
            .map(|f| {
                let first = f.eq_slab_info(0);
                match first {
                    EqSlabInfo::NotSlab => None,
                    _ => Some((0..f.num_vars()).map(|v| f.eq_slab_info(v)).collect()),
                }
            })
            .collect();
        let window = Some(
            Self::try_window_masks(spec, order, &masks, &public_tables)
                .expect("the mask window requires the windowable shape"),
        );
        SumcheckProver {
            spec_terms: spec.terms.clone(),
            dense: vec![Vec::new(); 3],
            dense_masks: Some(masks),
            public_tables,
            dense_bits: bits,
            slab_profile,
            window,
            coeffs: spec.terms.iter().map(|t| t.coeff).collect(),
            running: spec.claim,
            order,
            num_vars: spec.num_vars,
            round: 0,
            challenges: Vec::new(),
        }
    }

    /// The mask-mode window build: identical shape checks to
    /// [`Self::try_window`], then the packed pass.
    fn try_window_masks(
        spec: &SumcheckSpec,
        order: BindOrder,
        masks: &[Vec<u64>],
        public_tables: &[Vec<Gf128>],
    ) -> Option<crate::window::WindowState> {
        let high = match order {
            BindOrder::HighBitsFirst { high } => high,
            _ => return None,
        };
        let v = crate::window::window_size(high);
        if v == 0 || spec.num_vars < 2 {
            return None;
        }
        let shape = spec.dense.is_empty()
            && spec.terms.len() == 2
            && spec.terms[0].dense == vec![0, 1]
            && spec.terms[0].public == vec![0]
            && spec.terms[1].dense == vec![2]
            && spec.terms[1].public == vec![0]
            && spec.public.len() == 1
            && spec.public[0].num_vars() == spec.num_vars;
        if !shape {
            return None;
        }
        let tau = spec.public[0].eq_point()?;
        debug_assert_eq!(tau.len(), spec.num_vars);
        let mut w = crate::window::WindowState::new(spec.num_vars, v, &tau);
        w.run_pass_masks(&masks[0], &masks[1], &masks[2], &public_tables[0], &tau);
        Some(w)
    }

    fn with_dense_bits_opt_window(
        spec: &SumcheckSpec,
        order: BindOrder,
        bits: Vec<bool>,
        auto_window: bool,
    ) -> Self {
        spec.validate().expect("invalid sumcheck spec");
        assert_eq!(bits.len(), spec.dense.len(), "dense bit flags arity");
        let public_tables: Vec<Vec<Gf128>> = spec.public.iter().map(|f| f.materialize()).collect();
        let slab_profile: Vec<Option<SlabProfile>> = spec
            .public
            .iter()
            .map(|f| {
                let first = f.eq_slab_info(0);
                match first {
                    EqSlabInfo::NotSlab => None,
                    _ => Some((0..f.num_vars()).map(|v| f.eq_slab_info(v)).collect()),
                }
            })
            .collect();
        // The window: shape-checked, then the one-pass state build.
        let window = if auto_window {
            Self::try_window(spec, order, &bits, &public_tables)
        } else {
            None
        };
        SumcheckProver {
            spec_terms: spec.terms.clone(),
            dense: spec.dense.clone(),
            dense_masks: None,
            public_tables,
            dense_bits: bits,
            slab_profile,
            window,
            coeffs: spec.terms.iter().map(|t| t.coeff).collect(),
            running: spec.claim,
            order,
            num_vars: spec.num_vars,
            round: 0,
            challenges: Vec::new(),
        }
    }

    /// The windowable shape: `eq(τ,·) · (D0·D1 + D2)` with all three
    /// dense tables bit-flagged, bound high-bits-first. The window covers
    /// the first `min(v, high)` rounds (the top joint variables — the
    /// strided sub-cube structure).
    fn try_window(
        spec: &SumcheckSpec,
        order: BindOrder,
        bits: &[bool],
        public_tables: &[Vec<Gf128>],
    ) -> Option<crate::window::WindowState> {
        let high = match order {
            BindOrder::HighBitsFirst { high } => high,
            _ => return None,
        };
        let v = crate::window::window_size(high);
        if v == 0 || spec.num_vars < 2 {
            return None;
        }
        let shape = spec.dense.len() == 3
            && bits.len() == 3
            && bits.iter().all(|&b| b)
            && spec.terms.len() == 2
            && spec.terms[0].dense == vec![0, 1]
            && spec.terms[0].public == vec![0]
            && spec.terms[1].dense == vec![2]
            && spec.terms[1].public == vec![0]
            && spec.public.len() == 1
            && spec.public[0].num_vars() == spec.num_vars;
        if !shape {
            return None;
        }
        let tau = spec.public[0].eq_point()?;
        debug_assert_eq!(tau.len(), spec.num_vars);
        let mut w = crate::window::WindowState::new(spec.num_vars, v, &tau);
        w.run_pass(
            &spec.dense[0],
            &spec.dense[1],
            &spec.dense[2],
            &public_tables[0],
            &tau,
        );
        Some(w)
    }

    /// Number of remaining variables.
    pub fn remaining_vars(&self) -> usize {
        self.num_vars - self.round
    }

    /// Current round's message: coefficients of the round polynomial,
    /// computed by the **fused single-pass** kernel (the Horizon-1
    /// sum-check-arithmetic item): every factor's pair structure
    /// `(a, a+t1)` at the bind bit multiplies directly into per-term
    /// polynomial accumulators — one traversal per round, no per-point
    /// view materialization, no allocations (the previous kernel built
    /// `degree+1` half-table views per factor per round).
    ///
    /// **Small-value fast paths** (the sum-check-proving papers' bit-valued
    /// setting): a dense table flagged `dense_bits` has `{0, 1}` entries,
    /// so its pair products are AND/XOR and every multiply against a
    /// field-valued operand degenerates to a branchless select — no
    /// CLMUL at all. The paths are classified per term per round; a fold
    /// clears the flags (field-valued results), after which the generic
    /// kernel takes over. Bit-identical messages to the generic engine.
    pub fn message(&self) -> RoundMsg {
        if let Some(w) = &self.window {
            return w.message(&self.coeffs);
        }
        let degree = self
            .spec_terms
            .iter()
            .map(|t| t.dense.len() + t.public.len())
            .max()
            .unwrap_or(0);
        let bit = self.order.current_bit(self.num_vars, self.round);
        let block = 1usize << bit;
        let n = 1usize << (self.num_vars - self.round);
        let n_groups = n / (2 * block);
        debug_assert!(n_groups >= 1);
        let paths: Vec<Path> = self
            .spec_terms
            .iter()
            .map(|t| {
                let d = t.dense.len();
                let p = t.public.len();
                let bit_at = |k: usize| self.dense_bits[t.dense[k]];
                match (d, p) {
                    (1, 0) if bit_at(0) => Path::Bit1,
                    (1, 1) if bit_at(0) => Path::BitDensePub,
                    (2, 0) if bit_at(0) && bit_at(1) => Path::Bit2Dense,
                    (2, 1) if bit_at(0) && bit_at(1) => Path::Bit2DensePub,
                    _ => Path::Generic,
                }
            })
            .collect();
        let coeffs = {
            // Per-term accumulators over the coefficient vector (R1: the
            // per-chunk sums combine order-free by `+`).
            let terms = &self.spec_terms;
            let coeffs_of = &self.coeffs;
            let chunk_groups = if n_groups >= 2 && n >= PAR_MSG_THRESHOLD {
                (n_groups / (rayon::current_num_threads().max(1) * 4)).max(1)
            } else {
                n_groups.max(1)
            };
            (0..n_groups)
                .into_par_iter()
                .chunks(chunk_groups)
                .map(|groups| {
                    let mut local: Vec<Vec<Gf128>> =
                        terms.iter().map(|_| vec![ZERO; degree + 1]).collect();
                    for g in groups {
                        let gbase = g * 2 * block;
                        let n4 = block / 4;
                        // Term-outer: the fused VPCLMULQDQ kernels keep
                        // their accumulators in SIMD registers across the
                        // whole group; bit paths are scalar selects; the
                        // <= 3-position tail runs the per-position forms.
                        for (ti, t) in terms.iter().enumerate() {
                            let nf = t.dense.len() + t.public.len();
                            let c0 = coeffs_of[ti];
                            let acc = &mut local[ti];
                            match paths[ti] {
                                Path::Generic if nf == 2 => {
                                    let f1 = self.factor_table(t, 0);
                                    let f2 = self.factor_table(t, 1);
                                    let done = magnetar_field::avx::msg2_batch(
                                        acc,
                                        c0,
                                        f1,
                                        f2,
                                        gbase,
                                        block,
                                        4 * n4,
                                    );
                                    for o in done..4 * n4 {
                                        self.generic2_one(acc, c0, t, gbase + o, block);
                                    }
                                }
                                Path::Generic if nf == 3 => {
                                    let f1 = self.factor_table(t, 0);
                                    let f2 = self.factor_table(t, 1);
                                    let f3 = self.factor_table(t, 2);
                                    let done = magnetar_field::avx::msg3_batch(
                                        acc,
                                        c0,
                                        f1,
                                        f2,
                                        f3,
                                        gbase,
                                        block,
                                        4 * n4,
                                    );
                                    for o in done..4 * n4 {
                                        self.generic3_one(acc, c0, t, gbase + o, block);
                                    }
                                }
                                _ => {
                                    // Bit paths (scalar selects, cheap) and
                                    // the generic fallbacks: per position.
                                    for o in 0..4 * n4 {
                                        let base = gbase + o;
                                        self.position_contribution(
                                            acc, c0, t, base, block, degree, coeffs_of, ti, nf,
                                            &paths[ti],
                                        );
                                    }
                                }
                            }
                            // Scalar tail (block % 4 positions).
                            for o in 4 * n4..block {
                                let base = gbase + o;
                                self.position_contribution(
                                    acc, c0, t, base, block, degree, coeffs_of, ti, nf, &paths[ti],
                                );
                            }
                        }
                    }
                    local
                })
                .reduce(
                    || terms.iter().map(|_| vec![ZERO; degree + 1]).collect(),
                    |mut a, b| {
                        for (ta, tb) in a.iter_mut().zip(b.iter()) {
                            for (x, y) in ta.iter_mut().zip(tb.iter()) {
                                *x += *y;
                            }
                        }
                        a
                    },
                )
        };
        // Sum the per-term coefficient vectors into one message.
        let mut msg = vec![ZERO; degree + 1];
        for (e, slot) in msg.iter_mut().enumerate() {
            for c in &coeffs {
                *slot += c[e];
            }
        }
        RoundMsg(msg)
    }

    /// The `(a, d)` pair of term `t`'s factor `k` at position `base`
    /// (the MLE's value at bit 0 and its X coefficient).
    #[inline]
    fn pair_of_idx(&self, t: &crate::Term, k: usize, base: usize, block: usize) -> (Gf128, Gf128) {
        if k < t.dense.len() {
            let f = &self.dense[t.dense[k]];
            (f[base], f[base] + f[base + block])
        } else {
            let f = &self.public_tables[t.public[k - t.dense.len()]];
            (f[base], f[base] + f[base + block])
        }
    }

    /// The table of term `t`'s factor `k` (dense or public).
    #[inline]
    fn factor_table(&self, t: &crate::Term, k: usize) -> &Vec<Gf128> {
        if k < t.dense.len() {
            &self.dense[t.dense[k]]
        } else {
            &self.public_tables[t.public[k - t.dense.len()]]
        }
    }

    /// One position's message contribution, dispatching on the classified
    /// path (bit selects or the generic expansions). Bit-identical across
    /// paths; order-free accumulation.
    #[allow(clippy::too_many_arguments)]
    fn position_contribution(
        &self,
        acc: &mut [Gf128],
        c0: Gf128,
        t: &crate::Term,
        base: usize,
        block: usize,
        degree: usize,
        coeffs_of: &[Gf128],
        ti: usize,
        nf: usize,
        path: &Path,
    ) {
        match path {
            Path::Bit1 => {
                let f = &self.dense[t.dense[0]];
                let a = f[base];
                let d = Gf128(f[base].0 ^ f[base + block].0);
                acc[0] += mul_bit(c0, a);
                acc[1] += mul_bit(c0, d);
            }
            Path::BitDensePub => {
                let f = &self.dense[t.dense[0]];
                let a = f[base];
                let d = Gf128(f[base].0 ^ f[base + block].0);
                let g = &self.public_tables[t.public[0]];
                let e0 = g[base];
                let e1 = Gf128(g[base].0 ^ g[base + block].0);
                let (e0, e1) = if c0 == ONE {
                    (e0, e1)
                } else {
                    (mul(c0, e0), mul(c0, e1))
                };
                acc[0] += mul_bit(e0, a);
                acc[1] += mul_bit(e1, a) + mul_bit(e0, d);
                acc[2] += mul_bit(e1, d);
            }
            Path::Bit2Dense => {
                let f1 = &self.dense[t.dense[0]];
                let f2 = &self.dense[t.dense[1]];
                let a1 = f1[base];
                let d1 = Gf128(f1[base].0 ^ f1[base + block].0);
                let a2 = f2[base];
                let d2 = Gf128(f2[base].0 ^ f2[base + block].0);
                // Products of bits are bits: AND/XOR.
                let p0 = Gf128(a1.0 & a2.0);
                let p1 = Gf128((a1.0 & d2.0) ^ (d1.0 & a2.0));
                let p3 = Gf128(d1.0 & d2.0);
                acc[0] += mul_bit(c0, p0);
                acc[1] += mul_bit(c0, p1);
                acc[2] += mul_bit(c0, p3);
            }
            Path::Bit2DensePub => {
                let f1 = &self.dense[t.dense[0]];
                let f2 = &self.dense[t.dense[1]];
                let a1 = f1[base];
                let d1 = Gf128(f1[base].0 ^ f1[base + block].0);
                let a2 = f2[base];
                let d2 = Gf128(f2[base].0 ^ f2[base + block].0);
                let p0 = Gf128(a1.0 & a2.0);
                let p1 = Gf128((a1.0 & d2.0) ^ (d1.0 & a2.0));
                let p3 = Gf128(d1.0 & d2.0);
                let g = &self.public_tables[t.public[0]];
                let e0 = g[base];
                let e1 = Gf128(g[base].0 ^ g[base + block].0);
                // (p0 + X·p1 + X²·p3)·(e0 + X·e1), all bit×field selects.
                let (e0, e1) = if c0 == ONE {
                    (e0, e1)
                } else {
                    (mul(c0, e0), mul(c0, e1))
                };
                acc[0] += mul_bit(e0, p0);
                acc[1] += mul_bit(e1, p0) + mul_bit(e0, p1);
                acc[2] += mul_bit(e1, p1) + mul_bit(e0, p3);
                acc[3] += mul_bit(e1, p3);
            }
            Path::Generic => {
                if nf == 1 {
                    let (a, d) = self.pair_of_idx(t, 0, base, block);
                    acc[0] += mul(c0, a);
                    acc[1] += mul(c0, d);
                } else if nf == 2 {
                    self.generic2_one(acc, c0, t, base, block);
                } else if nf == 3 {
                    self.generic3_one(acc, c0, t, base, block);
                } else {
                    self.generic_n(acc, c0, t, base, block, degree, coeffs_of, ti, nf);
                }
            }
        }
    }

    /// One position's 2-factor generic expansion.
    fn generic2_one(
        &self,
        acc: &mut [Gf128],
        c0: Gf128,
        t: &crate::Term,
        base: usize,
        block: usize,
    ) {
        let (a1, d1) = self.pair_of_idx(t, 0, base, block);
        let (a2, d2) = self.pair_of_idx(t, 1, base, block);
        let ca = mul(c0, a1);
        let cd1 = mul(c0, d1);
        acc[0] += mul(ca, a2);
        acc[1] += mul(ca, d2) + mul(cd1, a2);
        acc[2] += mul(cd1, d2);
    }

    /// One position's 3-factor generic expansion.
    fn generic3_one(
        &self,
        acc: &mut [Gf128],
        c0: Gf128,
        t: &crate::Term,
        base: usize,
        block: usize,
    ) {
        let (a1, d1) = self.pair_of_idx(t, 0, base, block);
        let (a2, d2) = self.pair_of_idx(t, 1, base, block);
        let (a3, d3) = self.pair_of_idx(t, 2, base, block);
        // (a1 + X·d1)(a2 + X·d2)(a3 + X·d3)
        let a1a2 = mul(a1, a2);
        let a1d2 = mul(a1, d2);
        let d1a2 = mul(d1, a2);
        let d1d2 = mul(d1, d2);
        let s1 = a1d2 + d1a2;
        if c0 == ONE {
            acc[0] += mul(a1a2, a3);
            acc[1] += mul(a1a2, d3) + mul(s1, a3);
            acc[2] += mul(s1, d3) + mul(d1d2, a3);
            acc[3] += mul(d1d2, d3);
        } else {
            acc[0] += mul(c0, mul(a1a2, a3));
            acc[1] += mul(c0, mul(a1a2, d3) + mul(s1, a3));
            acc[2] += mul(c0, mul(s1, d3) + mul(d1d2, a3));
            acc[3] += mul(c0, mul(d1d2, d3));
        }
    }

    /// The generic high-degree fallback (one position).
    #[allow(clippy::too_many_arguments)]
    fn generic_n(
        &self,
        acc: &mut [Gf128],
        c0: Gf128,
        t: &crate::Term,
        base: usize,
        block: usize,
        degree: usize,
        coeffs_of: &[Gf128],
        ti: usize,
        nf: usize,
    ) {
        let mut poly = [ZERO; 17];
        poly[0] = coeffs_of[ti];
        let mut deg = 0usize;
        for k in 0..nf {
            let (a, dd) = self.pair_of_idx(t, k, base, block);
            let mut next = [ZERO; 17];
            for e in 0..=deg {
                next[e] += mul(poly[e], a);
                next[e + 1] += mul(poly[e], dd);
            }
            poly = next;
            deg += 1;
        }
        for e in 0..=deg.min(degree) {
            acc[e] += poly[e];
        }
        let _ = c0;
    }

    /// Absorb the message into the transcript, check nothing (prover side),
    /// sample the challenge, fold, and advance. Returns the challenge.
    pub fn absorb_and_fold(&mut self, t: &mut Transcript, label: &str) -> Gf128 {
        let msg = self.message();
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        let r = t.challenge_field(label);
        // Reuse the already-computed message (fold_msg) — the old path
        // recomputed the message inside fold().
        self.fold_msg(&msg, r);
        r
    }

    /// Fold all tables at `r` (the round variable) and update the running
    /// claim to `m(r)`. Used by the lockstep driver with *external*
    /// challenges (Virtual Spartan, paper §5.3 step 1).
    pub fn fold(&mut self, r: Gf128) {
        let msg = self.message();
        self.fold_msg(&msg, r);
    }

    /// Fold at `r` reusing an already-computed round message (the ZK driver
    /// masks the message it transmits but folds the *unmasked* polynomial —
    /// the mask never enters the prover's tables, only the transcript).
    ///
    /// Two prover-internal optimizations fire here (both leave the messages
    /// bit-identical to the generic engine's):
    /// * **Bit folds**: a `dense_bits`-flagged table folds with a
    ///   branchless select (`r·(lo+hi)` with `lo+hi ∈ {0,1}`) — no CLMUL.
    /// * **Eq slabs**: a slab-able public factor is never folded. Its
    ///   table is *gathered* (the bit-`s` slice, no multiplications) and
    ///   the bound variable's constant `eq(τ_b, r)/eq(τ_b, s)` moves into
    ///   the term coefficients — deleting the eq factor's per-round fold
    ///   multiplications entirely (the decomposable-equality discipline).
    pub fn fold_msg(&mut self, msg: &RoundMsg, r: Gf128) {
        self.challenges.push(r);
        // running := m(r)
        let mut mr = ZERO;
        let mut xp = Gf128(1);
        for &c in &msg.0 {
            mr += mul(c, xp);
            xp = mul(xp, r);
        }
        self.running = mr;
        // The round window path: bind the monomial state; the tables stay
        // un-folded until the window ends (then the one-shot
        // fast-forward).
        if self.window.is_some() {
            let v = self.window.as_ref().expect("checked").v;
            {
                let w = self.window.as_mut().expect("checked");
                w.bind(r);
            }
            if self.window.as_ref().expect("checked").round == v {
                self.finish_window();
            }
            self.round += 1;
            return;
        }
        let bit = self.order.current_bit(self.num_vars, self.round);
        // The slab profile is indexed by ORIGINAL variable; the round binds
        // `round_var` (the layout bit `bit` corresponds to it).
        let orig_var = self.order.round_var(self.num_vars, self.round);
        // Public factors: generic fold, or the slab gather + coefficient
        // scale. The per-term coefficient deltas accumulate first so each
        // term scales once per referenced slab factor.
        let mut coeff_scale: Vec<Gf128> = vec![ONE; self.spec_terms.len()];
        for (pi, prof) in self.slab_profile.iter().enumerate() {
            match prof.as_ref().map(|p| p[orig_var].clone()) {
                None => {
                    par_bind_var(&mut self.public_tables[pi], bit, r);
                }
                Some(crate::EqSlabInfo::NotSlab) => {
                    par_bind_var(&mut self.public_tables[pi], bit, r);
                }
                Some(crate::EqSlabInfo::Covered(tau_v)) => {
                    // Choose the slice s with eq(τ_v, s) ≠ 0: s = 1 when
                    // τ_v = ONE (eq(1,1) = 1); otherwise s = 0, where
                    // eq(τ_v, 0) = 1 + τ_v ≠ 0 exactly because τ_v ≠ 1.
                    // (The previous `τ_v != ONE` choice panicked on
                    // bit-valued τ — a latent edge case never hit by
                    // random challenges, exposed by the window's
                    // cross-check tests.)
                    let slice = tau_v == ONE;
                    par_gather_bit_at(&mut self.public_tables[pi], bit, slice);
                    // delta = eq(τ_v, r) / eq(τ_v, s).
                    let eq_r = ONE + tau_v + r; // eq(τ_v, r) in char 2
                    let eq_s = if slice { tau_v } else { ONE + tau_v };
                    let delta = mul(eq_r, inv(eq_s));
                    for (ti, t) in self.spec_terms.iter().enumerate() {
                        if t.public.contains(&pi) {
                            coeff_scale[ti] = mul(coeff_scale[ti], delta);
                        }
                    }
                }
                Some(crate::EqSlabInfo::Uncovered) => {
                    // Constant in this variable: any slice equals the true
                    // remaining table; no scale, no coefficient change.
                    par_gather_bit_at(&mut self.public_tables[pi], bit, false);
                }
            }
        }
        for (ti, sc) in coeff_scale.into_iter().enumerate() {
            if sc != ONE {
                self.coeffs[ti] = mul(self.coeffs[ti], sc);
            }
        }
        for (i, is_bit) in self.dense_bits.iter_mut().enumerate() {
            if *is_bit {
                par_bind_var_bit(&mut self.dense[i], bit, r);
                *is_bit = false; // field-valued from here on
            } else {
                par_bind_var(&mut self.dense[i], bit, r);
            }
        }
        self.round += 1;
    }

    /// The window's fast-forward (587 §C.5.6): reach, in one shot, the
    /// table state `v` generic rounds would have produced — the dense
    /// tables contracted at the window challenges with selects (the
    /// un-folded entries are bits), the eq factor's slab gathers applied
    /// with the combined coefficient delta.
    fn finish_window(&mut self) {
        let w = self.window.take().expect("window active");
        let eq_r: Vec<Gf128> = w.eq_r().to_vec();
        // 1. The dense tables: bound[x'] = the eq(r_<v)-weighted window
        //    contraction — a select per (m, x') since the entries are
        //    bits. In the mask-major mode the packed masks contract
        //    instead (one word per suffix).
        if let Some(masks) = self.dense_masks.take() {
            debug_assert_eq!(self.dense.len(), 3);
            for k in 0..3 {
                self.dense[k] = crate::window::contract_bit_masks(&masks[k], &eq_r, w.stride, w.v);
                self.dense_bits[k] = false;
            }
        } else {
            for k in 0..self.dense.len() {
                let old = std::mem::take(&mut self.dense[k]);
                self.dense[k] = crate::window::contract_bit_table(&old, &eq_r, w.stride, w.v);
                self.dense_bits[k] = false;
            }
        }
        // 2. The eq factor: v slab gathers + the combined delta
        //    (exactly what v rounds of fold_msg's slab path do).
        let mut delta_total = ONE;
        for i in 0..w.v {
            let orig_var = self.order.round_var(self.num_vars, i);
            let bit = self.order.current_bit(self.num_vars, i);
            let r_i = self.challenges[i];
            match self.slab_profile[0].as_ref().map(|p| p[orig_var].clone()) {
                Some(EqSlabInfo::Covered(tau_v)) => {
                    let slice = tau_v == ONE; // eq(τ, s) ≠ 0 at s = 1 iff τ = 1
                    par_gather_bit_at(&mut self.public_tables[0], bit, slice);
                    let eq_r_v = ONE + tau_v + r_i;
                    let eq_s = if slice { tau_v } else { ONE + tau_v };
                    delta_total = mul(delta_total, mul(eq_r_v, inv(eq_s)));
                }
                _ => {
                    par_gather_bit_at(&mut self.public_tables[0], bit, false);
                }
            }
        }
        if delta_total != ONE {
            for (ti, t) in self.spec_terms.iter().enumerate() {
                if t.public.contains(&0) {
                    self.coeffs[ti] = mul(self.coeffs[ti], delta_total);
                }
            }
        }
    }

    /// After all rounds: the dense factors' claimed evaluations at the
    /// challenge point (their folded single entries).
    pub fn finish(self) -> (Vec<Gf128>, Gf128) {
        debug_assert_eq!(self.round, self.num_vars);
        let claims = self.dense.iter().map(|d| d[0]).collect();
        (claims, self.running)
    }

    /// Snapshot of the dense tables (for compositional protocols that read
    /// intermediate folded tables — e.g. MosaicPCS's c*-folded witness).
    pub fn dense_tables(&self) -> &[Vec<Gf128>] {
        &self.dense
    }

    /// Current running claim.
    pub fn running(&self) -> Gf128 {
        self.running
    }

    /// The challenges sampled so far (in binding order) — for protocols
    /// that need the point during the run (e.g. address virtualization).
    pub fn challenges(&self) -> Vec<Gf128> {
        self.challenges.clone()
    }
}

/// Verifier-side driver: replays the round checks and computes the final
/// identity check. Returns `(point_in_original_coords, dense_factor_claims)`
/// on success — each claim is one value per dense factor at the point.
pub struct SumcheckVerifier<'a> {
    spec_public: &'a [Box<dyn crate::public::PublicFactor + Send + Sync>],
    terms: Vec<crate::Term>,
    num_vars: usize,
    order: BindOrder,
    running: Gf128,
    challenges: Vec<Gf128>,
}

impl<'a> SumcheckVerifier<'a> {
    pub fn new(
        num_vars: usize,
        claim: Gf128,
        public: &'a [Box<dyn crate::public::PublicFactor + Send + Sync>],
        terms: Vec<crate::Term>,
        order: BindOrder,
    ) -> Self {
        SumcheckVerifier {
            spec_public: public,
            terms,
            num_vars,
            order,
            running: claim,
            challenges: Vec::new(),
        }
    }

    /// Process one round: check the round identity, sample the challenge.
    pub fn round(
        &mut self,
        msg: &RoundMsg,
        t: &mut Transcript,
        label: &str,
    ) -> Result<(), &'static str> {
        self.round_capturing(msg, t, label).map(|_| ())
    }

    /// Process one round and return the sampled challenge (for protocols
    /// that thread sum-check challenges into later arguments, e.g. the PCS
    /// fold chain).
    pub fn round_capturing(
        &mut self,
        msg: &RoundMsg,
        t: &mut Transcript,
        label: &str,
    ) -> Result<Gf128, &'static str> {
        let expected_degree = self
            .terms
            .iter()
            .map(|trm| trm.dense.len() + trm.public.len())
            .max()
            .unwrap_or(0);
        if msg.0.len() != expected_degree + 1 {
            return Err("round message degree mismatch");
        }
        // Characteristic-two check: Σ_{j≥1} c_j = running.
        let mut s = ZERO;
        for &c in &msg.0[1..] {
            s += c;
        }
        if s != self.running {
            return Err("round sum check failed");
        }
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        let r = t.challenge_field(label);
        // Fold the running claim: m(r).
        let mut mr = ZERO;
        let mut xp = Gf128(1);
        for &c in &msg.0 {
            mr += mul(c, xp);
            xp = mul(xp, r);
        }
        self.running = mr;
        self.challenges.push(r);
        Ok(r)
    }

    /// The challenges sampled so far (in binding order).
    pub fn challenges(&self) -> &[Gf128] {
        &self.challenges
    }

    /// Process one round with an **externally supplied** challenge (for
    /// lockstep protocols where the driver manages the transcript: paper
    /// §5.3 step 1). Checks the degree and the round sum, folds the running
    /// claim, and records the challenge — but does not touch the
    /// transcript.
    pub fn round_external(&mut self, msg: &RoundMsg, r: Gf128) -> Result<(), &'static str> {
        let expected_degree = self
            .terms
            .iter()
            .map(|trm| trm.dense.len() + trm.public.len())
            .max()
            .unwrap_or(0);
        if msg.0.len() != expected_degree + 1 {
            return Err("round message degree mismatch");
        }
        let mut s = ZERO;
        for &c in &msg.0[1..] {
            s += c;
        }
        if s != self.running {
            return Err("round sum check failed");
        }
        let mut mr = ZERO;
        let mut xp = Gf128(1);
        for &c in &msg.0 {
            mr += mul(c, xp);
            xp = mul(xp, r);
        }
        self.running = mr;
        self.challenges.push(r);
        Ok(())
    }

    /// After all rounds: check the terminal identity with the prover's
    /// dense-evaluation claims, returning the point (original coordinates)
    /// and the accepted claims.
    pub fn finish(&self, dense_claims: &[Gf128]) -> Result<(Vec<Gf128>, Vec<Gf128>), &'static str> {
        self.finish_inner(dense_claims, ZERO)
    }

    /// The ZK terminal check: `Σ terms·claims + offset = running`, where
    /// `offset` is the prover-revealed residual of the per-round mask
    /// polynomials (uniform on its own, hiding the honest terminal).
    pub fn finish_offset(
        &self,
        dense_claims: &[Gf128],
        offset: Gf128,
    ) -> Result<(Vec<Gf128>, Vec<Gf128>), &'static str> {
        self.finish_inner(dense_claims, offset)
    }

    fn finish_inner(
        &self,
        dense_claims: &[Gf128],
        offset: Gf128,
    ) -> Result<(Vec<Gf128>, Vec<Gf128>), &'static str> {
        if self.challenges.len() != self.num_vars {
            return Err("rounds incomplete");
        }
        let n_dense = self
            .terms
            .iter()
            .flat_map(|t| t.dense.iter().copied())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        if dense_claims.len() != n_dense {
            return Err("dense evaluation claim count mismatch");
        }
        let point = self.order.point(self.num_vars, &self.challenges);
        // Evaluate public factors at the point.
        let public_vals: Vec<Gf128> = self.spec_public.iter().map(|f| f.eval_at(&point)).collect();
        // Terminal identity: Σ_t coeff_t · Π pub · Π dense = running.
        let mut total = ZERO;
        for t in &self.terms {
            let mut prod = t.coeff;
            for &p in &t.public {
                prod = mul(prod, public_vals[p]);
            }
            for &d in &t.dense {
                let v = dense_claims
                    .get(d)
                    .ok_or("missing dense evaluation claim")?;
                prod = mul(prod, *v);
            }
            total += prod;
        }
        total += offset;
        if total != self.running {
            return Err("terminal identity check failed");
        }
        Ok((point, dense_claims.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public::EqFactor;
    use magnetar_field::Gf128;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rand::RngExt::random::<u128>(rng))
    }

    fn bit_table(bits: &[bool]) -> Vec<Gf128> {
        bits.iter().map(|&b| Gf128(b as u128)).collect()
    }

    #[test]
    fn engine_roundtrip_simple() {
        // g(x) = eq(τ, x) · T(x), T a random bit table, claim = Σ eq·T.

        let mut rng = rand::rng();
        let m = 4;
        let tau: Vec<Gf128> = (0..m).map(|_| rg(&mut rng)).collect();
        let table: Vec<Gf128> = (0..(1 << m)).map(|_| rg(&mut rng)).collect();
        let eqt = magnetar_field::EqBits::build(&tau).values;
        let mut claim = ZERO;
        for i in 0..(1 << m) {
            claim += mul(eqt[i], table[i]);
        }
        let spec = SumcheckSpec {
            num_vars: m,
            dense: vec![table.clone()],
            public: vec![Box::new(EqFactor::new(tau.clone()))],
            terms: vec![crate::Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![0],
            }],
            claim,
        };
        // Prove.
        let mut t = magnetar_hash::Transcript::new("test-sc", &[b"simple"]);
        let mut prover = SumcheckProver::new(&spec, BindOrder::Increasing);
        for _ in 0..m {
            prover.absorb_and_fold(&mut t, "r");
        }
        let (claims, running) = prover.finish();
        // Verify.
        let mut tv = magnetar_hash::Transcript::new("test-sc", &[b"simple"]);
        let pubf: Vec<Box<dyn crate::public::PublicFactor + Send + Sync>> =
            vec![Box::new(EqFactor::new(tau.clone()))];
        let mut verifier =
            SumcheckVerifier::new(m, claim, &pubf, spec.terms.clone(), BindOrder::Increasing);
        // Replay messages by recomputing? The verifier consumes the proof
        // messages; here regenerate them via a second prover transcript.
        let mut tp = magnetar_hash::Transcript::new("test-sc", &[b"simple"]);
        let mut prover2 = SumcheckProver::new(&spec, BindOrder::Increasing);
        for _ in 0..m {
            let msg = prover2.message();
            verifier.round(&msg, &mut tv, "r").unwrap();
            let mut buf = Vec::new();
            for c in &msg.0 {
                buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
            }
            tp.absorb_bytes("r", &buf);
            let r = tp.challenge_field("r");
            prover2.fold(r);
        }
        let (claims2, running2) = prover2.finish();
        assert_eq!(claims, claims2);
        assert_eq!(running, running2);
        let (point, got_claims) = verifier.finish(&claims).unwrap();
        // The dense claim equals T̂ at the point.
        let expect = magnetar_field::eval_mle(&table, &point);
        assert_eq!(got_claims[0], expect);
    }

    #[test]
    fn tamper_detected() {
        let mut rng = rand::rng();
        let m = 5;
        let table: Vec<Gf128> = (0..(1 << m)).map(|_| rg(&mut rng)).collect();
        let tau: Vec<Gf128> = (0..m).map(|_| rg(&mut rng)).collect();
        let eqt = magnetar_field::EqBits::build(&tau).values;
        let mut claim = ZERO;
        for i in 0..(1 << m) {
            claim += mul(eqt[i], table[i]);
        }
        let spec = SumcheckSpec {
            num_vars: m,
            dense: vec![table],
            public: vec![Box::new(EqFactor::new(tau))],
            terms: vec![crate::Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![0],
            }],
            claim,
        };
        let mut t = magnetar_hash::Transcript::new("tamper", &[]);
        let mut prover = SumcheckProver::new(&spec, BindOrder::Increasing);
        let mut msgs = Vec::new();
        for _ in 0..m {
            let msg = prover.message();
            msgs.push(msg);
            prover.absorb_and_fold(&mut t, "r");
        }
        let (mut claims, _) = prover.finish();
        // Tamper with the final dense claim.
        claims[0] += Gf128(1);
        let mut tv = magnetar_hash::Transcript::new("tamper", &[]);
        let pubf: Vec<Box<dyn crate::public::PublicFactor + Send + Sync>> =
            vec![Box::new(EqFactor::new(Vec::new()))];
        let mut verifier = SumcheckVerifier::new(
            m,
            spec.claim,
            &pubf,
            spec.terms.clone(),
            BindOrder::Increasing,
        );
        // wrong public factor is fine — terminal check must fail on tampered claim
        for msg in &msgs {
            let _ = verifier.round(msg, &mut tv, "r");
        }
        // Rebuild the verifier with the right public factor.
        let mut tv2 = magnetar_hash::Transcript::new("tamper", &[]);
        let pubf2: Vec<Box<dyn crate::public::PublicFactor + Send + Sync>> = vec![Box::new(
            EqFactor::new((0..m).map(|_| rg(&mut rng)).collect()),
        )];
        let mut v2 = SumcheckVerifier::new(
            m,
            spec.claim,
            &pubf2,
            spec.terms.clone(),
            BindOrder::Increasing,
        );
        for msg in &msgs {
            let _ = v2.round(msg, &mut tv2, "r");
        }
        assert!(v2.finish(&claims).is_err(), "tampered claim accepted");
    }

    #[test]
    fn high_bits_first_order() {
        // Check BindOrder::HighBitsFirst maps challenges back correctly by
        // folding a table whose value depends on variable order.

        let mut rng = rand::rng();
        let m = 6;
        let table: Vec<Gf128> = (0..(1 << m)).map(|_| rg(&mut rng)).collect();
        // Summand: 1 · T(x) — claim = Σ T.
        let mut claim = ZERO;
        for &v in &table {
            claim += v;
        }
        let spec = SumcheckSpec {
            num_vars: m,
            dense: vec![table.clone()],
            public: vec![],
            terms: vec![crate::Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![],
            }],
            claim,
        };
        let order = BindOrder::HighBitsFirst { high: 3 };
        let mut t = magnetar_hash::Transcript::new("hb", &[]);
        let mut prover = SumcheckProver::new(&spec, order);
        let mut msgs = Vec::new();
        for _ in 0..m {
            let msg = prover.message();
            msgs.push(msg.clone());
            let mut buf = Vec::new();
            for c in &msg.0 {
                buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
            }
            t.absorb_bytes("r", &buf);
            let r = t.challenge_field("r");
            prover.fold(r);
        }
        let (claims, _) = prover.finish();
        // The claim must be T̂ at the mapped point.
        let point = order.point(m, &{
            // recompute challenges deterministically
            let mut t2 = magnetar_hash::Transcript::new("hb", &[]);
            let mut ch = Vec::new();
            for msg in &msgs {
                let mut buf = Vec::new();
                for c in &msg.0 {
                    buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
                }
                t2.absorb_bytes("r", &buf);
                ch.push(t2.challenge_field("r"));
            }
            ch
        });
        assert_eq!(claims[0], magnetar_field::eval_mle(&table, &point));
        let _ = bit_table;
    }
}
