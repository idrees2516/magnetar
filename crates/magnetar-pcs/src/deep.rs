//! The **BEEP layer** — m DEEP samples per fold round, with threshold
//! proofs of work ("Better Elimination of Pretenders", 2026; the
//! `worldfnd/BEEP` artifact, `beep-ligerito/flock/.../pcs/ligerito/beep.rs`).
//!
//! ## What BEEP gives this stack
//!
//! The ring switches' fold chains (a FRI-style walk over per-level
//! commitments) inherit their fold-round soundness from a
//! mutual-correlated-agreement-style analysis of the code. BEEP replaces
//! that dependency: **before every fold challenge** the verifier samples
//! `m` points, the prover answers the two halves' multilinear extensions
//! at them, and *then* the challenge is drawn (ground, behind a threshold
//! proof of work). The DEEP sampling error falls exponentially with `m`
//! (in practice `m = 2..5` makes it tiny), the subsequent folding error is
//! `(δ·n + 1)/|F|` for a proximity parameter δ, and the round needs only
//! the code's **list decodability** — not MCA. The affine variant (folding
//! challenges as affine polynomials, following CWYZ26) improves the
//! post-DEEP folding error to `2/|F|`; over GF(2^128), where this stack's
//! challenges already live in the field, the scalar form achieves the
//! same envelope and is what is implemented here.
//!
//! ## The construction (per fold round of one fold chain)
//!
//! State: a weight table `W` over the current level's codeword variables
//! (little-endian: the variable folded next is `point[0]`), and a running
//! claim `v` with `<f_i, W_i> = v_i` for the running codeword table `f_i`.
//!
//! 1. **Points.** One squeeze draws `m·dim` field elements — `m` points
//!    over the level-`(i+1)` layout (the *binding points in one message*
//!    discipline).
//! 2. **Answers.** `b_{j,0} = <f_0, eq(z_j)>`, `b_{j,1} = <f_1, eq(z_j)>`
//!    — the two halves' MLEs at each point, flattened
//!    `b_{1,0}, b_{1,1}, b_{2,0}, b_{2,1}, …`.
//! 3. **Round message.** `h(X) = Σ_q (f_0 + X(f_0+f_1))(W_0 + X(W_0+W_1))`
//!    as `[h(0), h_2]` — the fold is *the scalar fold of a sumcheck
//!    round*; the middle coefficient is determined by the running claim
//!    (`claim + h_2` in characteristic two).
//! 4. **Grind + challenge.** A proof of work with threshold `T` gates the
//!    squeeze of `(ρ, ζ_1..ζ_m)`: a squeeze of error `numerator/2^128`
//!    behind a pass-`T/2^64` nonce has error below `2^-target` whenever
//!    `numerator·T < 2^(64+128−target)` (BEEP's `Grind::exact_at`).
//! 5. **Fold.** `f' = (1+ρ)f_0 + ρf_1`, `W' = (1+ρ)W_0 + ρW_1 +
//!    Σ ζ_j eq(z_j)`, `v' = h(ρ) + Σ ζ_j·((1+ρ)b_{j,0} + ρb_{j,1})`.
//!
//! The level-0 committed table is first bound out-of-domain (`bind`): the
//! verifier samples `bind_points` points over its full layout, the prover
//! answers `<f_0, eq(z)>`, and a ground ζ batch folds them into `W`.
//!
//! **The final check.** Each sample's eq tensor folds through the later
//! rounds as a scalar `Π (1 + z_{x_r} + ρ_r)` times the eq tensor of its
//! ε coordinates, so the verifier rebuilds `W_final` over the `2^k` tail
//! itself and checks `<tail, W_final> = v_final` against the fully opened
//! tail. Combined with the existing correlated query walks (which pin the
//! per-level Merkle trees), the DEEP chain makes every fold round's
//! soundness rest on list decodability with exponentially small error —
//! without MCA and without per-round re-commitments.
//!
//! ## The ladder
//!
//! [`deep_ladder`] ports BEEP's per-level analysis (`sample_bits`,
//! `least_threshold`, `select_fold`): the tail bound on the number of
//! pretender challenges surviving `m` samples, and the fold grinding that
//! brings the `(t + 3L)/2^128` squeeze below the target. For this stack's
//! substitution code (relative distance ≈ 1 at rate `2^-k`, generic γ)
//! the Johnson-shape inputs at the profile rate are used conservatively;
//! the selected `(m, threshold, grind bits)` are the least-work parameters
//! reaching the round-by-round target.

use magnetar_field::{mul, EqBits, Gf128, ZERO};
use magnetar_hash::Transcript;
use rayon::prelude::*;

/// The log-size of the field: GF(2^128).
const LOG_Q: f64 = 128.0;

/// The most DEEP samples in one fold round (BEEP's `MAX_SAMPLES`).
pub const MAX_SAMPLES: usize = 8;

/// A proof-of-work difficulty: a nonce passes with probability
/// `2^-bits` (`Bits`) or `threshold / 2^64` (`Below`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grind {
    /// A nonce passes with probability `2^-bits`.
    Bits(u32),
    /// A probe below a 64-bit threshold: pass probability `t / 2^64`.
    Below(u64),
    /// No grinding at all (every nonce passes).
    Off,
}

impl Grind {
    /// The threshold for a squeeze of error `numerator / 2^128` that must
    /// land below `2^-target_bits` after the proof of work: the largest
    /// `T` with `numerator·T < 2^(64 + 128 − target)`. No proof of work
    /// when every nonce may pass (`numerator ≤ 2^(128 − target)`).
    pub fn exact_at(numerator: u64, target_bits: f64) -> Self {
        let slack = LOG_Q - target_bits;
        assert!(
            (0.0..64.0).contains(&slack) && slack.fract() == 0.0,
            "a target of {target_bits} bits over a field of 2^128 elements"
        );
        if numerator as f64 <= 2f64.powf(slack) {
            return Grind::Off;
        }
        let threshold = ((1u128 << (64 + slack as u32)) - 1) / u128::from(numerator);
        match u64::try_from(threshold) {
            Ok(threshold) => Grind::Below(threshold),
            Err(_) => Grind::Off,
        }
    }

    /// `b` bits of grinding (`Grind::Bits(b)`; zero is `Off`).
    pub fn bits(b: u32) -> Self {
        if b == 0 {
            Grind::Off
        } else {
            Grind::Bits(b)
        }
    }

    /// The transcript threshold (a probe passes when strictly below it).
    pub fn threshold(&self) -> u64 {
        match self {
            Grind::Bits(b) => 1u64.checked_shl(64 - *b).unwrap_or(0),
            Grind::Below(t) => *t,
            Grind::Off => u64::MAX,
        }
    }

    /// Expected hashes of the search (`1 / p` for pass probability `p`).
    pub fn work(&self) -> f64 {
        match self {
            Grind::Bits(0) | Grind::Off => 0.0,
            Grind::Bits(b) => f64::from(*b).exp2(),
            Grind::Below(t) => 64f64.exp2() / *t as f64,
        }
    }

    /// `log2` of the work — the bits the proof of work adds.
    pub fn added_bits(&self) -> f64 {
        match self {
            Grind::Bits(b) => f64::from(*b),
            Grind::Below(t) => 64.0 - (*t as f64).log2(),
            Grind::Off => 0.0,
        }
    }
}

/// The bits of the DEEP tail bound: `-log2 Pr[pretenders ≥ t]` for `m`
/// points at `variables` remaining variables, a list bound `L`, a line
/// bound `N`, and threshold `t` (BEEP's `sample_bits`, ported verbatim).
fn sample_bits(list: u64, variables: usize, samples: usize, line_bound: u64, t: u128) -> f64 {
    debug_assert!(t > u128::from(line_bound));
    samples as f64 * (LOG_Q - (variables as f64).log2())
        - 3.0 * (LOG_Q + (list as f64).log2())
        + (t as f64).log2()
        + ((t - 1) as f64).log2()
        + ((t - u128::from(line_bound)) as f64).log2()
}

/// The least threshold `t ≥ N + 1` at which `m` points reach the target,
/// or `None` if none below `2^100` does (BEEP's `least_threshold`).
fn least_threshold(
    list: u64,
    variables: usize,
    samples: usize,
    line_bound: u64,
    target_bits: f64,
) -> Option<u128> {
    const MARGIN_BITS: f64 = 2.0;
    let reaches =
        |t: u128| sample_bits(list, variables, samples, line_bound, t) >= target_bits + MARGIN_BITS;
    let lo = u128::from(line_bound) + 1;
    let hi = 1u128 << 100;
    if !reaches(hi) {
        return None;
    }
    let (mut lo, mut hi) = (lo, hi);
    if reaches(lo) {
        return Some(lo);
    }
    // Invariant: !reaches(lo), reaches(hi); the bits increase with t.
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if reaches(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

/// The least-work `(m, threshold, fold grinding bits)` for one fold round
/// (BEEP's `select_fold`, ported): for every `m ≤ MAX_SAMPLES` the least
/// threshold whose points reach the target, then the fold grinding that
/// brings the squeeze `(t + 3L)/2^128` below the target; among those the
/// least grinding.
fn select_fold(
    list: u64,
    variables: usize,
    line_bound: u64,
    target_bits: f64,
) -> Result<(usize, u128, u32), String> {
    const MAX_FOLD_GRINDING_BITS: u32 = 48;
    let mut best: Option<(usize, u128, u32)> = None;
    for m in 1..=MAX_SAMPLES {
        let Some(t) = least_threshold(list, variables, m, line_bound, target_bits) else {
            continue;
        };
        // The fold squeeze has error `(t + 3L) / 2^128`: the bits that
        // bring it below `2^-target_bits`.
        let grind = ((t + 3 * u128::from(list)) as f64).log2() - (LOG_Q - target_bits);
        let grind = grind.ceil().max(0.0) as u32;
        if grind > MAX_FOLD_GRINDING_BITS {
            continue;
        }
        let better = match &best {
            None => true,
            Some((_, _, g)) => grind < *g,
        };
        if better {
            best = Some((m, t, grind));
        }
    }
    best.ok_or_else(|| {
        format!(
            "no m <= {MAX_SAMPLES} reaches {target_bits} bits within \
             {MAX_FOLD_GRINDING_BITS} bits of fold grinding \
             (L = {list}, N = {line_bound}, {variables} variables)"
        )
    })
}

/// The DEEP-layer parameters for one ring switch, as derived by
/// [`deep_ladder`] or hand-set for tests.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepParams {
    /// DEEP samples per fold round (BEEP's `m`; 2..5 in practice).
    pub samples: usize,
    /// Out-of-domain binding points on the level-0 committed table.
    pub bind_points: usize,
    /// The proof of work gating every fold-round squeeze `(ρ, ζs)`.
    pub fold_grind: Grind,
    /// The proof of work gating the bind batch squeeze.
    pub bind_grind: Grind,
    /// The proof of work gating the query and anchor position draws.
    pub query_grind: Grind,
    /// The round-by-round soundness target, in bits.
    pub target_bits: f64,
}

impl DeepParams {
    /// The DEEP layer off — the legacy protocol, byte-identical proofs.
    pub fn off() -> Self {
        DeepParams {
            samples: 0,
            bind_points: 0,
            fold_grind: Grind::Off,
            bind_grind: Grind::Off,
            query_grind: Grind::Off,
            target_bits: 128.0,
        }
    }

    pub fn is_on(&self) -> bool {
        self.samples > 0
    }
}

impl Default for DeepParams {
    fn default() -> Self {
        // The 120-bit round-by-round operating point — the crate's stated
        // envelope — at m = 3 with a modest per-round grind (the ladder
        // derives the least-work 128-bit point per geometry).
        DeepParams {
            samples: 3,
            bind_points: 2,
            fold_grind: Grind::bits(12),
            bind_grind: Grind::bits(8),
            query_grind: Grind::bits(8),
            target_bits: 120.0,
        }
    }
}

/// One auditing line of [`deep_ladder`]'s round-by-round accounting.
#[derive(Clone, Debug)]
pub struct DeepLadderLine {
    pub samples: usize,
    pub threshold: u128,
    pub fold_grind_bits: u32,
    pub sample_bits: f64,
    pub fold_error_bits: f64,
    pub fold_grind_work: f64,
}

/// Derive the DEEP parameters for a ring switch of `n_total` block
/// variables at rate `2^-k` for a `target_bits` round-by-round target —
/// BEEP's per-level analysis adapted to this stack's single-level chain.
///
/// The Johnson-shape inputs at the profile rate are used conservatively
/// (the substitution code's generic-γ distance ≈ 1 only improves them):
/// `θ = √ρ + η`, list size `L = ⌊1/(2η√ρ)⌋`, line bound
/// `N = 2^v − ⌈θ·2^v⌉ + 1` at the round with the most variables. The
/// fold-round parameters `(m, t, grind)` are BEEP's least-work selection;
/// the bind and query grinds follow its claim-batch and query discipline.
pub fn deep_ladder(
    k: usize,
    n_total: usize,
    target_bits: f64,
) -> Result<(DeepParams, Vec<DeepLadderLine>), String> {
    let slack = LOG_Q - target_bits;
    if !(0.0..64.0).contains(&slack) || slack.fract() != 0.0 {
        return Err(format!(
            "a target of {target_bits} bits over a field of 2^128 elements"
        ));
    }
    if k == 0 || k > 6 {
        return Err(format!("rate 2^-{k} is outside the profiled ladder"));
    }
    // The rate-1/2^k Johnson agreement with η chosen to keep the list
    // size under 2^8 (BEEP runs η between 0.001 and 0.02).
    let rho = (-(k as f64)).exp2();
    let sqrt_rho = rho.sqrt();
    let eta = (2f64.powi(-8) / (2.0 * sqrt_rho)).min(0.02);
    let theta = sqrt_rho + eta;
    if !(eta > 0.0 && theta < 1.0) {
        return Err(format!("agreement {theta} is not below 1"));
    }
    // The worst round: the first, with `n_total - 1 + k` variables on the
    // level-1 table (BEEP's `max_variables = k + c - 1`).
    let max_variables = (n_total + k - 1).max(1);
    let n = 1u64 << max_variables;
    let agreement = (theta * n as f64).ceil() as u64;
    let line_bound = (n - agreement + 1).max(2);
    let list_size = ((1.0 / (2.0 * eta * sqrt_rho)).floor() as u64).max(1);
    let (samples, threshold, fold_grind_bits) =
        select_fold(list_size, max_variables, line_bound, target_bits)?;
    let fold_numerator = u64::try_from(threshold + 3 * u128::from(list_size))
        .map_err(|_| "fold numerator above 2^64")?;
    // The bind batch (the claim-batch discipline): one wrong candidate
    // passes with probability ≤ ⌈log2 bind⌉/|F| per point, list-bounded.
    let bind_points = 2usize;
    let claim_numerator = list_size + 2;
    let bind_grind_bits = ((claim_numerator as f64).log2() - slack)
        .ceil()
        .max(0.0) as u32;
    // The query draws: the grind adds the headroom the walk's per-level
    // catch leaves on top of the rejection-sampled positions.
    let query_grind_bits = 8u32;
    let line = DeepLadderLine {
        samples,
        threshold,
        fold_grind_bits,
        sample_bits: sample_bits(list_size, max_variables, samples, line_bound, threshold),
        fold_error_bits: LOG_Q - (fold_numerator as f64).log2() + f64::from(fold_grind_bits),
        fold_grind_work: f64::from(fold_grind_bits).exp2(),
    };
    let params = DeepParams {
        samples,
        bind_points,
        fold_grind: Grind::bits(fold_grind_bits),
        bind_grind: Grind::bits(bind_grind_bits),
        query_grind: Grind::bits(query_grind_bits),
        target_bits,
    };
    Ok((params, vec![line]))
}

// ----------------------------------------------------------------------
// The proof parts
// ----------------------------------------------------------------------

/// The DEEP layer's proof parts, carried by a ring-switch proof when the
/// DEEP parameters are on. One entry per fold chain — the monolithic
/// ring carries one chain, the split ring's two components carry two.
#[derive(Clone, Debug, Default)]
pub struct DeepLayerProof {
    /// Per chain: the bind answers (`bind_points` values each).
    pub bind_answers: Vec<Vec<Gf128>>,
    /// The bind batch's proof-of-work nonce.
    pub bind_nonce: u64,
    /// Per round, per chain: the DEEP round message `[h(0), h_2]`.
    pub rounds: Vec<Vec<[Gf128; 2]>>,
    /// Per round, per chain: the `2m` answer values.
    pub answers: Vec<Vec<Vec<Gf128>>>,
    /// Per round: the fold squeeze's proof-of-work nonce.
    pub nonces: Vec<u64>,
}

// ----------------------------------------------------------------------
// The prover's chain state
// ----------------------------------------------------------------------

/// One fold chain's DEEP state: the weight table `W` over the current
/// level's layout and the running claim `v = <f_i, W_i>`.
#[derive(Debug)]
pub struct DeepChain {
    w: Vec<Gf128>,
    claim: Gf128,
}

impl DeepChain {
    /// A chain over a level-0 table of `len` values (zero weight, zero
    /// claim — the bind and the rounds build both).
    pub fn new(len: usize) -> Self {
        DeepChain {
            w: vec![ZERO; len],
            claim: ZERO,
        }
    }

    /// The running claim (the DEEP sum-check's).
    pub fn claim(&self) -> Gf128 {
        self.claim
    }

    /// The current weight-table length (the current level's size).
    pub fn table_len(&self) -> usize {
        self.w.len()
    }

    /// Add a value to the running claim (the pair's shared-ζ bind).
    pub fn add_claim(&mut self, v: Gf128) {
        self.claim += v;
    }

    /// The level-0 out-of-domain bind: draw `points` points over the
    /// full level-0 layout, answer `<table, eq(z)>`, absorb, ground ζ
    /// batch, fold the eq weights into `W`, and add the ζ-weighted
    /// answers to the claim. Returns the proof parts
    /// `(answers, zetas, nonce)`.
    pub fn bind(
        &mut self,
        t: &mut Transcript,
        table: &[Gf128],
        points: usize,
        grind: Grind,
        label: &str,
    ) -> (Vec<Gf128>, Vec<Gf128>, u64) {
        assert_eq!(self.w.len(), table.len());
        let dim = table.len().trailing_zeros() as usize;
        let flat = t.challenge_field_vec(&format!("{label}-bind-z"), points * dim);
        let pts: Vec<Vec<Gf128>> = (0..points)
            .map(|i| flat[i * dim..(i + 1) * dim].to_vec())
            .collect();
        let answers: Vec<Gf128> = pts
            .iter()
            .map(|p| {
                let eq = EqBits::build(p).values;
                dot_tables(table, &eq)
            })
            .collect();
        t.absorb_field(&format!("{label}-bind-ans"), &answers);
        let nonce = t.grind_pow(&format!("{label}-bind-pow"), grind.threshold());
        let zetas = t.challenge_field_vec(&format!("{label}-bind-zeta"), points);
        self.add_eq_weights(&pts, &zetas);
        for (&z, &y) in zetas.iter().zip(&answers) {
            self.claim += mul(z, y);
        }
        (answers, zetas, nonce)
    }

    /// One full fold round's transcript interaction on `table` (the
    /// level-`i` codeword): draw the points, answer the halves' MLEs,
    /// absorb, send the round message, grind, and draw `(ρ, ζs)`. Folds
    /// `W` and the claim; **the caller folds the table at `ρ`**.
    /// Returns `(answers, msg, r, zetas, nonce)`.
    pub fn round(
        &mut self,
        t: &mut Transcript,
        table: &[Gf128],
        samples: usize,
        grind: Grind,
        label: &str,
    ) -> (Vec<Gf128>, [Gf128; 2], Gf128, Vec<Gf128>, u64) {
        assert_eq!(self.w.len(), table.len());
        let dim = (table.len() / 2).trailing_zeros() as usize; // level-(i+1) vars
        let flat = t.challenge_field_vec(&format!("{label}-deep-z"), samples * dim);
        let pts: Vec<Vec<Gf128>> = (0..samples)
            .map(|i| flat[i * dim..(i + 1) * dim].to_vec())
            .collect();
        let answers = Self::answers(table, &pts);
        t.absorb_field(&format!("{label}-deep-ans"), &answers);
        let msg = self.round_message(table);
        t.absorb_field(&format!("{label}-deep-msg"), &msg);
        let nonce = t.grind_pow(&format!("{label}-deep-pow"), grind.threshold());
        let r = t.challenge_field(&format!("{label}-deep-r"));
        let zetas = t.challenge_field_vec(&format!("{label}-deep-zeta"), samples);
        self.fold(r, msg, &pts, &zetas, &answers);
        (answers, msg, r, zetas, nonce)
    }

    /// `W += Σ_i ζ_i eq(z_i)` (points in the level-0 layout).
    pub fn add_eq_weights(&mut self, pts: &[Vec<Gf128>], zetas: &[Gf128]) {
        assert_eq!(pts.len(), zetas.len());
        let eqs: Vec<Vec<Gf128>> = pts.iter().map(|p| EqBits::build(p).values).collect();
        self.w
            .par_chunks_mut(1 << 10)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci << 10;
                for (q, dst) in chunk.iter_mut().enumerate() {
                    let idx = base + q;
                    let mut acc = *dst;
                    for (eq, &z) in eqs.iter().zip(zetas) {
                        acc += mul(z, eq[idx]);
                    }
                    *dst = acc;
                }
            });
    }

    /// The fold-round DEEP answers for `m` points over the level-`(i+1)`
    /// layout: `b_{j,0} = <f_0, eq(z_j)>`, `b_{j,1} = <f_1, eq(z_j)>`
    /// — flattened `b_{1,0}, b_{1,1}, b_{2,0}, b_{2,1}, …`.
    pub fn answers(table: &[Gf128], pts: &[Vec<Gf128>]) -> Vec<Gf128> {
        let eqs: Vec<Vec<Gf128>> = pts.iter().map(|p| EqBits::build(p).values).collect();
        let mut out = Vec::with_capacity(2 * pts.len());
        for eq in &eqs {
            let (a, b) = dot_halves(table, eq);
            out.push(a);
            out.push(b);
        }
        out
    }

    /// The fold-round message `[h(0), h_2]` of
    /// `h(X) = Σ_q (f_0 + X(f_0+f_1))(W_0 + X(W_0+W_1))`.
    pub fn round_message(&self, table: &[Gf128]) -> [Gf128; 2] {
        assert_eq!(self.w.len(), table.len());
        let (a, c) = table
            .par_chunks(1 << 10)
            .zip(self.w.par_chunks(1 << 10))
            .map(|(fc, wc)| {
                let (mut a, mut c) = (ZERO, ZERO);
                let mut i = 0;
                while i + 1 < fc.len() {
                    let f0 = fc[i];
                    let f1 = fc[i + 1];
                    let w0 = wc[i];
                    let w1 = wc[i + 1];
                    a += mul(f0, w0);
                    c += mul(f0 + f1, w0 + w1);
                    i += 2;
                }
                (a, c)
            })
            .reduce(|| (ZERO, ZERO), |x, y| (x.0 + y.0, x.1 + y.1));
        [a, c]
    }

    /// Fold the lowest bit at `ρ`: `W' = (1+ρ)W_0 + ρW_1 + Σ ζ_j eq(z_j)`
    /// (the points in the level-`(i+1)` layout), and
    /// `v' = h(ρ) + Σ ζ_j·((1+ρ)b_{j,0} + ρb_{j,1})`.
    pub fn fold(
        &mut self,
        r: Gf128,
        msg: [Gf128; 2],
        pts: &[Vec<Gf128>],
        zetas: &[Gf128],
        answers: &[Gf128],
    ) {
        assert_eq!(pts.len(), zetas.len());
        assert_eq!(answers.len(), 2 * pts.len());
        let half = self.w.len() / 2;
        let eqs: Vec<Vec<Gf128>> = pts.iter().map(|p| EqBits::build(p).values).collect();
        let old = std::mem::replace(&mut self.w, Vec::with_capacity(half));
        let mut w2 = vec![ZERO; half];
        w2.par_chunks_mut(1 << 10)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci << 10;
                for (q, dst) in chunk.iter_mut().enumerate() {
                    let j = base + q;
                    let w0 = old[2 * j];
                    let w1 = old[2 * j + 1];
                    let mut acc = w0 + mul(r, w0 + w1);
                    for (eq, &z) in eqs.iter().zip(zetas) {
                        acc += mul(z, eq[j]);
                    }
                    *dst = acc;
                }
            });
        self.w = w2;
        self.claim = round_value(self.claim, msg, r);
        for (j, &z) in zetas.iter().enumerate() {
            let b0 = answers[2 * j];
            let b1 = answers[2 * j + 1];
            self.claim += mul(z, b0 + mul(r, b0 + b1));
        }
    }
}

/// `h(ρ)` for the round polynomial `[h(0), h_2]` at running claim `v`
/// (characteristic two: the middle coefficient is `v + h_2`).
pub fn round_value(claim: Gf128, msg: [Gf128; 2], r: Gf128) -> Gf128 {
    msg[0] + mul(r, claim + msg[1]) + mul(mul(r, r), msg[1])
}

/// `<a, b>` over the same-layout tables (parallel).
fn dot_tables(a: &[Gf128], b: &[Gf128]) -> Gf128 {
    a.par_chunks(1 << 10)
        .zip(b.par_chunks(1 << 10))
        .map(|(x, y)| x.iter().zip(y).fold(ZERO, |acc, (&u, &v)| acc + mul(u, v)))
        .reduce(|| ZERO, |x, y| x + y)
}

/// `<f_0, eq>` and `<f_1, eq>` for the halves `f[2j]`, `f[2j+1]`.
fn dot_halves(f: &[Gf128], eq: &[Gf128]) -> (Gf128, Gf128) {
    debug_assert_eq!(f.len(), 2 * eq.len());
    f.par_chunks(2 << 10)
        .zip(eq.par_chunks(1 << 10))
        .map(|(fc, ec)| {
            let (mut a, mut b) = (ZERO, ZERO);
            for (pair, &e) in fc.chunks_exact(2).zip(ec) {
                a += mul(pair[0], e);
                b += mul(pair[1], e);
            }
            (a, b)
        })
        .reduce(|| (ZERO, ZERO), |x, y| (x.0 + y.0, x.1 + y.1))
}

// ----------------------------------------------------------------------
// The verifier
// ----------------------------------------------------------------------

/// One tracked sample: its batch weight, its point, the round whose fold
/// first applies to it, and the accumulated surviving scalar.
#[derive(Clone, Debug)]
struct Sample {
    zeta: Gf128,
    point: Vec<Gf128>,
    /// The first round whose fold applies (0 for bind samples).
    first_round: usize,
    /// `Π (1 + z_{x_r} + ρ_r)` over the rounds applied so far.
    scalar: Gf128,
}

/// The verifier's mirror of one fold chain's DEEP state.
#[derive(Debug)]
pub struct DeepVerify {
    claim: Gf128,
    samples: Vec<Sample>,
    n_total: usize,
    k: usize,
    rounds_seen: usize,
}

impl DeepVerify {
    /// A verifier for a chain of `n_total` block variables and `k`
    /// extension variables (the tail has `2^k` values).
    pub fn new(n_total: usize, k: usize) -> Self {
        DeepVerify {
            claim: ZERO,
            samples: Vec::new(),
            n_total,
            k,
            rounds_seen: 0,
        }
    }

    /// The running claim (auditing).
    pub fn claim(&self) -> Gf128 {
        self.claim
    }

    /// The bind points' dimension (the level-0 layout).
    pub(crate) fn bind_dim(&self) -> usize {
        self.n_total + self.k
    }

    /// The current round's sample-point dimension.
    pub(crate) fn dim_at_round(&self) -> usize {
        self.n_total - self.rounds_seen - 1 + self.k
    }

    /// The bind mirror: draw the points over the level-0 layout, absorb
    /// the prover's answers, verify the grind, draw the ζ batch, and
    /// record the samples. Returns whether the nonce passed.
    pub fn bind(
        &mut self,
        t: &mut Transcript,
        answers: &[Gf128],
        nonce: u64,
        grind: Grind,
        label: &str,
    ) -> bool {
        let points = answers.len();
        let dim = self.n_total + self.k;
        let flat = t.challenge_field_vec(&format!("{label}-bind-z"), points * dim);
        t.absorb_field(&format!("{label}-bind-ans"), answers);
        let ok = t.verify_pow(&format!("{label}-bind-pow"), nonce, grind.threshold());
        let zetas = t.challenge_field_vec(&format!("{label}-bind-zeta"), points);
        if ok {
            self.ledger_bind(&flat, dim, answers, &zetas);
        }
        ok
    }

    /// The ledger part of [`Self::bind`] (no transcript interaction).
    fn ledger_bind(&mut self, flat: &[Gf128], dim: usize, answers: &[Gf128], zetas: &[Gf128]) {
        for (i, &y) in answers.iter().enumerate() {
            self.claim += mul(zetas[i], y);
            self.samples.push(Sample {
                zeta: zetas[i],
                point: flat[i * dim..(i + 1) * dim].to_vec(),
                first_round: 0,
                scalar: Gf128(1),
            });
        }
    }

    /// One fold round's mirror: draw the points, absorb the answers and
    /// the round message, verify the grind, draw `(ρ, ζs)`, fold the
    /// claim, and advance every sample's surviving scalar. Returns the
    /// challenge (which the caller also uses for the ring sum-check) and
    /// the ζ batch.
    pub fn round(
        &mut self,
        t: &mut Transcript,
        msg: [Gf128; 2],
        answers: &[Gf128],
        nonce: u64,
        grind: Grind,
        label: &str,
    ) -> Result<(Gf128, Vec<Gf128>), &'static str> {
        if answers.len() % 2 != 0 {
            return Err("deep: odd answer count");
        }
        let m = answers.len() / 2;
        let dim = self.n_total - self.rounds_seen - 1 + self.k;
        let flat = t.challenge_field_vec(&format!("{label}-deep-z"), m * dim);
        t.absorb_field(&format!("{label}-deep-ans"), answers);
        t.absorb_field(&format!("{label}-deep-msg"), &msg);
        if !t.verify_pow(&format!("{label}-deep-pow"), nonce, grind.threshold()) {
            return Err("deep: fold grind nonce invalid");
        }
        let r = t.challenge_field(&format!("{label}-deep-r"));
        let zetas = t.challenge_field_vec(&format!("{label}-deep-zeta"), m);
        self.ledger_round(msg, answers, r, &zetas, &flat, dim);
        Ok((r, zetas))
    }

    /// The ledger part of [`Self::round`] (no transcript interaction).
    fn ledger_round(
        &mut self,
        msg: [Gf128; 2],
        answers: &[Gf128],
        r: Gf128,
        zetas: &[Gf128],
        flat: &[Gf128],
        dim: usize,
    ) {
        // Fold the claim: v' = h(ρ) + Σ ζ_j·((1+ρ)b_{j,0} + ρb_{j,1}).
        let mut next = round_value(self.claim, msg, r);
        for (j, &z) in zetas.iter().enumerate() {
            let b0 = answers[2 * j];
            let b1 = answers[2 * j + 1];
            next += mul(z, b0 + mul(r, b0 + b1));
        }
        self.claim = next;
        // Record this round's samples (their eq tensors enter W_{i+1} —
        // this round's fold does not apply to them).
        for (i, &z) in zetas.iter().enumerate() {
            self.samples.push(Sample {
                zeta: z,
                point: flat[i * dim..(i + 1) * dim].to_vec(),
                first_round: self.rounds_seen + 1,
                scalar: Gf128(1),
            });
        }
        // Fold every earlier sample's surviving scalar.
        for s in self.samples.iter_mut() {
            if s.first_round <= self.rounds_seen {
                let z_r = s.point[self.rounds_seen - s.first_round];
                s.scalar = mul(s.scalar, Gf128(1) + z_r + r);
            }
        }
        self.rounds_seen += 1;
    }

    /// The final check: rebuild `W_final` over the `2^k` tail layout from
    /// the samples' surviving scalars and ε coordinates, and check
    /// `<tail, W_final> = v_final`.
    pub fn final_check(&self, tail: &[Gf128]) -> Result<(), &'static str> {
        if self.rounds_seen != self.n_total {
            return Err("deep: rounds incomplete");
        }
        if tail.len() != (1usize << self.k) {
            return Err("deep: tail length mismatch");
        }
        let mut w_final = vec![ZERO; 1usize << self.k];
        for s in &self.samples {
            let eps = &s.point[self.n_total - s.first_round..];
            let eq = EqBits::build(eps).values;
            let scale = mul(s.zeta, s.scalar);
            for (dst, &e) in w_final.iter_mut().zip(&eq) {
                *dst += mul(scale, e);
            }
        }
        let ip = dot_tables(tail, &w_final);
        if ip != self.claim {
            return Err("deep: final tail inner product mismatch");
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------
// The two-chain pair (the split ring's merged (P, C) components)
// ----------------------------------------------------------------------

/// Two fold chains sharing every draw — the split ring's merged `(P, C)`
/// pair run as one BEEP layer: the same points, challenges, batch
/// coefficients, and grinding, with separate weights, round messages, and
/// claims.
#[derive(Debug)]
pub struct DeepPair {
    a: DeepChain,
    b: DeepChain,
}

impl DeepPair {
    /// A pair over two level-0 tables of `len` values each.
    pub fn new(len: usize) -> Self {
        DeepPair {
            a: DeepChain::new(len),
            b: DeepChain::new(len),
        }
    }

    /// The running claims.
    pub fn claims(&self) -> (Gf128, Gf128) {
        (self.a.claim(), self.b.claim())
    }

    /// The pair's level-0 bind: draw shared points over the full layout,
    /// answer both tables, absorb (a's, then b's), grind, draw the shared
    /// ζ batch, and fold both chains. Returns the per-chain answers, the
    /// ζ batch, and the nonce.
    pub fn bind(
        &mut self,
        t: &mut Transcript,
        table_a: &[Gf128],
        table_b: &[Gf128],
        points: usize,
        grind: Grind,
        label: &str,
    ) -> (Vec<Vec<Gf128>>, Vec<Gf128>, u64) {
        debug_assert_eq!(table_a.len(), table_b.len());
        let dim = table_a.len().trailing_zeros() as usize;
        let flat = t.challenge_field_vec(&format!("{label}-bind-z"), points * dim);
        let pts: Vec<Vec<Gf128>> = (0..points)
            .map(|i| flat[i * dim..(i + 1) * dim].to_vec())
            .collect();
        let eqs: Vec<Vec<Gf128>> = pts.iter().map(|p| EqBits::build(p).values).collect();
        let ans_a: Vec<Gf128> = eqs.iter().map(|e| dot_tables(table_a, e)).collect();
        let ans_b: Vec<Gf128> = eqs.iter().map(|e| dot_tables(table_b, e)).collect();
        t.absorb_field(&format!("{label}-bind-ans-a"), &ans_a);
        t.absorb_field(&format!("{label}-bind-ans-b"), &ans_b);
        let nonce = t.grind_pow(&format!("{label}-bind-pow"), grind.threshold());
        let zetas = t.challenge_field_vec(&format!("{label}-bind-zeta"), points);
        self.a.add_eq_weights(&pts, &zetas);
        self.b.add_eq_weights(&pts, &zetas);
        for (&z, &y) in zetas.iter().zip(&ans_a) {
            self.a.add_claim(mul(z, y));
        }
        for (&z, &y) in zetas.iter().zip(&ans_b) {
            self.b.add_claim(mul(z, y));
        }
        (vec![ans_a, ans_b], zetas, nonce)
    }

    /// One full fold round on both tables: draw the shared points, answer
    /// both tables' halves, absorb (a's, then b's answers; a's, then b's
    /// round messages), grind, draw `(ρ, ζs)`, and fold both chains.
    /// **The caller folds both tables at `ρ`.** Returns the per-chain
    /// answers, the per-chain messages, the challenge, the ζ batch, and
    /// the nonce.
    pub fn round(
        &mut self,
        t: &mut Transcript,
        table_a: &[Gf128],
        table_b: &[Gf128],
        samples: usize,
        grind: Grind,
        label: &str,
    ) -> (Vec<Vec<Gf128>>, [[Gf128; 2]; 2], Gf128, Vec<Gf128>, u64) {
        debug_assert_eq!(table_a.len(), table_b.len());
        debug_assert_eq!(table_a.len(), self.a.table_len());
        let dim = (table_a.len() / 2).trailing_zeros() as usize;
        let flat = t.challenge_field_vec(&format!("{label}-deep-z"), samples * dim);
        let pts: Vec<Vec<Gf128>> = (0..samples)
            .map(|i| flat[i * dim..(i + 1) * dim].to_vec())
            .collect();
        let ans_a = DeepChain::answers(table_a, &pts);
        let ans_b = DeepChain::answers(table_b, &pts);
        t.absorb_field(&format!("{label}-deep-ans-a"), &ans_a);
        t.absorb_field(&format!("{label}-deep-ans-b"), &ans_b);
        let msg_a = self.a.round_message(table_a);
        let msg_b = self.b.round_message(table_b);
        t.absorb_field(&format!("{label}-deep-msg-a"), &msg_a);
        t.absorb_field(&format!("{label}-deep-msg-b"), &msg_b);
        let nonce = t.grind_pow(&format!("{label}-deep-pow"), grind.threshold());
        let r = t.challenge_field(&format!("{label}-deep-r"));
        let zetas = t.challenge_field_vec(&format!("{label}-deep-zeta"), samples);
        self.a.fold(r, msg_a, &pts, &zetas, &ans_a);
        self.b.fold(r, msg_b, &pts, &zetas, &ans_b);
        (vec![ans_a, ans_b], [msg_a, msg_b], r, zetas, nonce)
    }
}

/// The verifier's mirror of a [`DeepPair`]: two ledgers sharing every
/// draw.
#[derive(Debug)]
pub struct DeepPairVerify {
    a: DeepVerify,
    b: DeepVerify,
}

impl DeepPairVerify {
    /// A pair verifier for a chain of `n_total` block variables and `k`
    /// extension variables.
    pub fn new(n_total: usize, k: usize) -> Self {
        DeepPairVerify {
            a: DeepVerify::new(n_total, k),
            b: DeepVerify::new(n_total, k),
        }
    }

    /// The bind mirror for both chains. Returns whether the nonce passed.
    pub fn bind(
        &mut self,
        t: &mut Transcript,
        answers: &[Vec<Gf128>],
        nonce: u64,
        grind: Grind,
        label: &str,
    ) -> bool {
        if answers.len() != 2 {
            return false;
        }
        let points = answers[0].len();
        debug_assert_eq!(points, answers[1].len());
        let dim = self.a.bind_dim();
        let flat = t.challenge_field_vec(&format!("{label}-bind-z"), points * dim);
        t.absorb_field(&format!("{label}-bind-ans-a"), &answers[0]);
        t.absorb_field(&format!("{label}-bind-ans-b"), &answers[1]);
        let ok = t.verify_pow(&format!("{label}-bind-pow"), nonce, grind.threshold());
        let zetas = t.challenge_field_vec(&format!("{label}-bind-zeta"), points);
        if ok {
            self.a.ledger_bind(&flat, dim, &answers[0], &zetas);
            self.b.ledger_bind(&flat, dim, &answers[1], &zetas);
        }
        ok
    }

    /// One fold round's mirror for both chains. Returns the shared
    /// challenge (which the caller also uses for the ring sum-check) and
    /// the ζ batch.
    pub fn round(
        &mut self,
        t: &mut Transcript,
        msgs: [[Gf128; 2]; 2],
        answers: &[Vec<Gf128>],
        nonce: u64,
        grind: Grind,
        label: &str,
    ) -> Result<(Gf128, Vec<Gf128>), &'static str> {
        if answers.len() != 2 {
            return Err("deep: pair answer chain count mismatch");
        }
        if answers[0].len() != answers[1].len() || answers[0].len() % 2 != 0 {
            return Err("deep: pair answer shape mismatch");
        }
        let m = answers[0].len() / 2;
        let dim = self.a.dim_at_round();
        let flat = t.challenge_field_vec(&format!("{label}-deep-z"), m * dim);
        t.absorb_field(&format!("{label}-deep-ans-a"), &answers[0]);
        t.absorb_field(&format!("{label}-deep-ans-b"), &answers[1]);
        t.absorb_field(&format!("{label}-deep-msg-a"), &msgs[0]);
        t.absorb_field(&format!("{label}-deep-msg-b"), &msgs[1]);
        if !t.verify_pow(&format!("{label}-deep-pow"), nonce, grind.threshold()) {
            return Err("deep: fold grind nonce invalid");
        }
        let r = t.challenge_field(&format!("{label}-deep-r"));
        let zetas = t.challenge_field_vec(&format!("{label}-deep-zeta"), m);
        self.a
            .ledger_round(msgs[0], &answers[0], r, &zetas, &flat, dim);
        self.b
            .ledger_round(msgs[1], &answers[1], r, &zetas, &flat, dim);
        Ok((r, zetas))
    }

    /// The final checks on both tails.
    pub fn final_check(&self, tail_a: &[Gf128], tail_b: &[Gf128]) -> Result<(), &'static str> {
        self.a.final_check(tail_a)?;
        self.b.final_check(tail_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        use rand::RngExt;
        Gf128(rng.random::<u128>())
    }

    fn fold_table(v: &[Gf128], r: Gf128) -> Vec<Gf128> {
        let mut out = Vec::with_capacity(v.len() / 2);
        let mut i = 0;
        while i < v.len() {
            let a = v[i];
            let b = v[i + 1];
            out.push(a + mul(r, a + b));
            i += 2;
        }
        out
    }

    /// The DEEP chain on a random table: the prover drives, the verifier
    /// mirrors, the claims agree, and the final tail check binds.
    #[test]
    fn deep_chain_roundtrip() {
        let mut rng = rand::rng();
        for (n, k, m) in [(6usize, 1usize, 2usize), (5, 2, 3), (4, 1, 4)] {
            let params = DeepParams {
                samples: m,
                bind_points: 2,
                fold_grind: Grind::bits(4),
                bind_grind: Grind::bits(2),
                query_grind: Grind::Off,
                target_bits: 128.0,
            };
            let len = 1usize << (n + k);
            let table: Vec<Gf128> = (0..len).map(|_| rg(&mut rng)).collect();
            let mut t = Transcript::new("deep-test", &[b"x"]);
            let mut tv = Transcript::new("deep-test", &[b"x"]);
            let mut chain = DeepChain::new(len);
            let mut ver = DeepVerify::new(n, k);
            // Bind (the prover on `t`, the verifier mirrors on `tv`).
            let (answers, _zetas, nonce) =
                chain.bind(&mut t, &table, params.bind_points, params.bind_grind, "dt");
            assert!(ver.bind(&mut tv, &answers, nonce, params.bind_grind, "dt"));
            assert_eq!(chain.claim(), ver.claim(), "bind claims must agree");
            // Rounds: shared challenges, mirrored transcripts, the table
            // folds at the prover's challenge.
            let mut cur = table.clone();
            for _ in 0..n {
                let (answers, msg, r, _zetas, nonce) =
                    chain.round(&mut t, &cur, m, params.fold_grind, "dt");
                let (rv, _) = ver
                    .round(&mut tv, msg, &answers, nonce, params.fold_grind, "dt")
                    .expect("deep round verifies");
                assert_eq!(r, rv, "the fold challenge must be shared");
                cur = fold_table(&cur, r);
            }
            assert_eq!(chain.claim(), ver.claim(), "rounds' claims must agree");
            // The final tail check passes on the honest tail.
            ver.final_check(&cur).expect("final check passes");
            // Tampering the tail breaks it.
            let mut bad = cur.clone();
            bad[0] += Gf128(1);
            assert!(ver.final_check(&bad).is_err());
            // Tampering one round's answer breaks the final check.
            let mut t2 = Transcript::new("deep-test", &[b"x"]);
            let mut tv2 = Transcript::new("deep-test", &[b"x"]);
            let mut chain2 = DeepChain::new(len);
            let mut ver2 = DeepVerify::new(n, k);
            let (answers, _z, nonce) =
                chain2.bind(&mut t2, &table, params.bind_points, params.bind_grind, "dt");
            assert!(ver2.bind(&mut tv2, &answers, nonce, params.bind_grind, "dt"));
            let mut cur2 = table.clone();
            let mut caught_early = false;
            for i in 0..n {
                let (mut answers, msg, r, _zetas, nonce) =
                    chain2.round(&mut t2, &cur2, m, params.fold_grind, "dt");
                if i == 0 {
                    answers[0] += Gf128(1); // lie about one answer
                }
                match ver2.round(&mut tv2, msg, &answers, nonce, params.fold_grind, "dt") {
                    // The lie can be caught at once: the tampered answer
                    // changes the absorbed state, so the ground nonce of
                    // the very round (or a later one) fails.
                    Err(_) => {
                        caught_early = true;
                        break;
                    }
                    Ok((r2, _)) => {
                        assert_eq!(r, r2);
                        cur2 = fold_table(&cur2, r);
                    }
                }
            }
            assert!(
                caught_early || ver2.final_check(&cur2).is_err(),
                "a lied answer must break the round mirror or the final check"
            );
        }
    }

    /// The ladder reaches the round-by-round target at every profiled
    /// rate, at the least work.
    #[test]
    fn deep_ladder_reaches_target() {
        for k in 1..=4usize {
            let (params, lines) = deep_ladder(k, 20, 128.0).expect("ladder at rate 1/2^k");
            assert!(params.samples >= 1 && params.samples <= MAX_SAMPLES);
            assert!(params.fold_grind.added_bits() <= 48.0);
            let line = &lines[0];
            assert!(
                line.sample_bits >= 128.0,
                "sample bits {line:?} must reach the target"
            );
            assert!(
                line.fold_error_bits >= 128.0,
                "post-grind fold error {line:?} must reach the target"
            );
        }
    }

    /// The grind threshold arithmetic (BEEP's `exact_at`).
    #[test]
    fn grind_exact_at() {
        assert_eq!(Grind::exact_at(1, 128.0), Grind::Off);
        // numerator·T < 2^64 for target 128: T = ⌊(2^64−1)/numerator⌋.
        let g = Grind::exact_at(1 << 20, 128.0);
        assert!(matches!(g, Grind::Below(_)));
        if let Grind::Below(t) = g {
            assert!(t < (1u64 << 44) && t >= (1u64 << 43));
        }
        assert_eq!(Grind::bits(0), Grind::Off);
        assert_eq!(Grind::bits(8).threshold(), 1u64 << 56);
    }

    /// The transcript PoW mirror: prover and verifier agree, and a wrong
    /// nonce fails.
    #[test]
    fn transcript_pow_mirror() {
        let mut t = Transcript::new("pow-test", &[b"x"]);
        let thr = 1u64 << 56; // 8 bits
        let nonce = t.grind_pow("site", thr);
        assert!(nonce > 0, "an 8-bit grind needs on average 256 tries");
        let mut v = Transcript::new("pow-test", &[b"x"]);
        assert!(v.verify_pow("site", nonce, thr));
        // A wrong nonce fails.
        let mut v2 = Transcript::new("pow-test", &[b"x"]);
        assert!(!v2.verify_pow("site", nonce.wrapping_add(1), thr));
        // The no-grind degenerate case: nonce 0 always passes.
        let mut t3 = Transcript::new("pow-test", &[b"x"]);
        assert_eq!(t3.grind_pow("any", u64::MAX), 0);
    }
}
