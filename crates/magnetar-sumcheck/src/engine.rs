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

use crate::{interpolate, BindOrder, RoundMsg, SumcheckSpec};
use magnetar_field::{bind_var, mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;

/// Below this many remaining entries the round passes run serially (the
/// rayon fork-join overhead dominates on small tables).
const PARALLEL_THRESHOLD: usize = 1 << 13;

/// Prover-side state: the folding tables plus the running claim.
pub struct SumcheckProver {
    spec_terms: Vec<crate::Term>,
    dense: Vec<Vec<Gf128>>,
    public_tables: Vec<Vec<Gf128>>,
    coeffs: Vec<Gf128>,
    running: Gf128,
    order: BindOrder,
    num_vars: usize,
    round: usize,
    challenges: Vec<Gf128>,
    /// The last computed round message, reused by `fold` so that the
    /// standard `message → absorb → fold` sequence computes the round once.
    cached_msg: Option<RoundMsg>,
}

impl SumcheckProver {
    pub fn new(spec: &SumcheckSpec, order: BindOrder) -> Self {
        spec.validate().expect("invalid sumcheck spec");
        let public_tables = spec.public.iter().map(|f| f.materialize()).collect();
        SumcheckProver {
            spec_terms: spec.terms.clone(),
            dense: spec.dense.clone(),
            public_tables,
            coeffs: spec.terms.iter().map(|t| t.coeff).collect(),
            running: spec.claim,
            order,
            num_vars: spec.num_vars,
            round: 0,
            challenges: Vec::new(),
            cached_msg: None,
        }
    }

    /// Build a prover **taking ownership** of the dense tables and terms —
    /// the hot-path constructor: no `O(2^m)` table copies. The public
    /// factors are materialized (they are folded like dense tables, so the
    /// prover needs its own copies regardless).
    pub fn new_owned(spec: SumcheckSpec, order: BindOrder) -> Self {
        spec.validate().expect("invalid sumcheck spec");
        let public_tables = spec.public.iter().map(|f| f.materialize()).collect();
        SumcheckProver {
            spec_terms: spec.terms,
            dense: spec.dense,
            public_tables,
            coeffs: Vec::new(),
            running: spec.claim,
            order,
            num_vars: spec.num_vars,
            round: 0,
            challenges: Vec::new(),
            cached_msg: None,
        }
        .with_coeffs()
    }

    fn with_coeffs(mut self) -> Self {
        self.coeffs = self.spec_terms.iter().map(|t| t.coeff).collect();
        self
    }

    /// Number of remaining variables.
    pub fn remaining_vars(&self) -> usize {
        self.num_vars - self.round
    }

    /// Current round's message: coefficients of the round polynomial.
    ///
    /// One fused pass over the half-table: for every remaining entry the
    /// per-factor slice pair `(a, d = a + a₁)` (the value at `X = 0` and
    /// its `X`-coefficient) yields the term's product at each node
    /// `X = 0..degree` with `degree` field multiplications per node — no
    /// intermediate views, no per-round allocations. The per-term `coeff`
    /// is applied once per node at the reduction, not per entry. Large
    /// tables are split across threads (the accumulators reduce at the
    /// end).
    pub fn message(&self) -> RoundMsg {
        if let Some(ref m) = self.cached_msg {
            return m.clone();
        }
        let degree = self
            .spec_terms
            .iter()
            .map(|t| t.dense.len() + t.public.len())
            .max()
            .unwrap_or(0);
        let npts = degree + 1;
        let bit = self.order.current_bit(self.num_vars, self.round);
        let block = 1usize << bit;
        let len = self
            .dense
            .first()
            .map(|d| d.len())
            .or_else(|| self.public_tables.first().map(|p| p.len()))
            .unwrap_or(0);
        let half = len / 2;
        let nterms = self.spec_terms.len();
        let terms = &self.spec_terms;
        let coeffs = &self.coeffs;
        let dense = &self.dense;
        let public = &self.public_tables;
        // The interpolation nodes X = 0, 1, 2, …, degree.
        let xs: Vec<Gf128> = (0..npts).map(|i| Gf128(i as u128)).collect();
        // Per (term, point) sums, possibly accumulated across threads.
        let mut acc = vec![ZERO; nterms * npts];
        // A factor-free (constant) term contributes `coeff` per Boolean
        // point of the post-round cube: `2^{remaining-1} · coeff`, nonzero
        // only on the final round (characteristic two).
        let const_contrib = self.remaining_vars() == 1;
        let term_has_factors = |t: &crate::Term| !t.dense.is_empty() || !t.public.is_empty();
        let run_range = |range: std::ops::Range<usize>, acc: &mut [Gf128]| {
            for n in range {
                // De-interleave: post-round entry `n` corresponds to the
                // pair (idx0, idx1) = (bit-0, bit-1) slices at the round bit.
                let lo = n & (block - 1);
                let hi = n >> bit;
                let idx0 = (hi << (bit + 1)) | lo;
                let j1 = idx0 | block;
                for (ti, t) in terms.iter().enumerate() {
                    if !term_has_factors(t) {
                        continue;
                    }
                    let base = ti * npts;
                    'points: for p in 0..npts {
                        let mut prod = ONE;
                        for &d in &t.dense {
                            let tab = &dense[d];
                            let a = tab[idx0];
                            let d1 = a + tab[j1];
                            let v = if p == 0 {
                                a
                            } else if p == 1 {
                                a + d1
                            } else {
                                a + mul(xs[p], d1)
                            };
                            prod = mul(prod, v);
                            if prod == ZERO {
                                continue 'points;
                            }
                        }
                        for &pf in &t.public {
                            let tab = &public[pf];
                            let a = tab[idx0];
                            let d1 = a + tab[j1];
                            let v = if p == 0 {
                                a
                            } else if p == 1 {
                                a + d1
                            } else {
                                a + mul(xs[p], d1)
                            };
                            prod = mul(prod, v);
                            if prod == ZERO {
                                continue 'points;
                            }
                        }
                        acc[base + p] += prod;
                    }
                }
            }
        };
        if half >= PARALLEL_THRESHOLD && nterms > 0 {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let merged = (0..half)
                .into_par_iter()
                .fold(
                    || vec![ZERO; nterms * npts],
                    |mut acc, j| {
                        run_range(j..j + 1, &mut acc);
                        acc
                    },
                )
                .reduce(
                    || vec![ZERO; nterms * npts],
                    |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b) {
                            *x += y;
                        }
                        a
                    },
                );
            for (i, v) in merged.into_iter().enumerate() {
                acc[i] += v;
            }
        } else {
            run_range(0..half, &mut acc);
        }
        // Scale by each term's coefficient and fold in the factor-free
        // constants.
        let mut values = Vec::with_capacity(npts);
        for p in 0..npts {
            let mut total = ZERO;
            for ti in 0..nterms {
                total += mul(coeffs[ti], acc[ti * npts + p]);
            }
            if const_contrib {
                for (ti, t) in terms.iter().enumerate() {
                    if !term_has_factors(t) {
                        total += coeffs[ti];
                    }
                }
            }
            values.push(total);
        }
        RoundMsg(interpolate(&values))
    }

    /// Compute and **cache** the current round message — for drivers that
    /// read the message before folding (`message → absorb_and_fold` and the
    /// Virtual-Spartan lockstep): the subsequent fold reuses it.
    pub fn message_cached(&mut self) -> RoundMsg {
        let m = self.message();
        self.cached_msg = Some(m.clone());
        m
    }

    /// Absorb the message into the transcript, check nothing (prover side),
    /// sample the challenge, fold, and advance. Returns the challenge.
    pub fn absorb_and_fold(&mut self, t: &mut Transcript, label: &str) -> Gf128 {
        let msg = self.message_cached();
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        let r = t.challenge_field(label);
        self.fold(r);
        r
    }

    /// Fold all tables at `r` (the round variable) and update the running
    /// claim to `m(r)`. Used by the lockstep driver with *external*
    /// challenges (Virtual Spartan, paper §5.3 step 1). Reuses the cached
    /// round message when the driver already computed it.
    pub fn fold(&mut self, r: Gf128) {
        self.challenges.push(r);
        let msg = self.cached_msg.take().unwrap_or_else(|| self.message());
        // running := m(r)
        let mut mr = ZERO;
        let mut xp = Gf128(1);
        for &c in &msg.0 {
            mr += mul(c, xp);
            xp = mul(xp, r);
        }
        self.running = mr;
        let bit = self.order.current_bit(self.num_vars, self.round);
        for d in self.dense.iter_mut() {
            bind_var(d, bit, r);
        }
        for p in self.public_tables.iter_mut() {
            bind_var(p, bit, r);
        }
        self.round += 1;
        self.cached_msg = None;
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

    /// The ZK-masked terminal check: the same identity as [`finish`]
    /// shifted by the revealed round-mask offset (`Σ terms + offset =
    /// running`) — see the `zk` module for the masking construction.
    pub fn finish_offset(
        &self,
        dense_claims: &[Gf128],
        offset: Gf128,
    ) -> Result<(Vec<Gf128>, Vec<Gf128>), &'static str> {
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
        let public_vals: Vec<Gf128> = self.spec_public.iter().map(|f| f.eval_at(&point)).collect();
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
        if total + offset != self.running {
            return Err("terminal identity check failed");
        }
        Ok((point, dense_claims.to_vec()))
    }

    /// After all rounds: check the terminal identity with the prover's
    /// dense-evaluation claims, returning the point (original coordinates)
    /// and the accepted claims.
    pub fn finish(&self, dense_claims: &[Gf128]) -> Result<(Vec<Gf128>, Vec<Gf128>), &'static str> {
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
        Gf128(rng.gen::<u128>())
    }

    fn bit_table(bits: &[bool]) -> Vec<Gf128> {
        bits.iter().map(|&b| Gf128(b as u128)).collect()
    }

    #[test]
    fn engine_roundtrip_simple() {
        // g(x) = eq(τ, x) · T(x), T a random bit table, claim = Σ eq·T.

        let mut rng = rand::thread_rng();
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
        let mut rng = rand::thread_rng();
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

        let mut rng = rand::thread_rng();
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

#[cfg(test)]
mod rewrite_tests {
    use super::*;
    use crate::public::{EqFactor, TableFactor};
    use magnetar_field::{mul, Gf128, ONE};

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rand::Rng::gen::<u128>(rng))
    }

    /// Reference (pre-rewrite) round message: evaluate the summand at
    /// X = 0..deg over the remaining cube, interpolating from per-point
    /// passes with explicit factor views.
    fn reference_message(
        dense: &[Vec<Gf128>],
        public_tables: &[Vec<Gf128>],
        terms: &[crate::Term],
        coeffs: &[Gf128],
        order: BindOrder,
        num_vars: usize,
        round: usize,
    ) -> RoundMsg {
        let degree = terms
            .iter()
            .map(|t| t.dense.len() + t.public.len())
            .max()
            .unwrap_or(0);
        let npts = degree + 1;
        let bit = order.current_bit(num_vars, round);
        let block = 1usize << bit;
        let view = |table: &Vec<Gf128>, x: Gf128| -> Vec<Gf128> {
            let mut out = Vec::with_capacity(table.len() / 2);
            let mut i = 0;
            while i < table.len() {
                for j in 0..block {
                    let a = table[i + j];
                    let t1 = table[(i + j) | block];
                    out.push(a + mul(x, a + t1));
                }
                i += 2 * block;
            }
            out
        };
        let mut values = Vec::with_capacity(npts);
        for p in 0..npts {
            let x = Gf128(p as u128);
            let dv: Vec<Vec<Gf128>> = dense.iter().map(|d| view(d, x)).collect();
            let pv: Vec<Vec<Gf128>> = public_tables.iter().map(|d| view(d, x)).collect();
            let mut total = ZERO;
            for (ti, t) in terms.iter().enumerate() {
                let mut prod: Option<Vec<Gf128>> = None;
                for &d in &t.dense {
                    prod = Some(match prod {
                        None => dv[d].clone(),
                        Some(p) => p
                            .iter()
                            .zip(dv[d].iter())
                            .map(|(&a, &b)| mul(a, b))
                            .collect(),
                    });
                }
                for &pf in &t.public {
                    prod = Some(match prod {
                        None => pv[pf].clone(),
                        Some(p) => p
                            .iter()
                            .zip(pv[pf].iter())
                            .map(|(&a, &b)| mul(a, b))
                            .collect(),
                    });
                }
                if let Some(p) = prod {
                    for &v in &p {
                        total += mul(coeffs[ti], v);
                    }
                }
            }
            values.push(total);
        }
        RoundMsg(interpolate(&values))
    }

    /// The rewrite must reproduce the reference messages round for round,
    /// serially and in the parallel regime, in both bind orders.
    #[test]
    fn rewrite_matches_reference_all_orders() {
        let mut rng = rand::thread_rng();
        for &(m, high) in &[(8, 3usize), (10, 0usize), (12, 5usize), (15, 6usize)] {
            let order = if high == 0 {
                BindOrder::Increasing
            } else {
                BindOrder::HighBitsFirst { high }
            };
            let d0: Vec<Gf128> = (0..(1usize << m)).map(|_| rg(&mut rng)).collect();
            let d1: Vec<Gf128> = (0..(1usize << m)).map(|_| rg(&mut rng)).collect();
            let d2: Vec<Gf128> = (0..(1usize << m)).map(|_| rg(&mut rng)).collect();
            let tau: Vec<Gf128> = (0..m).map(|_| rg(&mut rng)).collect();
            // Terms: (d0·d1·eq) + (d2·eq).
            let spec = SumcheckSpec {
                num_vars: m,
                dense: vec![d0.clone(), d1.clone(), d2.clone()],
                public: vec![Box::new(EqFactor::new(tau.clone()))],
                terms: vec![
                    crate::Term {
                        coeff: ONE,
                        dense: vec![0, 1],
                        public: vec![0],
                    },
                    crate::Term {
                        coeff: rg(&mut rng),
                        dense: vec![2],
                        public: vec![0],
                    },
                ],
                claim: ZERO,
            };
            let terms = spec.terms.clone();
            let coeffs: Vec<Gf128> = terms.iter().map(|t| t.coeff).collect();
            let pub_tables: Vec<Vec<Gf128>> = spec.public.iter().map(|f| f.materialize()).collect();
            let mut prover = SumcheckProver::new_owned(spec, order);
            // The reference keeps its own folded copies in lockstep.
            let mut ref_dense = vec![d0.clone(), d1.clone(), d2.clone()];
            let mut ref_pub = pub_tables.clone();
            for round in 0..m {
                let want =
                    reference_message(&ref_dense, &ref_pub, &terms, &coeffs, order, m, round);
                let got = prover.message();
                assert_eq!(got, want, "m={m} high={high} round={round}");
                // Advance with a fixed pseudo-random challenge.
                let rb = (round as u32).to_le_bytes();
                let mut tr = magnetar_hash::Transcript::new("t", &[&rb]);
                let r = tr.challenge_field("r");
                prover.fold(r);
                let bit = order.current_bit(m, round);
                for d in ref_dense.iter_mut() {
                    magnetar_field::bind_var(d, bit, r);
                }
                for p in ref_pub.iter_mut() {
                    magnetar_field::bind_var(p, bit, r);
                }
            }
        }
    }
}
