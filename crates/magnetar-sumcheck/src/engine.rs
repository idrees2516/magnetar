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
use magnetar_field::{mul, par_bind_var, Gf128, ZERO};
use magnetar_hash::Transcript;
use rayon::prelude::*;

/// Minimum remaining-cube size before the round message is computed with
/// chunked parallelism (R2). Field arithmetic reduces order-independently
/// (R1), so the parallel message is bit-identical to the sequential one.
const PAR_MSG_THRESHOLD: usize = 1 << 13;

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
        }
    }

    /// Number of remaining variables.
    pub fn remaining_vars(&self) -> usize {
        self.num_vars - self.round
    }

    /// Current round's message: coefficients of the round polynomial,
    /// computed by evaluating the summand at `X = 0, 1, …, d` on the
    /// remaining cube and interpolating.
    pub fn message(&self) -> RoundMsg {
        let degree = self
            .spec_terms
            .iter()
            .map(|t| t.dense.len() + t.public.len())
            .max()
            .unwrap_or(0);
        let npts = degree + 1;
        let mut values = Vec::with_capacity(npts);
        for i in 0..npts {
            let x = Gf128(i as u128);
            values.push(self.summand_sum_at(x));
        }
        RoundMsg(interpolate(&values))
    }

    /// `Σ_{rest} g(X, rest)` over the remaining Boolean cube, with the round
    /// variable fixed to the field value `X`.
    ///
    /// Parallel strategy (docs/PERFORMANCE.md R1/R3/R8): every factor's
    /// view is a disjoint chunk-wise map; each term's elementwise product
    /// is reduced per chunk and the chunk sums are XOR-combined — every
    /// reduction is over `+`/`×` (commutative, associative), so the result
    /// is bit-identical to the sequential computation regardless of thread
    /// scheduling.
    fn summand_sum_at(&self, x: Gf128) -> Gf128 {
        let bit = self.order.current_bit(self.num_vars, self.round);
        // Value of the MLE at (X, rest) = A + X·(A + T1) per entry.
        let view = |table: &Vec<Gf128>| -> Vec<Gf128> {
            let block = 1usize << bit;
            let out_len = table.len() / 2;
            if table.len() >= PAR_MSG_THRESHOLD {
                // One task per aligned span of pair-groups (R2/R8): the
                // chunk length is a multiple of `block`, so no group is
                // ever split across tasks.
                let threads = rayon::current_num_threads().max(1);
                let target = (out_len / (threads * 4)).clamp(block, out_len);
                let task = target.next_multiple_of(block).clamp(block, out_len);
                let mut buf = vec![ZERO; out_len];
                buf.par_chunks_mut(task)
                    .enumerate()
                    .for_each(|(ci, ochunk)| {
                        let groups = ochunk.len() / block;
                        let gi0 = (ci * task) / block;
                        for g in 0..groups {
                            let base = (gi0 + g) * 2 * block;
                            for j in 0..block {
                                let a = table[base + j];
                                let t1 = table[(base + j) | block];
                                ochunk[g * block + j] = a + mul(x, a + t1);
                            }
                        }
                    });
                buf
            } else {
                let mut out = Vec::with_capacity(out_len);
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
            }
        };
        let dense_views: Vec<Vec<Gf128>> = self.dense.iter().map(view).collect();
        let public_views: Vec<Vec<Gf128>> = self.public_tables.iter().map(view).collect();
        // Sum terms elementwise on the product of the views — chunked
        // with per-chunk XOR accumulators (R8: no shared mutable state).
        // The iteration count is the shortest view (the round variable is
        // already folded out of every view; zip-truncation semantics are
        // preserved for degenerate factors).
        let mut total = ZERO;
        for (ti, t) in self.spec_terms.iter().enumerate() {
            let coeff = self.coeffs[ti];
            let factor_views: Vec<&Vec<Gf128>> = t
                .dense
                .iter()
                .map(|&d| &dense_views[d])
                .chain(t.public.iter().map(|&p| &public_views[p]))
                .collect();
            if factor_views.is_empty() {
                if self.remaining_vars() == 1 {
                    total += coeff;
                }
                continue;
            }
            let n_rem = factor_views.iter().map(|f| f.len()).min().unwrap_or(0);
            if n_rem == 0 {
                continue;
            }
            if n_rem < PAR_MSG_THRESHOLD {
                // Sequential kernel (R2).
                let mut s = ZERO;
                for i in 0..n_rem {
                    let mut v = coeff;
                    for f in &factor_views {
                        v = mul(v, f[i]);
                    }
                    s += v;
                }
                total += s;
                continue;
            }
            // Per chunk: fold the elementwise product of all factor views
            // into one field element, then XOR-combine (R1: order-free).
            let chunk = (1usize << 15).min(n_rem);
            let terms_sum: Gf128 = (0..n_rem)
                .into_par_iter()
                .chunks(chunk)
                .map(|idxs| {
                    let mut s = ZERO;
                    for i in idxs {
                        let mut v = coeff;
                        for f in &factor_views {
                            v = mul(v, f[i]);
                        }
                        s += v;
                    }
                    s
                })
                .sum::<Gf128>();
            total += terms_sum;
        }
        total
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
        self.fold(r);
        r
    }

    /// Fold all tables at `r` (the round variable) and update the running
    /// claim to `m(r)`. Used by the lockstep driver with *external*
    /// challenges (Virtual Spartan, paper §5.3 step 1).
    pub fn fold(&mut self, r: Gf128) {
        self.challenges.push(r);
        let msg = self.message();
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
            par_bind_var(d, bit, r);
        }
        for p in self.public_tables.iter_mut() {
            par_bind_var(p, bit, r);
        }
        self.round += 1;
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
