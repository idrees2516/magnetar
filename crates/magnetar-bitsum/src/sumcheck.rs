//! Multilinear sumcheck over GF(2^128) with degree-1 (multilinear) rounds.
//!
//! # Statement
//!
//! Given a multilinear extension `f: {0,1}^m -> GF(2^128)` by its truth table
//! `evals` (length `2^m`), prove
//!
//! ```text
//! claimed_sum = XOR over b in {0,1}^m of f(b)
//! ```
//!
//! which is the *field sum* because the field has characteristic two. This is
//! the exact shape of Magnetar's MosaicPCS first-phase sum-check (a public
//! linear combination of table MLEs) and of Ligerito's proximity phase.
//!
//! # Round shape in characteristic two
//!
//! For a degree-1 round polynomial `g_r(x) = c0 + c1 * x`:
//!
//! ```text
//! g_r(0) ^ g_r(1) = c1      (so the verifier's round check is c1 == claim_r)
//! claim_{r+1}     = g_r(alpha_r) = c0 + c1 * alpha_r
//! bind:           f'[i] = f[i] + alpha_r * (f[i] + f[i + n/2])
//! ```
//!
//! The honest `c1` always equals the running claim, so the prover's real
//! message is `c0 = S0` (the XOR of the low half). Soundness is carried by
//! the final evaluation check (outer PCS binding): any false `c0` derails
//! the final claim with probability `1 - 2^-127` per round
//! (Schwartz-Zippel on the multilinear round polynomial), giving
//! `<= m / 2^128` total cheating probability for `m` rounds.
//!
//! # Layout convention
//!
//! Variable `x_r` is bound from the *most significant* table bit: round `r`
//! pairs table entries `(i, i + 2^{m-1-r})`. The returned evaluation point
//! is `[alpha_0, ..., alpha_{m-1}]` in binding order, and
//! [`eval_mle`] uses the matching convention.
//!
//! # Implementations
//!
//! * [`prove`] — scalar CPU reference (the Phase-1 CPU baseline).
//! * [`prove_planes`] — bitsliced CPU prover running the *exact* GPU op
//!   sequence (tree bind + plane reduction); used as the GPU parity oracle.
//! * [`verify`] — the verifier shared by all provers.

#![forbid(unsafe_code)]

use crate::bitslice::BitPlanes;
use crate::field::F128;
use crate::transcript::Transcript;
use crate::tree;

/// One round message: `g_r(x) = c0 + c1 * x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundMessage {
    /// Constant term (= `S0`, the XOR of the low half).
    pub c0: F128,
    /// Linear coefficient (= the running claim, for honest degree-1 rounds).
    pub c1: F128,
}

/// The sumcheck proof: one message per bound variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof {
    /// Number of variables `m` (rounds).
    pub num_vars: usize,
    /// Round messages in binding order.
    pub messages: Vec<RoundMessage>,
}

/// Everything a prover returns.
#[derive(Clone, Debug)]
pub struct ProveOutput {
    /// The proof itself.
    pub proof: SumcheckProof,
    /// The (recomputed) claimed sum: `XOR of all evals`.
    pub claimed_sum: F128,
    /// Challenges in binding order (`eval_point[r]` bound variable `r`).
    pub eval_point: Vec<F128>,
    /// The value of `f` at `eval_point`; must equal the verifier's final
    /// claim and is checked by the outer PCS.
    pub final_eval: F128,
}

/// Verifier outcome.
#[derive(Clone, Debug)]
pub struct VerifyOutcome {
    /// Drawn challenges in binding order.
    pub challenges: Vec<F128>,
    /// The evaluation point `f` is finally claimed at.
    pub eval_point: Vec<F128>,
    /// The final claim on `f(eval_point)` — hand this to the outer
    /// polynomial-evaluation check.
    pub final_claim: F128,
}

/// Prover-side failure modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProveError {
    /// The number of evaluations must be a power of two (and non-zero).
    NotPowerOfTwo(usize),
    /// Internal self-check failed (a bug, not a proof failure).
    SelfCheckFailed,
}

impl core::fmt::Display for ProveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProveError::NotPowerOfTwo(n) => {
                write!(f, "evaluation count {n} is not a power of two")
            }
            ProveError::SelfCheckFailed => {
                write!(f, "prover self-check failed (internal bug)")
            }
        }
    }
}

/// Verifier-side failure modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// `c1 != running claim` at the given round (the char-2 degree-1
    /// round check `g(0) ^ g(1) == claim` failed).
    RoundCheckFailed(usize),
    /// The proof is structurally inconsistent with its declared size.
    MalformedProof,
}

impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VerifyError::RoundCheckFailed(r) => {
                write!(f, "round {r} check failed: c1 != running claim")
            }
            VerifyError::MalformedProof => write!(f, "malformed proof"),
        }
    }
}

/// Evaluates the multilinear extension of `evals` at `point`.
///
/// Convention: `point[r]` is the challenge for variable `x_r`; variable `r`
/// corresponds to table bit `m-1-r` (MSB-first binding order).
#[must_use]
pub fn eval_mle(evals: &[F128], point: &[F128]) -> F128 {
    debug_assert_eq!(evals.len(), 1usize << point.len());
    // Lagrange over the hypercube in binding order: process variable 0 first
    // by folding pairs (i, i + n/2).
    let mut cur = evals.to_vec();
    for alpha in point.iter() {
        let half = cur.len() / 2;
        for i in 0..half {
            // f_r(alpha) = f_lo + alpha * (f_lo + f_hi)
            let lo = cur[i];
            let hi = cur[i + half];
            cur[i] = lo + *alpha * (lo + hi);
        }
        cur.truncate(half);
    }
    debug_assert_eq!(cur.len(), 1);
    cur[0]
}

/// Number of variables of a table of length `n` (None if not a power of 2).
#[must_use]
pub fn num_vars(n: usize) -> Option<usize> {
    if n == 0 || !n.is_power_of_two() {
        None
    } else {
        Some(n.trailing_zeros() as usize)
    }
}

/// Absorbs one round message and draws the binding challenge.
///
/// Shared verbatim by the CPU prover, the bitsliced prover, the GPU prover
/// and the verifier — one absorption order, everywhere.
pub fn absorb_round(t: &mut Transcript, msg: &RoundMessage) -> F128 {
    t.absorb_f128("sc-round-c0", &msg.c0);
    t.absorb_f128("sc-round-c1", &msg.c1);
    t.challenge_f128("sc-round-alpha")
}

/// Running prover state, shared by the planes and GPU provers so that the
/// protocol logic exists exactly once.
#[derive(Clone, Debug)]
pub struct ProverState {
    /// Running claim `C_r`.
    pub running: F128,
    /// Round messages so far.
    pub messages: Vec<RoundMessage>,
    /// Challenges so far.
    pub challenges: Vec<F128>,
}

impl ProverState {
    /// Fresh state from the initial claim.
    #[must_use]
    pub fn new(claim: F128) -> Self {
        Self {
            running: claim,
            messages: Vec::new(),
            challenges: Vec::new(),
        }
    }

    /// Pushes one round: message from `s0`, absorb, challenge, bind update.
    pub fn push_round(&mut self, s0: F128, t: &mut Transcript) -> F128 {
        let msg = RoundMessage {
            c0: s0,
            c1: self.running,
        };
        let alpha = absorb_round(t, &msg);
        self.messages.push(msg);
        self.challenges.push(alpha);
        self.running = msg.c0 + msg.c1 * alpha;
        alpha
    }
}

/// Finishes a prover from a scalar tail table (the remaining bound
/// evaluation table). Mutates `state` with the tail rounds and returns
/// `final_eval`.
pub fn prove_from_tail(
    tail: &[F128],
    t: &mut Transcript,
    state: &mut ProverState,
) -> Result<F128, ProveError> {
    debug_assert!(tail.len().is_power_of_two());
    let mut a = tail.to_vec();
    while a.len() >= 2 {
        let half = a.len() / 2;
        let mut s0 = F128::ZERO;
        for v in &a[..half] {
            s0 += *v;
        }
        let alpha = state.push_round(s0, t);
        for i in 0..half {
            let lo = a[i];
            let hi = a[i + half];
            a[i] = lo + alpha * (lo + hi);
        }
        a.truncate(half);
    }
    if a.len() == 1 && a[0] != state.running {
        return Err(ProveError::SelfCheckFailed);
    }
    Ok(if a.is_empty() { state.running } else { a[0] })
}

/// Scalar CPU reference prover (Phase-1 baseline).
pub fn prove(evals: &[F128], t: &mut Transcript) -> Result<ProveOutput, ProveError> {
    let m = num_vars(evals.len()).ok_or(ProveError::NotPowerOfTwo(evals.len()))?;
    t.absorb_u64("sc-num-vars", m as u64);

    let mut a = evals.to_vec();
    let mut claimed_sum = F128::ZERO;
    for v in &a {
        claimed_sum += *v;
    }

    let mut messages = Vec::with_capacity(m);
    let mut challenges = Vec::with_capacity(m);
    let mut running = claimed_sum;

    for _r in 0..m {
        let half = a.len() / 2;
        let mut s0 = F128::ZERO;
        for v in &a[..half] {
            s0 += *v;
        }
        let msg = RoundMessage {
            c0: s0,
            c1: running,
        };
        let alpha = absorb_round(t, &msg);

        messages.push(msg);
        challenges.push(alpha);
        running = msg.c0 + msg.c1 * alpha;

        for i in 0..half {
            let lo = a[i];
            let hi = a[i + half];
            a[i] = lo + alpha * (lo + hi);
        }
        a.truncate(half);
    }

    if a.len() == 1 && a[0] != running {
        return Err(ProveError::SelfCheckFailed);
    }
    Ok(ProveOutput {
        proof: SumcheckProof {
            num_vars: m,
            messages,
        },
        claimed_sum,
        eval_point: challenges,
        final_eval: if a.is_empty() { claimed_sum } else { a[0] },
    })
}

/// Bitsliced CPU prover: the *exact* GPU op sequence (pack once, then per
/// round: plane-reduce for `S0`, tree-multiply bind), finishing the tail
/// scalar. Proof is bit-identical to [`prove`] on the same transcript.
pub fn prove_planes(planes: &BitPlanes, t: &mut Transcript) -> Result<ProveOutput, ProveError> {
    let n = planes.groups * crate::bitslice::LANES;
    let m = num_vars(n).ok_or(ProveError::NotPowerOfTwo(n))?;
    t.absorb_u64("sc-num-vars", m as u64);

    // claimed sum: XOR over ALL groups
    let mut claimed_sum = F128::ZERO;
    for g in 0..planes.groups {
        claimed_sum += planes.group_sum(g);
    }

    let mut state = ProverState::new(claimed_sum);
    let mut cur: BitPlanes = planes.clone();
    while cur.groups >= 2 {
        let half_groups = cur.groups / 2;
        let s0 = tree::reduce_planes(&cur, half_groups);
        let alpha = state.push_round(s0, t);
        cur = tree::bind_planes(&cur, alpha);
    }

    // tail: single group (32 elements) unpacked and finished scalar
    let tail = cur.unpack();
    let final_eval = prove_from_tail(&tail, t, &mut state)?;

    Ok(ProveOutput {
        proof: SumcheckProof {
            num_vars: m,
            messages: state.messages,
        },
        claimed_sum,
        eval_point: state.challenges,
        final_eval,
    })
}

/// Verifier. Returns the challenges, the evaluation point and the final
/// claim to be checked by the outer PCS (`f(eval_point) == final_claim`).
pub fn verify(
    proof: &SumcheckProof,
    claimed_sum: F128,
    t: &mut Transcript,
) -> Result<VerifyOutcome, VerifyError> {
    if proof.num_vars != proof.messages.len() || proof.num_vars > 63 {
        return Err(VerifyError::MalformedProof);
    }
    t.absorb_u64("sc-num-vars", proof.num_vars as u64);
    let mut running = claimed_sum;
    let mut challenges = Vec::with_capacity(proof.num_vars);
    for (r, msg) in proof.messages.iter().enumerate() {
        // char-2 degree-1 identity: g(0) ^ g(1) = c1 must equal the claim.
        if msg.c1 != running {
            return Err(VerifyError::RoundCheckFailed(r));
        }
        let alpha = absorb_round(t, msg);
        challenges.push(alpha);
        running = msg.c0 + msg.c1 * alpha;
    }
    Ok(VerifyOutcome {
        challenges: challenges.clone(),
        eval_point: challenges,
        final_claim: running,
    })
}

/// Verifier with the outer evaluation check applied
/// (`f(eval_point) == final_claim` must hold for an honest proof).
pub fn verify_with_final_eval(
    proof: &SumcheckProof,
    claimed_sum: F128,
    final_eval: F128,
    t: &mut Transcript,
) -> Result<VerifyOutcome, VerifyError> {
    let outcome = verify(proof, claimed_sum, t)?;
    if outcome.final_claim != final_eval {
        return Err(VerifyError::RoundCheckFailed(proof.num_vars));
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SplitMix(u64);
    impl crate::Rng for SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x09E3_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn random_evals(m: usize, seed: u64) -> Vec<F128> {
        let mut r = SplitMix(seed);
        (0..(1usize << m)).map(|_| F128::random(&mut r)).collect()
    }

    #[test]
    fn end_to_end_cpu() {
        for m in [1usize, 2, 3, 5, 8, 10] {
            let evals = random_evals(m, 0x1000 + m as u64);
            let mut t = Transcript::new("e2e");
            let out = prove(&evals, &mut t).expect("prove");

            // claimed sum is the XOR of all entries
            let mut x = F128::ZERO;
            for v in &evals {
                x += *v;
            }
            assert_eq!(out.claimed_sum, x);

            // verify
            let mut tv = Transcript::new("e2e");
            let oc = verify(&out.proof, out.claimed_sum, &mut tv).expect("verify");
            // outer evaluation check
            assert_eq!(eval_mle(&evals, &oc.eval_point), oc.final_claim);
            assert_eq!(oc.final_claim, out.final_eval);
        }
    }

    #[test]
    fn zero_vars_edge() {
        let evals = vec![F128::from_u64(0x1234)];
        let mut t = Transcript::new("edge");
        let out = prove(&evals, &mut t).expect("prove");
        assert_eq!(out.proof.messages.len(), 0);
        assert_eq!(out.claimed_sum, evals[0]);
        let mut tv = Transcript::new("edge");
        let oc = verify(&out.proof, out.claimed_sum, &mut tv).expect("verify");
        assert_eq!(oc.final_claim, evals[0]);
        assert!(oc.eval_point.is_empty());
    }

    #[test]
    fn reject_bad_count() {
        let mut t = Transcript::new("bad");
        assert!(prove(&[F128::ONE; 3], &mut t).is_err());
        assert!(prove(&[], &mut t).is_err());
    }

    #[test]
    fn tampered_c1_fails_round_check() {
        let evals = random_evals(4, 42);
        let mut t = Transcript::new("tamper");
        let mut out = prove(&evals, &mut t).expect("prove");
        out.proof.messages[2].c1 += F128::ONE;
        let mut tv = Transcript::new("tamper");
        assert!(verify(&out.proof, out.claimed_sum, &mut tv).is_err());
    }

    #[test]
    fn tampered_midproof_c0_fails_round_check() {
        // Tampering c0 at a middle round changes every later challenge, so a
        // later round check (c1 == running claim) fails.
        let evals = random_evals(4, 43);
        let mut t = Transcript::new("tamper0");
        let mut out = prove(&evals, &mut t).expect("prove");
        out.proof.messages[1].c0 += F128::ONE;
        let mut tv = Transcript::new("tamper0");
        assert!(matches!(
            verify(&out.proof, out.claimed_sum, &mut tv),
            Err(VerifyError::RoundCheckFailed(_))
        ));
    }

    #[test]
    fn tampered_last_c0_fails_final_eval() {
        // Tampering the LAST round's c0 leaves all round checks passing
        // (c1's are untouched and no later rounds exist) but derails the
        // final claim, which the outer evaluation check catches.
        let evals = random_evals(4, 45);
        let mut t = Transcript::new("tamperlast");
        let mut out = prove(&evals, &mut t).expect("prove");
        let last = out.proof.messages.len() - 1;
        out.proof.messages[last].c0 += F128::ONE;
        let mut tv = Transcript::new("tamperlast");
        let oc = verify(&out.proof, out.claimed_sum, &mut tv).expect("round checks pass");
        assert_ne!(eval_mle(&evals, &oc.eval_point), oc.final_claim);
        assert_ne!(oc.final_claim, out.final_eval);
    }

    #[test]
    fn wrong_claimed_sum_fails() {
        let evals = random_evals(3, 44);
        let mut t = Transcript::new("wrong");
        let out = prove(&evals, &mut t).expect("prove");
        let mut tv = Transcript::new("wrong");
        let bad = out.claimed_sum + F128::ONE;
        // the honest proof under a false claim fails its first round check
        assert_eq!(
            verify(&out.proof, bad, &mut tv).unwrap_err(),
            VerifyError::RoundCheckFailed(0)
        );
    }

    #[test]
    fn planes_prover_bit_identical_to_scalar() {
        for m in [5usize, 6, 8] {
            let evals = random_evals(m, 0xabc + m as u64);
            let planes = BitPlanes::pack(&evals).expect("pack");
            let mut t1 = Transcript::new("parity");
            let mut t2 = Transcript::new("parity");
            let out_scalar = prove(&evals, &mut t1).expect("scalar");
            let out_planes = prove_planes(&planes, &mut t2).expect("planes");
            assert_eq!(out_scalar.proof, out_planes.proof, "m={m}");
            assert_eq!(out_scalar.claimed_sum, out_planes.claimed_sum);
            assert_eq!(out_scalar.eval_point, out_planes.eval_point);
            assert_eq!(out_scalar.final_eval, out_planes.final_eval);
        }
    }

    #[test]
    fn eval_mle_matches_direct_hypercube() {
        // small-case cross check of the binding-order convention
        let evals = random_evals(2, 77);
        let point = [F128::from_u64(5), F128::from_u64(9)];
        let got = eval_mle(&evals, &point);
        // direct: f(x0,x1) = sum_b f(b) * prod_i (bi ? x_i : 1+x_i), b0 = MSB
        let mut expect = F128::ZERO;
        for (b, &f_b) in evals.iter().enumerate() {
            let b0 = (b >> 1) & 1; // variable 0 <-> MSB of index
            let b1 = b & 1;
            let mut term = f_b;
            term *= if b0 == 1 {
                point[0]
            } else {
                point[0] + F128::ONE
            };
            term *= if b1 == 1 {
                point[1]
            } else {
                point[1] + F128::ONE
            };
            expect += term;
        }
        assert_eq!(got, expect);
    }
}
