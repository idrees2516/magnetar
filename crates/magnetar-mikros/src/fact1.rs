//! Fact 1 — the inner-product reduction (ePrint 2026/2646, §3.2).
//!
//! Over `log2(m)` rounds, the verifier samples challenges `y ∈ E^{log2 m}`
//! and the protocol reduces `v = 〈a, b〉` over `E` to the two terminal MLE
//! claims `v1 = 〈a, ts(y)〉`, `v2 = 〈b, ts(y)〉` ("the prover sends
//! v1, v2 ∈ E").
//!
//! **Round shape (degree 2).** With halves `a_lo|a_hi`, `b_lo|b_hi` and the
//! differences `ã = a_lo+a_hi`, `b̃ = b_lo+b_hi` (character 2), the round
//! polynomial is
//!
//! ```text
//! q(Z) = p0 + Z·(p1+p0+s) + Z²·s      p0 = 〈a_lo,b_lo〉, p1 = 〈a_hi,b_hi〉,
//!                                    s  = 〈ã, b̃〉
//! ```
//!
//! so the three-point message `[p0, p1, s]` (a) checks the Boolean sum
//! `p0 + p1 = running`, and (b) lets the verifier fold at `λ` without any
//! further prover values — `s` plays the role of the paper's folding cross
//! terms. Prover cost is 3 dots + 2 fold-passes per round ≈ 5m `E`-mults,
//! the same class as Fact 1's 4m bound.

use magnetar_field::mul;
use magnetar_field::Gf128;
use magnetar_hash::Transcript;

pub struct Fact1Proof {
    /// `[p0, p1, s]` per round.
    pub rounds: Vec<[Gf128; 3]>,
    /// Terminal MLE claims `(v1, v2)`.
    pub terminals: (Gf128, Gf128),
    /// The fold challenges (transcript-derived; carried for the prover's
    /// downstream use — the verifier re-derives its own and ignores these).
    pub challenges: Vec<Gf128>,
}

fn dot(a: &[Gf128], b: &[Gf128]) -> Gf128 {
    let mut acc = Gf128(0);
    for i in 0..a.len() {
        acc += mul(a[i], b[i]);
    }
    acc
}

/// Prove `v = 〈a, b〉` (the running claim must be supplied; the prover
/// asserts it). Returns the proof; the verifier derives `y` from the
/// transcript.
pub fn prove(
    a: &mut Vec<Gf128>,
    b: &mut Vec<Gf128>,
    claim: Gf128,
    t: &mut Transcript,
) -> Fact1Proof {
    let log_m = (a.len() as u128).trailing_zeros() as usize;
    debug_assert_eq!(a.len(), b.len());
    let mut running = claim;
    let mut rounds = Vec::with_capacity(log_m);
    let mut y = Vec::with_capacity(log_m);
    let mut len = a.len();
    while len > 1 {
        let half = len / 2;
        let (a_lo, a_hi) = a.split_at_mut(half);
        let (b_lo, b_hi) = b.split_at_mut(half);
        // messages
        let p0 = dot(a_lo, b_lo);
        let p1 = dot(&*a_hi, &*b_hi);
        let mut s = Gf128(0);
        for i in 0..half {
            s += mul(a_lo[i] + a_hi[i], b_lo[i] + b_hi[i]);
        }
        // NOTE: no prover-side assert `p0 + p1 == running` — `running` starts
        // at the CALLER's claim, and tamper tests deliberately feed claims
        // inconsistent with the vectors; the verifier's round check is the
        // soundness gate.
        t.absorb_field("fact1-msg", &[p0, p1, s]);
        let lambda = t.challenge_field("fact1-r");
        // fold in place
        for i in 0..half {
            a_lo[i] = a_lo[i] + mul(lambda, a_lo[i] + a_hi[i]);
            b_lo[i] = b_lo[i] + mul(lambda, b_lo[i] + b_hi[i]);
        }
        a.truncate(half);
        b.truncate(half);
        running = p0 + mul(lambda, p1 + p0 + s) + mul(mul(lambda, lambda), s);
        len = half;
        rounds.push([p0, p1, s]);
        y.push(lambda);
    }
    // terminal: running == a[0]·b[0]; prover sends (v1, v2)
    debug_assert_eq!(running, mul(a[0], b[0]), "terminal consistency");
    let terminals = (a[0], b[0]);
    t.absorb_field("fact1-term", &[terminals.0, terminals.1]);
    Fact1Proof {
        rounds,
        terminals,
        challenges: y,
    }
}

/// Verify: checks every round identity and the terminal product. Returns
/// the terminal point `y` and the two MLE claims.
pub fn verify(
    claim: Gf128,
    proof: &Fact1Proof,
    t: &mut Transcript,
) -> Result<(Vec<Gf128>, Gf128, Gf128), &'static str> {
    let mut running = claim;
    let mut y = Vec::with_capacity(proof.rounds.len());
    for &[p0, p1, s] in &proof.rounds {
        if p0 + p1 != running {
            return Err("fact1: round sum mismatch");
        }
        t.absorb_field("fact1-msg", &[p0, p1, s]);
        let lambda = t.challenge_field("fact1-r");
        running = p0 + mul(lambda, p1 + p0 + s) + mul(mul(lambda, lambda), s);
        y.push(lambda);
    }
    let (v1, v2) = proof.terminals;
    // MUST match prove's terminal absorb exactly (label + field layout):
    // batched protocols sample further challenges after this absorb, and the
    // label is part of the duplex's domain separation.
    t.absorb_field("fact1-term", &[v1, v2]);
    if running != mul(v1, v2) {
        return Err("fact1: terminal product mismatch");
    }
    Ok((y, v1, v2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_field::ZERO;

    fn test_transcript() -> Transcript {
        Transcript::new("mikros-fact1-test", &[])
    }

    #[test]
    fn roundtrip_and_tamper() {
        // length-32 vectors
        let mut r = 0x1234_5678_9abc_def0u128;
        let mut next = move || {
            r = r
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            Gf128(r)
        };
        let a0: Vec<Gf128> = (0..32).map(|_| next()).collect();
        let b0: Vec<Gf128> = (0..32).map(|_| next()).collect();
        let claim = dot(&a0, &b0);
        let mut t = test_transcript();
        let mut a = a0.clone();
        let mut b = b0.clone();
        let proof = prove(&mut a, &mut b, claim, &mut t);
        // verify
        let mut tv = test_transcript();
        let (y, v1, v2) = verify(claim, &proof, &mut tv).expect("verify");
        // terminal claims must equal the true MLE evaluations: with the fold
        // a ← a_lo + λ·ã, the final value is 〈a, ts(y)〉 (ts over the bound
        // variables, little-endian challenge order).
        let ts_eval = |v: &[Gf128], pts: &[Gf128]| -> Gf128 {
            let mut cur = v.to_vec();
            for (i, &p) in pts.iter().enumerate() {
                let half = cur.len() / 2;
                for k in 0..half {
                    cur[k] = cur[k] + mul(p, cur[k] + cur[k + half]);
                }
                cur.truncate(half);
                let _ = i;
            }
            cur[0]
        };
        assert_eq!(v1, ts_eval(&a0, &y), "v1 = <a, ts(y)>");
        assert_eq!(v2, ts_eval(&b0, &y), "v2 = <b, ts(y)>");

        // tamper: flip one round message coefficient
        for mut_idx in 0..proof.rounds.len() {
            let mut bad = Fact1Proof {
                rounds: proof.rounds.clone(),
                terminals: proof.terminals,
                challenges: proof.challenges.clone(),
            };
            bad.rounds[mut_idx][1] += Gf128(1);
            let mut tb = test_transcript();
            assert!(
                verify(claim, &bad, &mut tb).is_err(),
                "tamper round {mut_idx}"
            );
        }
        // tamper the terminal
        let bad = Fact1Proof {
            rounds: proof.rounds.clone(),
            terminals: (proof.terminals.0 + Gf128(1), proof.terminals.1),
            challenges: proof.challenges.clone(),
        };
        let mut tb = test_transcript();
        assert!(verify(claim, &bad, &mut tb).is_err(), "terminal tamper");
        // wrong claim
        let mut tw = test_transcript();
        assert!(
            verify(claim + Gf128(1), &proof, &mut tw).is_err(),
            "wrong claim"
        );
        let _ = ZERO;
    }
}
