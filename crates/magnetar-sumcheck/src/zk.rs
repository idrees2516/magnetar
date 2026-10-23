//! Zero-knowledge masking for the sum-check layer (the roadmap's
//! Horizon-1 ZK item: "blinding the sum-check messages" in the Ligero /
//! Brakedown tradition).
//!
//! **The construction — round-local mask polynomials.** Each round the
//! prover adds to its honest message a fresh random polynomial
//! `z_i(X) = Σ_j z_{i,j}·X^j` of the round's degree, sampled subject to
//! the single running constraint `Σ_{j≥1} z_{i,j} = z_{i-1}(r_{i-1})` (and
//! `= 0` at round 0). The constraint is exactly what the verifier's round
//! check sees: the masked messages satisfy `Σ_{j≥1} m_{i,j} = running`
//! with the running updated by `m_i(r_i)` throughout, so the rounds
//! verify unchanged. After the last round the running claim carries the
//! honest terminal plus `z_n(r_n)`; the prover reveals that one value —
//! uniform, hiding nothing — as the **finish offset**, and the verifier's
//! terminal identity becomes `Σ terms·claims + offset = running`
//! ([`crate::engine::SumcheckVerifier::finish_offset`]).
//!
//! **Hiding.** Given the transcript so far, every coefficient of `z_i` is
//! uniform except the one the constraint fixes, so each masked message is
//! uniform over its constraint coset — the honest message is
//! information-theoretically hidden. The total extra randomness is
//! `O(degree)` field elements per round: no big masking tables, so this
//! composes with the sparse router prover over large joint spaces (where
//! a full multilinear mask would cost `2^27` entries).
//!
//! **What this masks and what it does not.** The masked sum-check hides
//! the *round messages*. Full end-to-end ZK additionally requires
//! masking the commitment-side revelations (the PCS query symbols) —
//! the masked-commitment design is documented in `docs/STATUS.md`, and
//! the revealed *claims* (dense-factor evaluations at random points, the
//! protocol's public claim chain) remain public by design, exactly as
//! in the paper's protocol.

use crate::{BindOrder, RoundMsg, SumcheckSpec};
use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;

/// Per-round zero-knowledge mask state (prover side).
pub struct ZkMasker {
    /// The previous round's constraint: `z_{i-1}(r_{i-1})` (zero at round
    /// 0 — the honest claim already satisfies the round-0 sum check).
    prev: Gf128,
    /// The current round's mask coefficients (for `commit`).
    z: Vec<Gf128>,
}

impl Default for ZkMasker {
    fn default() -> Self {
        Self::new()
    }
}

impl ZkMasker {
    pub fn new() -> Self {
        ZkMasker {
            prev: ZERO,
            z: Vec::new(),
        }
    }

    /// Mask one round message in place: add `z_i` sampled subject to the
    /// running constraint.
    pub fn mask(&mut self, msg: &mut RoundMsg) {
        self.mask_with(msg, &mut rand::thread_rng());
    }

    /// The mask sampling, over an injectable source (tests use seeded
    /// generators; production uses fresh thread randomness).
    pub fn mask_with(&mut self, msg: &mut RoundMsg, rng: &mut impl rand::Rng) {
        let d = msg.0.len().saturating_sub(1);
        let mut z = vec![ZERO; d + 1];
        // The constant coefficient never enters the round check — sample
        // it for full hiding.
        if d + 1 > 0 {
            z[0] = Gf128(rng.gen::<u128>());
        }
        // z_1 .. z_{d-1} free; z_d carries the constraint
        // Σ_{j≥1} z_j = prev.
        let mut acc = ZERO;
        for j in 1..d {
            z[j] = Gf128(rng.gen::<u128>());
            acc += z[j];
        }
        if d >= 1 {
            z[d] = self.prev + acc;
        }
        for (c, zc) in msg.0.iter_mut().zip(z.iter()) {
            *c += *zc;
        }
        self.z = z;
    }

    /// After the round's challenge is sampled, fold the mask's running
    /// contribution: `prev := z(r)`.
    pub fn commit(&mut self, r: Gf128) {
        let mut v = ZERO;
        let mut xp = ONE;
        for &c in &self.z {
            v += mul(c, xp);
            xp = mul(xp, r);
        }
        self.prev = v;
    }

    /// The finish offset: `z_n(r_n)` after the last `commit` — the one
    /// revealed mask value the verifier needs for the terminal identity.
    pub fn offset(&self) -> Gf128 {
        self.prev
    }
}

impl crate::engine::SumcheckProver {
    /// The ZK absorb-and-fold: mask the round message, absorb the masked
    /// message, sample the challenge, and fold. The unmasked message is
    /// reused for the internal fold bookkeeping; the *transcript* and
    /// the verifier carry the masked values throughout.
    pub fn zk_absorb_and_fold(
        &mut self,
        masker: &mut ZkMasker,
        t: &mut Transcript,
        label: &str,
    ) -> Gf128 {
        let mut msg = self.message_cached();
        masker.mask(&mut msg);
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        let r = t.challenge_field(label);
        masker.commit(r);
        self.fold(r);
        r
    }
}

impl crate::sparse::SparseProver {
    /// The ZK absorb-and-fold for the sparse router driver (identical
    /// masking; the sparse enumeration is untouched).
    pub fn zk_absorb_and_fold(
        &mut self,
        masker: &mut ZkMasker,
        t: &mut Transcript,
        label: &str,
    ) -> Gf128 {
        let mut msg = self.message();
        masker.mask(&mut msg);
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        let r = t.challenge_field(label);
        masker.commit(r);
        self.fold(r);
        r
    }
}

/// Build a uniformly random masking table of `2^num_vars` entries (used
/// by callers that blind per-round *tables* rather than messages — e.g.
/// a test-level sanity check of the hiding argument).
pub fn random_table(num_vars: usize, rng: &mut impl rand::Rng) -> Vec<Gf128> {
    (0..(1usize << num_vars))
        .map(|_| Gf128(rng.gen::<u128>()))
        .collect()
}

/// A convenience: run a fully masked dense sum-check, returning
/// `(rounds, dense_claims, finish_offset)` for the proof object.
pub fn zk_prove_dense(
    spec: SumcheckSpec,
    order: BindOrder,
    t: &mut Transcript,
    label: &str,
) -> (Vec<Vec<Gf128>>, Vec<Gf128>, Gf128) {
    let mut prover = crate::engine::SumcheckProver::new_owned(spec, order);
    let mut masker = ZkMasker::new();
    let mut rounds = Vec::with_capacity(prover.remaining_vars());
    for _ in 0..prover.remaining_vars() {
        let mut msg = prover.message_cached();
        masker.mask(&mut msg);
        rounds.push(msg.0.clone());
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        let r = t.challenge_field(label);
        masker.commit(r);
        prover.fold(r);
    }
    let (claims, _) = prover.finish();
    (rounds, claims, masker.offset())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public::EqFactor;
    use crate::Term;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rng.gen::<u128>())
    }

    /// A masked sum-check verifies under `finish_offset`, and the offset
    /// is exactly the masker's final value.
    #[test]
    fn masked_roundtrip_with_offset() {
        let mut rng = rand::thread_rng();
        let m = 5;
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
            terms: vec![Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![0],
            }],
            claim,
        };
        let mut t = Transcript::new("zk-sc", &[b"t"]);
        let (rounds, claims, offset) = zk_prove_dense(spec, BindOrder::Increasing, &mut t, "r");
        assert_ne!(offset, ZERO, "the offset should be a random-looking value");
        // Verify with the standard engine + the offset.
        let pubf: Vec<Box<dyn crate::public::PublicFactor + Send + Sync>> =
            vec![Box::new(EqFactor::new(tau.clone()))];
        let make_terms = || {
            vec![Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![0],
            }]
        };
        let mut tv = Transcript::new("zk-sc", &[b"t"]);
        let mut v = crate::engine::SumcheckVerifier::new(
            m,
            claim,
            &pubf,
            make_terms(),
            BindOrder::Increasing,
        );
        for msg in &rounds {
            v.round_capturing(&RoundMsg(msg.clone()), &mut tv, "r")
                .unwrap();
        }
        v.finish_offset(&claims, offset)
            .expect("the masked proof must verify with the offset");
        // Without the offset it must fail.
        let mut tv = Transcript::new("zk-sc", &[b"t"]);
        let mut v2 = crate::engine::SumcheckVerifier::new(
            m,
            claim,
            &pubf,
            make_terms(),
            BindOrder::Increasing,
        );
        for msg in &rounds {
            v2.round_capturing(&RoundMsg(msg.clone()), &mut tv, "r")
                .unwrap();
        }
        assert!(v2.finish(&claims).is_err(), "the plain finish must reject");
        // A tampered masked message must fail the round check.
        let mut bad = rounds.clone();
        bad[0][1] += Gf128(1);
        let mut tv = Transcript::new("zk-sc", &[b"t"]);
        let mut v3 = crate::engine::SumcheckVerifier::new(
            m,
            claim,
            &pubf,
            make_terms(),
            BindOrder::Increasing,
        );
        let mut failed = false;
        for msg in &bad {
            if v3
                .round_capturing(&RoundMsg(msg.clone()), &mut tv, "r")
                .is_err()
            {
                failed = true;
                break;
            }
        }
        assert!(failed, "the tampered masked message must be rejected");
    }

    /// The hiding smoke test: the same witness, two masked runs — the
    /// first round messages must differ (fresh mask randomness), while
    /// both verify.
    #[test]
    fn masks_fresh_per_run() {
        let mut rng = rand::thread_rng();
        let m = 4;
        let tau: Vec<Gf128> = (0..m).map(|_| rg(&mut rng)).collect();
        let table: Vec<Gf128> = (0..(1 << m)).map(|_| Gf128(7)).collect();
        let eqt = magnetar_field::EqBits::build(&tau).values;
        let mut claim = ZERO;
        for i in 0..(1 << m) {
            claim += mul(eqt[i], table[i]);
        }
        let make_spec = || SumcheckSpec {
            num_vars: m,
            dense: vec![table.clone()],
            public: vec![Box::new(EqFactor::new(tau.clone()))],
            terms: vec![Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![0],
            }],
            claim,
        };
        let mut t1 = Transcript::new("zk-h", &[b"t"]);
        let (r1, c1, o1) = zk_prove_dense(make_spec(), BindOrder::Increasing, &mut t1, "r");
        let mut t2 = Transcript::new("zk-h", &[b"t"]);
        let (r2, c2, o2) = zk_prove_dense(make_spec(), BindOrder::Increasing, &mut t2, "r");
        assert_ne!(r1[0], r2[0], "fresh masks must change the messages");
        assert_eq!(c1, c2, "the dense claims are the witness's real values");
        assert_ne!(o1, o2, "the offsets are fresh randomness");
        // Both verify.
        let pubf: Vec<Box<dyn crate::public::PublicFactor + Send + Sync>> =
            vec![Box::new(EqFactor::new(tau.clone()))];
        let make_terms = || {
            vec![Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![0],
            }]
        };
        for (rounds, claims, offset) in [(r1, c1, o1), (r2, c2, o2)] {
            let mut tv = Transcript::new("zk-h", &[b"t"]);
            let mut v = crate::engine::SumcheckVerifier::new(
                m,
                claim,
                &pubf,
                make_terms(),
                BindOrder::Increasing,
            );
            for msg in &rounds {
                v.round_capturing(&RoundMsg(msg.clone()), &mut tv, "r")
                    .unwrap();
            }
            v.finish_offset(&claims, offset).expect("must verify");
        }
    }

    /// The masker works on messages of any degree, and the constraint
    /// chain stays consistent across rounds.
    #[test]
    fn masker_constraint_chain() {
        let mut rng = rand::thread_rng();
        let mut masker = ZkMasker::new();
        let mut prev_sum = ZERO;
        for degree in [1usize, 2, 3, 8] {
            let mut msg = RoundMsg(vec![Gf128(5); degree + 1]);
            masker.mask_with(&mut msg, &mut rng);
            // The round check: Σ_{j≥1} (honest_j + z_j) = honest_sum +
            // prev. The honest part sums to `prev` (inductively), so the
            // masked sum must be ZERO difference against
            // `degree·5 + ...` — check the constraint directly:
            let mut s = ZERO;
            for c in &msg.0[1..] {
                s += *c;
            }
            // honest sum = (degree)·Gf128(5) + prev (the previous
            // masked running); we verify the z-constraint instead:
            let mut zs = ZERO;
            for zc in &masker.z[1..] {
                zs += *zc;
            }
            assert_eq!(zs, prev_sum, "the mask must satisfy the constraint");
            prev_sum = {
                let r = rg(&mut rng);
                let mut v = ZERO;
                let mut xp = ONE;
                for &c in &masker.z {
                    v += mul(c, xp);
                    xp = mul(xp, r);
                }
                masker.commit(r);
                masker.offset()
            };
        }
    }
}
