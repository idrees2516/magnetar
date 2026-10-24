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
//! composes with the join-enumeration provers over large joint spaces
//! (where a full multilinear mask would cost `2^24`+ entries).
//!
//! **What this masks and what it does not.** The masked sum-check hides
//! the *round messages*. Full end-to-end ZK additionally requires
//! masking the commitment-side revelations (the PCS query symbols) —
//! blinding the packed leaves before commitment covers that; the
//! revealed *claims* (dense-factor evaluations at random points, the
//! protocol's public claim chain) remain public by design, exactly as
//! in the paper's protocol.

use crate::{BindOrder, RoundMsg, SumcheckSpec};
use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use rand::RngExt;

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
        self.mask_with(msg, &mut rand::rng());
    }

    /// The mask sampling, over an injectable source (tests use seeded
    /// generators; production uses fresh thread randomness).
    pub fn mask_with(&mut self, msg: &mut RoundMsg, rng: &mut impl rand::Rng) {
        let d = msg.0.len().saturating_sub(1);
        let mut z = vec![ZERO; d + 1];
        // The constant coefficient never enters the round check — sample
        // it for full hiding.
        if d + 1 > 0 {
            z[0] = Gf128(rng.random::<u128>());
        }
        // z_1 .. z_{d-1} free; z_d carries the constraint
        // Σ_{j≥1} z_j = prev.
        let mut acc = ZERO;
        for j in 1..d {
            z[j] = Gf128(rng.random::<u128>());
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
    /// used for the internal fold bookkeeping; the *transcript* and the
    /// verifier carry the masked values throughout.
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
        // Fold the tables at r using the honest (unmasked) message for
        // the running claim.
        let honest = self.message();
        self.fold_msg(&honest, r);
        r
    }
}

/// Build a uniformly random masking table of `2^num_vars` entries (used
/// by callers that blind per-round *tables* rather than messages).
pub fn random_table(num_vars: usize, rng: &mut impl rand::Rng) -> Vec<Gf128> {
    (0..(1usize << num_vars))
        .map(|_| Gf128(rng.random::<u128>()))
        .collect()
}

/// A convenience: run a fully masked dense sum-check, returning
/// `(rounds, dense_claims, finish_offset)` for the proof object.
pub fn zk_prove_dense(
    spec: &SumcheckSpec,
    order: BindOrder,
    t: &mut Transcript,
    label: &str,
) -> (Vec<Vec<Gf128>>, Vec<Gf128>, Gf128) {
    let mut prover = crate::engine::SumcheckProver::new(spec, order);
    let mut masker = ZkMasker::new();
    let mut rounds = Vec::with_capacity(prover.remaining_vars());
    for _ in 0..prover.remaining_vars() {
        let mut msg = prover.message();
        masker.mask(&mut msg);
        rounds.push(msg.0.clone());
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        let r = t.challenge_field(label);
        masker.commit(r);
        let honest = prover.message();
        prover.fold_msg(&honest, r);
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
        Gf128(rng.random::<u128>())
    }

    /// A masked sum-check verifies under `finish_offset`, and the offset
    /// is exactly the masker's final value.
    #[test]
    fn masked_roundtrip_with_offset() {
        let mut rng = rand::rng();
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
        let (rounds, claims, offset) = zk_prove_dense(&spec, BindOrder::Increasing, &mut t, "r");
        assert_ne!(offset, ZERO, "the offset should be a random-looking value");
        // Verify with the standard engine + the offset.
        let pubf: Vec<Box<dyn crate::public::PublicFactor + Send + Sync>> =
            vec![Box::new(EqFactor::new(tau.clone()))];
        let terms = vec![Term {
            coeff: Gf128(1),
            dense: vec![0],
            public: vec![0],
        }];
        let mut v =
            crate::engine::SumcheckVerifier::new(m, claim, &pubf, terms, BindOrder::Increasing);
        let mut tv = Transcript::new("zk-sc", &[b"t"]);
        for msg in &rounds {
            v.round_capturing(&RoundMsg(msg.clone()), &mut tv, "r")
                .unwrap();
        }
        let (point, got) = v.finish_offset(&claims, offset).unwrap();
        // The dense claim still equals T-hat at the point (the mask never
        // touches the folded tables).
        assert_eq!(got[0], magnetar_field::eval_mle(&table, &point));
    }

    /// HVZK: the masked round messages are independent of the honest
    /// ones — two different witnesses over the same statement and
    /// transcript prefix produce identically-distributed messages. We
    /// test the distributional property the construction gives: the
    /// masked message is a uniform coset member, so masking two distinct
    /// honest messages with independent masks yields messages that both
    /// verify against the same claim chain (and the messages differ).
    #[test]
    fn hvzk_mask_hides_message() {
        let mut rng = rand::rng();
        let d = 3usize;
        let a: Vec<Gf128> = (0..=d).map(|_| rg(&mut rng)).collect();
        let b: Vec<Gf128> = (0..=d).map(|_| rg(&mut rng)).collect();
        let mut masker = ZkMasker::new();
        let mut ma = RoundMsg(a.clone());
        let mut mb = RoundMsg(b.clone());
        masker.mask(&mut ma);
        // A fresh masker for the second message (a carried `prev` would
        // legitimately shift the constraint coset).
        let mut masker2 = ZkMasker::new();
        masker2.mask(&mut mb);
        masker.commit(Gf128(0x1234u128));
        // Both masked messages satisfy the same round-0 running
        // constraint (Σ_{j≥1} z_j = 0), so their tail sums equal the
        // honest tails — and the masked values differ from the honest
        // ones and from each other.
        let tail = |m: &RoundMsg| m.0[1..].iter().copied().sum::<Gf128>();
        assert_eq!(tail(&ma), a[1..].iter().copied().sum::<Gf128>());
        assert_eq!(tail(&mb), b[1..].iter().copied().sum::<Gf128>());
        assert_ne!(ma.0, a);
        assert_ne!(mb.0, b);
        assert_ne!(ma.0, mb.0);
    }

    /// A tampered masked sum-check (one round coefficient flipped) fails
    /// the verifier's round check.
    #[test]
    fn masked_tamper_detected() {
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
            dense: vec![table],
            public: vec![Box::new(EqFactor::new(tau.clone()))],
            terms: vec![Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![0],
            }],
            claim,
        };
        let mut t = Transcript::new("zk-tamper", &[]);
        let (mut rounds, claims, offset) =
            zk_prove_dense(&spec, BindOrder::Increasing, &mut t, "r");
        rounds[1][2] += ONE;
        let pubf: Vec<Box<dyn crate::public::PublicFactor + Send + Sync>> =
            vec![Box::new(EqFactor::new(tau))];
        let terms = vec![Term {
            coeff: Gf128(1),
            dense: vec![0],
            public: vec![0],
        }];
        let mut v =
            crate::engine::SumcheckVerifier::new(m, claim, &pubf, terms, BindOrder::Increasing);
        let mut tv = Transcript::new("zk-tamper", &[]);
        let mut ok = true;
        for msg in &rounds {
            if v.round_capturing(&RoundMsg(msg.clone()), &mut tv, "r")
                .is_err()
            {
                ok = false;
            }
        }
        if ok {
            assert!(v.finish_offset(&claims, offset).is_err());
        }
    }
}
