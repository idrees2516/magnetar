//! Fiat-Shamir transcript for the bitsliced sum-check.
//!
//! This is a thin adapter over the stack's standard SHA3-256 duplex
//! ([`magnetar_hash::Transcript`]): one transcript construction — domain
//! separation, ratcheting, length-bound absorption — for the whole prover.
//! (The Phase-1 study originally carried its own SHA-256 transcript and
//! listed "SHA-256 → SHA-3 sponge behind the same interface" as Phase-2
//! work; landing in this repository does it now.)
//!
//! Design notes:
//!
//! * Challenges are 128-bit duplex squeezes reinterpreted as tower
//!   coordinate vectors: a binary field consumes the *entire* output with
//!   no rejection sampling and no distribution bias — every 128-bit string
//!   is a valid field element.
//! * The same transcript type is used by the CPU reference prover, the GPU
//!   prover (rounds on device, challenges on host), and the verifier, which
//!   is what makes GPU/CPU proofs bit-identical.
//! * The bit pattern of a challenge is basis-dependent (tower coordinates
//!   here, polynomial coefficients in `K`), so transcripts stay within one
//!   basis per protocol; see [`crate::interop`] for the field-level bridge.

#![forbid(unsafe_code)]

use crate::field::F128;

/// Domain-separated Fiat-Shamir transcript (SHA3-256 duplex with ratcheting,
/// from `magnetar-hash`).
#[derive(Clone)]
pub struct Transcript {
    inner: magnetar_hash::Transcript,
}

impl Transcript {
    /// Creates a fresh transcript for a bitsliced-sumcheck instance
    /// identified by `domain` (e.g. `"mosaic-first-phase"`). Prover and
    /// verifier must use identical domain strings.
    #[must_use]
    pub fn new(domain: &str) -> Self {
        Self {
            inner: magnetar_hash::Transcript::new("bitsum/sumcheck/v1", &[domain.as_bytes()]),
        }
    }

    /// Absorbs raw bytes under a label.
    pub fn absorb_bytes(&mut self, label: &str, data: &[u8]) {
        self.inner.absorb_bytes(label, data);
    }

    /// Absorbs a field element (little-endian tower-coordinate bytes) under
    /// a label.
    pub fn absorb_f128(&mut self, label: &str, value: &F128) {
        self.inner.absorb_bytes(label, &value.to_le_bytes());
    }

    /// Absorbs a public parameter (e.g. the instance's variable count).
    pub fn absorb_u64(&mut self, label: &str, value: u64) {
        self.inner.absorb_u64(label, value);
    }

    /// Draws a uniformly random field element under a label.
    ///
    /// Successive calls with the same label are differentiated by the
    /// duplex's internal squeeze counter, so the transcript is replay-safe
    /// as long as the label sequence is deterministic (which both prover and
    /// verifier code guarantee by construction).
    pub fn challenge_f128(&mut self, label: &str) -> F128 {
        F128(self.inner.challenge_field(label).0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_domain_separated() {
        let mut t1 = Transcript::new("test");
        let mut t2 = Transcript::new("test");
        let mut t3 = Transcript::new("other");
        let a = t1.challenge_f128("alpha");
        let b = t2.challenge_f128("alpha");
        let c = t3.challenge_f128("alpha");
        assert_eq!(a, b);
        assert_ne!(a, c);

        // absorbing changes subsequent challenges
        t1.absorb_f128("x", &F128::ONE);
        let d = t1.challenge_f128("alpha");
        let e = t2.challenge_f128("alpha");
        assert_ne!(d, e);
        assert_ne!(d, a);
    }

    #[test]
    fn distinct_labels_and_replay() {
        let mut t = Transcript::new("entropy");
        let x = t.challenge_f128("a");
        let y = t.challenge_f128("b");
        assert_ne!(x, y);
        // consecutive same-label draws are distinct (squeeze counter)
        let z = t.challenge_f128("a");
        assert_ne!(x, z);

        fn run(t: &mut Transcript) -> Vec<F128> {
            t.absorb_u64("n", 1024);
            let c0 = t.challenge_f128("alpha0");
            t.absorb_f128("msg", &F128::from_u64(0xdead));
            let c1 = t.challenge_f128("alpha1");
            vec![c0, c1]
        }
        let mut t1 = Transcript::new("replay");
        let mut t2 = Transcript::new("replay");
        assert_eq!(run(&mut t1), run(&mut t2));
    }
}
