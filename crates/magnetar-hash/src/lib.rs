//! # magnetar-hash
//!
//! Hash layer for the Magnetar proof stack. Per the design requirement, the
//! system commits with **BLAKE3** (Merkle trees — the prover's dominant
//! hashing cost, where BLAKE3's SIMD throughput matters) and derives Fiat–
//! Shamir challenges with **SHA3-256** (Keccak, FIPS-202) — both standard,
//! well-scrutinized, non-algebraic hashes, matching the paper's post-quantum
//! "standard hashes only" requirement (§1.1).
//!
//! Everything is domain-separated and length-bound:
//! * every absorption is tagged with a protocol label;
//! * every Merkle leaf binds the tree domain, the tree size, and the leaf
//!   index (prevents cross-tree, size-confusion, and index-forgery attacks);
//! * the transcript ratchets its state after every squeeze.

#![forbid(unsafe_code)]
// Proof-system style: loops indexed over Boolean-cube coordinates and
// multi-input reductions are the domain's normal shape.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use blake3::Hasher as Blake3Hasher;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
use std::io::Write;

/// Digest size used throughout (256-bit → 128-bit collision resistance,
/// comfortably above the paper's 120-bit target).
pub const DIGEST: usize = 32;

/// A field element serialized little-endian (16 bytes).
#[inline]
pub fn fe_to_bytes(x: magnetar_field::Gf128) -> [u8; 16] {
    x.0.to_le_bytes()
}

/// Domain-separated one-shot SHA3-256 over labeled parts.
pub fn sha3_domain(parts: &[&[u8]]) -> [u8; DIGEST] {
    let mut h = Sha3_256::new();
    h.update((parts.len() as u64).to_le_bytes());
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p);
    }
    h.finalize().into()
}

/// Domain-separated one-shot BLAKE3 over labeled parts.
pub fn blake3_domain(parts: &[&[u8]]) -> [u8; DIGEST] {
    let mut h = Blake3Hasher::new();
    let _ = h.write(&(parts.len() as u64).to_le_bytes());
    for p in parts {
        let _ = h.write(&(p.len() as u64).to_le_bytes());
        let _ = h.write(p);
    }
    let mut out = [0u8; DIGEST];
    h.finalize_xof().fill(&mut out);
    out
}

/// Fiat–Shamir transcript (SHA3-256 duplex with ratcheting).
///
/// State `s` starts domain-bound; absorption sets `s = H(s ‖ label ‖ data)`;
/// squeezing sets `s = H(s ‖ "squeeze" ‖ ctr)` and returns `s`. After a
/// squeeze, further absorptions depend on all prior state — challenges are
/// bound to everything absorbed before them, and later messages are bound
/// to the challenges already drawn.
#[derive(Clone)]
pub struct Transcript {
    state: [u8; DIGEST],
    /// Number of field-element challenges drawn since the last absorption
    /// (used to make consecutive squeezes distinct).
    squeeze_ctr: u64,
}

impl Transcript {
    /// Create a transcript bound to a protocol identifier and its public
    /// parameters (deterministic derivation; no trusted setup).
    pub fn new(protocol: &str, public: &[&[u8]]) -> Self {
        let mut parts: Vec<&[u8]> = vec![b"magnetar/transcript/v1", protocol.as_bytes()];
        parts.extend_from_slice(public);
        Transcript {
            state: sha3_domain(&parts),
            squeeze_ctr: 0,
        }
    }

    /// Absorb labeled bytes.
    pub fn absorb_bytes(&mut self, label: &str, data: &[u8]) {
        self.state = sha3_domain(&[&self.state, b"absorb", label.as_bytes(), data]);
        self.squeeze_ctr = 0;
    }

    /// Absorb a labeled field element list.
    pub fn absorb_field(&mut self, label: &str, xs: &[magnetar_field::Gf128]) {
        let mut buf = Vec::with_capacity(xs.len() * 16);
        for &x in xs {
            buf.extend_from_slice(&fe_to_bytes(x));
        }
        self.absorb_bytes(label, &buf);
    }

    /// Absorb a labeled u64.
    pub fn absorb_u64(&mut self, label: &str, v: u64) {
        self.absorb_bytes(label, &v.to_le_bytes());
    }

    /// Absorb a digest (e.g. a Merkle root) under a label.
    pub fn absorb_digest(&mut self, label: &str, d: &[u8; DIGEST]) {
        self.absorb_bytes(label, d);
    }

    /// Draw one field-element challenge. Ratchets the state so two draws
    /// never repeat.
    pub fn challenge_field(&mut self, label: &str) -> magnetar_field::Gf128 {
        self.state = sha3_domain(&[
            &self.state,
            b"squeeze",
            label.as_bytes(),
            &self.squeeze_ctr.to_le_bytes(),
        ]);
        self.squeeze_ctr += 1;
        let mut lo = [0u8; 16];
        lo.copy_from_slice(&self.state[0..16]);
        magnetar_field::Gf128(u128::from_le_bytes(lo))
    }

    /// Draw `n` field-element challenges under one label.
    pub fn challenge_field_vec(&mut self, label: &str, n: usize) -> Vec<magnetar_field::Gf128> {
        (0..n).map(|_| self.challenge_field(label)).collect()
    }

    /// Draw `n` challenge *bits* packed into a u64 (used e.g. for query
    /// position selection). Uses fresh state per draw.
    pub fn challenge_u64(&mut self, label: &str) -> u64 {
        self.state = sha3_domain(&[
            &self.state,
            b"squeeze-u64",
            label.as_bytes(),
            &self.squeeze_ctr.to_le_bytes(),
        ]);
        self.squeeze_ctr += 1;
        u64::from_le_bytes(self.state[0..8].try_into().unwrap())
    }

    /// Uniform index in `[0, bound)` via rejection sampling on fresh
    /// challenges (no modulo bias for the accepted range).
    pub fn challenge_index(&mut self, label: &str, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        // Sample 64-bit values, reject >= largest multiple of bound.
        let limit = u64::MAX - (u64::MAX % bound) - 1; // inclusive bound
        loop {
            let v = self.challenge_u64(label);
            if v <= limit {
                return v % bound;
            }
        }
    }

    /// Current state digest (for testing / determinism checks).
    pub fn state_digest(&self) -> [u8; DIGEST] {
        self.state
    }
}

/// A Merkle tree over 16-byte field-element leaves, built with BLAKE3.
///
/// Leaf hash: `BLAKE3("magnetar/leaf/v1" ‖ domain ‖ n_leaves ‖ index ‖ value)`.
/// Node hash: `BLAKE3("magnetar/node/v1" ‖ domain ‖ level ‖ L ‖ R)`.
/// The domain + size + index binding defeats cross-tree substitution,
/// small-tree confusion, and index-forgery (duplicate-leaf) attacks; the
/// 128-bit collision resistance of BLAKE3-256 dominates the security.
/// Minimum leaves (or nodes) before a level is hashed in parallel (R2).
const PAR_TREE_THRESHOLD: usize = 1 << 13;

#[derive(Clone, Debug)]
pub struct MerkleTree {
    /// All levels, leaves first: levels[0] = leaf digests (n), levels[i] =
    /// level i's digests. Last level is the single root.
    levels: Vec<Vec<[u8; DIGEST]>>,
    domain: Vec<u8>,
    /// Number of leaves (power of two for the trees used in this stack).
    n_leaves: usize,
}

impl MerkleTree {
    /// The tree's domain-separation label.
    pub fn domain(&self) -> &[u8] {
        &self.domain
    }
    /// Build a tree over `leaves` (each 16 bytes, little-endian field
    /// elements) under a domain-separation string.
    ///
    /// Parallel (docs/PERFORMANCE.md R1/R5): the leaf level and every
    /// internal level are hashed with chunked parallelism; each level's
    /// digests depend only on the completed previous level, so the tree is
    /// built level-by-level (R5: one hash domain per level) and the result
    /// is bit-identical to the sequential build.
    pub fn build(domain: &str, leaves: &[magnetar_field::Gf128]) -> Self {
        let n = leaves.len();
        assert!(
            n.is_power_of_two() || n == 1,
            "Merkle leaves must be a power of two"
        );
        let dom = domain.as_bytes().to_vec();
        let leaf = |(i, &x): (usize, &magnetar_field::Gf128)| {
            blake3_domain(&[
                b"magnetar/leaf/v1",
                &dom,
                &(n as u64).to_le_bytes(),
                &(i as u64).to_le_bytes(),
                &fe_to_bytes(x),
            ])
        };
        let level: Vec<[u8; DIGEST]> = if n >= PAR_TREE_THRESHOLD {
            leaves.par_iter().enumerate().map(leaf).collect()
        } else {
            leaves.iter().enumerate().map(leaf).collect()
        };
        let mut levels = vec![level];
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let lvl = levels.len() - 1;
            let half = cur.len() / 2;
            let node = |j: usize| {
                blake3_domain(&[
                    b"magnetar/node/v1",
                    &dom,
                    &(lvl as u64).to_le_bytes(),
                    &cur[2 * j],
                    &cur[2 * j + 1],
                ])
            };
            let next: Vec<[u8; DIGEST]> = if half >= PAR_TREE_THRESHOLD {
                (0..half).into_par_iter().map(node).collect()
            } else {
                (0..half).map(node).collect()
            };
            levels.push(next);
        }
        MerkleTree {
            levels,
            domain: dom,
            n_leaves: n,
        }
    }

    /// The commitment (root).
    pub fn root(&self) -> [u8; DIGEST] {
        *self.levels.last().unwrap().first().unwrap()
    }

    /// Number of leaves.
    pub fn n_leaves(&self) -> usize {
        self.n_leaves
    }

    /// Open leaf `index`, returning the value digest and the sibling path
    /// (bottom-up).
    pub fn open(&self, index: usize) -> ([u8; DIGEST], Vec<[u8; DIGEST]>) {
        assert!(index < self.n_leaves);
        let mut path = Vec::with_capacity(self.levels.len() - 1);
        let mut idx = index;
        for level in &self.levels[..self.levels.len() - 1] {
            let sib = idx ^ 1;
            path.push(level[sib]);
            idx >>= 1;
        }
        (self.levels[0][index], path)
    }

    /// The digest of a leaf `value` (used by verifiers to recompute leaf
    /// hashes before path verification).
    pub fn leaf_digest_of(
        domain: &str,
        n_leaves: usize,
        index: usize,
        value: magnetar_field::Gf128,
    ) -> [u8; DIGEST] {
        blake3_domain(&[
            b"magnetar/leaf/v1",
            domain.as_bytes(),
            &(n_leaves as u64).to_le_bytes(),
            &(index as u64).to_le_bytes(),
            &fe_to_bytes(value),
        ])
    }

    /// Verify an opening. Returns the leaf digest on success.
    pub fn verify(
        domain: &str,
        n_leaves: usize,
        index: usize,
        leaf_digest: [u8; DIGEST],
        path: &[[u8; DIGEST]],
    ) -> Result<[u8; DIGEST], &'static str> {
        if !(n_leaves.is_power_of_two() || n_leaves == 1) {
            return Err("bad leaf count");
        }
        let expect_depth = if n_leaves == 1 {
            0
        } else {
            n_leaves.trailing_zeros() as usize
        };
        if path.len() != expect_depth {
            return Err("path length mismatch");
        }
        if index >= n_leaves {
            return Err("index out of range");
        }
        let dom = domain.as_bytes();
        let mut acc = leaf_digest;
        let mut idx = index;
        for (lvl, sib) in path.iter().enumerate() {
            let node = if idx & 1 == 0 {
                blake3_domain(&[
                    b"magnetar/node/v1",
                    dom,
                    &(lvl as u64).to_le_bytes(),
                    &acc,
                    sib,
                ])
            } else {
                blake3_domain(&[
                    b"magnetar/node/v1",
                    dom,
                    &(lvl as u64).to_le_bytes(),
                    sib,
                    &acc,
                ])
            };
            acc = node;
            idx >>= 1;
        }
        Ok(acc)
    }
}

/// Verify a full leaf opening (value level): recomputes the leaf digest for
/// a known leaf value and checks the path against the root.
pub fn verify_leaf(
    domain: &str,
    n_leaves: usize,
    index: usize,
    value: magnetar_field::Gf128,
    path: &[[u8; DIGEST]],
    root: &[u8; DIGEST],
) -> Result<(), &'static str> {
    let leaf = blake3_domain(&[
        b"magnetar/leaf/v1",
        domain.as_bytes(),
        &(n_leaves as u64).to_le_bytes(),
        &(index as u64).to_le_bytes(),
        &fe_to_bytes(value),
    ]);
    let acc = MerkleTree::verify(domain, n_leaves, index, leaf, path)?;
    if &acc == root {
        Ok(())
    } else {
        Err("merkle root mismatch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_is_deterministic_and_label_sensitive() {
        let mut t1 = Transcript::new("test", &[b"pub1"]);
        let mut t2 = Transcript::new("test", &[b"pub1"]);
        let mut t3 = Transcript::new("test", &[b"pub2"]);
        let c1 = t1.challenge_field("a");
        let c2 = t2.challenge_field("a");
        let c3 = t3.challenge_field("a");
        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
        // Label change → different challenge.
        let mut t4 = Transcript::new("test", &[b"pub1"]);
        let c4 = t4.challenge_field("b");
        assert_ne!(c1, c4);
        // Ratchet: two consecutive draws differ.
        let d1 = t1.challenge_field("a");
        let d2 = t1.challenge_field("a");
        assert_ne!(d1, d2);
        // Absorb order matters.
        let mut t5 = Transcript::new("test", &[b"pub1"]);
        t5.absorb_field("x", &[magnetar_field::Gf128(7)]);
        let e1 = t5.challenge_field("a");
        let mut t6 = Transcript::new("test", &[b"pub1"]);
        t6.absorb_field("y", &[magnetar_field::Gf128(7)]);
        let e2 = t6.challenge_field("a");
        assert_ne!(e1, e2);
    }

    #[test]
    fn merkle_open_verify() {
        use magnetar_field::Gf128;
        let leaves: Vec<Gf128> = (0..16u128).map(Gf128).collect();
        let tree = MerkleTree::build("mosaic", &leaves);
        let root = tree.root();
        for i in 0..16 {
            let (ld, path) = tree.open(i);
            let acc = MerkleTree::verify("mosaic", 16, i, ld, &path).unwrap();
            assert_eq!(&acc, &root);
            verify_leaf("mosaic", 16, i, leaves[i], &path, &root).unwrap();
        }
        // Tamper: wrong value must fail.
        let (_, path) = tree.open(3);
        assert!(verify_leaf("mosaic", 16, 3, Gf128(999), &path, &root).is_err());
        // Cross-domain substitution must fail.
        assert!(verify_leaf("other", 16, 3, leaves[3], &path, &root).is_err());
        // Index forgery must fail.
        assert!(verify_leaf("mosaic", 16, 4, leaves[3], &path, &root).is_err());
    }

    #[test]
    fn merkle_size_confusion_fails() {
        use magnetar_field::Gf128;
        let leaves: Vec<Gf128> = (0..8u128).map(Gf128).collect();
        let tree = MerkleTree::build("dom", &leaves);
        let root = tree.root();
        let (ld, path) = tree.open(1);
        // Claim a 4-leaf tree with the same root → must fail (path length
        // or digest mismatch).
        assert!(MerkleTree::verify("dom", 4, 1, ld, &path).is_err());
        let _ = root;
    }

    #[test]
    fn challenge_index_unbiased_small_range() {
        let mut t = Transcript::new("idx", &[]);
        let mut counts = [0u32; 7];
        for _ in 0..7000 {
            let v = t.challenge_index("q", 7);
            counts[v as usize] += 1;
        }
        for c in counts {
            assert!(c > 700, "distribution too skewed: {counts:?}");
        }
    }
}
