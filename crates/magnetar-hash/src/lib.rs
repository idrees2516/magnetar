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

/// Digest size used throughout (256-bit → 128-bit collision resistance,
/// comfortably above the paper's 120-bit target).
pub const DIGEST: usize = 32;

/// A field element serialized little-endian (16 bytes).
#[inline]
pub fn fe_to_bytes(x: magnetar_field::Gf128) -> [u8; 16] {
    x.0.to_le_bytes()
}

/// Domain-separated one-shot SHA3-256 over labeled parts.
///
/// The parts are serialized into ONE contiguous buffer first (the exact
/// byte stream the multi-update form would hash — one `update` call
/// instead of two per part; the output is bit-identical).
pub fn sha3_domain(parts: &[&[u8]]) -> [u8; DIGEST] {
    let mut h = Sha3_256::new();
    let mut buf =
        Vec::with_capacity(8 + parts.len() * 8 + parts.iter().map(|p| p.len()).sum::<usize>());
    serialize_parts(parts, &mut buf);
    h.update(&buf);
    h.finalize().into()
}

/// Serialize the labeled-parts byte stream: `[u64 nparts]` then per part
/// `[u64 len] [bytes]` — the exact stream the one-part-per-update form
/// feeds the hash.
#[inline]
fn serialize_parts(parts: &[&[u8]], buf: &mut Vec<u8>) {
    buf.extend_from_slice(&(parts.len() as u64).to_le_bytes());
    for p in parts {
        buf.extend_from_slice(&(p.len() as u64).to_le_bytes());
        buf.extend_from_slice(p);
    }
}

/// The prefix form: writes `[u64 total_nparts]` followed by only the
/// first parts' serializations — the caller appends the remaining parts'
/// `[len][bytes]` tails so the full stream is bit-identical to
/// `serialize_parts` over all `total_nparts` parts.
#[inline]
fn serialize_prefix(total_nparts: usize, parts: &[&[u8]], buf: &mut Vec<u8>) {
    debug_assert!(parts.len() <= total_nparts);
    buf.extend_from_slice(&(total_nparts as u64).to_le_bytes());
    for p in parts {
        buf.extend_from_slice(&(p.len() as u64).to_le_bytes());
        buf.extend_from_slice(p);
    }
}

/// Domain-separated one-shot BLAKE3 over labeled parts (single `update`).
pub fn blake3_domain(parts: &[&[u8]]) -> [u8; DIGEST] {
    let mut h = Blake3Hasher::new();
    let mut buf =
        Vec::with_capacity(8 + parts.len() * 8 + parts.iter().map(|p| p.len()).sum::<usize>());
    serialize_parts(parts, &mut buf);
    h.update(&buf);
    let mut out = [0u8; DIGEST];
    h.finalize_xof().fill(&mut out);
    out
}

/// One-shot BLAKE3 over a single pre-serialized byte stream (the hot
/// Merkle paths: the leaf/node prefixes are precomputed per tree/level,
/// and each hash is exactly one `update` + `finalize`).
#[inline]
fn blake3_one_shot(buf: &[u8]) -> [u8; DIGEST] {
    let mut h = Blake3Hasher::new();
    h.update(buf);
    *h.finalize().as_bytes()
}

/// The serialized leaf-hash prefix: `"magnetar/leaf/v1" ‖ domain ‖
/// n_leaves` — everything before the leaf's `index ‖ value` tail, so the
/// per-leaf hash is a 64-byte tail memcpy + one BLAKE3 compression.
pub(crate) struct LeafHashPrefix {
    buf: Vec<u8>,
}

/// The batched-leaf variant: `"magnetar/leafN/v1" ‖ domain ‖ n_values ‖
/// width` — each leaf digest covers `width` consecutive field elements
/// (the PCS fold-chain trees: one BLAKE3 call amortizes the per-leaf
/// fixed cost over 8 values, an ~8x cut of the fold-chain hashing at the
/// FastChunk profile).
pub(crate) struct BatchedLeafHashPrefix {
    buf: Vec<u8>,
    width: usize,
}

impl BatchedLeafHashPrefix {
    pub(crate) fn new(domain: &str, n_values: usize, width: usize) -> Self {
        let mut buf = Vec::with_capacity(112 + domain.len());
        serialize_prefix(
            5,
            &[
                b"magnetar/leafN/v1",
                domain.as_bytes(),
                &(n_values as u64).to_le_bytes(),
                &(width as u64).to_le_bytes(),
            ],
            &mut buf,
        );
        BatchedLeafHashPrefix { buf, width }
    }

    /// The leaf stream for group `group_index` covering `values` (exactly
    /// `width` elements): prefix ‖ width ‖ group_index ‖ byte_len ‖ values.
    pub(crate) fn hash_group(
        &self,
        group_index: usize,
        values: &[magnetar_field::Gf128],
    ) -> [u8; DIGEST] {
        debug_assert_eq!(values.len(), self.width);
        let bytes = 16 * self.width;
        let mut buf = Vec::with_capacity(self.buf.len() + 24 + bytes);
        buf.extend_from_slice(&self.buf);
        buf.extend_from_slice(&(self.width as u64).to_le_bytes());
        buf.extend_from_slice(&(group_index as u64).to_le_bytes());
        buf.extend_from_slice(&(bytes as u64).to_le_bytes());
        for v in values {
            buf.extend_from_slice(&fe_to_bytes(*v));
        }
        blake3_one_shot(&buf)
    }
}

impl LeafHashPrefix {
    pub(crate) fn new(domain: &str, n_leaves: usize) -> Self {
        let mut buf = Vec::with_capacity(96 + domain.len());
        serialize_prefix(
            5,
            &[
                b"magnetar/leaf/v1",
                domain.as_bytes(),
                &(n_leaves as u64).to_le_bytes(),
            ],
            &mut buf,
        );
        LeafHashPrefix { buf }
    }

    /// The full leaf stream for `(index, value)` in `out`; returns its length.
    #[inline]
    pub(crate) fn write_stream(
        &self,
        index: usize,
        value: magnetar_field::Gf128,
        out: &mut [u8; 160],
    ) -> usize {
        let pfx = self.buf.len();
        out[..pfx].copy_from_slice(&self.buf);
        let mut at = pfx;
        out[at..at + 8].copy_from_slice(&8u64.to_le_bytes());
        at += 8;
        out[at..at + 8].copy_from_slice(&(index as u64).to_le_bytes());
        at += 8;
        out[at..at + 8].copy_from_slice(&16u64.to_le_bytes());
        at += 8;
        out[at..at + 16].copy_from_slice(&fe_to_bytes(value));
        at += 16;
        at
    }

    #[inline]
    pub(crate) fn hash(&self, index: usize, value: magnetar_field::Gf128) -> [u8; DIGEST] {
        let mut buf = [0u8; 160];
        let at = self.write_stream(index, value, &mut buf);
        blake3_one_shot(&buf[..at])
    }
}

/// The serialized node-hash prefix: `"magnetar/node/v1" ‖ domain ‖ level`
/// (the level is fixed per build pass).
pub(crate) struct NodeHashPrefix {
    buf: Vec<u8>,
}

impl NodeHashPrefix {
    pub(crate) fn new(domain: &str, level: u64) -> Self {
        let mut buf = Vec::with_capacity(96 + domain.len());
        serialize_prefix(
            5,
            &[b"magnetar/node/v1", domain.as_bytes(), &level.to_le_bytes()],
            &mut buf,
        );
        NodeHashPrefix { buf }
    }

    #[inline]
    pub(crate) fn hash(&self, l: &[u8; DIGEST], r: &[u8; DIGEST]) -> [u8; DIGEST] {
        let pfx = self.buf.len();
        let mut buf = [0u8; 224];
        buf[..pfx].copy_from_slice(&self.buf);
        let mut at = pfx;
        buf[at..at + 8].copy_from_slice(&32u64.to_le_bytes());
        at += 8;
        buf[at..at + 32].copy_from_slice(l);
        at += 32;
        buf[at..at + 8].copy_from_slice(&32u64.to_le_bytes());
        at += 8;
        buf[at..at + 32].copy_from_slice(r);
        at += 32;
        blake3_one_shot(&buf[..at])
    }
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

    // ------------------------------------------------------------------
    // Proof-of-work grinding (the BEEP threshold discipline)
    // ------------------------------------------------------------------
    //
    // A *probe* is a detached digest of the current state and a nonce —
    // it never touches the transcript's own state, so a failed candidate
    // leaves no trace. A nonce *passes* when its probe is strictly below
    // `threshold`: a uniform probe passes with probability
    // `threshold / 2^64`, and `threshold == u64::MAX` accepts every nonce
    // (the no-grind degenerate case, one trivial hash of work).

    /// The detached proof-of-work probe for `nonce`: the first 8 bytes
    /// (little-endian) of a domain-separated digest of the current state,
    /// the label, and the nonce.
    pub fn pow_probe(&self, label: &str, nonce: u64) -> u64 {
        let d = sha3_domain(&[
            &self.state,
            b"pow",
            label.as_bytes(),
            &nonce.to_le_bytes(),
        ]);
        u64::from_le_bytes(d[0..8].try_into().unwrap())
    }

    /// Prover-side grinding: find the least passing nonce and absorb it.
    /// The accepted nonce enters the transcript, so every later draw is
    /// bound to it; the verifier mirrors this exactly. Returns the nonce.
    pub fn grind_pow(&mut self, label: &str, threshold: u64) -> u64 {
        let mut nonce = 0u64;
        if threshold != u64::MAX {
            while self.pow_probe(label, nonce) >= threshold {
                nonce = nonce.checked_add(1).expect("pow nonce space exhausted");
            }
        }
        self.absorb_u64(label, nonce);
        nonce
    }

    /// Verifier mirror of [`Self::grind_pow`]: check the nonce passes,
    /// then absorb it so the following draws agree with the prover's.
    pub fn verify_pow(&mut self, label: &str, nonce: u64, threshold: u64) -> bool {
        let ok = threshold == u64::MAX || self.pow_probe(label, nonce) < threshold;
        self.absorb_u64(label, nonce);
        ok
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

/// The batched-leaf width (values per BLAKE3 call) used by the PCS fold
/// trees. 8 keeps the leaf stream inside two BLAKE3 input blocks while
/// amortizing the per-call fixed cost over 8 values.
pub const BATCH_LEAF_WIDTH: usize = 8;

#[derive(Clone, Debug)]
pub struct MerkleTree {
    /// All levels, leaves first: levels[0] = leaf digests (n), levels[i] =
    /// level i's digests. Last level is the single root.
    levels: Vec<Vec<[u8; DIGEST]>>,
    domain: Vec<u8>,
    /// Number of leaves (power of two for the trees used in this stack).
    n_leaves: usize,
    /// Values per leaf group: 1 for the classic per-value trees, 8 for the
    /// batched PCS fold trees (n_leaves counts VALUES either way).
    width: usize,
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
        let leaf_pfx = LeafHashPrefix::new(domain, n);
        let leaf = |(i, &x): (usize, &magnetar_field::Gf128)| leaf_pfx.hash(i, x);
        let level: Vec<[u8; DIGEST]> = if n >= PAR_TREE_THRESHOLD {
            leaves.par_iter().enumerate().map(leaf).collect()
        } else {
            leaves.iter().enumerate().map(leaf).collect()
        };
        Self::finish_levels(domain, level, n, 1)
    }

    /// Build a batched-leaf tree: leaf i covers `values[i*width ..
    /// (i+1)*width]` (one BLAKE3 call per group). `values.len()` must be a
    /// power of two divisible by `width`; `n_leaves()` reports the VALUE
    /// count and `open(value_index)`/`verify_batched` address the group
    /// containing that value.
    pub fn build_batched(domain: &str, values: &[magnetar_field::Gf128], width: usize) -> Self {
        assert!(
            width == 1 || width.is_power_of_two(),
            "width must be 1 or a power of two"
        );
        let n = values.len();
        assert!(
            n.is_power_of_two(),
            "batched trees need a power-of-two value count"
        );
        assert_eq!(n % width, 0, "the groups must tile the value array");
        let n_groups = n / width;
        let leaf_pfx = BatchedLeafHashPrefix::new(domain, n, width);
        let level: Vec<[u8; DIGEST]> = if n_groups >= PAR_TREE_THRESHOLD {
            values
                .par_chunks_exact(width)
                .enumerate()
                .map(|(g, vals)| leaf_pfx.hash_group(g, vals))
                .collect()
        } else {
            values
                .chunks_exact(width)
                .enumerate()
                .map(|(g, vals)| leaf_pfx.hash_group(g, vals))
                .collect()
        };
        Self::finish_levels(domain, level, n, width)
    }

    /// The shared level-by-level node pass (R5: one hash domain per level).
    fn finish_levels(
        domain: &str,
        level: Vec<[u8; DIGEST]>,
        n_values: usize,
        width: usize,
    ) -> Self {
        let dom = domain.as_bytes().to_vec();
        let mut levels = vec![level];
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let lvl = levels.len() - 1;
            let half = cur.len() / 2;
            let node_pfx = NodeHashPrefix::new(domain, lvl as u64);
            let node = |j: usize| node_pfx.hash(&cur[2 * j], &cur[2 * j + 1]);
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
            n_leaves: n_values,
            width,
        }
    }

    /// The leaf-group width (1 for the classic per-value trees).
    pub fn width(&self) -> usize {
        self.width
    }

    /// The commitment (root).
    pub fn root(&self) -> [u8; DIGEST] {
        *self.levels.last().unwrap().first().unwrap()
    }

    /// Number of leaves.
    pub fn n_leaves(&self) -> usize {
        self.n_leaves
    }

    /// Open the leaf covering VALUE `index`, returning the group digest
    /// and the sibling path (bottom-up). For width-1 trees this is the
    /// classic per-value opening; for batched trees the digest is the
    /// GROUP's (the caller reveals the group's values alongside).
    pub fn open(&self, index: usize) -> ([u8; DIGEST], Vec<[u8; DIGEST]>) {
        assert!(index < self.n_leaves);
        let mut path = Vec::with_capacity(self.levels.len() - 1);
        let mut idx = if self.width == 1 {
            index
        } else {
            index / self.width
        };
        for level in &self.levels[..self.levels.len() - 1] {
            let sib = idx ^ 1;
            path.push(level[sib]);
            idx >>= 1;
        }
        (
            self.levels[0][if self.width == 1 {
                index
            } else {
                index / self.width
            }],
            path,
        )
    }

    /// The digest of a leaf `value` (used by verifiers to recompute leaf
    /// hashes before path verification).
    pub fn leaf_digest_of(
        domain: &str,
        n_leaves: usize,
        index: usize,
        value: magnetar_field::Gf128,
    ) -> [u8; DIGEST] {
        LeafHashPrefix::new(domain, n_leaves).hash(index, value)
    }

    /// The digest of the batched leaf covering VALUE `value_index` — the
    /// full group `values` (exactly `width` elements) is hashed in one
    /// BLAKE3 call.
    pub fn leaf_digest_of_batched(
        domain: &str,
        n_values: usize,
        width: usize,
        value_index: usize,
        values: &[magnetar_field::Gf128],
    ) -> [u8; DIGEST] {
        assert_eq!(
            values.len(),
            width,
            "the batched leaf covers `width` values"
        );
        BatchedLeafHashPrefix::new(domain, n_values, width).hash_group(value_index / width, values)
    }

    /// Verify a batched-leaf opening for VALUE `value_index` (its group's
    /// digest walked up `path`). Returns the recomputed root on success.
    pub fn verify_batched(
        domain: &str,
        n_values: usize,
        width: usize,
        value_index: usize,
        leaf_digest: [u8; DIGEST],
        path: &[[u8; DIGEST]],
    ) -> Result<[u8; DIGEST], &'static str> {
        if !n_values.is_power_of_two() || n_values == 0 {
            return Err("bad value count");
        }
        if width == 1 || width.is_power_of_two() {
            // ok
        } else {
            return Err("bad width");
        }
        if !n_values.is_multiple_of(width) {
            return Err("width does not tile");
        }
        let n_groups = n_values / width;
        if value_index >= n_values {
            return Err("index out of range");
        }
        let expect_depth = if n_groups == 1 {
            0
        } else {
            n_groups.trailing_zeros() as usize
        };
        if path.len() != expect_depth {
            return Err("path length mismatch");
        }
        let group = value_index / width;
        let mut acc = leaf_digest;
        let mut idx = group;
        for (lvl, sib) in path.iter().enumerate() {
            let node_pfx = NodeHashPrefix::new(domain, lvl as u64);
            let node = if idx & 1 == 0 {
                node_pfx.hash(&acc, sib)
            } else {
                node_pfx.hash(sib, &acc)
            };
            acc = node;
            idx >>= 1;
        }
        Ok(acc)
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
        let mut acc = leaf_digest;
        let mut idx = index;
        for (lvl, sib) in path.iter().enumerate() {
            let node_pfx = NodeHashPrefix::new(domain, lvl as u64);
            let node = if idx & 1 == 0 {
                node_pfx.hash(&acc, sib)
            } else {
                node_pfx.hash(sib, &acc)
            };
            acc = node;
            idx >>= 1;
        }
        Ok(acc)
    }

    /// Build a tree over **32-byte pair leaves** — each leaf is two field
    /// elements (the split-ring's merged `(P, C)` codeword entries, the
    /// per-segment commitment trees' leaves). The leaf digest binds the
    /// pair's 32 bytes (the parts-length prefix keeps it distinct from
    /// the 16-byte single-element leaf).
    pub fn build_pairs(
        domain: &str,
        pairs: &[(magnetar_field::Gf128, magnetar_field::Gf128)],
    ) -> Self {
        let n = pairs.len();
        assert!(
            n.is_power_of_two() || n == 1,
            "Merkle leaves must be a power of two"
        );
        let dom = domain.as_bytes().to_vec();
        let leaf = |(i, &(a, b)): (usize, &(magnetar_field::Gf128, magnetar_field::Gf128))| {
            leaf_digest_of_pair(domain, n, i, (a, b))
        };
        let level: Vec<[u8; DIGEST]> = if n >= PAR_TREE_THRESHOLD {
            pairs.par_iter().enumerate().map(leaf).collect()
        } else {
            pairs.iter().enumerate().map(leaf).collect()
        };
        Self::build_from_levels(dom, level, n)
    }

    /// Build a tree whose leaves are **raw digests** (the root layer over
    /// the per-cell subtree roots of the split ring).
    pub fn build_from_digests(domain: &str, leaves: &[[u8; DIGEST]]) -> Self {
        let n = leaves.len();
        assert!(
            n.is_power_of_two() || n == 1,
            "Merkle leaves must be a power of two"
        );
        let dom = domain.as_bytes().to_vec();
        let leaf = |(i, &d): (usize, &[u8; DIGEST])| leaf_digest_of_raw(domain, n, i, d);
        let level: Vec<[u8; DIGEST]> = if n >= PAR_TREE_THRESHOLD {
            leaves.par_iter().enumerate().map(leaf).collect()
        } else {
            leaves.iter().enumerate().map(leaf).collect()
        };
        Self::build_from_levels(dom, level, n)
    }

    /// The shared level-by-level construction from a precomputed leaf
    /// digest level.
    fn build_from_levels(dom: Vec<u8>, level: Vec<[u8; DIGEST]>, n_leaves: usize) -> Self {
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
            n_leaves,
            width: 1,
        }
    }
}

/// The leaf digest of a 32-byte **pair** value (two field elements — the
/// split ring's merged `(P, C)` leaves).
pub fn leaf_digest_of_pair(
    domain: &str,
    n_leaves: usize,
    index: usize,
    value: (magnetar_field::Gf128, magnetar_field::Gf128),
) -> [u8; DIGEST] {
    blake3_domain(&[
        b"magnetar/leaf/v1",
        domain.as_bytes(),
        &(n_leaves as u64).to_le_bytes(),
        &(index as u64).to_le_bytes(),
        &fe_to_bytes(value.0),
        &fe_to_bytes(value.1),
    ])
}

/// The leaf digest of a raw 32-byte value (the root layer's digest
/// leaves).
pub fn leaf_digest_of_raw(
    domain: &str,
    n_leaves: usize,
    index: usize,
    value: [u8; DIGEST],
) -> [u8; DIGEST] {
    blake3_domain(&[
        b"magnetar/leaf/v1",
        domain.as_bytes(),
        &(n_leaves as u64).to_le_bytes(),
        &(index as u64).to_le_bytes(),
        &value,
    ])
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
    let leaf = MerkleTree::leaf_digest_of(domain, n_leaves, index, value);
    let acc = MerkleTree::verify(domain, n_leaves, index, leaf, path)?;
    if &acc == root {
        Ok(())
    } else {
        Err("merkle root mismatch")
    }
}

// ---------------------------------------------------------------------------
// The batched multi-opening (the Canopy-class opening compression): several
// value indices authenticated against ONE root with the UNION copath —
// every root-path sibling serialized once, digest-only, in canonical
// (level, position) order the verifier recomputes from the indices.
// ---------------------------------------------------------------------------

/// A batched multi-opening of a set of VALUE indices.
#[derive(Clone, Debug)]
pub struct MultiOpen {
    /// The deduplicated, ascending covering group indices (level 0
    /// positions of the queried values' leaf groups).
    pub groups: Vec<usize>,
    /// The covering groups' leaf digests (parallel to `groups`).
    pub group_digests: Vec<[u8; DIGEST]>,
    /// The union copath positions `(level, position)`, ascending by level
    /// then position — the serialization order of `siblings`; the
    /// verifier recomputes this list from `groups`, so the proof carries
    /// digests only, no per-node metadata.
    pub copath: Vec<(usize, usize)>,
    /// The sibling digests (parallel to `copath`).
    pub siblings: Vec<[u8; DIGEST]>,
}

impl MerkleTree {
    /// Open several VALUE indices at once: the union of their root-path
    /// siblings. For `q` queries over a depth-`d` tree this serializes
    /// `|copath| ≤ q·d` digests (the shared top levels counted once)
    /// instead of `q·d` — the correlated-opening compression.
    pub fn open_multi(&self, value_indices: &[usize]) -> MultiOpen {
        assert!(!value_indices.is_empty(), "multi-open needs indices");
        let mut groups: Vec<usize> = value_indices
            .iter()
            .map(|&i| {
                assert!(i < self.n_leaves, "multi-open index out of range");
                if self.width == 1 { i } else { i / self.width }
            })
            .collect();
        groups.sort_unstable();
        groups.dedup();
        let group_digests = groups.iter().map(|&g| self.levels[0][g]).collect();
        let depth = self.levels.len() - 1;
        let mut seen = std::collections::BTreeSet::new();
        for &g in &groups {
            let mut idx = g;
            for lvl in 0..depth {
                seen.insert((lvl, idx ^ 1));
                idx >>= 1;
            }
        }
        let copath: Vec<(usize, usize)> = seen.into_iter().collect();
        let siblings = copath
            .iter()
            .map(|&(l, pos)| self.levels[l][pos])
            .collect();
        MultiOpen { groups, group_digests, copath, siblings }
    }

    /// Verify a batched multi-opening for VALUE indices: the caller
    /// supplies each covering group's values (exactly `width` per group,
    /// in ascending group order) and the union copath digests in the
    /// canonical (level, position) order recomputed from the indices.
    /// The leaf digests are hashed HERE from the revealed values, the
    /// root is reconstructed by pairing known nodes level by level, and
    /// the recomputed root is returned for the caller's comparison with
    /// the commitment — the standard Merkle binding: any deviation of a
    /// revealed group from the committed leaf breaks the reconstruction
    /// exactly as an individual opening would.
    pub fn verify_multi_batched(
        domain: &str,
        n_values: usize,
        width: usize,
        value_indices: &[usize],
        group_values: &[Vec<magnetar_field::Gf128>],
        copath: &[[u8; DIGEST]],
    ) -> Result<[u8; DIGEST], &'static str> {
        if !n_values.is_power_of_two() || n_values == 0 {
            return Err("multi: bad value count");
        }
        if !(width == 1 || width.is_power_of_two()) {
            return Err("multi: bad width");
        }
        if !n_values.is_multiple_of(width) {
            return Err("multi: width does not tile");
        }
        if value_indices.is_empty() {
            return Err("multi: no indices");
        }
        for &i in value_indices {
            if i >= n_values {
                return Err("multi: index out of range");
            }
        }
        // The expected group set (ascending, dedup).
        let mut want_groups: Vec<usize> =
            value_indices.iter().map(|&i| i / width).collect();
        want_groups.sort_unstable();
        want_groups.dedup();
        if group_values.len() != want_groups.len() {
            return Err("multi: group value count mismatch");
        }
        // The expected union copath positions (the canonical order the
        // digests must arrive in).
        let n_groups = n_values / width;
        let depth = if n_groups == 1 {
            0
        } else {
            n_groups.trailing_zeros() as usize
        };
        let mut seen = std::collections::BTreeSet::new();
        for &g in &want_groups {
            let mut idx = g;
            for lvl in 0..depth {
                seen.insert((lvl, idx ^ 1));
                idx >>= 1;
            }
        }
        let want_copath: Vec<(usize, usize)> = seen.into_iter().collect();
        if copath.len() != want_copath.len() {
            return Err("multi: copath length mismatch");
        }
        // Hash the revealed groups into the leaf level.
        let mut known: std::collections::BTreeMap<(usize, usize), [u8; DIGEST]> =
            std::collections::BTreeMap::new();
        for (gi, &g) in want_groups.iter().enumerate() {
            if group_values[gi].len() != width {
                return Err("multi: group width mismatch");
            }
            let d = if width == 1 {
                LeafHashPrefix::new(domain, n_values).hash(g, group_values[gi][0])
            } else {
                BatchedLeafHashPrefix::new(domain, n_values, width)
                    .hash_group(g, &group_values[gi])
            };
            known.insert((0, g), d);
        }
        for (k, &(l, pos)) in want_copath.iter().enumerate() {
            known.insert((l, pos), copath[k]);
        }
        // Reconstruct upward.
        for lvl in 0..depth {
            let at_level: Vec<usize> = known
                .keys()
                .filter(|&&(l, _)| l == lvl)
                .map(|&(_, p)| p)
                .collect();
            for p in at_level {
                if p % 2 == 1 {
                    continue; // handled from the even side
                }
                let sib = p + 1;
                if let Some(&right) = known.get(&(lvl, sib)) {
                    let node_pfx = NodeHashPrefix::new(domain, lvl as u64);
                    let left = known[&(lvl, p)];
                    let parent = node_pfx.hash(&left, &right);
                    known.insert((lvl + 1, p >> 1), parent);
                }
            }
        }
        known
            .get(&(depth, 0))
            .copied()
            .ok_or("multi: reconstruction did not reach the root")
    }
}

#[cfg(test)]
mod tests {
    /// The prefix-path digests must be bit-identical to the generic
    /// parts form (the optimization changes speed only).
    #[test]
    fn prefix_hashes_bit_identical() {
        use magnetar_field::Gf128;
        let domain = "some/level/7";
        let n = 4096usize;
        for idx in [0usize, 1, 2, 1234, 4095] {
            let v = Gf128(idx as u128 * 0x9E3779B97F4A7C15);
            let generic = blake3_domain(&[
                b"magnetar/leaf/v1",
                domain.as_bytes(),
                &(n as u64).to_le_bytes(),
                &(idx as u64).to_le_bytes(),
                &fe_to_bytes(v),
            ]);
            let pfx = LeafHashPrefix::new(domain, n).hash(idx, v);
            assert_eq!(generic, pfx, "leaf {idx}");
        }
        let l = [7u8; 32];
        let r = [9u8; 32];
        for lvl in [0usize, 3, 21] {
            let generic = blake3_domain(&[
                b"magnetar/node/v1",
                domain.as_bytes(),
                &(lvl as u64).to_le_bytes(),
                &l,
                &r,
            ]);
            let pfx = NodeHashPrefix::new(domain, lvl as u64).hash(&l, &r);
            assert_eq!(generic, pfx, "node {lvl}");
        }
    }

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


#[cfg(test)]
mod multi_open_tests {
    use super::*;
    use magnetar_field::Gf128;

    fn build_tree(n: usize, width: usize) -> (MerkleTree, Vec<Gf128>) {
        let mut rng_val = 0x9E3779B97F4A7C15u128;
        let values: Vec<Gf128> = (0..n)
            .map(|i| {
                rng_val = rng_val.wrapping_mul(0x2545F4914F6DD1BE).wrapping_add(i as u128);
                Gf128(rng_val)
            })
            .collect();
        let tree = if width == 1 {
            MerkleTree::build("multi-test", &values)
        } else {
            MerkleTree::build_batched("multi-test", &values, width)
        };
        (tree, values)
    }

    /// The multi-open's reconstruction must equal the individually verified
    /// roots, and the union copath must be smaller than q separate paths.
    #[test]
    fn multi_open_matches_individual() {
        for &(n, width, q) in &[
            (1024usize, 8usize, 14usize),
            (4096, 8, 14),
            (1024, 1, 5),
            (256, 8, 20),
            (64, 8, 3),
        ] {
            let (tree, values) = build_tree(n, width);
            // Deterministic pseudo-random query indices (distinct modulo n).
            let mut acc = 0x1234_5678_9abc_def0u64;
            let idx = |k: usize| -> usize {
                acc = acc.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407 + k as u64);
                (acc >> 33) as usize % n
            };
            let indices: Vec<usize> = (0..q).map(idx).collect();
            let mo = tree.open_multi(&indices);
            // The per-group values in ascending group order.
            let group_values: Vec<Vec<Gf128>> = mo
                .groups
                .iter()
                .map(|&g| values[g * width..g * width + width].to_vec())
                .collect();
            let root = MerkleTree::verify_multi_batched(
                "multi-test", n, width, &indices, &group_values, &mo.siblings,
            )
            .expect("multi-open verifies");
            assert_eq!(root, tree.root(), "root match (n={n})");
            // Each individual opening verifies to the same root.
            for &i in &indices {
                let (dg, path) = tree.open(i);
                let r2 = MerkleTree::verify_batched("multi-test", n, width, i, dg, &path)
                    .expect("individual verify");
                assert_eq!(r2, tree.root());
            }
            // The compression: union copath digests <= q * depth.
            let depth = if n / width == 1 { 0 } else { (n / width).trailing_zeros() as usize };
            let separate = q * depth;
            assert!(
                mo.siblings.len() <= separate,
                "copath {} <= q*d {} (n={n})",
                mo.siblings.len(),
                separate
            );
            if q >= 8 && depth >= 8 {
                assert!(
                    mo.siblings.len() < separate,
                    "strict compression at q={q}, depth={depth}: {} vs {}",
                    mo.siblings.len(),
                    separate
                );
            }
        }
    }

    /// Tamper classes: a wrong group value, a wrong sibling digest, a
    /// mismatched copath order, an out-of-range index.
    #[test]
    fn multi_open_tamper() {
        let (tree, values) = build_tree(1024, 8);
        let indices: Vec<usize> = vec![3, 100, 512, 999, 700];
        let mo = tree.open_multi(&indices);
        let group_values: Vec<Vec<Gf128>> = mo
            .groups
            .iter()
            .map(|&g| values[g * 8..g * 8 + 8].to_vec())
            .collect();
        assert!(MerkleTree::verify_multi_batched("multi-test", 1024, 8, &indices, &group_values, &mo.siblings).is_ok());
        // 1. A wrong group value.
        let mut gv2 = group_values.clone();
        gv2[0][0] += magnetar_field::ONE;
        // A wrong revealed group derives a different root (not an error).
        let rg2 = MerkleTree::verify_multi_batched("multi-test", 1024, 8, &indices, &gv2, &mo.siblings)
            .expect("reconstruction completes");
        assert_ne!(rg2, tree.root(), "the tampered root must differ");
        // 2. A wrong sibling digest: the reconstruction still completes
        // (every position accounted for) but derives a DIFFERENT root.
        let mut copath2 = mo.siblings.clone();
        copath2[0] = [7u8; DIGEST];
        let r2 = MerkleTree::verify_multi_batched("multi-test", 1024, 8, &indices, &group_values, &copath2)
            .expect("reconstruction completes");
        assert_ne!(r2, tree.root(), "the tampered root must differ");
        // 3. A reordered copath (positions recomputed — mismatch).
        // 3. A wrong copath length.
        let mut copath3 = mo.siblings.clone();
        if copath3.len() >= 2 {
            copath3.swap(0, 1); // same length, different order: the
                                // positions are derived, so a swap changes
                                // which digest lands where — a different root.
            let r3 = MerkleTree::verify_multi_batched("multi-test", 1024, 8, &indices, &group_values, &copath3)
                .expect("reconstruction completes");
            assert_ne!(r3, tree.root(), "the swapped copath must derive a different root");
        }
        copath3.truncate(copath3.len() - 1);
        assert!(MerkleTree::verify_multi_batched("multi-test", 1024, 8, &indices, &group_values, &copath3).is_err());
        // 4. An out-of-range index.
        let bad_idx: Vec<usize> = vec![3, 100, 5000];
        assert!(MerkleTree::verify_multi_batched("multi-test", 1024, 8, &bad_idx, &group_values, &mo.siblings).is_err());
        // 5. The multi-open of a DIFFERENT index set against this proof.
        let other: Vec<usize> = vec![5, 200];
        assert!(MerkleTree::verify_multi_batched("multi-test", 1024, 8, &other, &group_values, &mo.siblings).is_err());
    }
}
