# Security accounting

Magnetar targets the paper's 120-bit provable-security configuration.

## Hash layer

* Fiat–Shamir: SHA3-256 (Keccak, FIPS 202) duplex transcripts with
  domain separation (every absorption labeled), length binding, and
  state ratcheting after every squeeze. Challenge derivation collision
  resistance: 2^-128.
* Commitments: BLAKE3 Merkle trees. Leaf hashes bind the tree domain,
  the tree size, and the leaf index (defeating cross-tree substitution,
  size confusion, and index forgery); node hashes bind the level.
  Binding: 128-bit collision resistance.

## Sum-check layer

Each round of the multi-factor engine has soundness error at most
`(term degree)/|K| = 8/2^128` (the largest term in the stack is the
store-routing summand with 8 factors). The full protocol runs on the
order of a few hundred rounds; the total sum-check error is bounded by
`(#rounds × 8)/2^128 < 2^-115`, comfortably within the 120-bit target
with the paper's margin conventions.

## PCS / fold chain

The codeword family is the evaluation of the multilinear space at a
seed-generic affine image (rate 2^-k, dimension 2^n). Folding along a
cube variable at challenge r is the substitution `y_i → y_i + r` (the
family is closed); after all n folds the tail satisfies
`D_n(0) = T̂(r*)` exactly. The correlated path queries (FRI-style walk),
the fully revealed tail, and the level-0 commitment anchor give the
fold-chain soundness per the FRI/Ligerito proximity analysis within the
unique-decoding radius — the proven-gap standard the Magnetar paper
itself relies on (§6.1, citing Ben-Sasson et al., TR25-169).

## Ring switching

The naive "reveal folded slices as free witnesses" design is unsound
(a 126-dimensional cheat space over K); this implementation instead
anchors the c\*-folded table's codeword to the main commitment with
**exact φ_c bit extraction** at random (x, ε=0) positions: the verifier
opens the packed leaf, reads its 128 polynomial-basis bits, and checks
the dot product directly. Every accepted value is a fold of
Merkle-authenticated data or an exact bit extraction.

## Implementation hygiene

* `#![forbid(unsafe_code)]` in every crate except the audited
  carry-less-multiplication intrinsics module (guarded by cached
  `is_x86_feature_detected!` checks; SAFETY comments in place).
* Constant-time field arithmetic: branchless portable paths; the
  PCLMULQDQ path is data-independent by construction.
* No secret-dependent branching in transcripts or Merkle operations;
  the exponent in the field inversion is a public constant.
* Secret witness data is zeroized nowhere required to be persistent;
  proofs contain only public-coin responses.

## Known scope notes

* The interpreter materializes the Val history in BinaryTwist
  (O(K·L·T) time — the paper's prover work — with a space tradeoff the
  paper's production prover avoids by streaming).
* The end-to-end zkvm wiring currently closes Construction 3's steps
  1–3 and 5; the auxiliary arguments (Twist/Shout/Router) are
  implemented and tested per-component and compose through the same
  claim flow (documented in the README).
