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

## The virtual-claim discharge (step 4) — the identity inventory

Every reduction below is a sum-check whose round checks carry ≤
8/2^128 each; the listed identity is what binds the reduction's output
claims to committed data:

* **Ψ-sum-checks** (registers' indices, RAM addresses, the micro-PC):
  `c = Σ_{k,j} eq(r_cycle,j)·Ψ(k)·Π_i ra_i(k^{(i)},j)` with public
  `Ψ`. The register-index claims are checked *again* inside the bytecode
  Shout's Φ sum-check against the ROM's slices, and the RAM/pc claims
  are checked by exact equality against the committed segments' Virtual
  Spartan claims — the single eq-folded claim per port avoids the
  combination leak of per-bit claims (a per-bit lie that preserves the
  Φ combination but shifts the segment identity is caught by the
  equality check).
* **Bytecode Shout** (one joint (k, o, j) sum-check): the summand
  `eq(r_val,o)·eq(r_cycle,j)·[α₀ + w(o)·bus(o,k)]·Π ra_i` checks the
  fetched bus, the three register-index slices (with the
  slice-position-derived eq weights), and the folded non-emptiness
  (α₀ forces `Σ_k Π_i ra_i = 1` at the folded point) in one identity;
  Δ\* per component bounds the Hamming weight.
* **Selector checks** (per routing factor, over (k, h_f, j)): the
  summand covers **every** joint variable — a factor replicated over
  idle variables contributes `2^idle · v = 0` over the Boolean cube in
  characteristic two, so the per-factor design is a correctness
  requirement, not a convenience.
* **Routers**: the sparse prover evaluates each factor in its own
  variable space; the verifier replays the same round checks and
  evaluates the public routing tensor itself. The store router's
  offset-bit claims are solved from the eq-link terminals
  (`â_k = terminal + 1 + r_{a_k}`) and lift to the off segment at the
  Boolean unit points.
* **Reconstruction (Val) sum-checks in Twist**: the write-address
  terminal is now checked against the product of the per-component
  claims at the reconstruction's point (address virtualization) — this
  closes what was previously an unbound terminal.

## Known scope notes

* The interpreter materializes the Val history in BinaryTwist
  (O(K·L·T) time — the paper's prover work — with a space tradeoff the
  paper's production prover avoids by streaming).
* The cross-step micro-PC constraints are omitted exactly as the
  paper's Remark 4 omits them from the formal protocol; the
  decode-vs-flag correlation remains a documented engineering extension
  (the flags are witness-trusted; the fetched bus is authenticated by
  the Shout). The register-shift **stage-select** extension is now fully
  implemented: the trash-cell stage chain lives in the public ROM
  fields, the `sel_stage` one-hot is authenticated through the fetch
  like every other selector, and the stage-selector sum-check binds the
  conditional amount bits and composes the selector claims additively
  in characteristic two — see `docs/architecture/router.md` for the
  composition argument.
* Test traces are 2^3–2^5 cycles so CI stays fast; the trace-length
  parameter is a KeyGen input and the architecture scales linearly.
