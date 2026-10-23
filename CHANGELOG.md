# Changelog

## 0.2.0 (unreleased)

### The complete protocol

- **Construction 3, all seven steps end-to-end** — the virtual-claim
  discharge (step 4) is now wired into the proof: BinaryTwist for
  registers and RAM, BinaryRouter for shifts and sub-word stores, and
  BinaryShout for the bytecode, each reducing its claims to evaluations
  on the single Mosaic commitment.
- New `magnetar-memory::discharge` module: the Ψ-sum-checks binding
  eq-weighted address/index claims to the one-hot components, the
  joint-space bytecode Shout (bus + register-index slices + α₀
  non-emptiness in one sum-check), and the per-factor selector checks.
- **MosaicPCS v2**: the component layout — the direct segments plus the
  eight auxiliary polynomials (the one-hot address components and the
  increments) on aligned subcubes of one cube, with the Equation-4
  suffix lift.
- **The sparse sum-check prover** (`magnetar-sumcheck::sparse`): factors
  stay in their own variable spaces and fold only on their own rounds —
  the paper's linear-in-T prover work.
- Twist soundness fix: the reconstruction sum-check now emits and checks
  per-component claims at its point (address virtualization) — the write
  terminal was previously unbound against the components.
- Witness v2: virtual/direct segment split at 64-aligned offsets, the
  instruction bus with the redirected rd, the sub-word-load byte-lane
  mux, the router source copies, and the one-write-per-cycle register
  model with the full access log.
- **Proof-size instrumentation** (`magnetar-zkvm::stats`): the
  committed-bit inventory per component, the PCS geometry, and the byte
  breakdown per protocol component (the roadmap's cost model).
- 69 tests including four end-to-end programs and 15 tamper classes.

## 0.1.0

The initial release: the binary-field stack, the sum-check engine, the
PCS with ring switching, Spartan/Virtual Spartan, BinaryTwist/BinaryShout
with the refined one-hotness, BinaryRouter, the RV64I decoder and
interpreter, the dual SIMD R1CS circuit, and MosaicPCS.
