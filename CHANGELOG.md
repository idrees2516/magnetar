# Changelog

## 0.2.0 (unreleased)

### This cycle: soundness fix, throughput, and the horizon items

- **Soundness fix (the truncated ROM immediate)**: the ROM's
  instruction bus carried only the 13-bit immediate field while the
  witness's fetch segment carries the full sign-extended 36-bit
  pre-adjusted immediate — every program with a **negative immediate**
  (backward branches, `ADDI -1`, negative load/store offsets) desynced
  the bytecode Shout. The ROM now writes the identical 36-bit word;
  a witness/ROM consistency regression test guards the class.
- **The SUB carry-in row**: the class-gated T2 binding sets
  `T2 = rs2 + F_SUB` on the SUB/BRANCH classes (the two's-complement
  negation absorbed into the operand), and the stale duplicated
  constraint section was removed.
- **The sparse-anchor router prover** (`magnetar-sumcheck::sparse` v2):
  round messages enumerated through the routing tensors' nonzero lists
  (rebuilt per fold) instead of the full joint cube — the store routing
  runs 110× faster, the whole 32-cycle proof 8.4×.
- **PCS rate/query profiles** (Horizon 1): `PcsProfile::{FastChunk,
  Balanced, CompactWrap, DeepWrap}` with the per-query soundness
  accounting for the affine-substitution code; the dual-rate projection
  measured by the cost model and exposed as `magnetar profiles`.
- **The ZK round-mask primitive** (Horizon 1): `magnetar_sumcheck::zk`
  — per-round mask polynomials with the running constraint chain,
  perfect HVZK for the round messages at O(degree) randomness per
  round; wired into the Mosaic batching discharge behind
  `KeyGen::enable_zk` (the full site wiring is inventoried in
  `docs/STATUS.md`).
- **Segmented proving** (Horizon 2): `execute_chunks` with explicit
  state handoff; `segments.rs` proves S×T cycles under **one shared
  commitment** — the chunked witness equals the monolithic trace
  bit-for-bit (tested), the claims aggregate through the single
  batching discharge, and `aggregation_report` measures the saving.
- **The benchmark harness**: `magnetar bench <lo> <hi>` (the
  cycles/s sweep), `MAGNETAR_TIME=1` phase instrumentation, and
  `docs/BENCHMARKS.md` with the measured numbers versus the paper's
  Table 6.
- **Documentation**: `docs/ROADMAP.md` (the horizons + the GPU target
  build order), `docs/RECURSION.md` (the wrap/tree-IVC/streaming
  designs), `docs/STATUS.md` (the implemented/remaining inventory).

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
