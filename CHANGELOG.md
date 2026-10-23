# Changelog

## [0.4.0] — the soundness closure release

### Added
- **The cross-step transition argument** (the paper's Remark-4 mechanism,
  `magnetar-memory::cross_step`): the fetch chain
  `fetch(j+1) = TT[fetch(j), taken(j), halt(j)]` proven by two chained
  sum-checks (the transition Shout with α₀ sum-forcing + the
  predecessor-weight shift discharge), with the ROM transition table
  built at keygen. Closes the jumped-pc soundness gap — a swapped
  instruction order is now rejected (`tamper_jumped_pc_rejected`).
- **The flag-derivation constraints** (`magnetar-cpu::circuit`):
  `halt ⟺ opcode = SYSTEM` and `taken = branch-row ∧ comparison` (the
  full 64-bit comparison circuit: the zero-chain, the signed/unsigned
  less-than muxes, the funct3 selection) — a faked halt or branch
  direction is now unsatisfiable.
- **The ECALL output binding**: the halt row reads x10 (rs1 = 10 in the
  expansion) and `halt · (rd + rs1) = 0` ties the public output to the
  register file through the register Twist. `prove` additionally
  requires the execution to halt within the trace bound.
- **The initial-pc and final-halt public point claims** (single-segment
  and tree-IVC shapes).
- **Tree-IVC** (`magnetar-ivc`, new crate): segments share one
  MosaicPCS commitment; per-segment boundary statements chain through
  the final-state bindings; the root discharges all segments' claims
  with one batching sum-check and one PCS evaluation argument.
- **The final-state binding** (`magnetar-memory::final_state`): the
  Remark-4 initial/final register and RAM binding via the zero-check +
  the evolution reduction.
- **The join-enumeration router prover** (`magnetar-router::join`): the
  paper's linear-in-T BinaryRouter prover — byte-identical proofs to the
  generic sparse prover, ~600× faster at T = 2^12.
- **The benchmark harness and paper comparison**
  (`magnetar-cli::magnetar-bench`, `docs/BENCHMARKS.md`).
- `docs/STATUS.md`: the honest implemented/not-implemented accounting.

### Fixed
- The sub-word store tensor routed the source bit `o` instead of
  `o − 8a` at byte offset `a` — every store with a nonzero byte offset
  produced an unsound routing identity (all prior tests used offset 0).
- The store-router offset claims selected bit-column `2^k` instead of
  `k` (a unit-vector vs binary-index confusion in the claim point).
- The load zero-extension constraints were not gated on `is_load` and
  rejected any ALU result with bits above 15 set (e.g. sign-extended
  immediates).
- `prove` now pads the trace to exactly `trace_len` (the interpreter
  padded to the next power of two of the record count, desynchronizing
  every auxiliary argument at non-matching lengths).

### Changed
- The witness width doubled (W = 2048, M = 4096) to fit the comparison
  circuit; the `crossaux` virtual segment tiles the new region.
- The halt-latch padding cycles fetch the frozen row (the cross-step
  chain requires the freeze transition to be consistent).
- `JALR` is rejected at keygen: the static transition table cannot
  encode register-indirect targets (see `docs/STATUS.md`).


=======
## 0.3.0 (unreleased)

### The register-shift stage-select extension (complete)

- Register shifts (`SLL/SRL/SRA` by register) now prove end to end: the
  6-stage decomposition chains through the trash cell **in the public
  ROM fields** (stage 0 reads the source register, later stages read
  cell 32, intermediate stages write it back) — no scratch register, x0
  stays hardwired to zero, and the register Twist's one-write-per-cycle
  model authenticates the chain unchanged.
- The ROM gains the `sel_stage` one-hot (authenticated through the
  fetch like every other selector) and the shift-kind column goes live
  on stage rows; the router's amount selector becomes the
  witness-conditional pair `{identity, shift-by-2^k}`.
- The **stage-selector sum-check** (`magnetar-memory::discharge`) binds
  the condition bit to bit k of the amount register (the `store-src`
  copy segment) and yields the structured stage part's claim; the
  `sel_lo` selector check composes it additively in characteristic two,
  forcing `Q_lo = ROM(sel_lo) + STAGE` at the router's random point.
- Five new tamper classes (the stage claim, a round, both terminals,
  the stage selector) are rejected; the e2e register-shift program
  covers SLL, SRL, and SRA with negative sources.

### Claim aggregation (roadmap Horizon 2) and the dual-rate schedule

- **[legacy branch]** `keygen_aggregate` / `prove_multi` / `verify_multi`: K segments
  share ONE Mosaic commitment (the components' subcubes gain the
  `log2(K)` index dimension) and ONE evaluation argument — the
  byte-dominant part of every hash-based proof is paid once,
  independent of K. Measured: 3.46x smaller than the standalone
  counterfactual at K=4.
- **`PcsParams::chunk()` (rate 1/2) / `wrap()` (rate 1/4)**: the
  dual-rate profiles, with the honest analysis of what the rate does
  and does not move in the affine-substitution code
  (`docs/architecture/pcs.md`).
- The proof structure is now `Proof { commitment, segments: Vec<SegmentProof>,
  mosaic }` — the single-segment API is the K=1 wrapper.
- The **cost model** (`stats.rs`) prints the aggregation ledger:
  measured per-segment sum-check bytes, the measured single evaluation
  argument, and the analytic standalone counterfactual.
- **[legacy branch]** New CLI: `magnetar agg [segments] [chunk|wrap]`.

### Correctness fixes (found by the new full-circuit satisfiability harness)

- **Trace padding**: the witness cube is now filled exactly to the
  proving bound (`execute_padded`); previously, programs whose cycle
  count rounded below the bound left zero rows that violated the
  per-cycle constraints.
- **Inverted sub-word-load zero-extension masks**: byte loads zero
  `rd` bits 8–63 and half loads bits 16–63, masked by the load-width
  flags only — previously the mask also zeroed non-load cycles, making
  any ALU result with high bits set (e.g. a negative ADDI) unprovable.
- New regression tests evaluate both R1CS relations on every witness
  row of representative programs (`witness_sat_tests`).

### Documentation

- `docs/architecture/` — ten production-grade component deep-dives
  (fields, sum-check, PCS, Mosaic, Spartan, memory, router, CPU,
  protocol, cost model), each mapping code to paper sections.
- `ROADMAP.md` — the implemented-status table, the horizon program, and
  the 150M-cycles/s GPU study's build order as the future plans.


### Tree-IVC: chunks share one commitment (the continuity layer)

- **The tree-IVC layer** (`magnetar-zkvm::ivc`, `docs/ivc.md`): a long
  execution is cut into chunks at RV64I instruction boundaries
  (`execute_chunk`; the cut never splits a register-shift stage sequence)
  and each chunk is proven through the shared `SegmentProof` body on one
  coordinator-ordered transcript.
- **Boundary continuity**: the register files (x0..x31 + the trash cell)
  and micro-PCs are published per chunk and bound by a final-state
  identity sum-check (`exit = entry + Σ wfa·inc`, the sparse prover)
  against the committed write log; the micro-PC by fetch-component claims
  at the first/last cycle's Boolean points.
- **RAM continuity without publishing the state**: each chunk commits
  `ram-init`/`ram-final` image components; the RAM BinaryTwist gains a
  claimed-M0 mode (its `M̂0` evaluations are transmitted and discharged
  against `ram-init`); chunk 0's identity pins the entry image to the
  public input; every boundary checks
  `ram-final_b + ram-init_{b+1} = 0` at a random point.
- **Claim-batching at the tree node**: the k chunk layouts merge into
  one Mosaic cube (dynamic k — the merged layout concatenates the
  per-chunk component lists) and ONE BLAKE3 commitment; a single
  MosaicPCS evaluation argument discharges every claim of every chunk.
  Measured: the eval layer (60% of the aggregate) is paid once — 3.8x
  fewer bytes than k separate proofs. **[legacy branch]**
- **The boundary padding invariant**: the write port's one-hot targets
  the fetched row's redirected rd (the bytecode Shout binds it to the
  bus slice), so the frozen-PC padding writes that cell — the identity
  write (the cell's own content, zero increment) at mid-program
  boundaries, the latched output on halted machines. Pinned by
  write-log consistency regressions.
- **[legacy branch]** New: `keygen_chunked`, `magnetar chunked`, and `aggregate_stats` (the
  aggregate cost model: core/identity/eval layers + the batching saving
  on the exact per-chunk geometry).
- 16 new tests: cross-chunk RAM + register carryover, the public-input
  pin, single-chunk compatibility, a five-chunk fibonacci, two write-log
  consistency regressions, the cost model, and ten tamper classes.
>>>>>>> c5531013623f249f8b96155d4e6ddb1bf4d8819c

## 0.2.0 (unreleased)

> **The parallel-branch merge**: the entries below marked **[legacy
> branch]** describe the `ivc`/`prove_multi` implementation built on the
> pre-soundness-overhaul API. That code is preserved on the
> `tree-ivc-legacy` branch pending its port to the current core; the
> merged tree's equivalents are `magnetar-zkvm::segments` (the shared
> commitment + aggregation) and `PcsProfile` (the dual-rate dials) —
> see `docs/STATUS.md` for the full reconciliation.


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

### The end-to-end benchmark and the honest inventory

- `magnetar-zkvm::wire`: the exact uncompressed wire-size model over the
  real proof objects (`Proof` and the tree-IVC `AggregateProof`) — 16 B
  per committed field element, 32 B digests, 8 B length prefixes — so
  the benchmark reports measured bytes, not estimates.
- `magnetar bench [max-trace]`: the paper's Table-6 methodology plus the
  H2 artifacts (the aggregation ledger vs. the standalone
  counterfactual, the tree-IVC chunk count, the dual-rate schedule,
  peak RSS). Highlights: proof sizes 375.7–619.1 KiB across all
  artifacts (the paper's own range is 377–598 KiB at its operating
  points); claim aggregation saves 59% of wire bytes at K=4 (37% at
  K=2); tree-IVC chunking keeps the artifact growing in the chunk count
  at a ~13% overhead; the wrap rate costs +8.4% with flat verification.
- `docs/benchmarks.md`: the full measured tables with the hardware
  disclosure and the honest prover-throughput analysis.
- `docs/remaining-work.md`: the honest inventory — the sum-check-engine
  speed path (memoized/lazy round products, small-domain arithmetic,
  the streaming Twist, parallelism), the verifier-guest/wrap recursion,
  zero-knowledge, QROM, ISA completeness (LB/LH sign extension, the
  workload suite), and the production-engineering backlog.
