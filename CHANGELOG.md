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
