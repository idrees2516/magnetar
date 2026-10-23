# Status — what is implemented and what remains

The honest inventory. Everything listed as **implemented** is in this
repository, exercised by the workspace test suite (83 tests), and held
to the paper-mapping ground rules in `CONTRIBUTING.md`. Everything
listed as **remaining** carries its design location and its blocking
dependency.

## Implemented

**The complete Construction-3 protocol** (paper §5.5), end to end:

* The RV64I interpreter with the expansion pipeline (decode classes,
  register-shift stages, JAL/JALR link rows, load/store semantics), the
  full-access memory log, and halt-latch padding.
* The dual SIMD R1CS circuit (F₂ bit constraints + the K-packed
  products) with the ROM-authenticated decode-class one-hot, the
  class-gated T2 operand binding (including the SUB carry-in row), the
  logic-op constraints through the AND aux, the stage routing with the
  committed shamt, the load sign/zero extension, the store effective
  addresses, and the halt-latch I/O binding.
* Virtual Spartan (the outer/inner lockstep sum-checks) discharging one
  evaluation claim per segment.
* BinaryTwist for the register file and RAM (the LT recursions, the
  address components, the Ψ-sum-checks, the one-write-per-cycle model),
  BinaryShout for the bytecode bus with the α₀ non-emptiness trick, and
  BinaryRouter for shifts, register-shift stages, and sub-word stores.
* The decode-class, shift-amount, stage, and store-width selector
  checks against the ROM tables.
* MosaicPCS: aligned-subcube packing, the Equation-4 free lift, the
  batch sum-check, and the ring-switch discharge with exact φ bit
  extraction — one commitment, one evaluation argument.
* The Fiat–Shamir transcript (SHA3-256 duplex, labeled absorptions),
  the BLAKE3 Merkle commitments, the constant-time GF(2^128) field.
* The measured cost model (`stats.rs`) and the dual-rate profile
  projections.

**Horizon 1 (partial)**: the PCS rate/query profiles; the ZK
round-mask primitive and its first protocol wiring.

**Horizon 2 (partial)**: claim aggregation and segmented proving with
one shared commitment (`segments.rs`).

**Infrastructure**: the benchmark harness, the phase instrumentation,
the CLI (`fib`/`bench`/`check`/`profiles`), CI (fmt + clippy
`-D warnings` + release tests + the audit job), and the documentation
set you are reading.

## Remaining — soundness scope

* **Branch/jump targets** (the paper's own Remark-4 omission): the
  branch condition is constrained (the adder compares a−b) but the
  micro-PC sequence itself is not — the taken-branch target is
  witness-chosen. The paper defers this ("the next fetch address is
  not constrained" — its stated future work); closing it needs a
  pc-jump routing or a fetch-address sum-check analogous to the
  bytecode Shout.
* **QROM**: Fiat–Shamir is argued in the classical random-oracle model,
  as in the paper and every 2026-era system in its class; the quantum
  argument for the composed protocol is open (a shared debt with
  Akita, most meaningful at the wrap).

## Remaining — protocol features

* **Full end-to-end ZK.** The round-mask primitive
  (`magnetar_sumcheck::zk`) and its first wiring (the Mosaic batching
  discharge, `KeyGen::enable_zk`) are in. The remaining wiring, each
  site mechanical (wrap the prover's rounds with the masker, carry the
  offset, `finish_offset` on the verify side):
  the Virtual-Spartan outer/inner locksteps, the register and RAM
  Twists' runs, the three routers' sparse sum-checks, the bytecode bus
  Φ and its Ψ, the four selector checks, and the Ψ sum-checks.
  Beyond that, the **masked commitment** (the paper-standard
  Ligero-style trick: commit `T + R` for secret uniform `R` so every
  revealed query symbol is information-theoretically hidden; the
  evaluation claims adjust by revealed uniform `R̂` values) — designed,
  not yet implemented.
* **Streaming prover** (small-space proving within chunks): the design
  is in `docs/RECURSION.md`; the current prover materializes the
  witness and the packed table.
* **The verifier-guest wrap circuit** (the dual-rate wrap's guest):
  designed in `docs/RECURSION.md` with its cost arithmetic; not yet
  implemented. Until it lands, recursive composition — like the
  paper's own — remains a design, and this repository says so.

## Remaining — throughput

Measured status in `docs/BENCHMARKS.md`; the ranked items:

1. **The router join enumeration** — the shift/stage routers still
   enumerate `nonzeros × 2^remaining` where the factor-join costs
   `q × 64` per round; the dominant CPU item (worth an estimated
   30–60× on the two hot phases).
2. **Small-domain sum-check arithmetic** + the eq-optimization.
3. **Bolt-class encoder** for the commit phase.
4. **GPU offload** — the program's next milestone; the phased plan and
   the 150M cycles/s target arithmetic are in `docs/ROADMAP.md`.

## Remaining — code health

* The interpreter's scratch cell and the 64-cell register file are
  exercised by tests, but fuzz-style differential testing against a
  reference RV64I simulator is not yet in CI.
* `cargo audit` runs in CI; the `criterion`-style statistical
  benchmarks (versus the current single-shot harness) are not yet
  wired.

## The parallel-branch merge

A parallel development line landed three commits on the pre-overhaul
base (stage-select, its claim aggregation, `prove_multi`/
`verify_multi`, the `ivc` module, wire sizes, the architecture doc
set). This merge keeps the current (post-overhaul) core — the sounder,
faster lineage that also fixes the branch/jump immediate desync — and
takes from the parallel line:

* `docs/architecture/*` — the ten-part per-component documentation set
  (the big-component write-ups).
* `docs/ivc.md` — the tree-IVC design (the port's specification; the
  implementation itself lives on the `tree-ivc-legacy` branch until it
  is ported to the current `protocol` API).
* The Twist `m0_public` mechanism (committed initial-memory images for
  IVC chunks) — merged into `magnetar-memory`; inactive (`true`) in the
  monolithic protocol.

Its overlapping deliverables (the dual-rate profiles, claim aggregation,
the benchmark, the remaining-work inventory) are superseded by this
line's implementations on the fixed base.
