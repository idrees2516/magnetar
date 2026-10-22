# Magnetar roadmap

**Where the system is, where the research is going, and what the next
engineering milestones buy.** This document integrates the two deep
research reports (the proof-size/recursion study and the 150M-cycles/s
GPU study) into one program, and marks what the codebase already
implements versus what remains research.

The roadmap is **research-stage**: every projection below is a model
with error bars, the horizons will be re-ordered as instruments land,
and the architecture rule that governs all of it is modularity — each
lever swaps in at a crate boundary without touching the protocol
(see `docs/architecture/README.md` for the layer map).

---

## Status: implemented now

| Capability | State | Where |
|---|---|---|
| Full Construction 3 (7-step protocol, all claims through ONE commitment) | done, 108 tests, tamper classes | `protocol.rs` |
| Register-shift **stage-select** (the documented extension) | done — the trash-cell stage chain, the `sel_stage` ROM one-hot, the stage-selector sum-check, additive claim composition | `cpu/expand.rs`, `zkvm/rom.rs`, `memory/discharge.rs`, `protocol.rs` |
| **Claim aggregation** (Horizon 2, "aggregation before the wrap") | done — K segments, ONE commitment, ONE evaluation argument; measured 3.46x at K=4 | `prove_multi`, `docs/architecture/mosaic.md` |
| **Dual-rate schedule** (Horizon 1/2) | done — `PcsParams::chunk()` (1/2) / `wrap()` (1/4), measured by the CLI | `pcs/eval.rs`, `cost-model.md` |
| Measured cost model | done — committed-bit inventory, byte breakdown, the aggregation ledger, the standalone counterfactual | `stats.rs` |
| **Bitsliced sum-check kernel** (GPU program Phase 1) | done — tested host oracle + cuTile kernel spec (`magnetar-bitsum`); hardware bring-up pending (manual `bitsum-cuda` workflow + `gpu_parity` gate) | `magnetar-bitsum`, `docs/architecture/bitsum.md` |
| Production hardening | done — CI (fmt, clippy `-D warnings`, tests, CLI smoke), SECURITY/CONTRIBUTING/CHANGELOG, dependabot, issue/PR templates, the indexed docs site | repo root |

Two latent correctness bugs were found and fixed while landing the
stage-select extension (both pinned by new full-circuit satisfiability
regression tests): the trace-padding invariant and inverted sub-word-load
zero-extension masks (`docs/architecture/cpu.md`).

---

## Horizon 1 — parameterization (months)

**Goal:** shave constants on the *current* architecture. Projections
240–350 KiB final proofs at 10–20M cycles/s on CPU.

* **Rate/digest/field tuning** — the dual-rate profiles are in; the
  digest is BLAKE3 (16-byte elements vs the roadmap's 32-byte digest
  study); the next step is measuring the wrap profile at production
  trace lengths.
* **Small-domain sum-checks** — the sparse prover already keeps router
  factors in their own spaces; remaining wins are per-round laziness
  (evaluate factors only when the round's variable touches them).
* **ZK masking** — the current transcripts are honest-prover, not
  zero-knowledge. Masking (blinding terms in round messages) is
  localized to the sum-check engine; the protocol structure needs no
  change.
* **Bolt-class encoders** — the affine-substitution encoder is
  `O(n . 2^{n+k})` per table; a systematic linear-time encoder would
  cut the commit share of the chunk budget. Swap point: `pcs/code.rs`.

## Horizon 2 — recursion and aggregation (1–2 quarters)

**Goal:** 100–160 KiB final proofs, unbounded traces via chunks,
end-to-end/base ratio 0.85–0.95 (vs Airbender's 0.39 recursion tax).

* **Claim aggregation before the wrap** — **landed** (the K-segment
  shared commitment + single discharge). The byte-dominant evaluation
  argument is now paid once per batch, and the cost model prints the
  saving. What remains is the *continuity* flavor:
* **Tree-IVC across chunks of one execution** — segment i+1's initial
  RAM = segment i's final RAM, chained in a binary tree with
  claim-batching at each internal node (the roadmap's recommended
  schedule). The aggregation layout is the substrate; the missing
  piece is the cross-segment state hand-off argument (the paper's
  §6.2 sketch). Design note: our segments share ONE commitment, so the
  tree schedule is a *prover memory* strategy (fold pairwise, commit
  once) rather than a proof-structure change.
* **The verifier-guest + low-rate wrap** — the flagship: prove
  Magnetar's own verifier in-guest (it is BLAKE3/SHA-256-dominated
  work, exactly what BinarySpartan is fastest at), then wrap the small
  witness at a genuinely low RS-coupled rate where fewer queries buy a
  smaller final artifact. This is where the roadmap's 100–160 KiB
  projection lives. It is the largest single milestone and is
  deliberately sequenced after the instruments it needs.
* **Streaming within chunks** — the prover currently materializes
  tables; streaming (the paper's `O(T)`-space claim) touches the
  sum-check engine's factor handling and the Twist materialization.

## Horizon 3 — push (research)

* 64-bit challenge fields (coupled verifier cost), systematic-code
  sparsity (pay-per-bit à la lattice commitments), hybrid wrap only if
  the proven-soundness floor demands it.

---

## The 150M cycles/s GPU program (the future-plans study)

The second research report models **150M cycles/s on 1–2 Blackwell
GPUs** — 6.9x past Airbender's single-H100 21.8M baseline — as a
multiplicative stack:

| Factor | Multiplier (modeled) | Instrument |
|---|---|---|
| Parity base B | 18–25M on 1x H100 | a GPU port of the existing prover |
| Algorithmic A | 1.6–2.2x | AG-sumcheck (2026/2494), small-value provers, Flock-style elisions, log-star rounds (2025/1269) |
| Kernel K | 1.5–2.0x | fused folds, 2^13 SMEM tiles, persistent kernels, bit-sliced tower arithmetic |
| Bandwidth H | 2.39x (B300 8 TB/s) | the sum-check layer is HBM-bound streaming |
| Sharding G | 1.85x at 2 GPUs | segment-level parallelism; kilobytes of interconnect per chunk |

Landing zones: 1x B300 103–263M (expected case), 2x B300 191–487M
(committed design point), 4x H100 SXM the guaranteed air-cooled
fallback. The honest caveats: the witness tracer becomes the governor
at 150M (22M cycles/s today — needs the overlapped CPU pool or the GPU
tracer), and the Rust GPU toolchain (cuda-oxide/cutile-rs, official
since Sept 2026) is young.

**Build order this implies for the codebase** (all swap-points are
crate boundaries, none touch the protocol):

1. **GPU sum-check kernels** (the `magnetar-sumcheck` engine's factor
   interface is the natural FFI surface: factors in, round messages
   out) — cutile-rs for the bandwidth-bound folds, cudarc for the host
   plane, ICICLE as the reference to beat. *Phase 1 landed as
   `magnetar-bitsum`*: the bitsliced sum-check with a tested CPU oracle
   (1.5 Melem/s measured), the repo-standard SHA3-256 transcript, a
   verified tower↔K isomorphism, and the cuTile kernels behind the
   `cutile` feature — hardware bring-up checklist in
   `docs/architecture/bitsum.md`; wiring the engine's factor interface to
   it is the remaining step.
2. **GPU encode/commit** for the affine-substitution code (a streaming
   workload; the delta-shifted transform tiles cleanly).
3. **Multi-GPU sharding** — the aggregation layer's coordinator-star
   transcript already permits it: shards compute segment bodies, one
   coordinator hashes the round transcript in order, standard hashes
   only (the report's LtHash-free point of strategic significance).
4. **The AG-sumcheck / Flock instrument ports** — algebra swaps inside
   known kernel shapes; the engine's factor abstraction is where they
   land.
5. **Tensor-core experiments** (INT8 GEMM shapes for the F2^8 product
   tables and round extrapolations) — upside only, explicit kill
   criterion, never load-bearing.

## What we deliberately do not do yet

* **In-guest recursion before the verifier-guest design settles** —
  the roadmap's own risk register: wrap-witness growth can offset H1
  gains, and the mitigation is exactly the claim aggregation now
  implemented.
* **List-decoding regimes** — the paper's unique-decoding conservatism
  is a brand property; relaxed regimes enter only with proofs.
* **Any non-hash commitment** — the hash-only property is the
  differentiator the whole program protects.

---

## Reference instruments (from the studies)

AG-sumcheck ePrint 2026/2494 · Flock 2026/1329 · small-value/one-hot
provers 2026/587 · sum-check improvements 2024/1210 · log-star-round
IOPs 2025/1269 · Bolt 2026/310 · Arc accumulation 2024/1731 ·
bounded-depth accumulation 2024/474 · DeepFold 2024/1595 · FRI-Binius
2024/504 · Airbender (ZKsync, June 2025) · the SNARK.fast and HashSmash
community programs.
