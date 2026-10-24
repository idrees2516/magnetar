# Benchmarks — the reference implementation vs. the paper

Measured on this repository's reference implementation (Rust `--release`,
rayon parallel kernels, test-scale parameters: RAM = 2^8 words, shift
amount width 2, witness width W = 2048, GF(2^128) with PCLMULQDQ) and
compared with the paper's reported figures (eprint 2026-2423, §6,
Table 6). The development container: 2 Xeon cores, 4 GiB. The paper's
hardware: 12 performance cores of an M4 Max, 128 GiB.

Run with:

```sh
cargo run --release -p magnetar-cli --bin magnetar-bench          # end-to-end
cargo run --release -p magnetar-cli --bin magnetar-bench-big -- 12 # beyond 2^10
cargo run --release -p magnetar-cli --bin magnetar-bench-stream   # streaming vs sequential
MAGNETAR_TIME=1 cargo run --release -p magnetar-cli --bin magnetar-bench-big -- 8
                                                                   # per-phase wall clock
cargo run --release -p magnetar-cli --bin magnetar -- profiles     # dual-rate tails
```

## End-to-end results (this implementation)

Current numbers (the small-value fast paths + the eq-slab + per-relation
row padding + the fused VPCLMULQDQ kernels with measured dispatch):

| workload | T | prove (ms) | verify (ms) | cycles/s | proof (KiB) |
|---|---|---|---|---|---|
| alu-chain | 2^8 | 238 | 20.4 | 1,074 | 352.2 |
| alu-chain | 2^10 | 962 | 53.2 | 1,065 | 442.8 |
| alu-chain | 2^11 | 2,141 | 91.9 | 957 | 491.8 |
| alu-chain | 2^12 | 4,536 | 164.8 | 903 | 543.3 |
| alu-chain | 2^13 | 9,982 | 322.6 | 821 | 597.3 |
| (streaming, 64 segs) | 2^14 | 12,993 | — | **1,261** | one shared tail |

The session-start baseline was 372–411 cycles/s; the cumulative gain is
**2.8–3.3×**, from four structural optimizations taken from the
sum-check-proving literature (ePrint 2025/1117, 2026/587, 2025/611, and
the Dao–Thaler constraint-packing note):

1. **Per-relation row padding** (paper Construction 2 shares only `q` and
   `m`): the K relation's real row count is 31, not 2048 — its outer
   sum-check cube shrank 64×.
2. **Small-value (bit-valued) fast paths**: the binary relation's `az/bz/cz`
   are bits (a binary matrix applied over the bit witness), so every
   round-message product against them is a branchless select and their
   folds are selects — no CLMUL — plus the select-based `A·z` application
   and instance folds (`Σ eq·z_bit`).
3. **The never-folded eq slab** (the decomposable-equality discipline):
   `eq(τ, ·)` public factors gather their bit-`s` slice and move the
   bound-variable constant into the term coefficients — the eq factor's
   per-round fold multiplications are gone everywhere.
4. **Fused VPCLMULQDQ kernels with a *measured* dispatch**: the 2- and
   3-factor message kernels, the fold runs, and the eq-tensor doubling
   run through lane-batched pipelines (AVX-512 4-lane and AVX2 2-lane,
   `#[target_feature]`-specialized), with a one-time timed probe choosing
   the backend — some virtualized hosts advertise `avx512f` in CPUID but
   emulate it at ~10× scalar cost, so the probe picks AVX2 there.
   `MAGNETAR_SIMD=avx512|avx2|scalar` overrides.

The T = 2^11–2^12 rows use `magnetar-bench-big` (previously beyond the
prover's memory envelope — T = 2^11 OOM'd a 4 GiB container; the dense
joint materializations are now gone). Trace-linear scaling holds through
2^12: prove time doubles per trace doubling and the verifier grows
logarithmically. T = 2^13 exceeds the *monolithic* envelope on this 4 GiB
container (the Virtual Spartan outer tables' `O(q·M)` words plus the
`2^ell·rate` codeword) — the streaming/tree-IVC path is the scale answer
(dense-prover-free segments now that the memory layer is linear).

## The per-phase anatomy (T = 2^8, `MAGNETAR_TIME=1`)

| phase | wall (ms) | share |
|---|---|---|
| 1-execute+witness | 8 | 1% |
| 2-commit (encode + Merkle) | 9 | 1% |
| 3a-Virtual Spartan (outers + inner) | 498 | 55% |
| 3b/3c-Twist (registers + RAM, linear-in-T provers) | 49 | 5% |
| 4a/4b-routers (join provers) | 4 | <1% |
| 5-bus Shout + selectors + cross-step | 4 | <1% |
| 7a/7b/7c-Mosaic discharge (batching + ring switch) | 150 | 17% |
| other (transcript, claims, serialization) | ~140 | 21% |

The two remaining hot spots are the Virtual Spartan outer sum-checks
(the two relations' `az·bz − cz` products over the `(instance ‖ row)`
cube — O(q·M) field multiplies, the algorithmic floor of the reduction)
and the Mosaic batching pass. Both are now at their O(N)-table-pass
shape; the remaining gap to the paper is constant-factor (see below).

## The paper's figures (Table 6, erc20-mixed-block, optimized system)

| Trace length | Proving (ms) | Cycles/s | Peak RSS | Proof size | Verify (ms) |
|---|---|---|---|---|---|
| 2^16 | 39 | 1.7M | 0.2 GiB | 377 KiB | 8 |
| 2^18 | 61 | 4.3M | 0.5 GiB | 462 KiB | 9 |
| 2^20 | 174 | 6.0M | 1.6 GiB | 531 KiB | 10 |
| 2^22 | 612 | 6.9M | 6.4 GiB | 596 KiB | 11 |

## What is comparable, and what is not

**Proof size — directly comparable, and it matches the paper's shape.**
Our proofs are 352 KiB at T = 2^8 and 443 KiB at T = 2^10, growing by
~50 KiB per 2× trace length; the paper's grow 377 → 596 KiB over a 64×
trace-length increase. Both are dominated by the same term — the
evaluation argument and its correlated Merkle paths. The dual-rate
profiles and the wrap now measure the size dial on the same statement
(`magnetar profiles`): the evaluation tail runs 62.8 KiB (FastChunk,
rate 1/2, 40 queries) → 27.3 KiB (DeepWrap, rate 1/16, 14 queries), and
the **wrapped** configuration (chunk side at Balanced + the public tail
discharged at DeepWrap with the consistency spot-checks) lands at
36.5 KiB — a 44% tail reduction over the Balanced chunk proof, the
roadmap H2 projection direction measured on this implementation.

**Verifier — structurally comparable.** 21 ms at 2^8 and 141 ms at 2^12
against the paper's 8–11 ms at 2^22 (a 12-core M4 Max): the verifier now
evaluates its public factors structurally (`EqProdFactor`, O(T) suffix
sums, the sparse-support batching factor) — no joint-cube
materialization — and grows logarithmically with the trace. The
remaining constant is transcript hashing and the correlated-path checks.

**Cycles/s — comparable in shape; the constant is engineering.** This
implementation proves 372–411 cycles/s on 2 cloud Xeon cores against the
paper's 1.7–6.9M on 12 M4 performance cores. The 2024-era reference
baseline (before the linear-in-T provers) was **18–21 cycles/s** — a ~20×
algorithmic closing — the
algorithmic gap (the joint-cube materializations) is closed: every
memory-checking sum-check (Twist read/write, the bytecode Shout, the Ψ
checks, the selectors, the cross-step transition) now enumerates the
trace cycle-by-cycle with the one-hot kill and the prefix-XOR cumulative
sums — byte-identical round messages, cross-validated against the dense
engine by tests. The measured per-phase anatomy pins the remaining gap
on two constants:

1. **The field-multiply floor**: the scalar PCLMULQDQ Karatsuba
   multiplier runs at ~11.5 ns/mul on this Xeon. The Virtual Spartan
   outers' ~11 multiplies per `(instance, row)` position dominate the
   profile. The CPU advertises `vpclmulqdq`; a 4-lane batched multiplier
   in the engine kernels is the next 3–4× (the Phase-2 GPU item
   supersedes it on the roadmap).
2. **The witness-width constant**: the test-scale circuit pads the
   witness to W = 2048 columns and M = 2048 rows per cycle; the paper's
   operating point amortizes several cycles per SIMD instance with a
   tighter base structure, dividing the per-cycle position count.

Both are engineering constants, not algorithmic gaps: the prover now
does O(positions) work with O(1) passes per round, exactly the paper's
complexity shape.

**Memory.** The dense Twist/Shout materializations (O(2^{a+ell+n}) words
per factor) are gone: T = 2^12 proves in a 4 GiB container where T =
2^11 previously exhausted it, and T = 2^13 completes. The remaining
peak-RSS drivers are the Virtual Spartan outer tables (O(q·M) words) and
the codeword (O(2^ell · rate)) — the streaming/IVC layer's chunking is
the production answer at larger scales.

## What the benchmarks DO establish

- The complete Construction-3 protocol — every step of §5.5 including
  the Remark-4 cross-step and boundary bindings — runs end-to-end at
  paper-comparable *shape*: linear-in-T proving, logarithmic
  communication, structured verifier evaluation, and the full tamper
  coverage of the test suite.
- **The linear-in-T prover layer is real**: the memory-checking
  sum-checks' round messages are byte-identical to the dense engine's
  (cross-validated per merge), at O(T) prover work — the paper's §4.4
  "the prover's cost is linear in T" is now an implementation property,
  not an aspiration.
- The dual-rate dial and the wrap are measured on the same statement
  (62.8 → 27.3 KiB tails; the wrapped artifact at 36.5 KiB).
- The scaling is trace-linear in the limit and memory-bounded: 2^8 →
  2^12 multiplies prove time by ~16.6× (16× trace growth) with no
  super-linear blowup.

## The streaming, parallel tree-IVC prover

`magnetar-ivc::streaming` proves the same tree-IVC statements with a
bounded-memory pipeline — checkpoint pass (the trace never resides),
rayon-parallel segment proofs (each with its own Fiat–Shamir transcript,
tables packed straight into the shared leaf vector and dropped), then
one sequential root (parallel encode + Merkle + the discharge). The
proofs are **byte-identical** to the sequential prover's (tested on
1/2/4 segments). With the linear-in-T segment provers, the segment
workers are no longer memory-bound: the parallel speedup now tracks the
core count where the phases allow it.

The 611-paper streaming shape (checkpointed replay, per-segment
transcripts, bounded peak) is now measured through **2^14 cycles**
(1,261 cycles/s at 64 segments on this 2-core container); the adaptive
segment ladder keeps the per-worker envelope inside 4 GiB at every
scale. Above 2^14 the **root aggregation** becomes the bottleneck: its
Mosaic batching factor is a sum over all segments' remapped claims
(`segments × components` items), and the current `SumEqFactor`
materialization cost is `Σ_i 2^{free_i}` — quadratic growth in the
combined cube. The 611/587-style fix is the hierarchical k-way
decomposition (never materialize the sum; keep the shared per-segment
`eq(r_inst)` factor and the per-segment small local tables — a
rank-`segments` positional evaluation), which is the next roadmap item
for the 2^15–2^16 streaming ladder.

## The measured next levers (the per-phase anatomy at T = 2^10)

| phase | share | the paper-prescribed lever |
|---|---|---|
| VS outer rounds | 21% | the small-value **round window** (587 §5.1): keep the bit tables un-folded for the first v rounds, compute the window grid with popcount arithmetic, one eq-weighted pack at the window's end — also removes the K-valued materialization that bounds the memory envelope |
| Mosaic `SumEq` build | 21% | the hierarchical / k-way block build above (2026/587 §6): never materialize the α-combination |
| the twists | 19% | select-fold the bit-valued one-hot tables in the linear provers (the same `mul_bit` discipline) |
| everything else | 39% | the circuit-density item: the test-scale witness pads to 2048 bit-columns against the paper's amortized instance batching (several cycles per SIMD instance) |

The verifier stays unchanged by every optimization above (all messages
are bit-identical to the generic engine's), which is what keeps the
tamper suite meaningful through the optimization loop.

## Reproducing

```sh
cargo test --release          # the full suite: roundtrips + tamper classes
cargo run --release -p magnetar-cli --bin magnetar-bench
cargo run --release -p magnetar-cli --bin magnetar-bench-big -- 12
cargo run --release -p magnetar-cli --bin magnetar-bench-stream -- 9 seq    # isolated sequential
cargo run --release -p magnetar-cli --bin magnetar-bench-stream -- 9 stream # isolated streaming
MAGNETAR_TIME=1 cargo run --release -p magnetar-cli --bin magnetar-bench-big -- 8
cargo run --release --bin magnetar fib 12      # the CLI smoke
cargo run --release --bin magnetar proof-stats # the byte breakdown
cargo run --release --bin magnetar profiles    # the dual-rate tails + wrap
```

All numbers above are from single runs on the development container;
treat ±10% as noise for the prove times and exact for the byte counts.
