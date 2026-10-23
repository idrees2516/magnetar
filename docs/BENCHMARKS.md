# Benchmarks — the reference implementation vs. the paper

Measured on this repository's reference implementation (single-threaded
Rust, `--release`, test-scale parameters: RAM = 2^8 words, shift amount
width 2, witness width W = 2048, GF(2^128) with PCLMULQDQ) and compared
with the paper's reported figures (eprint 2026-2423, §6, Table 6).

Run with:

```sh
cargo run --release -p magnetar-cli --bin magnetar-bench
```

## End-to-end results (this implementation)

| workload | T | prove (ms) | verify (ms) | cycles/s | proof (KiB) |
|---|---|---|---|---|---|
| alu-chain | 2^8 | 18,021 | 303 | 14 | 352.6 |
| alu-chain | 2^10 | 82,122 | 1,438 | 12 | 443.1 |
| mem-mix | 2^8 | 17,331 | 303 | 15 | 351.9 |
| mem-mix | 2^10 | 80,288 | 1,442 | 13 | 442.4 |
| branchy | 2^8 | 16,984 | 306 | 15 | 351.9 |
| branchy | 2^10 | 80,182 | 1,478 | 13 | 442.4 |
| shift-mix | 2^8 | 17,069 | 306 | 15 | 351.9 |
| shift-mix | 2^10 | 80,778 | 1,442 | 13 | 442.4 |

Tree-IVC (T = 2^10 total, 4 segments of 2^8 sharing one commitment):
prove 89,874 ms, verify 1,229 ms, **proof body 90,304 B + ONE shared PCS
tail (1,008 B)** — versus four separate full proofs, each of which would
carry its own evaluation-argument tail. This is the roadmap §8.2
aggregation saving, measured.

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
~90 KiB per 4× trace length; the paper's grow 377 → 596 KiB over a 64×
trace-length increase. Both are dominated by the same term — the
evaluation argument and its correlated Merkle paths (~84% of bytes here,
`magnetar proof-stats`); the per-sum-check-round communication is O(log T)
field elements in both. The constant offset reflects our larger witness
width (2048 columns at the test scale) and our short-trace operating
point (fixed per-proof components amortize worse at 2^8 than at 2^22).

**Cycles/s — NOT comparable as a number; comparable in structure.** The
paper reports 1.7–6.9M cycles/s from an optimized system (streaming
sum-check provers, per-cycle-linear memory arguments, a laptop-class
multithreaded machine). This reference implementation is ~15 cycles/s,
for three identified reasons, in order of impact:

1. **The Twist/Shout provers materialize the joint space.** The
   BinaryTwist read/write runs and the bytecode bus sum-check run over
   the joint `(address, bit, cycle)` cube — our engine materializes the
   padded tables (O(2^(a+ell+n)) words and per-round views). The paper's
   provers are linear in T with small constants (§4: "the prover's cost
   is linear in T"). This is the dominant gap and the top optimization
   item (the router already has its linear-in-T prover —
   `magnetar-router::join` — the same treatment for the Twist/Shout is
   the roadmap's next engineering step).
2. **The verifier materializes public tables** (300 ms–1.4 s vs the
   paper's 8–11 ms): the reference verifier evaluates public factors by
   materialization rather than the paper's O(polylog) structured
   evaluation.
3. **Single-threaded, allocation-heavy Rust** — no rayon, no arena, no
   SIMD batching beyond the field multiplication.

**Memory.** T = 2^11 exceeds this implementation's envelope (OOM on a
commodity container) for the same reason as (1); the paper peaks at
6.4 GiB for 2^22 cycles (≈1.6 KiB/cycle).

**What the benchmarks DO establish:**

- The complete Construction-3 protocol — every step of §5.5 including
  the Remark-4 cross-step and boundary bindings — runs end-to-end, and
  rejects the jump-pc / fake-halt / fake-branch / wrong-output tamper
  classes (see the test suite).
- The measured cost anatomy matches the paper's: the evaluation argument
  dominates the proof bytes, sum-check communication is logarithmic, and
  the tree-IVC aggregation removes the per-segment PCS tail (the single
  largest byte component) exactly as the roadmap §8.2 projects.
- The scaling is trace-linear in the limit (2^8 → 2^10 multiplies prove
  time by ~4.7×, converging toward linear as the fixed components
  amortize).

## Reproducing

```sh
cargo test --release          # 79 tests: the tamper classes and roundtrips
cargo run --release -p magnetar-cli --bin magnetar-bench
cargo run --release --bin magnetar fib 12      # the CLI smoke
cargo run --release --bin magnetar proof-stats # the byte breakdown
```

All numbers above are from a single run on the development container;
treat ±10% as noise for the prove times and exact for the byte counts.
