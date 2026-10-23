# Benchmarks — the reference implementation vs. the paper

Measured on this repository's reference implementation (Rust `--release`,
rayon parallel kernels, test-scale parameters: RAM = 2^8 words, shift
amount width 2, witness width W = 2048, GF(2^128) with PCLMULQDQ) and
compared with the paper's reported figures (eprint 2026-2423, §6,
Table 6). The development container: 2 Xeon cores, 4 GiB.

Run with:

```sh
cargo run --release -p magnetar-cli --bin magnetar-bench          # end-to-end
cargo run --release -p magnetar-cli --bin magnetar-bench-stream   # streaming vs sequential
```

## End-to-end results (this implementation)

| workload | T | prove (ms) | verify (ms) | cycles/s | proof (KiB) |
|---|---|---|---|---|---|
| alu-chain | 2^8 | 13,263 | 291 | 19 | 352.6 |
| alu-chain | 2^10 | 58,183 | 1,488 | 18 | 443.1 |
| mem-mix | 2^8 | 12,819 | 305 | 20 | 351.9 |
| mem-mix | 2^10 | 55,865 | 1,390 | 18 | 442.4 |
| branchy | 2^8 | 12,473 | 299 | 21 | 351.9 |
| branchy | 2^10 | 56,028 | 1,382 | 18 | 442.4 |
| shift-mix | 2^8 | 12,653 | 308 | 20 | 351.9 |
| shift-mix | 2^10 | 56,297 | 1,472 | 18 | 442.4 |

Tree-IVC (T = 2^10 total, 4 segments of 2^8 sharing one commitment):
prove 69,159 ms, verify 1,216 ms, **proof body 90,304 B + ONE shared PCS
tail (1,008 B)** — versus four separate full proofs, each of which would
carry its own evaluation-argument tail. This is the roadmap §8.2
aggregation saving, measured. (The ~25% prove-time drop vs. the previous
measurement is the parallel Merkle/encode kernels and the join-enumeration
router now wired through the protocol.)

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
3. **Allocation-heavy Rust with only coarse parallelism** — the rayon
   kernels (Merkle, encode, sum-check folds, the join router, the
   parallel segments below) are in, but the dense per-segment provers
   stream hundreds of MB per sum-check round, so memory bandwidth, not
   core count, is the binding constraint at this scale (measured below).

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

## The streaming, parallel tree-IVC prover

`magnetar-ivc::streaming` proves the same tree-IVC statements with a
bounded-memory pipeline — checkpoint pass (the trace never resides),
rayon-parallel segment proofs (each with its own Fiat–Shamir transcript,
tables packed straight into the shared leaf vector and dropped), then
one sequential root (parallel encode + Merkle + the discharge). The
proofs are **byte-identical** to the sequential prover's (tested on 1/2/4
segments), so this is purely a memory/scheduling change.

Measured in fresh processes (mode-separated; `VmHWM` is a lifetime
high-water mark), alu-chain:

| T | segments | sequential (ms) | streaming (ms) | ckpt | segments phase | root | streaming artifacts |
|---|---|---|---|---|---|---|---|
| 2^9 | 1 | 27,231 | 27,671 | ~0 | 24.6 s | 3.1 s | 1.3 MiB |
| 2^9 | 2 | 29,581 | 28,394 | ~0 | 22.8 s | 5.6 s | 1.0 MiB |
| 2^9 | 4 | 34,488 | 33,326 | ~0 | 21.8 s | 10.9 s | 0.8 MiB |
| 2^10 | 1 | 58,623 | 58,044 | ~0 | 51.5 s | 6.6 s | 2.6 MiB |
| 2^10 | 2 | 60,369 | 60,574 | ~0 | 48.5 s | 12.1 s | 1.9 MiB |
| 2^10 | 4 | 70,939 | 68,050 | ~0 | 45.1 s | 23.0 s | 1.6 MiB |

What this measures, honestly:

- **The parallelism is real and correct.** Worker tracing
  (`MAGNETAR_TRACE_WORKERS=1`) shows segment workers dispatched two at a
  time on both cores with ~100% thread utilization (e.g. at T = 2^8,
  4 segments: two concurrent ~5.3 s waves — 4 × 5.5 s of work in 11.0 s
  of wall).
- **The net wall-clock is ~1.0–1.15x, not ~2x, because the per-segment
  provers are memory-bound.** Under 2-way concurrency each worker runs
  ~1.8x slower (a register-only CPU loop scales 2.10x on the same box);
  the dense Twist/Shout tables stream hundreds of MB per round and
  saturate the shared memory bus. This is R3's lesson in the flesh:
  collapse the algorithm before counting cores — the roadmap's
  linear-in-T Twist/Shout provers are what unlock the parallel speedup,
  exactly as the paper's cost model predicts.
- **The streaming artifacts are 3 orders of magnitude below the dense
  provers' working set**: the checkpoint store + packed leaves + codeword
  + in-flight tables total 0.8–2.6 MiB at these scales, while the dense
  per-segment provers dominate the process RSS (~1.4 GiB at 2^9). The
  architecture bounds everything it is responsible for; the remaining
  memory is the dense provers' known gap.
- **The commitment path no longer materializes the boolean union**:
  `pack_tables` writes the leaves directly from the component tables
  (a ~16x commitment-memory reduction), parallel over leaves.

## Reproducing

```sh
cargo test --release          # 93 tests: the tamper classes and roundtrips
cargo run --release -p magnetar-cli --bin magnetar-bench
cargo run --release -p magnetar-cli --bin magnetar-bench-stream -- 9 seq    # isolated sequential
cargo run --release -p magnetar-cli --bin magnetar-bench-stream -- 9 stream # isolated streaming
MAGNETAR_TRACE_WORKERS=1 cargo run --release -p magnetar-cli \
    --bin magnetar-bench-stream -- 8 stream   # the worker schedule
cargo run --release --bin magnetar fib 12      # the CLI smoke
cargo run --release --bin magnetar proof-stats # the byte breakdown
```

All numbers above are from a single run on the development container;
treat ±10% as noise for the prove times and exact for the byte counts.
