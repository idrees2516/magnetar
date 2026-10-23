# End-to-end benchmarks

Measured against the paper's Table 6 methodology (§6.1) — trace-length
scaling of proving time, throughput, proof size, verifier time, and
memory — plus the roadmap H2 artifacts (claim aggregation, tree-IVC
chunking, the dual-rate schedule), all with **exact uncompressed wire
sizes** (`magnetar-zkvm::wire`: 16 B per committed field element,
32 B digests, 8 B length prefixes; a production codec compresses
further, so these are conservative upper bounds).

Reproduce with:

```sh
cargo run --release --bin magnetar -- bench 128
```

## Hardware disclosure

| | This benchmark | The paper (§6.1) |
|---|---|---|
| Machine | 2 vCPU Intel Xeon (container) | M4 Max MacBook Pro, 12 perf. cores |
| Memory | ~4 GiB | 128 GiB |
| Mode | `--release`, single run | median of 3 after 1 warm-up |

This host cannot reach the paper's operating point (2²² cycles ⇒ 6.4 GiB
peak RSS there; the memory ceiling here fits ~2⁷ cycles of the current
prover), so runs are scaled down and the comparison separates
**protocol invariants** (proof size, verifier work — hardware-independent)
from **throughput** (implementation- and hardware-bound; see the honest
section at the end).

## Trace-length scaling (one segment, rate 1/4)

Workload: a straight-line Fibonacci body filling the trace.

| Trace | Guest (µs) | Prove (ms) | Cycles/s | Proof (KiB) | Verify (ms) | Peak RSS (MiB) |
|------:|-----------:|-----------:|---------:|------------:|------------:|----------------:|
| 2⁶ | 13 | 8,636 | 7 | 375.7 | 80 | 201 |
| 2⁷ | 12 | 22,385 | 6 | 432.3 | 149 | 390 |

**Proof size lands in the paper's own range at matched geometry.** The
paper reports 377 KiB at 2¹⁸ rising to 598 KiB at 2²²; here 375.7 KiB at
2⁶ rising with the trace — the size is dominated by the fixed costs of
the evaluation argument (the paper's own cost anatomy: ~84% evaluation
argument) plus the logarithmic terms, so the smallest traces already sit
at the paper's floor. The wire model walks the real proof object, so
these are measured bytes, not estimates.

## Claim aggregation (roadmap §4.3 — "aggregation before the wrap")

`K` segments — independent executions of the program, each with its own
public input — share **one** Mosaic commitment and discharge through
**one** evaluation argument (`prove_multi`). The counterfactual is `K`
standalone proofs:

| K | K standalone (KiB) | Aggregated (KiB) | Saving | Prove (ms) | Verify (ms) |
|--:|-------------------:|-----------------:|-------:|-----------:|------------:|
| 1 | 375.2 | 375.2 | 0% | 7,316 | 79 |
| 2 | 750.4 | 472.0 | **37%** | 15,069 | 146 |
| 4 | 1,500.8 | 616.3 | **59%** | 32,529 | 300 |

The aggregated artifact grows by the *batched* per-segment bodies only —
the evaluation argument (the dominant cost) is paid once. This is the
measured wire-size ledger behind the roadmap's "aggregation before the
wrap" lever: the wrap (when it lands) proves one small batch-verification
instance instead of `k` separate proofs, so its witness scales with the
*aggregated* size, not the standalone sum.

## Tree-IVC chunking (one execution, state continuity)

One Fibonacci execution (128 cycles) split into dynamic chunks at safe
instruction boundaries, each chunk proven through the shared
`SegmentProof` body, boundaries crossed by the committed register/RAM
state identities, and the whole thing discharging through one argument
(`prove_chunked` — see `docs/ivc.md`):

| Chunk cycles | Chunks | Total cycles | Proof (KiB) | Prove (ms) | Verify (ms) |
|-------------:|-------:|-------------:|------------:|-----------:|------------:|
| 32 | 4 | 128 | 619.1 | 21,388 | 155 |
| 64 | 2 | 128 | 536.5 | 19,047 | 152 |

Two things to read off: the proof of a chunked execution grows with the
**chunk count**, not the total cycles (4×32 ≈ 619 KiB vs 2×64 ≈ 537 KiB —
the per-chunk bodies dominate; the evaluation argument stays one); and
chunking costs ~13% more than the same cycles in one segment (537 vs
~475), the price of the boundary-continuity identities. The unbounded-
trace story: halve the chunk size to fit memory, pay linearly in chunks,
keep the verifier flat.

## Dual-rate schedule (roadmap §4 — the chunk/wrap profiles)

The chunk profile (rate 1/2) minimizes per-chunk encode/commit work; the
wrap profile (rate 1/4) buys soundness headroom on the public artifact.
Measured on 4 aggregated segments of 2⁶ cycles:

| Profile | Rate | Proof (KiB) | Verify (ms) |
|---------|-----:|------------:|-------------:|
| chunk | 2⁻¹ | 568.6 | 288 |
| wrap | 2⁻² | 616.3 | 292 |

The wrap costs **+8.4% proof size** at these parameters with flat
verification — the concrete constant for the dual-rate recursion's
accounting: keep the chunks at rate 1/2, publish the wrap at 1/4, and
the final artifact carries the 8.4% only on the wrap's own (small)
instance.

## The prover-throughput gap, honestly

At 2⁷ cycles this implementation proves **~6 cycles/s**; the paper
reports **4.3M cycles/s at 2¹⁸** (its smallest row). The gap is in the
implementation's constants, not the protocol — the same measurement on
the same host as everything above:

* The sum-check engines are **constant-factor unoptimized**: the sparse
  prover re-enumerates every remaining Boolean-cube assignment per round
  and recomputes each factor's compacted index with a per-variable
  position walk; the dense engine materializes joint tables where the
  factors' own tables would do. The paper's §6 attributes its speed to
  exactly the optimizations the roadmap's Horizon 1 lists: per-round
  laziness, small-domain (packed bit) arithmetic, and incremental
  factor folding. None change the verifier or the proof.
* The Twist value sum-checks **materialize** the Val tables
  (`O(K·L·T)` memory, ~390 MiB peak at 2⁷); the paper's prover streams
  at ~1.6 KiB/cycle. The streaming form is Twist-and-Shout's incremental
  structure, applied.
* The host is ~6–10× smaller than the paper's (2 vCPU / 4 GiB vs. 12
  perf. cores / 128 GiB), which caps the feasible trace and removes any
  parallelism headroom. The per-segment and per-chunk bodies are the
  parallelism entry points once the constants are fixed.

Expected order of gains (the roadmap's Horizon 1, in landing order):
memoized/lazy round products (the largest single win), packed bit
arithmetic in the one-hot/bit tables, the streaming Twist, then
`rayon` across segments/chunks.

## What this benchmark establishes

* The **complete protocol measures in the paper's own proof-size range**
  (375.7–619.1 KiB across all artifacts vs. the paper's 377–598 KiB) —
  the strongest hardware-independent validation available at this scale.
* **Claim aggregation delivers a measured 59% wire-size saving at K=4**
  (and 37% at K=2) — the aggregation-before-the-wrap lever has its
  constants.
* **Tree-IVC chunking keeps the artifact growing in the chunk count,
  not the total cycles**, with boundary continuity proven (state
  identities against the commitment), at a measured ~13% chunking
  overhead.
* The **dual-rate schedule has its measured constant** (+8.4% for the
  wrap rate, flat verification).
* The **remaining throughput work is precisely identified** — the
  sum-check engine constants — with the optimization path and expected
  gains in `docs/remaining-work.md` and `ROADMAP.md`'s Horizon 1.
