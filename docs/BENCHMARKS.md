# Benchmarks — cycles/s versus the paper

This page reports this implementation's measured performance under the
paper's own headline metric — **proving throughput in cycles per second**
— together with the paper's reference numbers, the per-phase breakdown,
and the specific engineering items that close the gap. Every number below
is measured, on this repository's current code; nothing is extrapolated
silently.

## The reference (the paper, Table 6)

The Magnetar paper (eprint 2026-2423, §6.4) measures the
`erc20-mixed-block` workload on an Apple M4 Max (12 performance cores,
128 GiB), at trace lengths 2^16–2^22 application cycles:

| Trace | Prove | Cycles/s | Peak RSS | Proof | Verify |
|----------|--------|----------|----------|-----------|--------|
| 2^16 | 39 ms | 1.7 M | 0.2 GiB | 377 KiB | 8 ms |
| 2^18 | 61 ms | 4.3 M | 0.5 GiB | 462 KiB | 9 ms |
| 2^20 | 174 ms | 6.0 M | 1.6 GiB | 531 KiB | 10 ms |
| 2^22 | 612 ms | 6.9 M | 6.4 GiB | 596 KiB | 11 ms |

## This implementation (measured)

Workload: a compact Fibonacci loop (`ADDI/ADDI/ADD/ADDI/BNE` body — the
paper's workloads are loops too, but note the difference: the paper's
`erc20-mixed-block` is a far denser memory workload; a loop of ALU ops is
the *easiest* shape for a prover, so the comparison below is, if anything,
generous to this implementation).

Hardware: a 2-core cloud container (the paper: 12 performance cores).
Traces below 2^16 are necessarily far below the paper's operating point —
the paper's own throughput *grows* 4× from 2^16 to 2^22 because the
per-cycle costs amortize; the same effect applies here in reverse.

| Trace | Prove | Cycles/s | Proof | Verify |
|-------|--------|----------|-------|--------|
| 2^8 | 14.9 s | 17 | 351 KiB | 0.31 s |
| 2^9 | 31.2 s | 16 | 395 KiB | 0.67 s |
| 2^10 | 66.4 s | 15 | 441 KiB | 1.44 s |

Measured with `cargo run --release -- bench 8 10` (per-phase
instrumentation: `MAGNETAR_TIME=1`).

### The per-phase breakdown (2^8 cycles)

| Phase | Time | Share |
|----------------------|--------|-------|
| shift router | 5.4 s | 36% |
| stage router | 7.2 s | 48% |
| store router | 0.07 s | <1% |
| RAM Twist | 1.3 s | 9% |
| MosaicPCS | 1.3 s | 9% |
| Virtual Spartan | 0.5 s | 3% |
| register Twist | 0.5 s | 3% |
| bytecode Shout | 0.1 s | <1% |
| selectors, witness, commit | 0.1 s | <1% |

(The store router already runs the **sparse-anchor** enumeration added
this cycle — the same treatment for the shift/stage routers is the top
throughput item below.)

## Where the gap comes from — and what closes it

1. **Router enumeration (the dominant gap).** The BinaryRouter
   sum-checks run over a joint space of ~27 variables; the sparse prover
   anchors on the routing tensors' nonzero lists (a 16×–110× speedup for
   the store router) but the shift/stage routers still enumerate
   `nonzeros × 2^(remaining)` per round where the optimal **factor-join**
   enumeration — selector one-hots joined with the tensor's per-routing
   slices — costs only `q × 64` per round. Implementing the join
   enumeration is the single largest CPU-side lever, worth an estimated
   30–60× on the two hot phases (roadmap: `magnetar-sumcheck` join
   enumeration).
2. **Small-domain sum-check arithmetic** (roadmap Lever 4): the engine's
   round messages multiply full `GF(2^128)` elements even where the
   operands are bits; the packed/bit-sliced paths (the 2025/1117
   small-field sum-check, the eq-optimization) are worth 1.3–2× on the
   sum-check-dominated phases.
3. **The trace-length operating point.** The paper's 6.9 M cycles/s is
   measured at 2^22 cycles where the fixed protocol costs amortize; at
   2^16 the paper itself reports 1.7 M. This implementation's small-trace
   numbers carry the full per-round protocol overhead on every proof;
   the amortization only begins at lengths this repository's benchmarks
   have not yet reached.
4. **Hardware and parallelism.** 2 cores versus 12, and the current
   prover parallelizes only the dense engine's large rounds (`rayon`
   above 2^13 entries); the routers and twists run single-threaded.
5. **GPU offload** (the roadmap's next target): the commit/Merkle phase,
   the fold chain, and the router enumerations are throughput-shaped
   workloads — see the GPU chapter of `docs/ROADMAP.md` for the 150M
   cycles/s target analysis.

## Proof size (already in the paper's regime)

This implementation's proofs are 351–441 KiB at 2^8–2^10 — the paper's
grow 377→596 KiB over 2^16→2^22, and the cost anatomy matches: the
MosaicPCS evaluation argument dominates (~84% of bytes; the paper's
§"proof size" attributes ~550 of 598 KiB to the single Ligerito
opening). The `magnetar profiles` command prints the measured dual-rate
projection: the same proof at every PCS profile runs 216 KiB
(`FastChunk`, rate 1/2) down to 128 KiB (`DeepWrap`, rate 1/16 with 14
queries) — the Horizon-2 wrap strategy's size arithmetic, measured on
real proofs.

## Reproducing

```sh
cargo run --release -- bench 8 10        # the sweep above
MAGNETAR_TIME=1 cargo run --release --example bench_dbg   # the phase table
cargo run --release -- profiles           # the dual-rate cost model
cargo test --workspace                    # 76 tests, all passing
```
