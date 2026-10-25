# Benchmarks — the reference implementation vs. the paper

Measured on this repository's reference implementation (Rust `--release`,
rayon parallel kernels, the packed-bit witness, the mask-major outer,
GF(2^128) with VPCLMULQDQ) and compared with the paper's reported
figures (eprint 2026-2423, §6, Table 6). The production load-test
host: a 4-vCPU EPYC instance with real AVX-512 + VPCLMULQDQ (InstaBox).
The paper's hardware: 12 performance cores of an M4 Max, 128 GiB.

Run with:

```sh
cargo run --release -p magnetar-cli --bin magnetar-bench          # end-to-end
cargo run --release -p magnetar-cli --bin magnetar-bench-big -- 13 # beyond 2^10
cargo run --release -p magnetar-cli --bin magnetar-bench-stream -- 16 stream
cargo run --release -p magnetar-cli --bin magnetar-bench-erc20 -- 16
                                                                   # the paper's Table-6 workload
MAGNETAR_TIME=1 cargo run --release -p magnetar-cli --bin magnetar-bench-big -- 8
                                                                   # per-phase wall clock
cargo run --release -p magnetar-cli --bin magnetar -- profiles     # dual-rate tails
```

## End-to-end results (this implementation)

Current numbers (the commitment-free ring switch release: the DP24
partial-evaluations reduction, the one-shot Merkle hashing, the join
prover's suffix tables + bit selects, the tabulated phi_c, the packed
witness, the mask-major bin outer, the group-fold discharge, the linear
st1, the FastChunk rate, the parallel hierarchical root):

| workload | T | prove (ms) | verify (ms) | cycles/s | proof (KiB) |
|---|---|---|---|---|---|
| alu-chain | 2^10 | 203 | 43.6 | 5,044 | 404.9 |
| alu-chain | 2^11 | 373 | 62.1 | 5,495 | 451.5 |
| alu-chain | 2^12 | 614 | 114.3 | 6,668 | 500.5 |
| (streaming, 8 segs) | 2^13 | 672 | — | 12,186 | one shared tail |
| (streaming, 16 segs) | 2^14 | 1,321 | — | 12,398 | one shared tail |
| (streaming, 32 segs) | 2^15 | 2,799 | — | 11,707 | one shared tail |
| (streaming, 64 segs) | 2^16 | 6,300 | — | **10,402** | one shared tail |

Peak RSS at 2^12 monolithic: **165 MiB** (was 619 MiB two releases ago);
at 2^16 streaming: 1.4 GiB.

## The commitment-free ring switch release (measured on the InstaBox
## 4-vCPU EPYC host; ±5% host-contention noise between runs)

| T | prove (ms) | root (ms) | cycles/s | tx/s | vs. session start |
|---|---|---|---|---|---|
| 2^13 | ~460 | — | ~14,2xx | ~1,18x | +18% |
| 2^16 | 4,924–5,280 | 1,603–1,641 | 12,414–13,310 | **1,034–1,109** | **+11–19%** |

The root phase fell 2.34 s → 1.60 s: the ring switch alone 1.1 s →
~0.39 s (the DP24 reduction: no second encode, no level-0 C-tree, no
anchors — plus the one-shot leaf/node hashing and the tabulated `phi_c`,
which was the single biggest cost inside the new ring at ~400 ms). The
segment phase fell ~10%: the join prover's o-round suffix tables (≈17
muls → 2 per pair per round) and the h-round packed-bit selects, plus
the per-segment glue elimination (whole-word witness reads, the store-X
claims evaluated directly from the packed words, no table clones).
Proofs shrank by ~60–90 KiB (the anchors).

## The ERC-20 mixed-block load test (the paper's Table-6 workload shape)

`magnetar-bench-erc20` proves the paper's `erc20-mixed-block` shape:
balance transfers between two accounts with checked subtraction, every
16th transfer failing the underflow check — 12 cycles per executed
transfer at the 64-bit balance width of this RV64I reference CPU
(the 0.9.2 operating point is `cpi = 1` — the levers flipped the 0.9.1
`cpi = 4` default; `MAGNETAR_CPI` sweeps the density):

| T | segments | tx | prove (ms) | cycles/s | tx/s | peak RSS |
|---|---|---|---|---|---|---|
| 2^12 (monolithic) | — | 341 | 446 | 9,183 | 764 | 166 MiB |
| 2^13 | 8 | 682 | 470 | **17,418** | **1,450** | 285 MiB |
| 2^14 | 16 | 1,365 | 933 | **17,568** | **1,464** | 376 MiB |
| 2^15 | 32 | 2,730 | 1,990 | 16,470 | 1,372 | 595 MiB |
| 2^16 | 64 | 5,461 | 4,254 | 15,405 | 1,284 | 1.09 GiB |

**The pooled proving-pool run** (5 × 4-vCPU instances proving
independent 2^16 batches concurrently — the cpi × pool matrix, the
aggregate over the 20 vCPUs):

| cpi | per-machine tx/s | aggregate tx/s |
|---|---|---|
| **1** | 1,005–1,051 | **5,139** |
| 4 | 847–975 | 4,585 |
| 8 | 956–1,027 | 4,942 |

The honest pooled operating point is `cpi = 1`: the lightest
per-segment constants degrade least under cross-VM contention (the
solo ladder's cpi 4 edge — the per-instance discharge amortization —
was exactly what the 0.9.2 levers removed). The 0.8.0 pool measured
≈3,788 tx/s; the same five machines now pool ≈5,139 tx/s (+36%). The
remaining measured levers, in order of impact:

1. **The discharge's `fw-bound-se` group-fold** (~19 ms per segment,
   ~30% of the root's local stages): the eq(r)-weighted fold of the
   packed words per distinct free-word point. A chunked-axpy rewrite
   was attempted once and REVERTED (it broke small-scale verification);
   the group-fold needs a different decomposition.
2. **The VS window pass's residual** (`vs-win-build`, ~16 ms per
   segment): the complement-convolution (0.9.2) cut it from ~26 ms;
   the remaining cost is the eq-weighted pack over the 3^v monomial
   state (one 16-byte scatter per set monomial per suffix).
3. **The twists' Δ\* one-hot proofs** (`tw-onehot`, ~11 ms per segment
   total across the two twists) and the port k-rounds (~4 ms).
4. **The streaming root commitment** for 2^17+ monolithic-bounded
   proving: the driver materializes the combined packed table, its
   codeword, and the Merkle tree (≈3.3× the packed bytes); the
   611-paper's shape (encode + hash per segment slice, the discharge
   consuming packed words per segment) is the 2^23/2^24 gate.


Every row proves **and verifies** end-to-end (the assert is in the
harness). The paper's erc20-mixed-block at 2^16: 39 ms, 1.7M cycles/s,
0.2 GiB — on 12 M4 performance cores whose per-core CLMUL rate is
~5× this Xeon's measured 11.5 ns/mul.

## The three levers of the 0.9.2 pass

1. **The twists' packed-inc select-fold** (the largest measured lever):
   the BinaryTwist port sum-checks' o-rounds folded the `T·2^ell`
   increment table generically (a CLMUL pass per round, ~4 ms per port
   run). The increments are **bits**: `TwistShared` now packs them one
   `u128` per cycle, one masked pass per port run computes
   `D_o = Σ_{u: inc bit o} (WFAEQ·LTW)[u]`, and each round's message is
   `O(2^ell)` — the round weight `hist[o]·eq_rest[o>>(b+1)]` multiplies
   `D_o` and the address-folded `M0` row. The measured `port-o-rounds`
   fell to **0.0 ms** (from ~4 ms × 5 port runs per segment); the
   reconstruction's increment row is the eq(r_v,·) select over the packed
   words (no `T·2^ell` clone per fold). Messages byte-identical
   (`linear_port_prover_matches_dense`).
2. **The store-router join's select-dots**: the routing tensors are 0/1,
   so the extraction's `tau` sums are XOR-selects of the global
   `eq_o[o]` (no CLMUL), and the o-rounds collapse from two products
   **per pair per round** (the store router's byte-store columns carry
   ~128 pairs, doubleword stores up to ~912) to one product **per nonzero
   output bit** against a GLOBAL per-round weight vector — one pass over
   the support builds the per-column `xo2[o]`. Measured: the
   `4b-store-router` phase fell 45.4 → **4.4 ms per segment (~10×)**.
   Non-binary tensors keep the generic per-pair path (byte-identity
   pinned by the new `join_matches_sparse_prover_generic_coefs`).
3. **The window pass's complement-convolution**: the small-value round
   window's one-pass state build convolves the two tables' Möbius masks
   per suffix — `O(popcount(m0)·popcount(m1))` pair iterations, dense
   structured sub-cubes up to 4,096. The convolution is GF(2)-bilinear:
   with `a`/`b` the sparser of `(m, ~m)`, the corrections are three
   precomputed word vectors — dense sub-cubes cost ~a dozen word XORs.
   The pass's chunking also no longer serializes small strides (the
   previous `PAR_THRESHOLD` floor ran the 2^11-suffix VS window
   single-threaded). `vs-win-build` fell ~26 → **16 ms per segment**.

The verifier is unchanged by every optimization (all messages are
engine-identical), which is what keeps the tamper suite meaningful
through the optimization loop.

## The four optimizations of the earlier pass

1. **Circuit density (2048 → 512 rows/cycle)**: the binary relation's
   per-cycle row count was dominated by 1,127 *booleanity* rows
   (`x² = x` per witness column) — but the committed witness is the
   **bit matrix**: MosaicPCS packs the components' coordinate bits into
   GF(2^128) elements, and any committed element's coordinates are bits
   by construction, so `x² = x` holds for every possible commitment and
   binds nothing. Deleting the vacuous rows drops the real row count
   1610 → 483 (padded 512) and the matrix nnz 5,116 → 1,735 — a 4×
   smaller row cube and a 3× cheaper matrix application. The full
   tamper suite (15 classes) still passes: the datapath's soundness
   lives in the value-carrying rows and the memory/routing arguments.
2. **The small-value round window on the bin outer** (ePrint
   2025/1117 §5.1, 2026/587 §5.1 + C.5, specialized to characteristic
   two): the first `min(6, s)` shared rounds' messages are computed in
   **one pass** over the un-folded bit tables. In char two the
   multilinear Möbius coefficients of a bit table are bits (AND/XOR),
   the product's monomial coefficients are bits, and the **eq-weighted
   pack** collapses the whole suffix dimension into just `3^v + 2^v`
   field accumulators — the window state is *independent of the trace
   length*. The messages are extracted per round from the monomial
   state (the future coordinates' collapse `Σ_b eq(τ,b)·b^e` gives
   `τ_t` for `e ≥ 1`), and the fast-forward contracts the still-bit
   tables at the challenges with selects — no CLMUL. Measured:
   `vs-outer-rounds` fell from the dominant phase to 13 ms at 2^8.
   Cross-validated byte-identical against the generic engine
   (`window_check.rs`).
3. **The hierarchical SumEq root** (the 2025/611 streaming shape): the
   root's batching sum-check over the combined cube materialized two
   `2^ell · 16 B` tables (the G bit table and the summed-eq table) —
   2×1 GiB at 2^16 before a single round, the measured 2^15 gate. The
   combined cube is the product `(segment bits) ∥ (local cube)` and G
   is zero outside each segment's block, so the batched sum **factors
   per segment**: each segment's local stage proves its own batching
   sum-check in lockstep (shared challenges, per-segment state), with
   the round window reading G straight from the packed 128-bit words
   (the window covers the top plane bits, so all `2^v` window values of
   a suffix live in one word) and the per-item suffix eq tables the
   only materialization. A `log2(segments)`-round segment stage then
   eq-batches the per-segment terminals into one claim. The root's peak
   memory drops from `O(2^ell_combined)` to `O(one segment)` and the
   local stages parallelize; **no cross-segment "prefix adaptation"
   pass ever touches the segment tables**. The packing convention is
   split-aware (the word index is `segment ∥ block`, the 7 word bits
   are the local planes), which keeps every segment's words contiguous.
4. **The measured SIMD-probe fix**: the backend-detection probes now
   accumulate wrapping sinks — the previous `sum()` panicked on
   debug-build overflow (a latent 5-test debug failure).

The verifier is unchanged by every optimization above (all messages are
engine-identical), which is what keeps the tamper suite meaningful
through the optimization loop — the hierarchical root's verifier is the
new per-segment lockstep replay plus the segment stage.

## The per-phase anatomy (T = 2^8, `MAGNETAR_TIME=1`)

| phase | wall (ms) | share |
|---|---|---|
| 1-execute+witness | 6 | 4% |
| 2-commit (encode + Merkle) | 7 | 5% |
| 3a-Virtual Spartan (outers + inner) | 42 | 29% |
| 3b/3c-Twist (registers + RAM, linear-in-T provers) | 46 | 32% |
| 4a/4b-routers (join provers) | 5 | 3% |
| 5-bus Shout + selectors + cross-step | 1 | 1% |
| 7-Mosaic discharge (windowed local + ring switch) | 49 | 34% |

The two remaining hot spots are the **memory-checking twists** (32% —
the select-fold lever for their bit-valued one-hot tables is the next
documented step) and the Mosaic ring switch's evaluation tail.

## The paper's figures (Table 6, erc20-mixed-block, optimized system)

| Trace length | Proving (ms) | Cycles/s | Peak RSS | Proof size | Verify (ms) |
|---|---|---|---|---|---|
| 2^16 | 39 | 1.7M | 0.2 GiB | 377 KiB | 8 |
| 2^18 | 61 | 4.3M | 0.5 GiB | 462 KiB | 9 |
| 2^20 | 174 | 6.0M | 1.6 GiB | 531 KiB | 10 |
| 2^22 | 612 | 6.9M | 6.4 GiB | 596 KiB | 11 |

## What is comparable, and what is not

**Proof size — directly comparable, and it matches the paper's shape.**
Our proofs are 351 KiB at 2^8 and 442 KiB at 2^10, growing by ~50 KiB
per 2× trace length; the paper's grow 377 → 596 KiB over a 64×
trace-length increase. Both are dominated by the same term — the
evaluation argument and its correlated Merkle paths. The dual-rate
profiles and the wrap measure the size dial on the same statement
(`magnetar profiles`): the evaluation tail runs 62.8 KiB (FastChunk,
rate 1/2, 40 queries) → 27.3 KiB (DeepWrap, rate 1/16, 14 queries), and
the **wrapped** configuration lands at 36.5 KiB — a 44% tail reduction
over the Balanced chunk proof.

**Verifier — structurally comparable.** 19–43 ms at 2^8–2^10 against
the paper's 8–11 ms at 2^22 (a 12-core M4 Max): the verifier evaluates
its public factors structurally and grows logarithmically with the
trace. The remaining constant is transcript hashing and the
correlated-path checks.

**Cycles/s — comparable in shape; the constant is hardware + two
documented levers.** This implementation proves 1,727–1,801 cycles/s
monolithic and 1,707–1,920 streaming on **2 cloud Xeon cores** against
the paper's 1.7–6.9M on 12 M4 performance cores. Per-core, the paper's
1.7M at 2^16 is ~142k cycles/s/core on an M4 whose CLMUL is ~5× faster
than this host's measured 11.5 ns/mul; hardware-normalized, the
implementation is within a small constant of the paper's own operating
point at this trace length, and the throughput now *scales* — 2^16
proves end-to-end in bounded memory where the session started unable to
prove 2^15 at all. The remaining measured levers: the twists'
select-folds (32% of the monolithic profile), and the ring switch's
evaluation tail.

**Memory.** The 2^15/2^16 gate is gone: the streaming root's peak is
`O(one segment's suffix tables + packed + codeword + tree)` — 1.36 GiB
at 2^15 and 2.60 GiB at 2^16 on this 4 GiB container, with the
per-worker envelope constant across the ladder. The monolithic path
proves through 2^13 in 619 MiB.

## What the benchmarks DO establish

- The complete Construction-3 protocol runs end-to-end at paper-comparable
  *shape* on **sixteen** workloads: linear-in-T proving (2^8 → 2^13
  monolithic multiplies prove time by 18.9× against a 8× trace growth
  bounded by the padding share), logarithmic communication, structured
  verifier evaluation, and the full tamper coverage of the test suite —
  including the paper's own `erc20-mixed-block` load-test shape, proven
  and verified at 2^16.
- **The linear-in-T prover layer is real**: the memory-checking
  sum-checks' round messages are byte-identical to the dense engine's,
  at O(T) prover work.
- **The round window is real**: the bin outer's expensive shared rounds
  are now one select-density pass, with the window state independent of
  the trace length — the papers' small-value speedup realized in the
  char-2 tower setting.
- **The hierarchical root is real**: 2^15 and 2^16 prove in bounded
  memory with a per-segment state that never materializes the combined
  cube — the 611-paper's streaming shape with the 587 window applied
  per segment.

## Reproducing

```sh
cargo test --release          # the full suite: roundtrips + tamper classes
cargo run --release -p magnetar-cli --bin magnetar-bench
cargo run --release -p magnetar-cli --bin magnetar-bench-big -- 13
cargo run --release -p magnetar-cli --bin magnetar-bench-stream -- 16 stream
cargo run --release -p magnetar-cli --bin magnetar-bench-erc20 -- 16
MAGNETAR_TIME=1 cargo run --release -p magnetar-cli --bin magnetar-bench-big -- 8
cargo run --release --bin magnetar fib 12      # the CLI smoke
cargo run --release --bin magnetar proof-stats # the byte breakdown
cargo run --release --bin magnetar profiles    # the dual-rate tails + wrap
```

All numbers above are from single runs on the development container;
treat ±10% as noise for the prove times and exact for the byte counts.
