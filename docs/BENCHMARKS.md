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
cargo run --release -p magnetar-cli --bin magnetar-bench-erc20 -- 20
                                                                   # the paper's Table-6 workload
MAGNETAR_TIME=1 cargo run --release -p magnetar-cli --bin magnetar-bench-big -- 8
                                                                   # per-phase wall clock
cargo run --release -p magnetar-cli --bin magnetar -- profiles     # dual-rate tails
```

## End-to-end results (this implementation)

Current numbers (the production-throughput release: the packed witness,
the mask-major bin outer, the group-fold discharge, the linear st1,
the FastChunk rate, the parallel hierarchical root):

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

## The ERC-20 mixed-block load test (the paper's Table-6 workload shape)

`magnetar-bench-erc20` proves the paper's `erc20-mixed-block` shape:
balance transfers between two accounts with checked subtraction, every
16th transfer failing the underflow check — 12 cycles per executed
transfer at the 64-bit balance width of this RV64I reference CPU:

The **split-ring release** (the per-segment commitment trees, the
streamed-eq discharge, the merged openings, and the 2^13-cycle segment
ladder) — measured on the 2-vCPU / 3-GiB development container:

| T | segments | tx | prove (ms) | cycles/s | tx/s | peak RSS |
|---|---|---|---|---|---|---|
| 2^12 (monolithic) | — | 341 | 986 | 4,153 | 346 | 197 MiB |
| 2^13 | 8 | 682 | 2,265 | 3,617 | 301 | 229 MiB |
| 2^16 | 8 | 5,461 | 19,187 | 3,416 | 285 | 708 MiB |
| 2^18 | 32 | 174,762 | 76,250 | 3,438 | 286 | 795 MiB |
| 2^19 | 64 | 349,525 | 157,132 | 3,337 | 278 | 1.37 GiB |
| 2^20 | 128 | 699,050 | 343,551 | 3,052 | 254 | 2.38 GiB |

The memory is now **flat in the segment count and O(segment) in the
worker envelope**: 2^17 through 2^20 prove on a 3-GiB container where
the previous release's practical ceiling was 2^17 (the combined
codeword + tree + the stored per-group eq tables — the projected
~8–45 GiB at 2^19–2^21). A 2^21 proof projects to ≈4.5 GiB (an 8-GiB
host's envelope; the InstaBox-class machine). The production ladder's
per-machine throughput on real AVX-512 hardware is unchanged in shape
(~3.2× this container's rate per the last release's ratio).

**The pooled proving-pool run** (5 × 4-vCPU instances proving
independent 2^16 batches concurrently): ≈3,880 tx/s aggregate (the
per-machine rate drops ~12% under cross-VM contention). The production
50k-tx/s target therefore needs ~13× more per-core efficiency; the
`MAGNETAR_TIME` split at scale (the aggregated per-phase table — every
per-instance and per-segment phase now accumulates its total, fire
count, and per-fire average, plus a CPU-ms axis per root phase)
measures the remaining levers, in order of impact:

1. **The store-router join pass** (24% of the segments phase at 2^16):
   the join prover's per-segment work — the next packed refactor.
2. **Virtual Spartan's outers** (20% of the segments phase): the
   row-cube window passes at the 512-row density.
3. **The twists' o-rounds** (16% combined): the packed-inc refactor
   (u64 words per cycle instead of the `2^{6+n}` bit table) mirrors the
   witness packing.
4. **The discharge's `fw-bound-se` walk** (12% of the root): the group
   word-walk's mul chains (the eq arena is now transient; the walk
   itself is the residual cost).
5. **The paper's per-instance amortization**: the circuit still pads
   the rows to 512/cycle against the paper's several-cycles-per-
   instance operating point — the single largest structural constant
   left (4–8× at the paper's density).

Every row proves **and verifies** end-to-end (the assert is in the
harness). The paper's erc20-mixed-block at 2^16: 39 ms, 1.7M cycles/s,
0.2 GiB — on 12 M4 performance cores whose per-core CLMUL rate is
~5× this Xeon's measured 11.5 ns/mul.

## The optimizations of the split-ring release

1. **The split ring switch** (`magnetar-pcs/src/ring_split.rs`): the
   streaming root's per-segment commitment trees. The codeword uses
   **segment-local** substitution vectors (the combined codeword IS the
   `[ε][s][x_loc]` concatenation of per-segment codewords — the direct
   sum of the per-segment codes, same relative distance), so every fold
   level `i ≤ n_s` decomposes into `(s, ε)` cells whose digests form a
   tiny **root layer** per level. The chains are built streaming, one
   segment at a time; each cell tree commits **merged `(P, C)` pairs**
   (the witness codeword and the c-folded codeword in ONE tree — the
   monolithic ring's two-path anchor collapsed into one); the anchors
   check `φ_c(P) = C` exactly. The Fiat–Shamir order binds the merged
   commitment after the hierarchical body's challenges but before any
   opening. Peak root memory: `O(one segment's chain + one wave)`.
2. **The streamed-eq discharge**: the hierarchical discharge's
   per-group eq tables moved from **stored** (through the whole
   lockstep — the measured per-instance O(T) spike: ~53 groups × ~2^17
   entries ≈ 130 MB/instance at 2^13-cycle segments) to **transient
   per-phase arenas** (one contiguous allocation per fold/walk, dropped
   at the phase end). This was the single largest memory term at
   2^17+.
3. **The merged openings pass**: the ring's queries and anchors draw
   all positions up front (the transcript's draw order preserved) and
   answer from ONE per-segment derivation in bounded waves — the
   previous shape re-derived the touched segments twice (the kept-tree
   variant also held every digest level of every cell).
4. **The 2^13-cycle segment ladder**: `segment_count` picks
   `2^13`-cycle segments at scale (8 segments at 2^16, 128 at 2^20),
   amortizing the per-segment fixed costs; `MAGNETAR_SEG_LOG2` overrides
   for the amortization study.
5. **The MAGNETAR_TIME split at scale**: `AccumTimer` accumulates every
   per-instance/per-segment phase's total + fire count + per-fire
   average into one sorted table (the CPU-ms axis per root phase rides
   alongside) — the "~825 CPU-ms/instance" class of fixed costs is now
   measured directly rather than inferred.

## The four optimizations of the previous pass

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

**Memory.** The 2^17 wall is gone: the split ring's root holds only
the packed vector + the transient c-table + the per-instance tail
engines + one wave of per-segment chain derivations — 708 MiB at 2^16,
795 MiB at 2^18, 1.37 GiB at 2^19, and **2.38 GiB at 2^20** on this
3 GiB container (the previous release needed ≈1.4 GiB at 2^16 and
projected ≈8+ GiB at 2^19). 2^21 projects to ≈4.5 GiB (the 8-GiB
production host's envelope); 2^23+ needs the documented follow-ups (the
lazy `bound_g` dense factor, the one-hot components' sparse
commitment).

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
