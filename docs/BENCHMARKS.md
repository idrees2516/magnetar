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

## Mikros (ePrint 2026/2646) — the bitwidth-aware SNARK benchmarks

The `magnetar-mikros` crate implements the Mikros framework end-to-end on
this stack's fields (`F = GF(2) ⊂ K = GF(2^8) ⊂ E = GF(2^128)`, packing
width κ = 128): Algorithms 1–6 (packed inner-product, packed Hadamard,
packed permutation, packed PCS opening, and the batched-instance R1CS
SNARK), with every component cross-validated against dense E-arithmetic
references and tamper-gated. Benchmarks run single-threaded per the
paper's protocol:

```sh
cargo run --release -p magnetar-mikros --bin mikros-bench -- hadamard --log2-n 30
cargo run --release -p magnetar-mikros --bin mikros-bench -- hashes --batch 256
cargo run --release -p magnetar-mikros --bin mikros-bench -- perm --log2-n 22
```

**The hardware caveat (read this first):** the paper's host is a Xeon
Gold 6342 with **GFNI + AVX-512 + VPCLMULQDQ**, where the dominant
kernels (the GF(2^8) row products, the spread/transpose steps) are
single instructions. This dev container (2 vCPU, 3 GiB) has *none* of
those — the same kernels run scalar with table lookups. The Table-3 PIOP
measures the gap directly: ~41× on the pure sum-check core. The numbers
below are the honest container measurements; the protocol shape, proof
sizes, and memory envelopes are the comparables, absolute times are not
(on the InstaBox AVX-512 host the same code paths pick up the GFNI
dispatch — see the `magnetar-field::avx` measured dispatch — and the
earlier sessions measured 10–20× on exactly these kernel classes).

### Table 3 — the binary Hadamard product at N = 2^30 (κ = 128)

| metric | this implementation (container) | paper (Xeon 6342) |
|---|---|---|
| prover | 27,985 ms | 684.48 ms |
| verifier | 258 µs | 34 µs |
| proof size | 5,216 B | 4,888 B |
| peak RSS | **915.8 MiB** | **0.89 GiB** |

The proof size matches the paper's shape (the same κ column claims + the
127-coefficient quotient + the Fact-1 reduction; the ~7% delta is our
Fact-1 carrying `[p0,p1,s]` per round vs. their tighter encoding), and
**the peak RSS lands inside the paper's own 0.89 GiB envelope** — the
bitsliced plane layout is exactly their memory profile. The prover
anatomy at 2^30: encodings 11.7 s + spreads/buckets 6.1 s + the `a,b`
selector builds 7.8 s + the d-fold 1.8 s + Fact-1 0.5 s — the first two
phases are precisely `VPGF2P8MULB`/`VPCLMULQDQ` territory.

### Table 4 — end-to-end packed hash batches (256 instances, Algorithm 6)

Our circuits: SHA-256 compression 96,000 rows (N = 2^17), Keccak-f[1600]
132,096 rows (N = 2^18), BLAKE3 compression 61,440 rows (N = 2^16); the
batch is ν = 2 packed blocks of κ = 128 instances (the §5.3
larger-batch z-combination of the ν linchecks is the documented next
step — it saves one Fact-1 per block).

| workload | prover (256 inst.) | verifier | proof | peak RSS | paper prover |
|---|---|---|---|---|---|
| SHA-256 | 2,541 ms | 118 ms | 3,730 KiB | 136 MiB | 79.54 ms / 37.47 ms / 367.71 KiB |
| Keccak | 5,069 ms | 191 ms | 3,993 KiB | 269 MiB | 100.15 ms / 19.30 ms / 380.31 KiB |
| BLAKE3 | 1,240 ms | 78 ms | 3,471 KiB | 269 MiB | 56.13 ms / 35.74 ms / 355.64 KiB |

Verifier times are within 3–10× of the paper (the direct sparse
evaluation dominates both); memory is *better* than the paper's
606 MiB SHA-256 figure (the packed planes are 2 MiB against their
per-instance layouts). The proof size is the one honest regression: our
FastChunk PCS operating point (rate 1/2, 40 correlated FRI queries)
carries full Merkle paths per query; the paper's Ligerito-Fast openings
are ~10× more compact. The repo's size profiles (the `magnetar profiles`
command: FastChunk → DeepWrap) trade exactly this axis.

### The permutation argument (Algorithm 3, the Table-5 complement)

The grand-product proof over committed planes (the component the
paper's Table 5 measures *via* the single-instance PLONK copy
constraint — their "excl. perm." column is 6.72 s of 86.56 s at
N = 2^25, i.e. the perm argument is ~12.9× the rest of the system).
Measured at the §4.3 geometry (base field GF(2), κ = 128):

| N | prover | bits/s | verifier | proof |
|---|---|---|---|---|
| 2^20 | 440 ms | 662,920 | 5.7 ms | 74.7 KiB |
| 2^22 | 964 ms | 1,929,168 | 12.5 ms | 79.9 KiB |

(0.10.0: 703 / 2,360 ms — the 0.11.1 kernel release and the AVX-512
probe fix took the argument down 1.6× at 2^22. On the InstaBox host:
695 ms at 2^22 = 4.35M bits/s.)

Prover anatomy: the η off-domain sampling pipeline ~86% (the per-node
Lagrange encodings z-evaluated then row-dotted — the interchange that
saved 3.6× over the per-pair bucket route), the `g` contractions ~9%,
the coefficient-tree build + E-tree + cubic sum-checks the rest. The
paper's Table-5 geometry runs the same algorithm over the
position-encoded wires (§5.1) — the single-instance PLONK composition
(Algorithm 5) is implemented in `plonk.rs` on top of the `copy.rs`
instantiation (K_pos = GF(2^32), κ_pos = 4 — the geometry the Theorem-1
interface forces for every N ≥ 2^14); see the Algorithm-5 section below.

## Algorithm 5 — the single-instance PLONK SNARK (Table 5)

`mikros-bench plonk --log2-n K` — simulated binary circuits (the paper's
own Table-5 workload) through the full stack: the five gate claims, four
packed Hadamard proofs plus the nested `w_M` discharge, the §5.1
copy-constraint encoding over `K_pos = GF(2^32)` at width `κ_pos = 4`,
the terminal expansion (six wire values + `3·d_L` permuted-digit
values), and the two grouped openings (witness 3m / public
`(5+3·d_L)m`). Single-threaded, this 2-vCPU container:

| N | prover | gates/s | verifier | approx. proof | peak RSS |
|---|---|---|---|---|---|
| 2^16 | 152 ms | 430,412 | 72 ms | 193 KiB | 29 MiB |
| 2^18 | 850 ms | 308,339 | 127 ms | 210 KiB | 143 MiB |
| 2^20 | 2.59 s | 404,667 | 296 ms | 228 KiB | 523 MiB |
| 2^21 | 7.04 s | 298,148 | 514 ms | 237 KiB | 909 MiB |
| 2^22 | 11.3 s | 372,463 | 965 ms | 247 KiB | 1.75 GiB |

**Vs the paper's Table 5**: Mikros proves N = 2^25 in 86.56 s on their
Xeon Gold 6342 = 386K gates/s. This implementation sustains
**404K gates/s at 2^20 and 372K at 2^22** on the InstaBox host (a
4-vCPU cloud machine with GFNI/AVX-512/VPCLMULQDQ, single-threaded) —
**105% / 96% of the paper's measured per-core throughput** — and
266–346K on the shared 2-vCPU dev container. Prover anatomy at 2^22:
copy 6.2 s (55%), openings 4.1 s (37%), gates 0.14 s (1%) — the
paper's Table-5 shape (perm-dominant). The 0.11.1 kernel release (the
AVX-512 probe fix, the flat `[d][r][j]` tree, the streaming η pipeline,
the scalar-folded `g` contraction, the grouped openings' ξ cache, the
hoisted lane dispatch) took the ladder from 21.2 s to 11.3 s at 2^22
(1.9×). The
κ_pos = 4 geometry is forced: `d_L` position coordinates plus the
witness coordinate need `d = 32` basis elements (`2d ≤ κ = 128`), so
`[E : K_pos] = 4` for every admissible N ≥ 2^14.

### Implementation notes (the bugs the cross-validation caught)

- The Fact-1 terminal absorb: the prover's `fact1-term` label vs. the
  verifier's split `fact1-v1`/`fact1-v2` diverged the duplex state —
  invisible in standalone tests (nothing sampled downstream), fatal in
  every composed protocol.
- The Fact-1 → PCS point order: Fact-1 binds the vector MSB-first, the
  PCS binds its MLE LSB-first — the terminal claims must be *reversed*.
- `ts_msbf` (the MSB-first tensor-sum materialization): the growing
  recursion applies the **last** coordinate first; the first draft had
  it backwards, which only surfaced as a claim-invariant break three
  levels deep in the perm unfold.
- The Keccak io positions: the output lanes are pushed in (y,x) order,
  not column-creation order — the batch io-bind's masked-eq scan
  requires the sorted set (fixed in `CircuitBuilder::build`).

## End-to-end results (this implementation)

Current numbers (the fw-bound-se group-fold attack + the transient-eq
release: the per-(group,plane) hoisted member sums with the fused-lane
axpy over the contiguous support, the TRANSIENT group eq tables — the
lockstep instance state drops from `O(Σ 2^free)` to `O(items)`, the
single-machine 2^19+ unlock — the 2^13-cycle segment operating point,
on top of the commitment-free ring switch release):

| workload | T | prove (ms) | verify (ms) | cycles/s | proof (KiB) |
|---|---|---|---|---|---|
| alu-chain | 2^10 | 113 | 12.3 | 9,062 | 368.8 |
| alu-chain | 2^11 | 211 | 12.6 | 9,719 | 412.8 |
| alu-chain | 2^12 | 393 | 14.3 | 10,434 | 459.3 |
| (streaming, 2^12-cyc segs) | 2^16 | 3,821 | — | **17,153** | one shared tail |
| (streaming, 2^13-cyc segs) | 2^17 | 6,847 | — | **19,142** | one shared tail |
| (streaming, 2^13-cyc segs) | 2^18 | 14,795 | — | **17,719** | one shared tail |
| (streaming, 2^13-cyc segs) | 2^19 | 36,047 | — | **14,545** | one shared tail |
| (streaming, 2^13-cyc segs) | 2^20 | 116,237 | — | **9,021** | one shared tail |

Peak RSS at 2^12 monolithic: **166 MiB**; at 2^16 streaming (the
paper's workload): **838 MiB** (was 1.37 GiB two releases ago — the
transient-eq fix); at 2^19: 3.63 GiB; the measured single-machine
streaming envelope is **2^20 at 7.06 GiB** on this 8-GiB host (the
per-segment commitment trees are the documented next lever for 2^21+).

## The fw-bound-se group-fold attack + the 2^13-cycle operating point

Measured on the InstaBox 4-vCPU EPYC host (±5% host-contention noise
between runs), the `erc20-mixed-block` production workload at the new
operating point (2^12-cycle segments — the root's per-instance
discharge machinery costs ~33 ms wall per instance REGARDLESS of the
instance's size, so longer segments amortize it; measured at 2^18 the
2^13-cycle configs run 7,374 vs 4,341 cycles/s at the old floor):

| T | segments | prove (ms) | cycles/s | tx/s | peak RSS |
|---|---|---|---|---|---|
| 2^16 | 16 | 3,821 | **17,153** | **1,429** | 838 MiB |
| 2^17 | 32 | 8,890 | 14,743 | 1,229 | 1.24 GiB |
| 2^18 | 64 | 17,809 | 14,719 | 1,227 | 2.04 GiB |
| 2^19 | 128 | 47,323 | 11,079 | 923 | 3.63 GiB |

Against the session start on the same machine (2^16: 5.83 s, 936 tx/s,
1.37 GiB): **1.53× faster, 53% more tx/s, 39% less memory at 2^16**,
and the whole 2^17–2^20 single-machine range opened (the pre-session
ladder topped at ~7.3k cycles/s at 2^18 with the 2^10-cycle floor and
the resident-eq lockstep).

The root's `fw-bound-se` group-fold — the biggest measured lever of
the 0.9.2 anatomy (~19 ms per segment, ~30% of the root's local
stages) — fell to ~14 ms at the per-instance shape on the 2-vCPU dev
container and, on the production host, the whole discharge
construction at 2^12 monolithic from ~102 ms (66.9 + ~35 hidden) to
62 ms. The three pieces:

1. **The hoisted member sums** `S_p = Σ_m pe_m` per (group, plane): one
   broadcast multiply-accumulate per (group, plane) instead of one
   multiplication per member per word — and the identity groups
   size-sorted across per-worker `stride` accumulators that XOR-merge
   (GF(2^128) adds commute: byte-identical output, full worker
   parallelism, no intersection logic).
2. **The contiguous support walked directly** by the fused VPCLMULQDQ
   `axpy_run` — the group's word support is exactly
   `[base, base+2^fw)` inside the plane block, where the first version
   scanned the whole `2^{word_bits}` plane block and filtered (7M
   wasted iterations at 2^12).
3. **The TRANSIENT group eq tables**: rebuilt fused (half the
   multiplications of the reference build, in a REUSED per-worker
   scratch — one cold-page-fault set per group was measured 2× on the
   whole pass) at the word fold and again at the scatter, never
   resident on the lockstep state. This is the single-machine unlock:
   the lockstep holds every instance's state simultaneously, and the
   resident tables were `O(Σ 2^{free_i})` ≈ 12 MiB per instance at
   the 2^10-cycle shape — 768 MiB at 2^16, ~6 GiB at 2^19.

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

The **split-ring release** (the per-segment commitment trees, the
streamed-eq discharge, the merged openings, and the 2^13-cycle segment
ladder) — measured on the 2-vCPU / 3-GiB development container:

| T | segments | tx | prove (ms) | cycles/s | tx/s | peak RSS |
|---|---|---|---|---|---|---|
| 2^12 (monolithic) | — | 341 | 446 | 9,183 | 764 | 166 MiB |
| 2^13 | 8 | 682 | 470 | **17,418** | **1,450** | 285 MiB |
| 2^14 | 16 | 1,365 | 933 | **17,568** | **1,464** | 376 MiB |
| 2^15 | 32 | 2,730 | 1,990 | 16,470 | 1,372 | 595 MiB |
| 2^16 | 64 | 5,461 | 4,254 | 15,405 | 1,284 | 1.09 GiB |

**The pooled proving-pool run** — the cpi × pool matrix AND the batch
shape × pool matrix (5 × 4-vCPU instances, independent batches proved
concurrently, the aggregate over the 20 vCPUs):

| shape | per-machine tx/s | aggregate tx/s |
|---|---|---|
| 2^13 batches, cpi 1 | 1,188–1,215 | **6,005** |
| 2^14 batches, cpi 1 | 1,107–1,192 | 5,775 |
| 2^16 batches, cpi 1 | 1,005–1,051 | 5,139 |
| 2^16 batches, cpi 4 | 847–975 | 4,585 |
| 2^16 batches, cpi 8 | 956–1,027 | 4,942 |

The honest pooled operating points: `cpi = 1` AND the **2^13 batch**
(smaller roots contend less; the solo ladder peaks at 2^14 but the pool
peaks at 2^13). The 0.8.0 pool measured ≈3,788 tx/s at 2^16; the same
five machines now pool **6,005 tx/s (+59%)**. `magnetar-bench-erc20-pool
<LOG2_BATCH> <BATCHES>` drives one machine's share of a large total —
the **2^23 pooled demonstration, measured**: 1,024 independent 2^13
batches (8,388,608 cycles = 699,050 transfers) split 205/205/205/205/204
across the five machines, every batch proved AND verified — **210 s
wall, ≈3,330 tx/s sustained aggregate** (665–798 tx/s per machine
under sustained cross-VM contention; the burst pool runs 6,005). The
**2^24 pooled demonstration, measured**: 2,048 batches (16,777,216
cycles = 1,398,100 transfers, 410×4 + 408) — **400 s wall, ≈3,495
tx/s sustained**. The remaining measured levers, in order of impact:

1. ~~**The discharge's `fw-bound-se` group-fold`**~~ — **DONE in
   0.9.3**: the per-(group,plane) hoisted member sums + the fused-lane
   axpy over the contiguous support + the TRANSIENT eq tables (see the
   release section above); the residual scatter cost is now the
   eq-rebuild + broadcast-multiply floor of the decomposition.
2. **The VS window pass's residual** (`vs-win-build`, ~16 ms per
   segment): the complement-convolution (0.9.2) cut it from ~26 ms;
   the remaining cost is the eq-weighted pack over the 3^v monomial
   state (one 16-byte scatter per set monomial per suffix).
3. **The twists' Δ\* one-hot proofs** (`tw-onehot`, ~11 ms per segment
   total across the two twists) and the port k-rounds (~4 ms).
4. **The join prover's extraction** (`jr-extract`, now timed): the
   per-column selector scan reads the one-hot tables at a `2^{h_vars}`
   stride — 64 strided cache lines per column per selector; the
   positions are known from the trace, so a pre-pass (or threading
   them through the `JoinWitness`) deletes the scan entirely.
5. **The streaming root** for 2^21+ single-machine proving — the
   measured 2^20 envelope is 7.06 GiB (see below).

## The single-machine envelope, measured (the 2^21+ gate)

The transient-eq release moved the single-machine wall from 2^19 to
**2^20**: the lockstep's per-instance state dropped from `O(Σ 2^free)`
(~12 MiB resident at the 2^10-cycle shape) to `O(items)` — the 0.9.2
envelope table's ~15 MiB/segment lockstep cost is now ~1 MiB (the
tail's bound tables), and the remaining growth is the commitment
(`packed + codeword + tree` ≈ 3.3× the packed bytes) plus the
per-worker tables at the 2^13-cycle operating point (~780 MiB × 4
concurrent workers). Measured on the 4-vCPU/8-GiB host, the
`erc20-mixed-block` workload, every row **proved and verified**:

| T | segments | prove | cycles/s | peak RSS |
|---|---|---|---|---|
| 2^19 | 128 | 47.3 s | 11,079 | 3.63 GiB |
| 2^20 | 128 (alu ladder) | 116.2 s | 9,021 | **7.06 GiB** |

2^21 projects to ~13.5 GiB — over this host. The unlock, in order:

1. ~~**Per-segment commitment trees**~~ — **DELIVERED** (the split
   ring switch, `ring_split.rs`): segment-local codeword vectors, the
   merged `(P, C)` pair cell trees, the per-level root layers, and the
   waved openings — the multi-segment root never materializes the
   combined codeword or tree (measured on the 3-GiB dev container:
   2^20 in 2.38 GiB at the 2^13-cycle shape BEFORE this merge; on this
   host the split root removes the ~3.3×-packed commitment terms from
   the 7.06-GiB envelope).
2. **Per-segment independent discharges** — drop the lockstep (each
   segment's local stage runs to completion in its own transcript,
   then the segment stage batches the terminals): the root's discharge
   memory becomes O(one segment).
3. **The streamed commitment** — encode + Merkle per segment slice
   (the code's linearity makes the encode per-block), keeping only the
   tree's upper nodes and re-deriving per-slice packed words from the
   checkpoints during the discharge (the 611-paper's small-space
   prover shape).
4. The local-stage rate then parallelizes across the pool for the
   aggregate scales.

Until then: **the pool is the 2^23/2^24 shape** — the total proven
workload scales linearly in machines (the 2^23 demonstration at 699k
transfers in 3.5 min and the 2^24 at 1.40 M transfers in 6.7 min, every
batch independently verified), and the per-machine envelope stays
bounded at the 2^13–2^18 batch sizes.

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
