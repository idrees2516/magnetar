# bitsum — the bitsliced sum-check kernel (GPU program, Phase 1)

`magnetar-bitsum` is the first crate of the **150M-cycles/s GPU proving
program** (the [roadmap](../../ROADMAP.md) study's build order, item 1:
"GPU sum-check kernels — cutile-rs for the bandwidth-bound folds"). It
implements the *bitsliced multilinear sum-check over GF(2^128)* as a
production-shaped, fully tested host oracle plus NVlabs **cuTile**
(`cutile-rs`) GPU kernels that mirror the oracle one-to-one behind the
`cutile` feature.

Phase-1 exit criterion (from the study): **≥ 2× over the CPU baseline on
1× H100** — the CPU baseline is measured below at 1.5 Melem/s; the cost
model puts the kernel at ~10–30 **G**elem/s.

## The protocol

Multilinear sum-check over the tower GF(2^128) in the statement shape of
MosaicPCS's first-phase sum-check (a public linear combination of table
MLEs) and Ligerito's proximity phase:

```text
claim: XOR over b in {0,1}^m of f(b)        (field sum; char 2)
round: g_r(x) = c0 + c1*x  with  c0 = S0 (low-half XOR),  c1 = running claim
check: c1 == claim_r        (char-2 degree-1 identity g(0)^g(1) = c1)
bind:  f'[i] = f[i] + alpha * (f[i] + f[i + n/2])
soundness: <= m / 2^128 (the final-eval check carries the soundness)
```

Variables bind most-significant-bit first; `eval_mle` uses the matching
convention (`magnetar-field`'s little-endian tables enter through
[`interop`] and are re-ordered at that boundary — tested).

## The tower field (verified, not assumed)

GF(2^128) is the quadratic tower F2 ⊂ F4 ⊂ F16 ⊂ F256 ⊂ F65536 ⊂ F2^32 ⊂
F2^64 ⊂ F2^128 with constants **`NUS[k] = 1 << (2^k − 1)`** (the full
generator product at every level — single-coordinate constants, which is
what keeps the kernel's ν-multiplications cheap). These were found by an
exhaustive basis-trace search and are re-verified by tests at build time:
trace-1 (irreducibility) per level, field axioms on random triples,
`a^(2^128−1) = 1`, and the tree multiplier against scalar multiplication.

## Synchronization with the rest of the stack

Two of the original kernel-spec's Phase-2 items are done as part of landing
in this repository, because the repo's architecture rule is that components
interoperate at crate boundaries:

* **Fiat–Shamir**: the transcript is a thin adapter over the repo-standard
  SHA3-256 duplex (`magnetar-hash`), with the sum-check's own domain
  separation (`"bitsum/sumcheck/v1"`) and labels. Challenges remain full
  128-bit draws — binary fields consume the whole digest with no rejection
  sampling. One transcript construction, one audit surface, repo-wide.
* **Field interop**: [`interop`](#the-interop-bridge) is a *verified
  isomorphism* between this crate's tower basis and the stack's
  `K = F_{2^128}` (the GCM polynomial basis of `magnetar-field`). Tables in
  `K` enter through `interop::prove_gcm`; final claims map back with
  `interop::to_gcm` and discharge against `magnetar-field`'s own
  `eval_mle` — the e2e test does exactly that, cross-crate.

The default build has **no CUDA requirement**: `magnetar-hash`,
`magnetar-field`, and pure-Rust protocol code only. The `cutile` feature is
opt-in, so CI (fmt / clippy / test) stays green on stock runners.

### The interop bridge

The tower and the polynomial basis are two coordinate systems for the same
field. The bridge is built once per process by embedding the tower
generators into `K`: level by level, solving `z^2 + z = p_k` (the image of
`NUS[k]`) by GF(2) Gaussian elimination in a fixed order — deterministic,
no trusted setup, a few hundred `K`-multiplications. Solvability is
structural (`p_k` lies in a proper subfield ⇒ trace 0), and each root lies
*outside* that subfield (the tower constant has subfield-trace 1), so the
128 embedded basis elements are independent and the map inverts. Tests
check the round trip, that multiplication commutes with the map in both
directions (a field isomorphism, not just a linear map), the generator
relations, and the end-to-end proof over a `K` table.

## Bitslicing + the tree multiplier (the kernel's algorithm)

32 elements share one `u32` per coordinate plane; a group is `[128]` words,
laid out group-major to match the GPU tensor `[groups, 128]`. Field
addition of two groups is 128 word-XORs. Multiplication by the round
challenge α — identical for every element of a round — is a GF(2)-linear
map, computed by the Karatsuba tower recursion (`tree.rs`, mirrored
*literally* by the `tower_w1..tower_w128` device functions in the GPU
module):

```text
split value+mask tiles into coordinate halves:
  r0 = tree(a_lo, m_lo)   r1 = tree(a_hi, m_hi)
  r2 = tree(a_lo ^ a_hi, m_lo ^ m_hi)
  nu = tree(r1, NUS[k-1])            <- constant masks, compiler-folded
  out = (r0 ^ nu) ++ (r2 ^ r0)
leaf: a & mask
```

Why the tree and not a dense XOR schedule: cuTile's tile model has no
dynamic-index register access, so a runtime-compiled dense circuit (the
~270 word-ops/element Phase-2 form) is not expressible in the safe DSL —
but the recursion with runtime α-masks and constant ν-masks is, with
identical outputs.

## GPU kernels (the `cutile` feature)

| Kernel | Shape | Work |
|---|---|---|
| `pack_planes` | `[groups,32,4] → [groups,128]`, tile `[16,128]` | 4 unrolled 32-plane chunks, `reduce_sum` bit assembly (exact integer identity) |
| `bind_round` | tile `[32,128]` per program | `out = lo ^ tree(lo ^ hi, alpha)` — coalesced 512 B row bursts |
| `reduce_partial` | tile `[64,128]` blocks, `[1,128]` partial rows | XOR-scan row reduction, hierarchical to one row |

Round orchestration on the host: `S0` via the reduction hierarchy → absorb
→ α → upload 512 B of masks → bind. The tail (< 2048 elements) finishes on
the CPU with the *shared* protocol code (`prove_from_tail`), which is what
makes GPU and CPU proofs bit-identical on the same transcript.

The kernels are written against the published cuTile DSL/host API and are
compiled and gated by the manual `bitsum-cuda` CI workflow (this sandbox
and stock CI runners have no CUDA 13.x toolkit — see the bring-up
checklist below).

## Correctness gates

| Test | Gate |
|---|---|
| 32 unit tests + 1 doc-test (default build) | tower constants (trace-1 per level, single coordinates), field axioms, group order, inverses, pack/unpack round trip, transpose simulator vs naive, tree-vs-scalar (pruned == unpruned), bind/reduce vs scalar, protocol e2e, tamper classes (c1, mid-proof c0, last c0, wrong claim), planes-vs-scalar prover parity, transcript determinism/domain separation, interop round trip + field isomorphism + generator relations + e2e proof over a `K` table |
| `gpu_parity` integration test (CUDA hosts, `BITSUM_GPU_TESTS=1`) | GPU pack / bind / reduce vs the host oracle; full prover bit-identical to CPU; GPU proof passes the CPU verifier + outer eval |

The design rule: **every GPU op sequence exists as a tested host function
first** (`BitPlanes::pack`, `tree::bind_planes`, `tree::reduce_planes`),
and the kernel is a transliteration. GPU correctness therefore reduces to
"the kernel compiles to this op sequence", which `print_ir` + the parity
test verify on hardware.

## Cost model (modeled; measure with `bench_gpu`)

Per element per bind round: 48 B of HBM traffic (read lo + hi, write
bound). H100 SXM (3.35 TB/s) → ~62 G elem-round/s memory ceiling. The
pruned tree (~28 K word-ops per 32-element group, measured on the host
reference) makes the kernel ALU-bound at ~15–30 G elem-round/s on H100's
~29.6 T u32-ops/s. A 2^20-entry instance costs ~2n elem-rounds ≈ 0.1–0.2
ms → **10–20 G elem/s instance throughput** — three to four orders above
the Phase-1 exit criterion, which is the point: the criterion is a floor,
not a ceiling.

Measured CPU baseline (`cargo run --release -p magnetar-bitsum --example
bench_cpu -- 16`, single thread): **1.5 Melem/s** (44 ms at m=16, 686 ms at
m=20). Memory: ≤ 40n bytes peak (640 MiB at 2^24 entries).

Scaling to the 150M-c/s program: this kernel is the per-round engine of
the MosaicPCS/Ligerito sum-checks; instance-level cycles/s is a
whole-prover composition (commitment hashing, many interleaved
sum-checks, tracer) per the roadmap's multiplicative stack.

## GPU bring-up checklist

1. CUDA 13.2+ toolkit (`CUDA_TOOLKIT_PATH`), Rust stable ≥ 1.89, sm_80+ GPU.
2. `BITSUM_GPU_TESTS=1 cargo test -p magnetar-bitsum --features cutile
   --test gpu_parity --release` — the acceptance gate.
3. Inspect `bind_round` with `#[cutile::entry(print_ir = true)]`: confirm
   the constant-mask ν-subtrees folded (dead `AND 0` eliminated). Unfolded
   = still correct, ~4× more tile ops.
4. Confirm the `scan`-based reduction return shape (the one DSL primitive
   the reference docs leave ambiguous) and the `reduce_sum` axis-collapse
   in `pack_planes`.
5. Run `cargo run --release -p magnetar-bitsum --features cutile --example
   bench_gpu -- 20` and log Melem/s against the modeled 10–30 G.

## Phase-2 roadmap

* **Dense CSE circuits** (~270 word-ops/element, ~4× the tree):
  host-compile the mul-by-α XOR circuit per round; requires the raw Tile
  IR surface (gather views / unsafe pointer tiles) or a cudarc+PTX sidecar
  kernel.
* **Fused single-pass bind+reduce** (48 B/elem-round, no separate
  reduction read) — the study's "fused message+fold" item.
* **2^13-plane SMEM tiles running 13 rounds on-chip** — keep the bound
  table in shared memory across rounds, launching once.
* **TMA loads, persistent CUDA Graphs, async streams** between
  pack/reduce/bind.
* **Device-side sponge transcript** (SHA-3/Keccak kernel) so challenges
  stop round-tripping through the host.
* **ZK masking** — the sum-check messages leak round sums; a blinding
  layer is a Phase-5 research item per the program plan.

(Done while landing in this repository, from the same list: the SHA3-256
transcript and the tower↔`K` interop bridge.)
