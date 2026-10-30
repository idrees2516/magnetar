# Changelog

## 0.12.0 — BinarySpartan on the Mikros stack: the production SNARK path for R1CS

*BinarySpartan* (Setty, *BinarySpartan: Spartan over binary fields*, IACR
ePrint 2026/1656) instantiated on this repo's packed multi-instance
layout: **Spartan's outer + inner shape** for the bit-level R1CS with the
witness committed through the fold-chain PCS (Ligerito's role), the
matrices as verifier-side public factors at the real-Spartan operating
point (direct sparse-MLE evaluation, O(nnz)), and the witness claim
opened through Algorithm 4's packed opening. Transparent, hash-secure,
polylog proofs, a linear-work prover, constant-extra-memory.

### The protocol (`magnetar-mikros::spartan`)
1. **The public-I/O bind** — Algorithm 6's steps 3–5 verbatim (the
   masked-eq Fact-1 → a plain PCS opening).
2. **The outer sum-check** — `0 = Σ_{(j,i)} eq(τ_j,j)·eq(τ_i,i)·[L·R + O]`
   over the joint (instance ‖ constraint-row) cube, in two phases:
   - **The instance phase** (7 rounds, degree-3 messages): the prover
     never materializes an E-valued cube — per free plane it forms the
     challenge-weighted combinations of the *original* GF(2) planes
     (XOR-selects) and takes the `l = eq(τ_i,·)`-weighted dots; the bound
     coordinates' `eq(τ_j, ρ)` products ride the running claim. Constant
     memory, work halving per round.
   - **The row phase** (log N rounds): the images fold once at
     `eq(j*,·)` into E-valued row tables and the engine finishes the
     sum-check — the public `eq(τ_i,·)` factor rides the never-folded
     eq-slab discipline; the term coefficients carry `eq(τ_j, j*)`.
3. **The inner sum-check** — `⋆̂(j*,i*) = Σ_y Ŝ⋆(i*,y)·Ŵ(y,j*)`: an
   α-batched Fact-1 over the witness columns; the verifier
   direct-evaluates the matrix MLEs (O(nnz)), the witness side is the
   `eq(j*,·)`-folded packed table.
4. **The openings** — the io claim through the plain fold-chain
   evaluation; `Ŵ(y_in, j*)` through Algorithm 4's packed opening (the U
   coordinate claims + the char-2 W recursion).

### Three bugs found by the composed protocols (the repo's discipline)
- **`pcs_open`'s verify-side point order** (the first end-to-end caller):
  the W-recursion and the terminal PCS opening paired the *un-reversed*
  Fact-1 challenge list — the prove side had the reversals, the verify
  side skipped both, and no earlier test ever drove `verify_packed_open`
  (plonk carries its own grouped opening). Every honest proof failed at
  exactly the W-recursion check.
- **The missing accumulated eq factor**: each instance-round message must
  carry `Π_{m<k} eq(τ_j[m], ρ_m)`; the first draft dropped it, and the
  toy circuit validated *trivially zero on both sides* of the round-sum
  identity — only the N = 2^17 SHA differential caught it (round 1's sum
  disagreed with the chain while round 0's message matched the
  reference).
- **The bound-pattern bit order**: the c-tensor update shifted the old
  bits up (`nc[2b]`), reversing the bound pattern from round 2 on; the
  new variable must land at the *top* bit (`nc[b]`, `nc[b+half]`). Also
  invisible at the toy's 2-row circuit, decisive at round 2 of the SHA
  differential.

### The BinarySpartan technique map (2026/1656 on this stack)
| BinarySpartan | Here |
|---|---|
| Spartan over binary fields (outer + inner) | `magnetar-mikros::spartan` (this release) |
| Ligerito + DP24 ring switching | `magnetar-pcs` (the fold-chain backend every opening uses) |
| Phalanx SIMD R1CS | the packed `S⋆W` planes (Algorithm 6's layout) |
| eq/sum-check optimizations (Gruen 2023; Dao–Thaler 2024/1210; BDDT) | the engine's never-folded eq slabs + `gf8`'s 64K byte tables (landed in earlier releases; this protocol routes its row phase through them) |
| Binius64 byte lookup tables | `gf8::mul_tab` — the T16 bucket route (0.10.0) |
| SuperSpartan's next multilinear extension | **not implemented** — the paper text was not retrievable (eprint PDF access is Cloudflare-gated this session); documented in `docs/STATUS.md` rather than guessed at |

### Measured (the Table-4 workloads, both paths on the same instances)
Single-threaded, this 2-vCPU container (the GFNI host comparison is the
usual honest hardware-gap anatomy — `docs/BENCHMARKS.md`):

| SHA-256 (N = 2^17) | Algorithm 6 | BinarySpartan |
|---|---|---|
| prover | 468 ms | 1362 ms |
| verifier | 53.7 ms | **27.6 ms** |
| proof | 960 KiB | 953 KiB |

| Keccak (N = 2^18) | Algorithm 6 | BinarySpartan |
|---|---|---|
| prover | 942 ms | 1807 ms |
| verifier | 83.8 ms | **47.9 ms** |
| proof | 1061 KiB | 1055 KiB |

| BLAKE3 (N = 2^16) | Algorithm 6 | BinarySpartan |
|---|---|---|
| prover | 239 ms | 769 ms |
| verifier | 33.2 ms | **17.5 ms** |
| proof | 863 KiB | 856 KiB |

(The 0.11.2 batched-leaf PCS halves every proof and speeds both provers;
these are the re-measured numbers on the rebased base — the first draft
of this entry carried the pre-rebase figures.)

The verifier wins 1.7–1.95× (the direct sparse-MLE route replaces the U⋆
coordinate messages and the per-matrix public-value scans); the proof
size is a hair smaller. **The prover is 1.9–3.2× slower in this first,
deliberately plain form** — the instance phase's per-free-plane dots are
scalar E-multiplies; the levers (the VPCLMULQDQ lane kernels on the
combinations, the fused dot3, the l-weight folded once into the images)
are the same class the 0.11.x releases pulled on Algorithm 5, and the
per-phase anatomy is the next release's material. RSS at Keccak: 260 MiB
(the same envelope as Algorithm 6).

### Tests
- `spartan_roundtrip_and_tamper` — the honest verify + seven tamper
  classes (the io words, an instance-round coefficient, a row message,
  the terminal claims, the inner Fact-1, the packed opening's v_msg, the
  witness claim) + the unsatisfiable-batch rejection.
- `spartan_dense_reference_and_inner_identity` — the lazy instance-round
  messages byte-matched against a materialized joint-cube reference; the
  terminal claims brute-forced against the direct joint MLEs; the
  Spartan inner identity asserted brute-force on both sides.
- `spartan_sha256_dense_differential` (`--ignored`, ~6 s) — the same
  machinery at N = 2^17 with a per-round chain oracle (the true
  remaining sum from the folded state): the invariant that caught two
  of the three bugs.
- `spartan_cross_check_algorithm6` — the same instances verify through
  both SNARKs.
- `engine_finish_vs_manual_fold` — the engine's terminal claims vs the
  plain interleaved fold (the cross-crate regression guard).

# Changelog
## 0.11.2 — the remaining-parts release: the batched-leaf PCS + the ERC-20 real-time benchmark

Measured on the AVX-512 host (2 vCPU, single-threaded, before → after):
- **Algorithm 5 (Table 5)**: 2^20: 3.75→2.51 s (418K gates/s = **108%
  of the paper's 386K per-core**), 2^22: 15.6→10.6 s (396K = 103%);
  peak RSS at 2^22 down 15% (1.89→1.60 GiB); setup (pack+commit)
  1.47→0.61 s at 2^20.
- **The openings' anatomy**: the public group's PCS fold+trees
  1,175→331 ms, the PCS sum-check 271→152 ms, the grouped Fact-1
  66→49 ms (`dot_vv`).
- **The expansion phase (steps 15–16)**: 183→55 ms (the AVX-512 tile
  transpose + one popcount walk over the hoisted row words).
- **The ERC-20 state-transition core**: NEW — `mikros-bench erc20`,
  2816 rows/transfer, 142–152 transfers/s single-core at every size
  2^16..2^22, ~7.5K cycles/gate, ~21.5M cycles/transfer at 3.20 GHz
  (TSC-bound). **A full 600-transfer Ethereum block's state-transition
  core proves in 4.0 s — real time against the 12 s slot, 3.0×
  headroom.** The extended stacks (keccak storage proofs, ECDSA, full
  EVM) are quantified honestly in docs/BENCHMARKS.md — not real-time
  on one core, with the levers named.

### The batched-leaf PCS trees (magnetar-hash + magnetar-pcs)
One BLAKE3 call now covers 8 codeword values (`width 8` once a fold
level is ≥ 16 wide, pairs below that — the tail included, never
width-1: an even `a_idx` always shares its group with the odd `b_idx`,
so ONE group reveal + ONE path covers the fold pair). The
fold-chain's Merkle work drops ~4× (the leaf stream amortizes the
per-call fixed cost over 8 values); `MerkleTree` gained
`build_batched`/`verify_batched`/`leaf_digest_of_batched` with a fresh
`magnetar/leafN/v1` domain. The correlated queries reveal the a/b
group + the c group (slightly SMALLER proofs than the old separate
a/b paths). The ring and the zkVM wrap (the deep consistency opens)
moved to the same group format; eval, ring, and ivc/zkvm all derive
the width from the level length deterministically.

### The single-message sum-check (magnetar-pcs eval.rs)
`prove_eval`'s round loop called `prover.message()` and then
`absorb_and_fold` — which computed the message a SECOND time. One
message per round now (absorb in the identical byte format, fold via
`fold_msg`): the public group's sum-check 271→152 ms.

### The VPCLMULQDQ-laned Möbius shift (magnetar-pcs code.rs)
`shift_monomials`' pair loop is the strided-axpy shape — now
`avx::axpy_run` per block (bit-identical to the scalar fold): the
codeword encode 127→~50 ms at the 2^20 public-group geometry.

### The AVX-512 tile transpose (magnetar-mikros r1cs.rs)
`transpose_ps` called the recursive scalar `planes::flip64` reference;
it now calls the AVX-512 block-swap `simd::flip64_v`. The expansion
phase's 69 per-plane transposes: 211→6 ms; the setup's pack+commit
path got the same win for free (742→611 ms → with the encode kernel
0.61 s).

### The expansion rewrite (magnetar-mikros plonk.rs)
Steps 15–16 walked `packed_row_word` per row per claim (78 passes ×
m rows × 128 bit tests). Now: one transpose per plane (hoisted),
one popcount walk per claim over the row words, and `dot_vv` for the
value dots. A masked-gather AVX-512 kernel was tried for the operand
build and LOST to the scalar popcount walk (219 vs 49 ms per 71
claims) — the walk stays; the lesson is recorded in the code.

### dot_vv (magnetar-field avx.rs)
The vectorized inner product (one XOR-accumulating pass through the
measured backend, horizontal-reduced 512→256→128) — Fact-1's dots,
the cubic sum-checks' terminals, and the expansion's value dots now
run on it (bit-identical; unit-gated against the scalar fold).

## 0.11.1 — the copy-layer kernel release: the AVX-512 probe fix + the flat tree + the streaming η pipeline

Measured (single thread, the default dispatch):
- **Algorithm 5 (Table 5)**: 1.9× across the ladder — 2^16: 267→152 ms
  (430K gates/s), 2^20: 4.62→2.59 s (405K gates/s), 2^22: 21.2→11.3 s
  (372K gates/s) on the InstaBox GFNI host; 266–346K gates/s on the
  shared 2-vCPU dev container. **The paper's own Table-5 throughput is
  386K gates/s (Xeon Gold 6342 at N = 2^25) — this release reaches
  105% of it at 2^20.**
- **The perm argument (§4.3)**: 2^20: 579→440 ms, 2^22: 1.89→0.96 s
  (local); 695 ms at 2^22 on the GFNI host (4.35M bits/s).
- The Hadamard Table-3 shape: 427 ms at 2^26 on the GFNI host (725 ms
  on the dev container).

### The AVX-512 probe fix (magnetar-field `avx.rs`)
The runtime micro-probe that picks the VPCLMULQDQ width compared UNEQUAL
work: the scalar probe did 16,384 multiplies, avx2 16,384 (8,192 ×
mul2), but **avx512 did 65,536 (16,384 × mul4) — 4× the work of the
scalar baseline**, silently biasing the dispatch against AVX-512 on
every capable host. The probe now does 16,384 multiplies on every
candidate (4,096 × mul4); this host's sustained kernels measured 2× in
favor of AVX-512, and the dispatch now agrees (the `MAGNETAR_SIMD`
override still wins).

### The gf32 lane kernels (the width-4 K_pos forms)
- **The lane-packing bug**: `_mm512_clmulepi64_epi128` pairs operands per
  128-bit LANE (the low 64 of each under imm 0x00) — the draft's
  `[a0..a3, 0,0,0,0]` packing computed only TWO of the four products
  (the odd slots never paired). The operands now sit at the EVEN 64-bit
  slots (`[a0, ·, a1, ·, a2, ·, a3, ·]`) and the results read from the
  even slots — caught by running the whole suite with
  `MAGNETAR_SIMD=avx512` (the default dispatch never exercised the path:
  a lesson in dispatch-gated test matrices).
- **The target_feature discipline**: the lane bodies live in
  `#[target_feature(enable = ...)]` functions (the repo's avx.rs
  pattern) — inlined-in-place intrinsics without the attribute compile
  to memory round-trips per instruction (a measured 4× regression
  before the restructure).
- **The 256-bit form first**: the ymm VPCLMULQDQ (2 products/register,
  `avx512vl`) is preferred over the zmm form when both exist — the
  measured dispatch and this host's sustained rates both favor it for
  the 32-bit field; the zmm form remains for hosts where it wins.
- `dot4`'s horizontal fold: the two-stage lane tree (i ↔ i^1 then
  i ↔ i^2), no scalar tail.

### The copy-layer kernels (magnetar-mikros `copy.rs`)
- **The flat tree**: the coefficient tree's nested `Vec<Vec<Vec<u32>>>`
  became one flat `Vec<u32>` per level in `[d][r][j]` order — the four
  columns of a (d, r) pair sit in ONE cache line (the dot4/encode
  kernels' access pattern); the draft pointer-chased three allocations
  per element.
- **The streaming η pipeline**: the off-domain f-sampling materialised
  `ncoef·rows` E-vectors per node (hundreds of MB of traffic at
  `m_pos = 2^20`); it now streams fixed 2,048-row blocks — enc (dot4) →
  embed → the `z^d` broadcast-accumulate → the block's triple dot,
  all L2-resident, the triple dots XOR-folding into the node scalar.
- **The `g` contraction**: `ρ_j·z^d` folded into one scalar per (j, d)
  — one E-mult per row (the scale-then-axpy draft spent two).
- **The grouped openings' ξ cache**: the `3·d_L` digit claims share one
  ξ — the words expansion and the plane transpose now run once per
  DISTINCT factor list, not once per claim.

### Verified on real GFNI/AVX-512 hardware (the InstaBox cloud host)
The full workspace suite (214 tests) green, and the benchmark ladder
(single-threaded, 4-vCPU host with GFNI/AVX-512/VPCLMULQDQ):

| N | prover | gates/s | verifier |
|---|---|---|---|
| 2^16 | 152 ms | 430,412 | 72 ms |
| 2^20 | 2.59 s | **404,667** | 296 ms |
| 2^22 | 11.3 s | 372,463 | 965 ms |

**404K gates/s at 2^20 exceeds the paper's own measured Table-5
throughput** (86.56 s at N = 2^25 = 386K gates/s on their Xeon Gold
6342). The Table-3 Hadamard PIOP: 427 ms at 2^26; the perm argument:
695 ms at 2^22; the Table-4 256-instance hash batches: SHA-256 1.13 s /
Keccak 2.35 s / BLAKE3 0.58 s (the paper's 80/100/56 ms run on their
dedicated host with the Ligerito Fast-rate parameters — the gap there
is the PCS commitment share, not the PIOP).

Gate: 214 workspace tests green in debug AND release (including the
full suite under `MAGNETAR_SIMD=avx512`), fmt clean, clippy
--workspace --all-targets clean.

## 0.11.0 — Algorithm 5 (the single-instance PLONK SNARK) + the copy layer over K_pos; the Mikros stack complete

The last protocol piece of ePrint 2026/2646: the end-to-end
single-instance composition (§5.1–§5.2, Algorithm 5) on this stack's
fields — the gate layer, the §5.1 copy-constraint encoding over
`K_pos = GF(2^32)` at width `κ_pos = 4`, and the two grouped PCS
openings (Appendix C.6). With this, all six algorithms of the paper are
implemented, cross-validated, tamper-gated, and benchmarked.

### The new modules
- **`gf32`** — the position-encoding field `K_pos = GF(2^32)` (the unique
  order-2^32 subfield of `E`): the GF(2)-power-basis code representation
  (so the joint value–position label of position `n` with witness bit `w`
  is *literally* `code = n | (w << 31)` — Lemma 2's injectivity for free),
  PCLMULQDQ multiplication with a **two-CLMUL Barrett reduction** (valid
  for any modulus shape — no sparse-polynomial search), the 4-lane
  VPCLMULQDQ batch kernel (`mul4`/`dot4`), and byte-table E-embedding.
- **`copy`** — Algorithm 3 at the §5.1 geometry: the packed permutation
  proof over `K_pos` at width 4 (the coefficient tree as u32 codes over
  VPCLMULQDQ lanes, the width-4 Lagrange interpolation data, the E-tree
  on the shared cubic engine, two coefficient transitions, the terminal
  packed claims) — plus **Proposition 5** (the verifier's identity-label
  evaluation in O(κ_pos + log L + n_io): the column/row label split with
  the public-I/O partial column).
- **`plonk`** — Definition 1 (the constraint system + a builder with
  wiring-cycle σ), the full Algorithm 5 driver: the five gate claims
  (Eq. 12–13), four Algorithm-2 reductions + the nested `w_M` discharge,
  the copy layer with the **terminal expansion** (the column=segment
  structure regrouping `l₀`'s tensor factors to the witness packing
  width), the six wire values + `3·d_L` permuted-digit values with both
  recombinations, and the **two grouped openings** (witness group 3m /
  public group `(5+3·d_L)m`): per-claim `U^⊤β` column messages,
  independent `r ← E^κ` projections after all messages are fixed, ONE
  Fact-1 per group, one PCS opening per group — the paper's
  `16 + 3·d_L` batched-terminal structure.

### The Table-5 benchmark (`mikros-bench plonk --log2-n K`)
Simulated binary circuits (the paper's own Table-5 workload), the
`d = 32 / κ_pos = 4` geometry the Theorem-1 interface forces for every
`N ≥ 2^14` (`d_L + 1 ≤ 28`), single-threaded on this 2-vCPU container:

| N | prover | gates/s | verifier | approx. proof | peak RSS |
|---|---|---|---|---|---|
| 2^16 | 267 ms | 245,677 | 91 ms | 193 KiB | 29 MiB |
| 2^18 | 1.17 s | 56,246 | 161 ms | 251 KiB | 134 MiB |
| 2^20 | 4.61 s | 227,680 | 361 ms | 361 KiB | 524 MiB |
| 2^21 | 10.1 s | 207,091 | 615 ms | 437 KiB | 922 MiB |
| 2^22 | 21.2 s | 197,895 | 1.13 s | 585 KiB | 1.79 GiB |

The anatomy matches the paper's shape: the copy argument dominates
(62% at 2^22 — the paper's Table 5: 92% perm share at 2^25 on their
GFNI Xeon, whose per-core throughput this container's shared vCPU does
not match), the gate layer (the GFNI-accelerated Algorithm 2) is the
small share the paper's "excl. perm." column reports, and the grouped
openings carry the preprocessed public commitment.

### Fixed (found by the composed protocol)
- The copy tree's row pairing must be the **contiguous halves** (the
  cubic engine's U/D convention) — the interleaved draft produced a
  grand product that was accidentally correct but structurally
  inconsistent with the transition invariants.
- The Barrett constant's missing `t_hi` term (`q̂ = t_hi ^ ((t_hi·μ_low)
  >> 32)`, not the bare product half) — caught by the loop-reference
  cross-test.
- The verifier's group opening must fix **all** column messages before
  sampling any projection (the paper's explicit ordering — an
  interleaved absorb/sample silently forked the duplex).
- The PCS opening point reversal at the group boundary (the Fact-1
  MSB-first vs PCS LSB-first convention — the batch.rs lesson applied).

Gate: 42 mikros tests green (the new gf32/copy/plonk suites with dense
E-arithmetic references and six tamper classes), cargo fmt clean,
clippy --workspace --all-targets clean, the full workspace suite green.

## 0.10.0 — Mikros (ePrint 2026/2646): the bitwidth-aware SNARK suite, complete and benchmarked

The paper's full protocol stack on this repo's fields
(`F = GF(2) ⊂ K = GF(2^8) ⊂ E = GF(2^128)`, packing width κ = 128):
algorithms 1–6 implemented, cross-validated against dense E-arithmetic
references, tamper-gated, and benchmarked against the paper's Tables 3–4.

### The new crate (`magnetar-mikros`)
- **Algorithm 1** (`innerprod`): the packed inner-product proof — the
  monomial/reciprocal packing, the Laurent correction polynomials
  `h±` (2·127 GF(2) coefficients), `x ←$ E×`, Fact-1 to the terminal
  packed claims with the monomial column weights.
- **Algorithm 3** (`perm`): the packed permutation proof — the GF(2)
  coefficient-tree `C_τ(Z)` (7 levels, the AND-convolution build with
  tail-safe row-half extraction), the column-major flatten, the E-tree
  of cubic weighted sum-checks down to `l₇^⊤C₇(z)r₇`, the 7 coefficient
  transitions (the `u` column claims via aggregate-then-evaluate, the
  Lagrange-residual quotient `h = f/Ψ`, the per-node `η` encodings
  z-evaluated then row-dotted — 3.6× over the per-pair GF(2^8) bucket
  route on scalar hardware), and the terminal packed-PCS claims
  `v₀ − z = l₀^⊤Dρ(x)` per operand.
- **The cubic weighted sum-check engine** (`perm::cubic_*`): the
  degree-3 round messages with the char-2 Boolean check, the child MLEs
  at the shared row point, the verifier's ξ̂ fold.
- **The benchmark binary** (`mikros-bench`): Table 3 (the 2^30 binary
  Hadamard), Table 4 (end-to-end packed batches of 256 SHA-256 / Keccak
  / BLAKE3 instances through Algorithm 6), and the perm anatomy ladder.

### Fixed (found by the composed protocols and the dense references)
- **The Fact-1 terminal absorb**: prove absorbed `fact1-term` (one
  duplex call) while verify absorbed `fact1-v1` + `fact1-v2` — the
  labels are part of the duplex domain separation, so every protocol
  that samples a challenge *after* a Fact-1 silently forked its
  transcript. Invisible in Fact-1's own tests; fatal in Algorithm 6.
- **The Fact-1 → PCS point order**: Fact-1 binds its vector MSB-first,
  the fold-chain PCS binds LSB-first — the witness claims are now
  reversed at the opening boundary (batch + packed-open).
- **`ts_msbf` coordinate order**: the MSB-first tensor-sum materializer
  grew the recursion with the last coordinate first; the first draft
  had it inverted, which only surfaced three transitions deep in the
  perm unfold as a broken `v_{τ+1} = 〈u, r_{τ+1}〉` invariant.
- **The Keccak io positions**: the output lanes are io-marked in (y,x)
  order, not column-creation order — the batch io-bind's masked-eq scan
  requires the sorted set; `CircuitBuilder::build` now sorts + dedups.
- Prover-side claim asserts removed where tamper tests deliberately
  feed dishonest claims (the established repo pattern: the verifier's
  rejection is the test).

### Benchmarks (single-threaded, the 2-vCPU container; the paper's host
is a Xeon Gold 6342 with GFNI/AVX-512 — see `docs/BENCHMARKS.md` for
the honest hardware-gap anatomy; ~41× on the pure PIOP core)
- **Table 3 (N = 2^30 Hadamard)**: prover 27.99 s, verifier 258 µs,
  proof 5,216 B, **peak RSS 915.8 MiB inside the paper's 0.89 GiB
  envelope**.
- **Table 4 (256-instance hash batches)**: SHA-256 2.54 s / Keccak
  5.07 s / BLAKE3 1.24 s prover, verifier 78–191 ms, RSS ≤ 269 MiB —
  the verifier within 3–10× of the paper, memory better; the proof
  size 3.5–4.0 MiB is the FastChunk operating point (the size profiles
  are the knob).
- **The perm argument**: 703 ms at N = 2^20 / 2.36 s at 2^22 (the η
  pipeline 86%, the `g` contractions 9%).

### The remaining paper piece
Algorithm 5 (the single-instance PLONK composition: the gate layer +
the §5.1 copy-constraint encoding over `K_pos` at width κ_pos, feeding
`perm`) — the Table-5 protocol. Everything below that composition is
done: `perm.rs` exposes the exact terminal interface it consumes.

## 0.9.3 — the fw-bound-se group-fold attack + the transient-eq unlock + the 2^13-cycle operating point

The biggest measured root lever of the 0.9.2 anatomy (the `fw-bound-se`
group-fold, ~19 ms per segment, ~30% of the root's local stages) plus
the lockstep-memory unlock that opened the single-machine 2^19–2^20
range, plus the streaming operating point that amortizes the root's
per-instance fixed costs.

### The group-fold attack (magnetar-field `eq.rs` + magnetar-zkvm `discharge.rs`)
- **`eq_build_fused`/`eq_build_fused_into`**: the transient fused
  eq-table build — half the multiplications of the reference
  `EqBits::build` (the broadcast-multiply + XOR doubling), two reusable
  ping-pong buffers with the in-place DESCENDING interleaved doubling
  (position `i` is read at iteration `i` and written at `i/2`, which
  runs later — no clobber), zero per-level allocations. Unit-gated
  byte-identical to the reference.
- **Transient group eq tables**: rebuilt fused at the word fold and
  again at the scatter, never resident on the lockstep state — the
  per-instance steady memory drops from `O(Σ 2^{free_i})` (~12 MiB at
  the 2^10-cycle shape) to `O(items)`. The lockstep holds every
  instance simultaneously, so this is the single-machine 2^19+ unlock:
  the 2^16 peak RSS fell 1.37 GiB → 838 MiB, and 2^19 (3.63 GiB) / 2^20
  (7.06 GiB) now fit the 8-GiB production host.
- **The bound-SumEq scatter rewrite**: per-(group,plane) HOISTED member
  sums `S_p = Σ pe_m` (one broadcast multiply-accumulate instead of one
  multiplication per member per word), the group's CONTIGUOUS support
  `[base, base+2^fw)` walked directly by the fused VPCLMULQDQ `axpy_run`
  (the first version scanned the whole `2^{word_bits}` plane block and
  filtered), and the identity groups size-sorted across per-worker
  `stride` accumulators that XOR-merge (GF(2^128) adds commute —
  byte-identical output, verified by the generic-engine cross-check,
  the brute-force monomial fold, and the streaming IVC roundtrip).
- Measured: the whole discharge construction at 2^12 monolithic
  ~102 ms (66.9 + ~35 hidden) → 62 ms; the per-instance shape
  24.0 → 14.0 ms on the 2-vCPU dev container.

### The 2^13-cycle operating point (magnetar-cli benches)
- The streaming ladder only ever tested ≤ 2^10-cycle segments; the
  root's per-instance discharge machinery costs ~33 ms wall per
  instance REGARDLESS of the instance's size, so longer segments
  amortize it — measured at 2^18: 17,719 vs 4,341 cycles/s. The
  erc20 bench's old `clamp(1, 128)` would also have grown the
  per-worker envelope to 2^17-cycle segments at 2^24 (dangerous); the
  ladders now floor at 2^12/2^13-cycle segments and the erc20 ladder
  extends to 2^24 (`<MAX_LOG2> <MIN_STREAM_LOG2>`).
- The join prover gains `MAGNETAR_TIME` phase timers
  (`jr-extract/a/xo2/o/h/u/j`); `jr-extract`'s strided selector scan
  is the documented next lever.

### The measured ladder (the InstaBox 4-vCPU AVX-512 host, every row proved AND verified)
- `erc20-mixed-block`: 2^16 3.82 s = 17,153 cycles/s = **1,429 tx/s**
  (the session start on this machine: 5.83 s, 936 tx/s, 1.37 GiB);
  2^17 8.89 s = 14,743; 2^18 17.8 s = 14,719; 2^19 47.3 s = 11,079,
  3.63 GiB. The alu ladder: 2^17 19,142; 2^18 17,719; 2^19 14,545;
  **2^20 116.2 s = 9,021 cycles/s, 7.06 GiB** — the measured
  single-machine streaming envelope (2^21+ needs the per-segment
  commitment trees, the documented next structural lever; the pooled
  2^23/2^24 demonstrations stand at 3,330/3,495 tx/s sustained).

## 0.9.2 — the three-lever pass + the pooled operating point

The three measured levers of the 0.9.1 anatomy (the twists' packed-inc
o-rounds ~420 ms, the store-router join pass ~308 ms, the outer-rounds'
window pass ~311 ms at 2^13), plus the cpi × pool matrix that flipped
the default back to `cpi = 1`.

### The levers
- **The twists' packed-inc select-fold** (magnetar-memory): the port
  sum-checks' o-rounds never touch the `T·2^ell` increment table — the
  increments are bits, packed one `u128` per cycle in `TwistShared`;
  one masked pass per port run builds the per-value-bit dots
  `D_o = Σ_{u: inc bit o} (WFAEQ·LTW)[u]`, and every round is
  `O(2^ell)` field ops (`port-o-rounds` 4 ms → 0.0 ms per port run).
  The reconstruction's increment row is the eq(r_v,·) select over the
  packed words.
- **The store-router join select-dots** (magnetar-router): the 0/1
  routing tensors make the extraction's `tau` XOR-selects of the global
  `eq_o[o]`, and the o-rounds one-pass per-column `xo2[o]` dots against
  GLOBAL round weights (the per-pair pref/suffix products are gone;
  `4b-store-router` 45.4 → 4.4 ms per segment, ~10×). Non-binary
  tensors keep the generic path — pinned byte-identical by the new
  `join_matches_sparse_prover_generic_coefs` (every existing suite was
  binary).
- **The window's complement-convolution** (magnetar-sumcheck): the
  small-value round window's one-pass build is GF(2)-bilinear in the
  two Möbius masks — the sparser of `(m, ~m)` on each side plus three
  precomputed correction vectors; dense sub-cubes cost ~a dozen word
  XORs instead of up to 4,096 pair iterations, and the pass no longer
  serializes strides below `PAR_THRESHOLD` (`vs-win-build` ~26 → 16 ms
  per segment).
- `MAGNETAR_TIME` sub-timers for the VS anatomy (`vs-win-build`,
  `vs-shared-rounds`, `vs-row-rounds-*`) and `examples/window_bench`.

### The measured effect (the same 4-vCPU EPYC host)
| T | 0.9.1 (cpi 4) | 0.9.2 (cpi 1) | tx/s |
|---|---|---|---|
| 2^13 | 546 ms / 15,006 | **470 ms / 17,418** | **1,450** |
| 2^14 | 1,088 ms / 15,061 | **933 ms / 17,568** | **1,464** |
| 2^15 | 2,483 ms / 13,198 | **1,990 ms / 16,470** | **1,372** |
| 2^16 | 4,954 ms / 13,228 | **4,254 ms / 15,405** | **1,284** |

**The cpi × pool matrix** (5 × 4-vCPU instances, 2^16 batches,
aggregate tx/s): cpi 1 **5,139** / cpi 4 4,585 / cpi 8 4,942 — the
levers removed the per-instance amortization edge, and the lightest
per-segment constants degrade least under cross-VM contention:
`DEFAULT_CPI` returns to **1** with the measurement recorded in
`WITNESS_LAYOUT`. The batch-shape × pool sweep: **2^13 batches pool
best — 6,005 tx/s burst** (0.8.0's pool: 3,788). The large-scale pooled
demonstrations (`magnetar-bench-erc20-pool`): **2^23** — 1,024 × 2^13
batches = 699,050 transfers, every batch proved AND verified, 210 s
(≈3,330 tx/s sustained); **2^24** — 2,048 batches = 1,398,100
transfers, 400 s (≈3,495 tx/s). The single-machine envelope extended
to 2^18 (the segments phase bounded at 261 MiB; the root's lockstep
discharge + combined commitment measured as the 2^19+ gate, with the
unlock path documented). Gate: 169 tests (debug + release), clippy
clean, fmt clean.

## 0.9.1 — the instance-batching release (the paper's §5.1 operating point)

**`WITNESS_LAYOUT::DEFAULT_CPI = 4` cycles per SIMD instance** — the
paper's "each SIMD instance covers a fixed number of consecutive
cycles", as a runtime parameter through `keygen_with` (the benches
sweep it via `MAGNETAR_CPI`; `cpi = 1` recovers the one-cycle layout
byte-identically).

### The restructure
- `InstanceGeometry` (magnetar-cpu): the per-instance column map
  `col' = cpi·col + δ` — the cycle-within-instance coordinate is the
  LOW bits, **one variable shared by every segment** (a per-segment δ
  placement breaks the pc-Ψ and the ram-addr-Ψ at cpi > 1: the
  auxiliary arguments consume claims of *different* segments against
  one cycle cube).
- `build_circuit_batched(cpi)`: both relations' row indexing reshapes
  over `(δ ‖ row-within-cycle)` blocks; the constraint set is the
  per-cycle relation replicated — no new rows, the same soundness
  surface (an assembled instance satisfies the batched circuit iff its
  cycles satisfy the per-cycle circuit; tested at cpi 1/2/4/8).
- The segment claims' local point is `[δ-vars ‖ col-vars]`; the cycle
  point of cycle `u·cpi + δ` is `[δ-vars ‖ instance-vars]` — threaded
  through the twists, the Ψs, the routers (the X claims reordered into
  the components' `[δ ‖ bit-index ‖ instance]` convention), the shout,
  the boundary claims, and the cross-step flags.
- The Mosaic components keep their total cube (`len·T` bits at every
  cpi); the direct tables' local index is `col·cpi + δ`.
- The streaming ladder proves + verifies at cpi 2 and 4 end-to-end.

### The measured effect (4-vCPU EPYC, AVX-512)
| cpi | 2^13 | 2^14 | 2^15 | 2^16 | root(2^16) |
|---|---|---|---|---|---|
| 1 | 15,009 | 14,980 | 13,999 | 12,641 | 1,930 ms |
| 2 | 14,907 | 14,972 | 14,002 | 12,656 | 1,814 ms |
| **4** | **15,077** | **15,004** | **14,028** | **13,079** | **1,620 ms** |
| 8 | 14,825 | 14,475 | 13,645 | 12,700 | 1,658 ms |

The Spartan flop count is cpi-invariant (the joint cube, the applies,
and the folds are the same total size) — the gain is the per-instance
discharge work (the local `bound_se` walk) amortizing over 4 cycles;
at cpi 8 the per-segment inner/m_combo constants already eat it.

### Also in this release
- The misplaced prover-side honesty debug-assert in the cross-step S1
  (the tamper tests legitimately prove false claims; the verifier
  rejects them — release builds compiled it out, which is why the
  0.9.0 gate missed it on the debug side).
- The newer toolchain's clippy lints (div_ceil, clamp, late-init,
  cfg(test) helpers, items-after-test-module) — the gate is
  `-D warnings` clean again.
- The word-based witness loads (`wword_cycle`): the 64-bit word fast
  path at cpi = 1, the strided gather at cpi > 1.
## [0.9.0] — the split-ring release (the per-segment commitment trees)

### Added
- **The split ring switch** (`ring_split`): the streaming root's
  per-segment commitment trees — segment-local codeword vectors, merged
  `(P, C)` pair leaves, per-level root layers, and the shared upper
  levels. Peak root memory `O(one segment's chain)`; 2^20 proves in
  2.38 GiB on a 3-GiB container (2^17 was the previous ceiling).
- **`AccumTimer`** — the `MAGNETAR_TIME` split at scale: aggregated
  per-phase totals + fire counts + per-fire averages; a CPU-ms axis per
  root phase (`magnetar_ivc::cpu_ms`).
- `magnetar_ivc::streaming::segment_count` — the adaptive segment
  ladder (2^13-cycle segments at scale; `MAGNETAR_SEG_LOG2` override).
- `MAGNETAR_RSS` — per-phase RSS traces on the streaming path.

### Changed
- The hierarchical discharge's per-group eq tables are transient
  per-phase arenas (the stored tables were the measured per-instance
  O(T) spike at 2^17+).
- The split ring's queries and anchors share one per-segment derivation
  in bounded waves.
- `MosaicCommitment.codeword`/`tree` are `Option` (the multi-segment
  root never materializes the combined codeword or tree).
- The streaming root's transcript binds the commitment through the
  split ring's `ring-root-0` (no separate `cm-g` absorb on the
  multi-segment path); the sequential and streaming provers remain
  byte-identical.

### Fixed
- A latent debug-only `debug_assert` in the cross-step S1 prover fired
  on the tamper tests' deliberately inconsistent inputs (removed — the
  verifier's rejection is the test).
- The newer toolchain's clippy lints (div_ceil, needless late init,
  items-after-test-module, type complexity).

## [0.8.0] — the production-throughput release

### Added
- **The InstaBox load-test pool**: the ERC-20 mixed-block ladder measured
  on real AVX-512 hardware (4-vCPU EPYC instances), including the pooled
  5-machine (20-vCPU) concurrent run — the production proving-pool
  throughput shape.
- **The packed witness** (`WitnessData::w_packed`): the bit-packed w-half
  (128 witness columns per `u128`) + the shared public-x vector. The
  `q × m × 16 B` joint table — every bit a full field element — is gone;
  the eq(r_inst)-fold is one tzcnt select-XOR pass (`fold_packed_w`),
  the x-fold is the identity (`Σ eq = 1`, the shared-x observation),
  and the matrices apply over packed bits with the shared x-part
  hoisted out of the per-instance loop (`apply_simd_packed`).
- **The linear sum-to-one prover** (`prove_st1_linear`): the twist st1
  sum-checks over the one-hot components via the one-hot kill — O(T) per
  round over the per-cycle address traces, never the `2^{ad+n}` cube
  (the RAM twist's st1 was 291 ms of a 541 ms twist). Byte-identical to
  the dense engine (cross-validated).
- **The sparse component evaluations**:
  `row_at_sparse`/`column_at_sparse`/`eval_at_sparse` from the address
  traces (the dense `O(2^a · T)` table dots were the onehot/claims
  cost), with the address traces computed once and threaded through the
  claims, st1, onehot, and reconstruction paths.
- **The group-fold Mosaic discharge**: the eq(r)-weighted fold of the
  packed words per **distinct free-word point** — the claims' word
  coordinates coincide within families (the witness segments share
  `r_low ‖ r_inst`; the memory claims share `r_k ‖ r_cycle`), so ~53
  groups cover 61 claims — then the per-claim plane folds (the
  bind-shrink-adjusted pattern positions) + the window monomial
  butterfly, and `bound_se` as the group outer product. The per-suffix
  64-plane extraction and the `2^{ell-v}`-per-item suffix tables are
  gone; a brute-force cross-validation test gates the fold.
- **The mask-major bin outer** (`apply_simd_bit_masks` +
  `SumcheckProver::with_mask_window`): the three bit tables arrive as
  packed window truth-table words (one `u64` per suffix — the
  constant-1 column as a fixed row flip), the window pass reads one
  word per (table, suffix) instead of `2^v` strided 16-byte gathers,
  and the fast-forward contracts the masks. vs-apply 3.4×, vs-outer
  2.1×, and the az/bz/cz materialization (q·rows·48 B) is gone.
- **The fused `axpy_run` kernel**: the broadcast multiply-accumulate
  (AVX-512 4-lane / AVX-2 2-lane + the scalar reference), unit-tested —
  the bound_se inner op's kernel.
- The twist + port-linear + discharge phase instrumentation
  (`MAGNETAR_TIME=1`) used to pin every optimization this release.

### Changed
- **The FastChunk default** (`k = 1`, rate 1/2): every codeword and
  Merkle tree halves. Each profile's per-query far-word rejection is
  the Schwartz–Zippel bound `n/2^128` with `n ≤ 64` (≥ 122 bits per
  query), so every profile holds the 120-bit envelope on its own; the
  wrap re-proves the public tail at a deep rate for size.
- **The hierarchical discharge parallelized**: the per-segment local
  stages (construction, per-round messages, binds) run rayon-parallel
  — they were sequential; the streaming root halved.
- The window size stays `min(6, high)`: the SWAR monomial machinery is
  `u64`-bound at v ≤ 6, and the pack cost grows as `(3/2)^v` — the
  masks are the real window lever (measured).

### Measured (4-vCPU EPYC, AVX-512)
- Monolithic: 5,044–6,668 cycles/s at 2^10–2^12 (release start 3,541).
- Streaming: 12,186 at 2^13 / 12,398 at 2^14 / 11,707 at 2^15 /
  **10,276 at 2^16** (10.4–12.4k cycles/s; the release start measured
  3,943 on the same host).
- **The ERC-20 mixed-block load test** (12 cycles/tx): 856–1,033 tx/s
  per machine across the ladder; the pooled 5-machine concurrent run
  measured **≈3,880 tx/s** (20 vCPUs, with cross-VM contention).
- Memory: the 2^12 monolithic prove's peak RSS fell 619 → 165 MiB (the
  packed witness + the mask-major outer + the group-fold discharge).


## [0.7.0] — the streaming-scale release

### Added
- **The hierarchical Mosaic discharge** (`magnetar-zkvm::discharge`): the
  streaming root's batching sum-check factors per segment (the combined
  cube is `(segment bits) ∥ (local cube)` and cross-segment eq terms
  vanish on each segment's block), so the root now runs **per-segment
  local stages in lockstep** — each proving its own batching sum-check
  against its packed slice, with the round window reading G straight
  from the commitment's 128-bit words — plus a `log2(segments)`-round
  segment stage that eq-batches the per-segment terminals into one
  claim. The root's peak memory drops from `O(2^ell_combined)` (two
  `2^ell · 16 B` tables — 2×1 GiB at 2^16, the 2^15 gate) to
  `O(one segment)`, and no cross-segment "prefix adaptation" pass ever
  touches the segment tables. **2^15 and 2^16 now prove end-to-end in
  bounded memory** (1.36 / 2.60 GiB peak on a 4 GiB container).
- **The small-value round window** (`magnetar-sumcheck::window`, ePrint
  2025/1117 §5.1 + 2026/587 §5.1/C.5 in characteristic two): the first
  `min(6, s)` rounds of the binary relation's outer sum-check are
  computed in **one pass** over the un-folded bit tables — the char-2
  monomial-basis window with the eq-weighted pack (the window state is
  `3^v + 2^v` field accumulators, *independent of the trace length*) and
  a select-only fast-forward. Messages byte-identical to the generic
  engine's (cross-validated per shape); measured `vs-outer-rounds`
  collapsed from the dominant phase to 13 ms at 2^8.
- **The ERC-20 mixed-block load test** (`magnetar-bench-erc20`): the
  paper's Table-6 workload shape — checked-balance transfers with every
  16th transfer failing — proven and verified end-to-end through 2^16
  (1,447 cycles/s, 121 tx/s at 2^16 on this container), plus the
  `end_to_end_erc20_mixed_block` protocol test with a public-output
  tamper class.
- The window cross-validation tests (`window_check.rs`: full-proof
  byte-identity vs the generic engine and a real-verifier roundtrip)
  and the discharge regression test (per-instance messages vs the
  generic engine + the full hierarchical body verify).

### Changed
- **Circuit density 2048 → 512 rows/cycle**: the 1,127 booleanity rows
  (`x² = x` per witness column) were **vacuous** — the committed witness
  is the bit matrix (any GF(2^128) element's coordinates are bits), so
  the constraint holds for every possible commitment and binds nothing.
  Deleted; the binary relation's real rows drop 1610 → 483 (padded 512)
  and nnz 5,116 → 1,735. The full 15-class tamper suite passes
  unchanged — the datapath's soundness lives in the value-carrying rows
  and the memory/routing arguments.
- **The block-aligned streaming layout + split packing**: the combined
  Mosaic layout places segment `s`'s components at `s·2^{ell_seg} +
  off_c` (the combined cube is the clean product), and the packing is
  split-aware — the word index is `segment ∥ block`, the 7 word bits
  are the local planes — so every segment's packed words stay
  contiguous (the window's one-word-per-suffix read depends on it).
- **The eq-slab slice choice fixed** (`fold_msg` and the window's
  fast-forward): the previous `slice = τ ≠ 1` gather panicked on
  bit-valued τ coordinates (`inv(0)`); the correct choice is `s = 1`
  iff `τ = 1`, else `s = 0` (eq(τ,0) = 1+τ ≠ 0 exactly because τ ≠ 1).
  A latent edge case never hit by random challenges, exposed by the
  window's bit-τ cross-check tests.
- **The SIMD-detection probes use wrapping sinks**: the timing-only
  accumulators previously `sum()`-panicked on debug-build overflow (a
  latent 5-test debug failure, now fixed).

### Measured (2 cloud Xeon cores, 4 GiB)
- Monolithic: 1,727–1,801 cycles/s at 2^8–2^10 (session start 1,065),
  1,558 at 2^13 (session start 821).
- Streaming: 1,920 at 2^14 / 1,833 at 2^15 / **1,707 at 2^16** — the
  2^15/2^16 scales were **not provable** at session start (the root's
  combined-cube tables exhausted the envelope).
- The ERC-20 mixed-block load test: verified end-to-end through 2^16.

## [0.6.0] — the linear-in-T release

### Added
- **The linear-in-T memory provers** (`magnetar-memory::linear`): the
  Twist read/write port sum-checks, the bytecode Shout Φ, the Ψ
  address-claim checks, the selector checks, and the cross-step S1
  transition argument all enumerate the trace cycle-by-cycle — the
  one-hot kill over the address rounds (the address factors' MLE at a
  partially bound point is the per-cycle product `Π (1 + c_β + q_β(j))`,
  and the unbound-bit sum selects the addressed cell) plus the
  prefix-XOR cumulative sums for the reconstruction identity
  (`Σ_{j'<j, same address}` running XOR). O(T) prover work, O(K·L + T)
  state — the joint `(k, o, j)` materializations are gone. Round
  messages are **byte-identical** to the dense engine's, cross-validated
  per shape by tests. Measured: prove 6.3× faster at T = 2^10 (58.2 s →
  2.9 s), the memory envelope extends from OOM-at-2^11 to 2^13 completing.
- **The dual-rate wrap** (`magnetar-zkvm::wrap`, roadmap H2):
  `prove_with_wrap` discharges the public evaluation tail at `DeepWrap`
  (rate 1/16, 14 queries) with transcript-sampled consistency
  spot-checks (packed-leaf dual openings) binding the deep commitment to
  the chunk commitment — the bounded-depth spot-check accumulation
  shape. Measured tails: 62.8 KiB (FastChunk) → 27.3 KiB (DeepWrap);
  the wrapped artifact 36.5 KiB vs the Balanced chunk's 65.6 KiB.
- **ZK masking, recovered and wired** (`magnetar-sumcheck::zk`): per-round
  mask polynomials subject to the running round constraint, a uniform
  finish offset, and `SumcheckVerifier::finish_offset`. Perfect HVZK of
  the round messages at O(degree) randomness per round; wired into the
  Mosaic batching discharge behind `KeyGen::enable_zk` (the finish had
  been lost in the v2 merge; the docs now match the tree again).
- **The dual-rate PCS profiles, recovered** (`PcsProfile::{FastChunk,
  Balanced, CompactWrap, DeepWrap}`) and the `magnetar profiles`
  command measuring the tail dial on one statement.
- **The phase instrumentation** (`MAGNETAR_TIME=1`): the per-phase wall
  clock across the whole prover (the honest-measurement hook the
  performance rulebook mandates).

### Changed
- **The sum-check engine's round kernel is fused**: one traversal per
  round computing the message coefficients directly (specialized
  1/2/3-factor products, no per-point half-table views, no allocations),
  and `absorb_and_fold`/the lockstep drivers no longer recompute the
  message inside `fold`.
- **The batching factor builds sparsely** (`SumEqFactor`): the Mosaic
  claims' lifted points carry Boolean suffixes (the aligned-subcube
  offsets), so each α·eq table scatters over its `2^{vars_i}` support
  subcube — `Σ 2^{vars_i}` work instead of `S·2^ℓ` (measured 1.19 s →
  33 ms on the T = 2^8 discharge).
- **The verifier evaluates its public factors structurally**: the new
  `EqProdFactor` (products of eq over disjoint variable blocks) replaces
  the joint-cube materializations in the Twist verification; the
  reconstruction's LT columns are O(T) suffix sums (no 2^{2n} tables);
  the reconstruction's address row is the one-hot product (O(T·a), not
  O(T·K)). Verify: 1,488 ms → 46 ms at T = 2^10.
- `MosaicProof.ring` is `Option<RingProof>` with the `wrap` alternative;
  the byte accounting in `proof-stats` covers both.

## [0.5.0] — the parallel streaming release

### Added
- **The performance rulebook** (`docs/PERFORMANCE.md`, R1–R10): the
  mandatory engineering rules for every prover-side hot path —
  order-free parallel reductions (char-2 determinism), work thresholds,
  structure-collapse-before-parallelism, branch-free SIMD-friendly
  kernels, batching, memoization, streaming memory discipline, disjoint
  writes, the parallel-kernel shape, and honest measurement. Every PR
  touching prover code is reviewed against its checklist.
- **Parallel kernels throughout the stack** (rayon): the parallel
  sum-check fold and round messages (bit-identical to the sequential
  engine, tested), the parallel Merkle tree (level-by-level with
  chunked leaf/node hashing), the parallel subspace-extension encode
  (independent ε-blocks), `par_bind_var`/`par_dot_xor` field kernels
  with measured thresholds, and `Gf128: Sum` for order-free reductions.
- **The parallel join-enumeration router prover**
  (`magnetar-router::join`): every routing sum-check phase enumerates
  the trace column-by-column with per-chunk coefficient accumulators
  (τ algebraic folding over the store offsets, sparse support pairs,
  the one-hot kill, lockstep u-folds, pair-folded j-rounds) —
  byte-identical round messages to the sparse prover (cross-validated),
  O(c·T) work per routing instance.
- **The streaming, parallel tree-IVC prover**
  (`magnetar-ivc::streaming`): checkpointed execution
  (`magnetar-cpu::StreamCursor` — the trace never resides; segment
  replay byte-identical to the full padded trace, tested at the record
  and proof levels), rayon-parallel segment proofs with per-segment
  Fiat–Shamir transcripts, tables packed straight into the shared
  commitment leaves and dropped in flight (`pack_tables`/
  `pack_tables_slice` — no boolean union, ~16x less commitment memory),
  and one sequential root. `StreamingStats` measures per-phase wall
  clock, peak RSS (VmHWM), and the byte inventories. The proofs are
  byte-identical to the sequential prover's on 1/2/4 segments.
- **The streaming-prover benchmark**
  (`magnetar-bench-stream`): mode-separated sequential-vs-streaming
  measurement (fresh processes for honest VmHWM) with the worker
  schedule tracer (`MAGNETAR_TRACE_WORKERS=1`) and the honest analysis
  in `docs/BENCHMARKS.md` — the parallelism is thread-correct (~100%
  utilization), the net speedup is bounded by the dense per-segment
  provers' memory bandwidth (the roadmap's linear-in-T provers are the
  unlock).

### Fixed
- The join prover's u-rounds split the round variable from the wrong
  end of the working table (`split_at` on the high bit instead of the
  strided bit 0 the folds bind) and the j-rounds double-counted each
  pair (self-cancelling in characteristic two) while missing the folded
  R/eq_o scale — found by the byte-identity cross-check against the
  sparse prover, now green.

## [0.4.0] — the soundness closure release

### Added
- **The cross-step transition argument** (the paper's Remark-4 mechanism,
  `magnetar-memory::cross_step`): the fetch chain
  `fetch(j+1) = TT[fetch(j), taken(j), halt(j)]` proven by two chained
  sum-checks (the transition Shout with α₀ sum-forcing + the
  predecessor-weight shift discharge), with the ROM transition table
  built at keygen. Closes the jumped-pc soundness gap — a swapped
  instruction order is now rejected (`tamper_jumped_pc_rejected`).
- **The flag-derivation constraints** (`magnetar-cpu::circuit`):
  `halt ⟺ opcode = SYSTEM` and `taken = branch-row ∧ comparison` (the
  full 64-bit comparison circuit: the zero-chain, the signed/unsigned
  less-than muxes, the funct3 selection) — a faked halt or branch
  direction is now unsatisfiable.
- **The ECALL output binding**: the halt row reads x10 (rs1 = 10 in the
  expansion) and `halt · (rd + rs1) = 0` ties the public output to the
  register file through the register Twist. `prove` additionally
  requires the execution to halt within the trace bound.
- **The initial-pc and final-halt public point claims** (single-segment
  and tree-IVC shapes).
- **Tree-IVC** (`magnetar-ivc`, new crate): segments share one
  MosaicPCS commitment; per-segment boundary statements chain through
  the final-state bindings; the root discharges all segments' claims
  with one batching sum-check and one PCS evaluation argument.
- **The final-state binding** (`magnetar-memory::final_state`): the
  Remark-4 initial/final register and RAM binding via the zero-check +
  the evolution reduction.
- **The join-enumeration router prover** (`magnetar-router::join`): the
  paper's linear-in-T BinaryRouter prover — byte-identical proofs to the
  generic sparse prover, ~600× faster at T = 2^12.
- **The benchmark harness and paper comparison**
  (`magnetar-cli::magnetar-bench`, `docs/BENCHMARKS.md`).
- `docs/STATUS.md`: the honest implemented/not-implemented accounting.

### Fixed
- The sub-word store tensor routed the source bit `o` instead of
  `o − 8a` at byte offset `a` — every store with a nonzero byte offset
  produced an unsound routing identity (all prior tests used offset 0).
- The store-router offset claims selected bit-column `2^k` instead of
  `k` (a unit-vector vs binary-index confusion in the claim point).
- The load zero-extension constraints were not gated on `is_load` and
  rejected any ALU result with bits above 15 set (e.g. sign-extended
  immediates).
- `prove` now pads the trace to exactly `trace_len` (the interpreter
  padded to the next power of two of the record count, desynchronizing
  every auxiliary argument at non-matching lengths).

### Changed
- The witness width doubled (W = 2048, M = 4096) to fit the comparison
  circuit; the `crossaux` virtual segment tiles the new region.
- The halt-latch padding cycles fetch the frozen row (the cross-step
  chain requires the freeze transition to be consistent).
- `JALR` is rejected at keygen: the static transition table cannot
  encode register-indirect targets (see `docs/STATUS.md`).


=======
## 0.3.0 (unreleased)

### GPU proving program, Phase 1: the bitsliced sum-check (`magnetar-bitsum`)

- New crate `magnetar-bitsum` (roadmap build-order item 1, "GPU
  sum-check kernels"): the bitsliced multilinear sum-check over GF(2^128)
  in the tower basis — the statement shape of MosaicPCS's first-phase
  sum-check and Ligerito's proximity phase.
- **Tower field with computationally verified constants**
  (`NUS[k] = 1 << (2^k - 1)`, single-coordinate at every level):
  trace-1 irreducibility per level, field axioms, group order, and
  inverses are all re-checked by unit tests.
- **The kernel algorithm as tested host functions**: 32-way bitsliced
  groups (group-major `[groups, 128]`, identical to the GPU tensor
  layout), the Karatsuba tower-multiplier tree (pruned == unpruned,
  tree == scalar multiplication), bind/reduce vs scalar references, and
  the planes prover bit-identical to the scalar prover.
- **Repo synchronization**: Fiat-Shamir now runs over the repo-standard
  SHA3-256 duplex (`magnetar-hash`) instead of the study's private
  SHA-256 transcript, and a **verified tower ↔ K isomorphism**
  (`interop`) bridges the kernel's tower basis to the stack's GCM
  polynomial basis — round trip, homomorphism, and generator relations
  tested; `prove_gcm` / `eval_mle_gcm` prove over `K` tables, and the
  e2e test discharges the final claim with `magnetar-field`'s own
  `eval_mle`.
- **NVlabs cuTile GPU kernels behind the `cutile` feature**
  (`pack_planes`, `bind_round`, `reduce_partial`): a 1:1 transliteration
  of the tested host functions; the `gpu_parity` integration test is the
  hardware acceptance gate. The default build is pure Rust — CI stays
  green without CUDA; the feature-on path is run by the manual
  `bitsum-cuda` workflow.
- Tamper coverage per the contributing rules: c1, mid-proof c0, last-c0
  (caught by the outer evaluation check), and wrong-claim rejections.

### The register-shift stage-select extension (complete)

- Register shifts (`SLL/SRL/SRA` by register) now prove end to end: the
  6-stage decomposition chains through the trash cell **in the public
  ROM fields** (stage 0 reads the source register, later stages read
  cell 32, intermediate stages write it back) — no scratch register, x0
  stays hardwired to zero, and the register Twist's one-write-per-cycle
  model authenticates the chain unchanged.
- The ROM gains the `sel_stage` one-hot (authenticated through the
  fetch like every other selector) and the shift-kind column goes live
  on stage rows; the router's amount selector becomes the
  witness-conditional pair `{identity, shift-by-2^k}`.
- The **stage-selector sum-check** (`magnetar-memory::discharge`) binds
  the condition bit to bit k of the amount register (the `store-src`
  copy segment) and yields the structured stage part's claim; the
  `sel_lo` selector check composes it additively in characteristic two,
  forcing `Q_lo = ROM(sel_lo) + STAGE` at the router's random point.
- Five new tamper classes (the stage claim, a round, both terminals,
  the stage selector) are rejected; the e2e register-shift program
  covers SLL, SRL, and SRA with negative sources.

### Claim aggregation (roadmap Horizon 2) and the dual-rate schedule

- **[legacy branch]** `keygen_aggregate` / `prove_multi` / `verify_multi`: K segments
  share ONE Mosaic commitment (the components' subcubes gain the
  `log2(K)` index dimension) and ONE evaluation argument — the
  byte-dominant part of every hash-based proof is paid once,
  independent of K. Measured: 3.46x smaller than the standalone
  counterfactual at K=4.
- **`PcsParams::chunk()` (rate 1/2) / `wrap()` (rate 1/4)**: the
  dual-rate profiles, with the honest analysis of what the rate does
  and does not move in the affine-substitution code
  (`docs/architecture/pcs.md`).
- The proof structure is now `Proof { commitment, segments: Vec<SegmentProof>,
  mosaic }` — the single-segment API is the K=1 wrapper.
- The **cost model** (`stats.rs`) prints the aggregation ledger:
  measured per-segment sum-check bytes, the measured single evaluation
  argument, and the analytic standalone counterfactual.
- **[legacy branch]** New CLI: `magnetar agg [segments] [chunk|wrap]`.

### Correctness fixes (found by the new full-circuit satisfiability harness)

- **Trace padding**: the witness cube is now filled exactly to the
  proving bound (`execute_padded`); previously, programs whose cycle
  count rounded below the bound left zero rows that violated the
  per-cycle constraints.
- **Inverted sub-word-load zero-extension masks**: byte loads zero
  `rd` bits 8–63 and half loads bits 16–63, masked by the load-width
  flags only — previously the mask also zeroed non-load cycles, making
  any ALU result with high bits set (e.g. a negative ADDI) unprovable.
- New regression tests evaluate both R1CS relations on every witness
  row of representative programs (`witness_sat_tests`).

### Documentation

- `docs/architecture/` — ten production-grade component deep-dives
  (fields, sum-check, PCS, Mosaic, Spartan, memory, router, CPU,
  protocol, cost model), each mapping code to paper sections.
- `ROADMAP.md` — the implemented-status table, the horizon program, and
  the 150M-cycles/s GPU study's build order as the future plans.


### Tree-IVC: chunks share one commitment (the continuity layer)

- **The tree-IVC layer** (`magnetar-zkvm::ivc`, `docs/ivc.md`): a long
  execution is cut into chunks at RV64I instruction boundaries
  (`execute_chunk`; the cut never splits a register-shift stage sequence)
  and each chunk is proven through the shared `SegmentProof` body on one
  coordinator-ordered transcript.
- **Boundary continuity**: the register files (x0..x31 + the trash cell)
  and micro-PCs are published per chunk and bound by a final-state
  identity sum-check (`exit = entry + Σ wfa·inc`, the sparse prover)
  against the committed write log; the micro-PC by fetch-component claims
  at the first/last cycle's Boolean points.
- **RAM continuity without publishing the state**: each chunk commits
  `ram-init`/`ram-final` image components; the RAM BinaryTwist gains a
  claimed-M0 mode (its `M̂0` evaluations are transmitted and discharged
  against `ram-init`); chunk 0's identity pins the entry image to the
  public input; every boundary checks
  `ram-final_b + ram-init_{b+1} = 0` at a random point.
- **Claim-batching at the tree node**: the k chunk layouts merge into
  one Mosaic cube (dynamic k — the merged layout concatenates the
  per-chunk component lists) and ONE BLAKE3 commitment; a single
  MosaicPCS evaluation argument discharges every claim of every chunk.
  Measured: the eval layer (60% of the aggregate) is paid once — 3.8x
  fewer bytes than k separate proofs. **[legacy branch]**
- **The boundary padding invariant**: the write port's one-hot targets
  the fetched row's redirected rd (the bytecode Shout binds it to the
  bus slice), so the frozen-PC padding writes that cell — the identity
  write (the cell's own content, zero increment) at mid-program
  boundaries, the latched output on halted machines. Pinned by
  write-log consistency regressions.
- **[legacy branch]** New: `keygen_chunked`, `magnetar chunked`, and `aggregate_stats` (the
  aggregate cost model: core/identity/eval layers + the batching saving
  on the exact per-chunk geometry).
- 16 new tests: cross-chunk RAM + register carryover, the public-input
  pin, single-chunk compatibility, a five-chunk fibonacci, two write-log
  consistency regressions, the cost model, and ten tamper classes.
>>>>>>> c5531013623f249f8b96155d4e6ddb1bf4d8819c

## 0.2.0 (unreleased)

> **The parallel-branch merge**: the entries below marked **[legacy
> branch]** describe the `ivc`/`prove_multi` implementation built on the
> pre-soundness-overhaul API. That code is preserved on the
> `tree-ivc-legacy` branch pending its port to the current core; the
> merged tree's equivalents are `magnetar-zkvm::segments` (the shared
> commitment + aggregation) and `PcsProfile` (the dual-rate dials) —
> see `docs/STATUS.md` for the full reconciliation.


### This cycle: soundness fix, throughput, and the horizon items

- **Soundness fix (the truncated ROM immediate)**: the ROM's
  instruction bus carried only the 13-bit immediate field while the
  witness's fetch segment carries the full sign-extended 36-bit
  pre-adjusted immediate — every program with a **negative immediate**
  (backward branches, `ADDI -1`, negative load/store offsets) desynced
  the bytecode Shout. The ROM now writes the identical 36-bit word;
  a witness/ROM consistency regression test guards the class.
- **The SUB carry-in row**: the class-gated T2 binding sets
  `T2 = rs2 + F_SUB` on the SUB/BRANCH classes (the two's-complement
  negation absorbed into the operand), and the stale duplicated
  constraint section was removed.
- **The sparse-anchor router prover** (`magnetar-sumcheck::sparse` v2):
  round messages enumerated through the routing tensors' nonzero lists
  (rebuilt per fold) instead of the full joint cube — the store routing
  runs 110× faster, the whole 32-cycle proof 8.4×.
- **PCS rate/query profiles** (Horizon 1): `PcsProfile::{FastChunk,
  Balanced, CompactWrap, DeepWrap}` with the per-query soundness
  accounting for the affine-substitution code; the dual-rate projection
  measured by the cost model and exposed as `magnetar profiles`.
- **The ZK round-mask primitive** (Horizon 1): `magnetar_sumcheck::zk`
  — per-round mask polynomials with the running constraint chain,
  perfect HVZK for the round messages at O(degree) randomness per
  round; wired into the Mosaic batching discharge behind
  `KeyGen::enable_zk` (the full site wiring is inventoried in
  `docs/STATUS.md`).
- **Segmented proving** (Horizon 2): `execute_chunks` with explicit
  state handoff; `segments.rs` proves S×T cycles under **one shared
  commitment** — the chunked witness equals the monolithic trace
  bit-for-bit (tested), the claims aggregate through the single
  batching discharge, and `aggregation_report` measures the saving.
- **The benchmark harness**: `magnetar bench <lo> <hi>` (the
  cycles/s sweep), `MAGNETAR_TIME=1` phase instrumentation, and
  `docs/BENCHMARKS.md` with the measured numbers versus the paper's
  Table 6.
- **Documentation**: `docs/ROADMAP.md` (the horizons + the GPU target
  build order), `docs/RECURSION.md` (the wrap/tree-IVC/streaming
  designs), `docs/STATUS.md` (the implemented/remaining inventory).

### The complete protocol

- **Construction 3, all seven steps end-to-end** — the virtual-claim
  discharge (step 4) is now wired into the proof: BinaryTwist for
  registers and RAM, BinaryRouter for shifts and sub-word stores, and
  BinaryShout for the bytecode, each reducing its claims to evaluations
  on the single Mosaic commitment.
- New `magnetar-memory::discharge` module: the Ψ-sum-checks binding
  eq-weighted address/index claims to the one-hot components, the
  joint-space bytecode Shout (bus + register-index slices + α₀
  non-emptiness in one sum-check), and the per-factor selector checks.
- **MosaicPCS v2**: the component layout — the direct segments plus the
  eight auxiliary polynomials (the one-hot address components and the
  increments) on aligned subcubes of one cube, with the Equation-4
  suffix lift.
- **The sparse sum-check prover** (`magnetar-sumcheck::sparse`): factors
  stay in their own variable spaces and fold only on their own rounds —
  the paper's linear-in-T prover work.
- Twist soundness fix: the reconstruction sum-check now emits and checks
  per-component claims at its point (address virtualization) — the write
  terminal was previously unbound against the components.
- Witness v2: virtual/direct segment split at 64-aligned offsets, the
  instruction bus with the redirected rd, the sub-word-load byte-lane
  mux, the router source copies, and the one-write-per-cycle register
  model with the full access log.
- **Proof-size instrumentation** (`magnetar-zkvm::stats`): the
  committed-bit inventory per component, the PCS geometry, and the byte
  breakdown per protocol component (the roadmap's cost model).
- 69 tests including four end-to-end programs and 15 tamper classes.

## 0.1.0

The initial release: the binary-field stack, the sum-check engine, the
PCS with ring switching, Spartan/Virtual Spartan, BinaryTwist/BinaryShout
with the refined one-hotness, BinaryRouter, the RV64I decoder and
interpreter, the dual SIMD R1CS circuit, and MosaicPCS.

### The end-to-end benchmark and the honest inventory

- `magnetar-zkvm::wire`: the exact uncompressed wire-size model over the
  real proof objects (`Proof` and the tree-IVC `AggregateProof`) — 16 B
  per committed field element, 32 B digests, 8 B length prefixes — so
  the benchmark reports measured bytes, not estimates.
- `magnetar bench [max-trace]`: the paper's Table-6 methodology plus the
  H2 artifacts (the aggregation ledger vs. the standalone
  counterfactual, the tree-IVC chunk count, the dual-rate schedule,
  peak RSS). Highlights: proof sizes 375.7–619.1 KiB across all
  artifacts (the paper's own range is 377–598 KiB at its operating
  points); claim aggregation saves 59% of wire bytes at K=4 (37% at
  K=2); tree-IVC chunking keeps the artifact growing in the chunk count
  at a ~13% overhead; the wrap rate costs +8.4% with flat verification.
- `docs/benchmarks.md`: the full measured tables with the hardware
  disclosure and the honest prover-throughput analysis.
- `docs/remaining-work.md`: the honest inventory — the sum-check-engine
  speed path (memoized/lazy round products, small-domain arithmetic,
  the streaming Twist, parallelism), the verifier-guest/wrap recursion,
  zero-knowledge, QROM, ISA completeness (LB/LH sign extension, the
  workload suite), and the production-engineering backlog.

## 0.9.0 — the commitment-free ring switch release

- **The Diamond–Posen (ePrint 2024/504) ring switch**: the c-chain
  redundancy eliminated. The prover sends the 128 per-plane partial
  evaluations, the verifier checks the recombination and samples the row
  batch, and the claim discharges through ONE sum-check with the packed
  table itself as the dense factor and the transparent tensor-algebra
  factor A — whose correct multilinear extension is the DP24 tensor
  recursion (the *formula* φ_rb(eq(z,·)) agrees on the Boolean cube but
  is not multilinear off it; cross-checked against the materialized
  table's MLE). The main codeword's own fold chain carries the terminal:
  no second encode, no level-0 C-tree, no c-table, no anchors (~60–90
  KiB off every proof; the ring phase 1.1 s → ~0.39 s at 2^16).
- **One-shot Merkle hashing**: every leaf/node digest is a single
  BLAKE3 update over a pre-serialized buffer with per-tree/per-level
  prefixes (bit-identity locked by a differential test) — ~2–3× on every
  tree in the system.
- **The join prover**: the o-round suffix tables (the per-pair-per-round
  suffix product → one table per round) and the h-round packed-bit
  selects (the X columns pack into u64 words once; the contraction is a
  popcount-driven select) — the store router halved. A store-shape
  regression (two-word columns + the a-rounds) is pinned in
  join_seeded_regress.
- **The per-segment glue**: whole-word witness reads (wword), the
  store-X claims evaluated directly from the packed words via the
  tabulated φ, the x/selector tables moved into the join witnesses.
- **PhiTab**: the tabulated φ_c (32 nibble lookups vs the 128-iteration
  branch loop) — ~100× on the ring's A-factor materialization.
- Gate repairs for the current toolchain (the cs-s1 honest-data assert
  that fired on the adversarial sub-proofs, the 1.99 clippy lints).
- Measured on the 4-vCPU EPYC host: 2^16 erc20-mixed-block at
  12,4–13,3k cycles/s = 1,034–1,109 tx/s per machine (was 936), the
  root 2.34 s → 1.60 s; pooled ~3,788 tx/s across the 5 free-tier
  instances (the workers are slower hosts with ~35–40% cross-VM
  contention).
