# Magnetar

**A post-quantum RV64I zkVM from standard hashes** — a complete Rust
implementation of *"Magnetar: High-speed post-quantum RV64 zkVM from
standard hashes"* (eprint 2026-2423).

Magnetar proves executions of programs compiled to the general-purpose
RV64I ISA, with all computational security rooted in **standard hash
functions** (SHA-3 for Fiat–Shamir transcripts, BLAKE3 for Merkle
commitments) — no elliptic curves, lattices, algebraic hashes, trusted
setup, or conjectured Reed–Solomon proximity gaps.

```
$ magnetar fib 2
Magnetar fib(2) — 16 virtual cycles, output = 3
prove: 1.7 s, verify: 29 ms, accepted: true
```

## Architecture

Following the paper's layering (in the spirit of LambdaVM's
field/utils/proof/arith/cpu split), one crate per component:

| Crate | Paper § | Contents |
|---|---|---|
| `magnetar-field` | §3 | `B = F₂`, `K = F_{2^128}` (the GCM polynomial, polynomial basis), PCLMULQDQ + portable carry-less multiplication, characteristic-two `eq`, multilinear folding, 128-bit packing with the `φ_c` bit-extraction functional |
| `magnetar-hash` | §6 | SHA3-256 Fiat–Shamir transcripts (domain-separated, ratcheting, length-bound), BLAKE3 Merkle trees (index/size/domain-bound leaves) |
| `magnetar-sumcheck` | §3.2 | The general multi-factor sum-check engine over characteristic two: `Σ_{j≥1} c_j = running` round checks, dense/public factors, `HighBitsFirst` binding for MosaicPCS's packing-first order, a round-by-round API for Virtual Spartan's lockstep |
| `magnetar-pcs` | §3.6, §5.2 | The hash-based multilinear commitment in Ligerito's role: affine-substitution codewords `C(x,ε) = T̂(x + L(ε))`, the FRI-style fold chain with correlated path queries and fully-revealed tails, plus **ring switching** (c\*-folded tables anchored to the main commitment with exact `φ_c` bit extraction) |
| `magnetar-spartan` | §4.1–§4.3, §5.3 | Sparse R1CS, the SIMD (Phalanx) reduction, **Virtual Spartan** (Construction 2): two relations in lockstep over one shared witness, α-batched single inner sum-check, per-segment claims via `eq(q_i, b_i)` |
| `magnetar-memory` | §4.4 | **BinaryTwist** (Val reconstruction with LT, read/write identities over the joint (k,o,j) space, sum-to-one, increments) and **BinaryShout** (the Φ-combination check with α₀ sum-forcing), both with the paper's **refined one-hotness** (Frobenius Δ\*: `s_h = r_h²`, `r'_h = r_h³/c_h`), plus the **discharge helpers** (§5.5 steps 3–6: the Ψ-sum-checks binding eq-weighted address/index claims, the joint-space bytecode Shout, the per-factor selector checks) |
| `magnetar-router` | §5.4 | **BinaryRouter**: routing-identity sum-checks over the joint (o,j,u,h) space for shifts (`R_sh`, Eq. 5, two one-hot selector fields) and sub-word stores (`R_st`, Eq. 6, byte-offset variables linked via `eq(a_k, â_k(j))`) |
| `magnetar-cpu` | §5.1 | The complete RV64I decoder, microcode expansion (register shifts → 6 static-routing stages), the interpreter with the virtual-cycle trace, and the **dual SIMD R1CS circuit** (the F₂ bit constraints: booleanity + the 64-bit adder carry chain; the K packed products: the word-wide write mux, the packed sum identity, the halt-latch I/O binding) |
| `magnetar-zkvm` | §5.2, §5.5 | **MosaicPCS** with the component layout (direct segments + the auxiliary one-hot/increment polynomials on aligned subcubes, the Equation-4 suffix lift), the **complete Construction-3 protocol** (all seven steps), the bytecode ROM tables, and the **proof-size instrumentation** |
| `magnetar-cli` | — | The `magnetar` binary |

## The protocol (Construction 3)

1. **Execute** — the interpreter runs the expanded program and records the
   trace of virtual cycles (the CPU circuit's witness matrix `W`) — per
   segment in the aggregated mode, padded to fill the witness cube exactly.
2. **Commit** — `W`'s bits are packed 128 per `K` element over the top 7
   index bits; the affine-substitution codeword of the packed vector is
   Merkle-committed with BLAKE3. This is the protocol's only commitment,
   sent before any verifier challenge — shared by every segment in the
   claim-aggregated mode.
3. **Virtual Spartan** — the two SIMD R1CS relations (F₂ and K) are
   reduced jointly — lockstep outer sum-checks sharing the instance
   challenges, one α-batched inner sum-check — to one evaluation claim
   per witness segment.
4. **Reduce the virtual claims** — BinaryTwist (registers, RAM),
   BinaryRouter (shifts, sub-word stores) and BinaryShout (bytecode) own
   the virtual segments; their reductions turn the claims into component
   claims on committed data. The register-shift **stage-select** binds
   the routers' conditional amount selectors: the trash-cell stage chain
   (public ROM fields), the `sel_stage` one-hot (ROM-authenticated), and
   the stage-selector sum-check composing `Q_lo = ROM(sel_lo) + STAGE`
   additively in characteristic two.
5. **Prove the committed evaluations** — MosaicPCS lifts every claim to
   `G` for free (aligned subcubes), batches them with one sum-check
   (binding the 7 packing variables first — after those rounds the dense
   factor *is* the c\*-folded table), and discharges the terminal via ring
   switching: the c\*-codeword's fold chain, anchored to the main
   commitment with exact `φ_c` bit-extraction checks at random
   `(x, ε=0)` positions. In the aggregated mode this single discharge
   covers **all** segments' claims (roadmap Horizon 2, "aggregation
   before the wrap"): the byte-dominant evaluation argument is paid
   once — measured 3.46x smaller than the standalone counterfactual at
   K=4 segments.

## Horizon-2 instruments

* **Claim aggregation** — `keygen_aggregate` + `prove_multi`/
  `verify_multi`: `K` segments share ONE Mosaic commitment (the
  components' subcubes gain the `log2(K)` index dimension) and ONE
  evaluation argument. `magnetar agg 4` proves 4 executions of a
  program against one commitment and prints the amortization ledger.
* **Tree-IVC** — `keygen_chunked` + `prove_chunked`/`verify_chunked`
  ([`docs/ivc.md`](docs/ivc.md)): a long execution is cut into chunks at
  instruction boundaries; the state crosses every boundary through
  arguments — published register files and micro-PCs bound by the
  final-state identity sum-check, RAM carried by committed
  `ram-init`/`ram-final` image components with claimed-M0 twists and
  per-boundary continuity checks at random points — and the `k` chunks
  merge into ONE Mosaic cube and ONE commitment with a single shared
  evaluation argument. `magnetar chunked 32 96` proves 5 chunks and
  prints the batching win (the eval layer paid once, 3.8x).
* **Dual-rate PCS schedule** — `PcsParams::chunk()` (rate 1/2, internal
  artifacts: half the commit work, one level shorter paths) vs
  `PcsParams::wrap()` (rate 1/4, the conservative public profile);
  `magnetar agg N chunk|wrap` measures both.
* **The measured cost model** — `stats.rs` now reports the aggregation
  ledger (per-segment sum-check bytes, the single evaluation argument,
  the analytic standalone counterfactual, the saving ratio).

The full engineering program — what is done, what the horizons are, and
the 150M-cycles/s GPU study's build order — is in [`ROADMAP.md`](ROADMAP.md)
and [`docs/architecture/`](docs/architecture/) (one deep-dive per component).

## Security

* **Hashes only**: SHA-3 (FIPS 202) for Fiat–Shamir, BLAKE3 for
  commitments — both standard, heavily scrutinized, non-algebraic.
* **Transparent**: no trusted setup; the code's public γ-parameters are
  hash-derived from a public seed.
* **Constant-time** field arithmetic (branchless, PCLMULQDQ fast path
  with a portable fallback); `#![forbid(unsafe_code)]` everywhere except
  the audited clmul intrinsics module.
* **120-bit target** (the paper's configuration): sum-check soundness
  `≈ (#rounds × degree)/|K|` with `|K| = 2^128`; BLAKE3 Merkle binding
  at 128-bit collision resistance; the fold chain follows the
  FRI/Ligerito proximity analysis within the unique-decoding radius (the
  same standard the paper relies on, citing the proven gaps of
  Ben-Sasson et al.) — see `docs/security.md` for the full accounting.

## Building and testing

```sh
cargo build --release
cargo test --workspace      # 75 tests: field, hash, sum-check (dense +
                             # sparse), PCS + ring switching, Virtual Spartan,
                             # Twist, Shout, the discharge helpers, Router, CPU,
                             # full-circuit satisfiability regressions,
                             # end-to-end + 20 tamper classes, claim
                             # aggregation, dual-rate profiles
cargo run --release --bin magnetar -- fib 5
cargo run --release --bin magnetar -- agg 4      # claim-aggregated proof
cargo run --release --bin magnetar -- agg 4 chunk # at the chunk rate
```

## Status and scope

All paper components are implemented and tested: the binary-field stack,
the sum-check engine (dense + the sparse linear-in-T prover), the PCS with
ring switching, Spartan/Virtual Spartan, BinaryTwist/BinaryShout with the
refined one-hotness, BinaryRouter for both instance families, the RV64I
decoder/interpreter with microcode expansion and the full memory-access
log, the dual SIMD R1CS circuit, MosaicPCS with the component layout, and
the **complete Construction-3 prover/verifier — all seven steps
end-to-end, with the virtual-claim discharge (step 4) integrated**:
BinaryTwist for registers and RAM, BinaryRouter for shifts and sub-word
stores, and BinaryShout for the bytecode, each reducing its claims to
evaluations on the single commitment. Five programs (arithmetic,
load/store with sub-word semantics, immediate shifts, **register shifts
via the stage-select extension**, a hot-cell with the public input
binding) prove and verify, **20 tamper classes are rejected**, and
**full-circuit satisfiability regressions** pin every witness row of the
test programs against both R1CS relations.

The **register-shift stage-select** extension is implemented end to end:
register shifts (SLL/SRL/SRA by register) expand into 6 conditional
routing stages chained through the trash cell, the ROM carries a
`sel_stage` one-hot authenticated through the fetch, the routers' amount
selector is the witness-conditional pair `{identity, shift-by-2^k}`, and
the stage-selector sum-check binds the condition bit to the amount
register's bits — composing the selector claims additively in
characteristic two.

**Claim aggregation** (roadmap Horizon 2) is implemented: K segments
share one commitment and one evaluation argument (`prove_multi`), with
the measured amortization ledger (3.46x at K=4) and the analytic
standalone counterfactual printed by the cost model. The **dual-rate
schedule** (`chunk` at rate 1/2 vs `wrap` at 1/4) is implemented and
measured (`docs/architecture/pcs.md` carries the honest analysis of what
the rate does and does not move in the affine-substitution code).

The `magnetar-zkvm::stats` module measures the cost model directly (the
committed-bit inventory per component, the PCS geometry, the byte
breakdown): the PCS evaluation argument dominates, exactly the paper's
cost-anatomy prediction, making it the target of the
proof-size roadmap.

Scope notes (paper-faithful): the cross-step micro-PC constraints are
omitted exactly as the paper's Remark 4 omits them; the decode-vs-flag
correlation remains a documented engineering extension. Aggregated
segments are independent (input, output) pairs — cross-segment state
continuity is the tree-IVC milestone on the roadmap. Trace lengths in
tests are 2^3–2^5 cycles (CI-scale); the paper's production envelope is
2^22 cycles at ~6.5–7.7M cycles/s with ~600 KiB proofs and 10–12 ms
verification.

```sh
$ cargo run --release --bin magnetar -- fib 2
Magnetar fib(2) — 16 virtual cycles, output = 3
prove: 1.7 s, verify: 29 ms, accepted: true
committed bits: 22272 across 19 components
virtual (uncommitted) bits: 6272 (22.0% of the witness)
PCS: n = 8 vars, rate 2^-2 (codeword 1024 entries), 40 queries
proof: 199.4 KiB (mosaic+pcs-eval: 84%)
```

## License

MIT OR Apache-2.0.
