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
   trace of virtual cycles (the CPU circuit's witness matrix `W`).
2. **Commit** — `W`'s bits are packed 128 per `K` element over the top 7
   index bits; the affine-substitution codeword of the packed vector is
   Merkle-committed with BLAKE3. This is the protocol's only commitment,
   sent before any verifier challenge.
3. **Virtual Spartan** — the two SIMD R1CS relations (F₂ and K) are
   reduced jointly — lockstep outer sum-checks sharing the instance
   challenges, one α-batched inner sum-check — to one evaluation claim
   per witness segment.
4. **Reduce the virtual claims** — BinaryTwist (registers, RAM),
   BinaryRouter (shifts, sub-word stores) and BinaryShout (bytecode) own
   the virtual segments; their reductions turn the claims into component
   claims on committed data.
5. **Prove the committed evaluations** — MosaicPCS lifts every claim to
   `G` for free (aligned subcubes), batches them with one sum-check
   (binding the 7 packing variables first — after those rounds the dense
   factor *is* the c\*-folded table), and discharges the terminal via ring
   switching: the c\*-codeword's fold chain, anchored to the main
   commitment with exact `φ_c` bit-extraction checks at random
   `(x, ε=0)` positions.

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

## Documentation

- `docs/BENCHMARKS.md` — the measured cycles/s versus the paper's
  Table 6, the per-phase breakdown, and the gap analysis.
- `docs/ROADMAP.md` — the three-horizon schedule and the **GPU target**
  (150M cycles/s on the least GPUs: the five-phase build order).
- `docs/RECURSION.md` — the dual-rate wrap, tree-IVC, claim batching,
  and streaming designs (Horizon 2).
- `docs/STATUS.md` — the honest implemented-versus-remaining inventory.
- `docs/security.md` — the soundness accounting.

## Building and testing

```sh
cargo build --release
cargo test --workspace      # 69 tests: field, hash, sum-check (dense +
                             # sparse), PCS + ring switching, Virtual Spartan,
                             # Twist, Shout, the discharge helpers, Router, CPU,
                             # end-to-end + 15 tamper classes
cargo run --release --bin magnetar -- fib 5
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
evaluations on the single commitment. Four programs (arithmetic,
load/store with sub-word semantics, shifts, a hot-cell with the public
input binding) prove and verify, and **15 tamper classes are rejected** —
one per transmitted claim and round-message family.

The `magnetar-zkvm::stats` module measures the cost model directly (the
committed-bit inventory per component, the PCS geometry, the byte
breakdown): the PCS evaluation argument dominates at ~84% of the proof,
exactly the paper's cost-anatomy prediction, making it the target of the
proof-size roadmap.

Scope notes (paper-faithful): the cross-step micro-PC constraints are
omitted exactly as the paper's Remark 4 omits them; the decode-vs-flag
correlation and the register-shift stage-select are documented engineering
extensions. Trace lengths in tests are 2^3–2^5 cycles (CI-scale); the
paper's production envelope is 2^22 cycles at ~6.5–7.7M cycles/s with
~600 KiB proofs and 10–12 ms verification.

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
