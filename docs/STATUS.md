# Implementation Status

An honest accounting of what is implemented, what is not, and why —
maintained per the repo's ground rules (every claim here is backed by
code and tests, or explicitly marked as not).

The reference is the Magnetar paper (eprint 2026-2423). Sections marked
**[paper §x]** cite the construction being discussed.

## Implemented and tested

| Component | Paper ref | Where | Tests |
|---|---|---|---|
| GF(2^128) field (PCLMULQDQ + portable Karatsuba, constant time) | §2 | `magnetar-field` | field axioms, cross-reference |
| SHA3-256 Fiat–Shamir duplex transcript; BLAKE3 Merkle trees | §2 | `magnetar-hash` | duplex + tree tests |
| Sum-check engine (char-2 round convention, arbitrary bind orders) | §3.2 | `magnetar-sumcheck` | roundtrip + tamper |
| Sparse sum-check prover (factors in their own variable spaces) | §4 | `magnetar-sumcheck::sparse` | dense-engine cross-check |
| BinarySpartan / Construction 1 (outer+inner sum-checks) | §4.1 | `magnetar-spartan` | roundtrip + tampers |
| SIMD R1CS + Virtual Spartan (Construction 2, shared-instance lockstep outers, single inner) | §4.2, §5.3 | `magnetar-spartan` | roundtrip + 4 tampers |
| BinaryTwist (registers d=1 p=2; RAM d=1 p=1; LT recurrences, Frobenius refinement, address virtualization, Δ* one-hotness) | §4.4 | `magnetar-memory::twist` | folds, tables, materialization, tampers |
| BinaryShout (bytecode bus Φ + α₀ sum-forcing; Ψ index/address claims; selector checks) | §4.4 | `magnetar-memory::shout`, `::discharge` | roundtrip + tampers |
| BinaryRouter R_sh / R_st (routing tensors, offset-eq links, X split) | §5.4 | `magnetar-router` | tensor shape, roundtrips, tampers |
| **Join-enumeration router prover** (linear-in-T, the paper's §5.4 prover-work claim) | §5.4 | `magnetar-router::join` | byte-identical cross-check vs the sparse prover; ~600× faster at T=2^12 |
| **Final-state binding** (the Remark-4 initial/final memory binding: the zero-check + the evolution reduction) | Rmk 4 | `magnetar-memory::final_state` | roundtrip + 4 tampers |
| End-to-end benchmarks with the paper comparison | §6 | `magnetar-cli::magnetar-bench`, `docs/BENCHMARKS.md` | the CLI harness |
| MosaicPCS (aligned subcubes, Equation-4 lift, α-batching, ring switch, single PCS evaluation argument) | §5.2 | `magnetar-zkvm::mosaic` | roundtrip + tamper |
| **The hierarchical Mosaic discharge** (per-segment lockstep local stages + the segment stage; the 611-paper's bounded-memory root) | §5.2 / 611 | `magnetar-zkvm::discharge` | per-instance byte-identity vs the generic engine; seq/streaming identity; 2^15–2^16 e2e |
| **The small-value round window** (char-2 monomial window + eq-weighted pack + select fast-forward) | 1117 §5.1 / 587 §5.1+C.5 | `magnetar-sumcheck::window` | byte-identity vs the generic engine (shapes × orders); real-verifier roundtrips |
| **The ERC-20 mixed-block load test** (the paper's Table-6 workload shape, proven and verified through 2^16) | §6 | `magnetar-cli::magnetar-bench-erc20` | `end_to_end_erc20_mixed_block` + the harness asserts |
| RV64I decoder, microcode expansion (register-shift stages), interpreter with the full memory-access log | §5.1 | `magnetar-cpu` | decode/expansion/interp tests |
| Dual SIMD R1CS circuit (bit + packed-field relations, carry chain, byte-lane muxes, halt-latch I/O) | §5.1 | `magnetar-cpu::circuit` | circuit-satisfaction tests |
| Construction 3, all seven steps, end-to-end | §5.5 | `magnetar-zkvm::protocol` | fibonacci, load/store, sub-word, shift, hot-cell; 15+ tamper classes |
| **Cross-step constraints** (the Remark-4 mechanism: the next-fetch transition argument + the flag-derivation constraints + the ECALL output binding) | §5.5 / Rmk 4 | `magnetar-memory::cross_step`, `magnetar-cpu::circuit` | jumped-pc, fake-halt, fake-taken, wrong-output tampers |
| **Tree-IVC with shared commitments** (segments share one MosaicPCS commitment; boundary chaining via Remark-4 bindings; claim batching at the root) | roadmap §8.2 | `magnetar-ivc` | 1/4/8-segment roundtrips; 11 tamper classes |
| **The streaming, parallel tree-IVC prover** (checkpointed trace replay, rayon segment parallelism, tables packed directly into the commitment leaves, byte-identical proofs) | §1.3 (the IVC layer) | `magnetar-ivc::streaming` | byte-equality vs. the sequential prover on 1/2/4 segments; record-level replay equality; tamper classes |
| **The parallelism rulebook + kernels** (R1–R10: rayon Merkle/encode/sum-check/join/segments, thresholds, batching, caching) | engineering | `docs/PERFORMANCE.md`, the crates' kernels | `par_bind_matches_seq`, `pack_tables_matches_assemble_and_pack`, `join_matches_sparse_prover` |
| Proof-size instrumentation (committed-bit inventory, per-component bytes) | roadmap §7 | `magnetar-zkvm::stats` | CLI smoke |
| **The linear-in-T memory provers** (Twist read/write ports, the bytecode Shout Φ, the Ψ checks, the selector checks, the cross-step S1 transition): cycle enumeration with the one-hot kill and the prefix-XOR cumulative sums — O(T) prover work, byte-identical round messages | §4.4 | `magnetar-memory::linear` | byte-identity cross-checks vs the dense engine (multiple shapes); the end-to-end tamper suite |
| **The fused sum-check kernel** (one traversal per round, zero per-point view allocations, specialized 1/2/3-factor products, no double message computation) | roadmap H1 | `magnetar-sumcheck::engine` | the engine roundtrips + the full protocol suite |
| **The sparse-support batching factor** (the Mosaic α-combination over lifted claims with Boolean suffixes: Σ 2^{vars_i} work, not S·2^ℓ) | roadmap H1 | `magnetar-sumcheck::public::SumEqFactor` | `sum_eq_matches_sum_of_eq` + end-to-end |
| **The structured verifier factors** (`EqProdFactor`, O(T) suffix-sum LT columns): the verifier's public-factor evaluation is polylogarithmic — no joint-cube materialization | roadmap H1 | `magnetar-sumcheck::public`, `magnetar-memory` | verify-time measurements (21 ms at 2^8); tamper suite |
| **ZK masking, recovered and wired** (`magnetar_sumcheck::zk`: per-round mask polynomials, perfect HVZK of the round messages, O(degree) randomness/round; `finish_offset` on the verifier) — wired into the Mosaic batching discharge behind `KeyGen::enable_zk` | roadmap H1 | `magnetar-sumcheck::zk`, `magnetar-zkvm::mosaic` | `end_to_end_zk_masked_batching` (+ offset and round tampers) |
| **The dual-rate PCS profiles** (`PcsProfile::{FastChunk, Balanced, CompactWrap, DeepWrap}`) and the `magnetar profiles` measurement (tails 62.8 → 27.3 KiB) | roadmap H1/H2 | `magnetar-pcs::eval`, `magnetar-cli` | `end_to_end_wrapped_deep`; the CLI harness |
| **The dual-rate wrap** (roadmap H2): the chunk side keeps its fast profile; the public evaluation tail is discharged at `DeepWrap` with transcript-sampled consistency spot-checks binding the deep commitment to the chunk commitment (the bounded-depth spot-check accumulation shape; ARC's distance-preserving fold documented as the unbounded-depth follow-on) | roadmap H2 / RECURSION §1 | `magnetar-zkvm::wrap`, `prove_with_wrap` | `end_to_end_wrapped_deep` (roundtrip + 3 tamper classes + the size assertion) |
| **The phase instrumentation** (`MAGNETAR_TIME=1`): the per-phase wall clock of the full prover | engineering | `magnetar-zkvm::protocol::PhaseTimer` | the BENCHMARKS anatomy table |

## Explicitly not implemented

These are the honest gaps. Each is tracked with its reason and its plan.

### 1. Zero-knowledge (ZK) — roadmap H1 (primitive landed; wiring partial)
The round-message masking primitive is implemented and tested
(`magnetar_sumcheck::zk`: per-round mask polynomials subject to the
running constraint, a uniform finish offset, `finish_offset` on the
verifier — perfect HVZK of the round messages at O(degree) randomness
per round). It is wired into the **Mosaic batching discharge** (the
dominant witness-derived sum-check) behind `KeyGen::enable_zk`,
end-to-end tested. **Remaining sites:** the Twist/Shout/router round
messages (now linear provers — the mask composes with them the same way)
and the commitment-side revelations (blinded packed leaves before the
commit; the tail reveal padding). The revealed evaluation *claims*
remain public by design, as in the paper.

### 2. The dual-rate wrap — landed (spot-check shape); the verifier-guest circuit remains
The **dual-rate wrap** is implemented and measured
(`magnetar-zkvm::wrap`): the chunk side keeps its fast profile, the
public evaluation tail is re-committed and discharged at `DeepWrap`
(rate 1/16, 14 queries), and transcript-sampled consistency spot-checks
open packed-leaf positions in both trees, binding the deep commitment to
the chunk commitment the transcript already absorbed (the bounded-depth
spot-check accumulation shape of [BMNW24]; measured 44% tail reduction).
**Not implemented:** (a) the ARC-style distance-preserving fold — the
attached ARC paper (eprint 2024/1731) achieves unbounded accumulation
depth via Reed–Solomon quotients and out-of-domain sampling; our
affine-substitution code is multilinear-over-Boolean-cubes, so the
quotient machinery does not transfer directly — the engineering path is
documented in `docs/RECURSION.md`; (b) the hash-circuit verifier-guest
(SHA3/BLAKE3 in-circuit) — the alternative wrap that removes the
spot-check loss entirely; (c) cross-*proof* aggregation of independently
stated proofs (batching exists **within** one execution via the tree-IVC
root).

### 3. Recursion / verifier-as-guest — roadmap H2
No recursive composition. The verifying key's verifier work is measured
(see `docs/BENCHMARKS.md`) and is small enough to target a
verifier-as-guest wrap, which is the roadmap's approach.

### 4. GPU acceleration — roadmap H3
CPU only. The prover's kernels are rayon-parallel across cores (the
sum-check folds, the join enumeration, the Merkle hashing, the codeword
encode, the tree-IVC segments — docs/PERFORMANCE.md); no CUDA/OpenCL
backend exists.

### 5. Streaming — partially implemented (the paper's §1.3 sum-check
### provers are not)
**Implemented** — the tree-IVC layer now streams: the trace never
resides (checkpointed `StreamCursor` replay), segments prove in parallel
with their tables packed straight into the shared commitment leaves and
dropped, and the commitment path never materializes the boolean union
(`magnetar-ivc::streaming`, byte-identical to the sequential prover,
tested on 1/2/4 segments). **Not implemented** — the per-segment
Twist/Shout/Spartan provers still materialize their dense tables; the
paper's linear-in-T per-cycle provers are the remaining work (they are
also what unlocks the parallel speedup — the measured segment workers
are memory-bandwidth-bound, docs/BENCHMARKS.md).

### 6. Production-scale parameters
Test-scale configuration throughout: RAM = 2^8 words, registers padded to
64 cells, the shift-amount routing at `AMOUNT_BITS = 2` (the production
envelope is 6), the micro-PC segment 8 bits wide. All are parameterized
constants; none are raised to the paper's operating point because the
prover is single-threaded Rust and the wall-clock would not be
representative. **Benchmarks in `docs/BENCHMARKS.md` state this
explicitly and compare structure, not marketing numbers.**

### 7. `JALR` (register-indirect jump)
Rejected at `keygen` with a clear error. The cross-step transition table
is a *static* function of the fetch address and the branch-taken flag;
`JALR`'s target is data-dependent (`(rs1 + imm) >> 2`), which needs the
next-pc column constrained by R1CS (the adder exists) plus the next-MLE
shift argument over a *data* one-hot — the documented extension path.
Everything else in RV64I (including `JAL`, all branches, all loads and
stores, register shifts via stage expansion) is proven with the full
cross-step chain. Fail-closed beats silent unsoundness.

### 8. The paper's 15-workload benchmark suite
We benchmark four representative programs (fibonacci-style ALU chains,
load/store with sub-word lanes, shifts, a running-sum loop with branches)
at multiple trace lengths, plus the tree-IVC configuration. Porting the
paper's exact workload binaries (erc20, sha256, ...) requires the
toolchain (ELF loading) that is out of scope for the reference
implementation.

## Soundness notes (what a reviewer should check first)

1. **The char-2 idle-variable rule**: every sum-check term must cover
   all joint variables through its factors (a factor idle in a summed
   variable contributes its value times 2 = 0). Enforced by construction
   in every reduction; tested by the roundtrip tests.
2. **The cross-step chain** (this release): pc(0) = 0 (public point
   claim), the per-cycle transition `fetch(j+1) = TT[fetch(j), taken(j),
   halt(j)]` (the transition argument), `halt ⟺ opcode = SYSTEM` and
   `taken = branch ∧ comparison` (R1CS flag-derivation constraints),
   and the ECALL output binding (`halt ⇒ rd = rs1 = x10` through the
   register Twist). A jumped pc, a faked halt, a faked branch direction,
   or a wrong output each break a distinct check.
3. **The 120-bit soundness budget**: ~200 sum-check rounds at degree
   ≤ 8 over |K| = 2^128 — the paper's §1.3 accounting applies unchanged.
4. **Regression-tested invariants** (each was a real bug found by the
   benchmark workloads and is now pinned by a test): the sub-word store
   tensor routes the source bit `o − 8a` (not `o`) at byte offset `a`;
   the store-router offset claims select bit-column `k` (the binary of
   k, not the unit vector); the load zero-extension is gated on
   `is_load` (an ungated rule would zero ALU results with high bits).
