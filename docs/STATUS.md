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
| RV64I decoder, microcode expansion (register-shift stages), interpreter with the full memory-access log | §5.1 | `magnetar-cpu` | decode/expansion/interp tests |
| Dual SIMD R1CS circuit (bit + packed-field relations, carry chain, byte-lane muxes, halt-latch I/O) | §5.1 | `magnetar-cpu::circuit` | circuit-satisfaction tests |
| Construction 3, all seven steps, end-to-end | §5.5 | `magnetar-zkvm::protocol` | fibonacci, load/store, sub-word, shift, hot-cell; 15+ tamper classes |
| **Cross-step constraints** (the Remark-4 mechanism: the next-fetch transition argument + the flag-derivation constraints + the ECALL output binding) | §5.5 / Rmk 4 | `magnetar-memory::cross_step`, `magnetar-cpu::circuit` | jumped-pc, fake-halt, fake-taken, wrong-output tampers |
| **Tree-IVC with shared commitments** (segments share one MosaicPCS commitment; boundary chaining via Remark-4 bindings; claim batching at the root) | roadmap §8.2 | `magnetar-ivc` | 1/4/8-segment roundtrips; 11 tamper classes |
| **The streaming, parallel tree-IVC prover** (checkpointed trace replay, rayon segment parallelism, tables packed directly into the commitment leaves, byte-identical proofs) | §1.3 (the IVC layer) | `magnetar-ivc::streaming` | byte-equality vs. the sequential prover on 1/2/4 segments; record-level replay equality; tamper classes |
| **The parallelism rulebook + kernels** (R1–R10: rayon Merkle/encode/sum-check/join/segments, thresholds, batching, caching) | engineering | `docs/PERFORMANCE.md`, the crates' kernels | `par_bind_matches_seq`, `pack_tables_matches_assemble_and_pack`, `join_matches_sparse_prover` |
| Proof-size instrumentation (committed-bit inventory, per-component bytes) | roadmap §7 | `magnetar-zkvm::stats` | CLI smoke |

## Explicitly not implemented

These are the honest gaps. Each is tracked with its reason and its plan.

### 1. Zero-knowledge (ZK) — roadmap H1
The paper's construction is a plain (transparent, post-quantum) SNARK;
ZK masking of witness-dependent sum-check messages is an H1 roadmap item
(masking polynomials in the sum-check rounds + blinded Merkle openings).
Not started. **Plan:** mask the router/twist/shout round messages and the
MosaicPCS query paths; the Spartan outer is already public-coin.

### 2. Dual-rate PCS wrap and cross-proof claim aggregation — roadmap H2
`PcsParams` already carries the rate `k` (the codeword extension), so a
low-rate *wrap* instance is parameterizable, but the wrap circuit (the
verifier-as-guest that compresses a proof into a recursively verifiable
one) does not exist. Cross-*proof* aggregation (batching independent
proofs' evaluation claims by random linear combination) exists **within**
one execution (the tree-IVC root batches all segments' claims into one
PCS evaluation argument) but not across independently-stated proofs.
**Plan:** the H2 wrap instance reuses `magnetar-ivc`'s shared-commitment
aggregation over the wrap circuit's trace.

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
