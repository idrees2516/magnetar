# Remaining work — the honest inventory

This repository implements the Magnetar paper's complete protocol
(Construction 3, all seven online-phase steps) **plus** the roadmap's
Horizon-2 aggregation layer: the register-shift stage-select extension,
K-segment claim aggregation, tree-IVC chunking with proven state
continuity, the dual-rate PCS schedule, a measured cost model, and the
end-to-end benchmark. `ROADMAP.md` is the research program; this page is
the production-style accounting of **what is not implemented yet**, in
impact order, with the concrete path for each item. Nothing here is
hidden — every gap is visible in `docs/benchmarks.md`'s measured numbers.

## 1. Prover throughput (the largest gap — engineering, not research)

**Measured**: ~6 cycles/s on the benchmark host (2 vCPU / 4 GiB) vs. the
paper's 4.3M cycles/s at 2¹⁸ on an M4 Max. The protocol, the proof
sizes, and the verification are in the paper's range; the *engines* have
unoptimized constants.

**Remaining** (roadmap Horizon 1, in expected-gain order):
1. **Memoized / lazy round products** in the sum-check provers: fold
   each factor only when the round's variable touches it, hoist partial
   products per variable group, and stop re-enumerating the remaining
   Boolean cube per round with per-variable index walks. The standard
   production prover shape; removes the dominant constant. Verifier
   unchanged.
2. **Small-domain arithmetic**: the committed tables are mostly bits and
   one-hot indicators. Keep them packed (u64 words, XOR) and pay
   `GF(2^128)` multiplications only at the eq-weighted boundaries.
3. **Streaming BinaryTwist**: the Val materialization is `O(K·L·T)`
   memory (390 MiB peak at 2⁷). The streaming form computes the value
   sum-check rounds incrementally over the access sequence — the
   structure Twist-and-Shout already provides.
4. **Parallelism**: `rayon` across segments/chunks (the bodies are
   independent on one coordinator-ordered transcript) and across cube
   halves inside the round products; Merkle hashing parallelizes over
   leaves.

## 2. Recursion: the verifier-guest and the wrap (Horizon 2's flagship)

**Done**: the *claim-side* — K-segment aggregation (59% measured
wire-size saving at K=4), tree-IVC chunking with boundary continuity,
the dual-rate profiles (+8.4% measured for the wrap rate), and the
aggregation ledger the wrap will consume.

**Remaining**:
1. **The verifier-guest**: the Construction-3 verifier as an RV64
   program running inside the VM (the paper's recursion-friendly
   property: the verifier is hash-sum-check arithmetic — the natural
   guest). A substantial build: the round checks, the eq-table
   evaluations, and the Merkle path verification as a checksummed RV64
   program with the child proof fed in as RAM words.
2. **The wrap circuit** at the wrap rate: the small instance proving
   "the guest verified the batch", sized by the *aggregated* artifact
   (the measured 59% saving is what keeps its witness small). The
   final proof becomes wrap-sized, independent of `k` — the roadmap's
   100–160 KiB projection.
3. **Distributed Fiat-Shamir**: today's aggregation assumes the
   segments' claim lists arrive in the deterministic in-order
   traversal. Out-of-order/parallel leaf provers need the
   coordinator-star (a Merkle-tree transcript for partial commitments —
   roadmap §8.3's v1 recommendation), then PCD falls out of the tree
   recursion.

## 3. Zero-knowledge

**Status**: not implemented — the protocol as given is honest-prover
(the paper defers ZK like its baselines). **Remaining**: blinding terms
in the sum-check round messages (localized to the engine; the protocol
structure needs no change) and randomized PCS openings, with care at
the ring-switch's φ-anchors (they open packed leaves publicly, so the
mask covers the committed table, not the openings).

## 4. Fiat-Shamir in the QROM

**Status**: classical ROM, like every system in the paper's comparison.
**Remaining**: the QROM composition argument for the full protocol —
worth paying at the recursion milestone, where the argument concentrates
in the wrap (the roadmap's risk register).

## 5. ISA completeness

**Status**: RV64I implemented and proven end to end (register shifts
through the stage-select extension, sub-word loads/stores, the halt
latch), with the satisfiability harness pinning the circuit.
**Remaining**:
1. **LB/LH sign extension**: sub-word loads currently use unsigned
   (LBU/LHU) semantics, consistent between the interpreter, the
   circuit, and the proofs — the ISA-correct sign extension needs the
   sign-bit routing in the byte-lane mux plus its e2e tests.
2. **RV64M and beyond** (mul/div, atomics, compressed, FP): outside the
   paper's stated scope; M extends naturally through lookup arguments.
3. **The 15-workload suite** (paper Table 1: sha256, keccak-f1600,
   lz4-decode, radix-sort, pointer-chase, ERC-20 …): needs an RV64
   assembler/toolchain story and the larger traces that become
   feasible after item 1's speed work.

## 6. Production engineering

1. **A real wire codec**: `wire.rs` counts the uncompressed size.
   A production format (length-prefixed, entropy-aware field-element
   compression, optional zstd framing) is a contained, testable module.
2. **Constant-time pass**: the field arithmetic is constant-time
   (PCLMUL + portable fallback); a formal pass over the transcript
   hashing and Merkle path building (the `subtle`-crate discipline) is
   due before any deployment that treats the witness as secret.
3. **Fuzzing + property-based tamper tests**: the decoder and the
   interpreter's access log are the input-facing surfaces
   (`cargo-fuzz`); the tamper classes should become a property-based
   harness that flips random proof fields and asserts rejection.
4. **Bench CI**: gate performance regressions on a pinned workload
   once the speed work lands (today's noise floor is the unoptimized
   engine itself).

## 7. Research-adjacent (deliberately out of scope for this milestone)

The roadmap's Horizon 3 horizon, recorded so the boundary is explicit:
64-bit challenge fields, systematic-code sparsity, Bolt-class sketch
codes, the GPU program (the 150M-cycles/s study), and the hybrid
lattice wrap fallback (the documented fallback if the measured wrap
misses the 160 KiB proven-soundness line).
