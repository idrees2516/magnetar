# Tree-IVC: chunked proving with state continuity

This document specifies the tree-IVC layer — the incrementally-verifiable
computation architecture the research roadmap selects for Magnetar (its
§8.2: *"the correct default for Magnetar is tree-IVC with claim-batching at
each internal node, because both mechanisms reuse MosaicPCS machinery it
already has"*). The implementation is `magnetar-zkvm::ivc`; the entry
points are [`keygen_chunked`](
https://docs.rs/magnetar-zkvm/latest/magnetar_zkvm/protocol/fn.keygen_chunked.html),
[`prove_chunked`](crate::ivc::prove_chunked), and
[`verify_chunked`](crate::ivc::verify_chunked).

## 1. The problem

A single Construction-3 proof covers one power-of-two trace window: the
prover's memory and work are linear in the trace, and a long computation
must be split. Splitting a zkVM trace is not free — the machine state
(registers, RAM, micro-PC) flows across every cut, and a verifier given
`k` independent proofs has no reason to believe the pieces concatenate into
one honest execution. The roadmap prices the three hash-only options:

| Schedule | Prover shape | Verifier | Constraint |
|---|---|---|---|
| Sequential IVC | chain the accumulator through every chunk | `O(1)` beyond the wrap | fully serial |
| **Tree-IVC + claim batching** | pairs/sets of chunks merge at tree nodes | `O(k)` small checks + one shared eval argument | ordered Fiat–Shamir (the coordinator-star) |
| Aggregation-then-wrap only | batch `k` finished proofs | one wrap | needs the wrap first |

Magnetar's segments already share one commitment inside a chunk
(Virtual Spartan's lockstep outers + the Mosaic component layout). The
tree-IVC layer generalizes that sharing one level up: **chunks share one
commitment**, and the claim-batching at the tree node is exactly the
component-merge the Mosaic layout already performs.

## 2. Chunking

`execute_chunk` runs the machine for at most `trace_len − 8` real cycles
(the margin covers the longest microcode expansion — the 6-stage register
shifts), stopping at the first **safe instruction boundary** at or after
the budget: a cut never lands inside a register-shift stage sequence, whose
intermediate stages chain through x0's scratch slot. The trace is padded
to exactly `trace_len` cycles with **frozen-PC no-op cycles**:

* the padding fetches the ROM row at the frozen exit micro-PC (so the
  committed PC column *carries* the boundary address the arguments bind),
* the write port writes the row's redirected destination — the bytecode
  Shout binds the write one-hot to the fetched bus's rd slice, so the
  padding must write *that* cell:
  * a **halted** machine latches the output (`halted_latch = 1`, the
    halt-latch I/O binding),
  * a **mid-program boundary** writes the cell's current content back — a
    zero-increment identity write that leaves the architectural state
    untouched;
* reads log the row's registers' current values, keeping the memory
  arguments consistent.

`prove_chunked` cuts chunks until the machine halts. All chunks share one
KeyGen, so `k` is whatever the program needs.

## 3. Boundary continuity

The state crossing a boundary splits by cost:

### 3.1 Registers (published, identity-bound)

Each chunk publishes its entry and exit register files — the 32
architectural words plus the trash cell (32) that x0-redirected writes
target (cells 33..64 are never written; x0's scratch slot is not
architectural and resets at every cut). The **final-state identity
sum-check** binds the published pair to the committed write log:

```
Σ_{k,o,j} eq(r_k,k)·eq(r_v,o)·wfa(k,j)·inc(o,j) = exit(r_k,r_v) + entry(r_k,r_v)
```

over the joint `(k, o, j)` space, with the write one-hot `wfa` and the
increment `inc` as the two sparse dense factors (the prover's work is
linear in the trace; the verifier materializes only the two small eq
public factors). A forged boundary cannot be made consistent with the
committed log, and chunk `c+1`'s entry must equal chunk `c`'s exit (a
public check the verifier performs on the published words).

### 3.2 Micro-PC (published, fetch-bound)

Each chunk publishes its entry/exit micro-PC. The fetch one-hot component
is claimed at the first and last cycle's Boolean points
(`e₀ = 0…0`, `e_last = 1…1`), where its multilinear evaluation is exactly
the eq-packing of the published micro-PC — a forged PC fails the
committed-component discharge. The interpreter's padding freezes the PC
column at the exit address, so the last cycle genuinely carries it.

### 3.3 RAM (committed, chained)

RAM is the large state and is **not** published. Each chunk commits two
image components — `ram-init` and `ram-final`, bit tables over the
`(address, bit)` cube — and the same final-state identity binds
`ram-final = ram-init + Σ wfa·inc` inside the chunk. The chain closes
with:

* **chunk 0's identity pins its entry image to the public input** (its
  init term is the verifier's own evaluation of the input image);
* **each boundary adds a continuity check**: at a random point `r_b`,
  chunk `b`'s `ram-final` and chunk `b+1`'s `ram-init` are claimed with
  values summing to zero (characteristic-2 equality). Two distinct images
  agree at a random point with probability `≤ 14/2¹²⁸`.

The RAM BinaryTwist runs in **claimed-M₀ mode** for every chunk of the
aggregation: the reconstruction sum-checks transmit their `M̂₀(r_k, r_v)`
values (absorbed into the transcript) and leave PCS claims at those
points, discharged against the `ram-init` component — so every read is
explained by the chained image even though the verifier never sees it.
An entry image that disagrees with the truth at a cell the chunk never
accesses is inert: the reads it must explain are checked, the exit it
must produce is checked, and the continuity check consumes only the
committed values.

### 3.4 The identity sum-check (shared machinery)

The RAM and register identities share one sparse sum-check shape
(`magnetar_sumcheck::sparse`): each factor stays in its own variable
space and folds only on its own rounds — the paper's linear-in-`T`
prover work. The terminals are claims on the *already committed*
`wfa`/`inc` components, so the identity adds no new commitments.

## 4. Claim-batching — the tree node operation

The `k` per-chunk Mosaic component lists concatenate (name-prefixed) and
pack onto one cube through the standard aligned-subcube builder, and the
`k` chunks' bit tables are committed in **one** BLAKE3 Merkle tree.
Claim `i` of chunk `c` is merged claim `c·n_comp + i`, and every claim of
the aggregation lifts into the merged cube by appending its merged
Boolean suffix — the same Equation-4 lift the single-chunk protocol uses (and the same per-segment body machinery the claim-aggregation path exposes as `SegmentProof`).

The claim set is the union of:

1. every chunk's step 1–4 discharges (the direct-segment and
   memory-argument claims, exactly the single-chunk collection),
2. every chunk's `ram-init`/`ram-final` claims at its identity point and
   its twist's claimed-M₀ claims,
3. every chunk's identity terminals (`wfa`, `inc`) and PC claims,
4. every boundary's continuity pair.

**One** MosaicPCS evaluation argument — the batching sum-check over the
merged cube's `ℓ + ⌈log₂ k⌉` variables, the ring switch, one query set —
discharges all of it.

### 4.1 What it saves (measured)

The cost model (`magnetar_stats::aggregate_stats`) measures the layers
directly. At the test envelope (5 chunks × 32 cycles):

```
aggregate: 517,752 bytes = core 36% + identities 3% + eval 61%
claim-batching: the eval layer is paid ONCE —
  314,624 bytes shared vs 1,187,040 for 5 separate proofs (3.8x)
```

The evaluation layer dominates a chunk proof (the single-chunk cost
anatomy: 84%); batching removes its `k`-scaling, leaving the per-chunk
step 1–4 transcripts — `O(k)` in bytes — as the remaining scaling term.
The `log₂ k` growth of the shared argument (deeper tree, `log₂ k` more
batching rounds and path levels) is the only overhead the merge adds.

### 4.2 The tree schedule and out-of-order chunks

The merge is associative: aggregating chunk sets is the same component
concatenation, so the distributed schedule can combine subtrees as data
arrives and the root runs the single step-5 — the binary-tree IVC shape
(the schedule CycleFold popularized): wall-clock parallelism across the
leaves, `log k` depth, final proof independent of `k` once the wrap
absorbs the per-chunk transcripts. The Fiat–Shamir constraint (the
roadmap's §8.3) is that challenges derive from the *ordered* transcript,
so shards produce partial round messages and a single coordinator
serializes them — the hash-only coordinator-star. The Merkle-tree
transcript is the documented scale-up path; nothing in the cryptography
depends on the topology.

## 5. The coordinator-ordered transcript

Both sides derive the transcript identically:

```
Transcript::new("magnetar-ivc", [b"tree-ivc-v1",
                                 program, input, output, boundaries])
absorb("cm-g", merged_root)
for c in 0..k:
    step 1–4 core of chunk c        (the single-chunk labels, shared)
    sample  ram-state-point-c       absorb ram-state-claims-c
            the RAM final-state identity sum-check
    sample  reg-state-point-c       the register identity sum-check
    sample  pc-point-in-c, pc-point-out-c
for b in 0..k−1:
    sample  cont-point-b             absorb cont-claims-b
one MosaicPCS discharge over every claim
```

The per-chunk cores run sequentially on the shared transcript (the
coordinator serialization); execution and witness generation parallelize
freely, and the partial-message machinery for lockstep distributed proving
already exists (`magnetar_sumcheck`'s round-capturing API).

## 6. Security notes

* **Boundary binding.** Each chunk's reads are bound (by its Twists) to
  its committed entry image; the identity binds entry → exit through the
  committed write log; chunk 0's identity binds the entry image to the
  public input; the continuity checks chain the images at random points
  (`≤ 14/2¹²⁸` per boundary, multilinear Schwartz–Zippel). By induction,
  every read is explained by the true accumulated state and every
  published boundary equals the committed images' evolution.
* **The padding is part of the machine.** The frozen-PC no-op cycles are
  real cycles: their writes are committed (zero increments on
  boundaries), their fetches are the frozen row, and their latches bind
  the output only on the halting chunk. A boundary cut therefore changes
  nothing about the program's semantics — the next chunk resumes at the
  exit micro-PC with the exact architectural state the last real cycle
  left.
* **The merged commitment.** The root is absorbed before any challenge;
  the batched evaluation argument's soundness is the single-chunk
  argument's (the FRI-style fold-chain proximity analysis), now over a
  cube `log₂ k` deeper. The per-chunk claims' binding is unchanged: each
  still lifts from its own aligned subcube.
* **Aggregate soundness budget.** The construction adds per chunk: two
  identity sum-checks (≈ `14 + n` rounds), two PC claims, and per
  boundary one continuity check — all characteristic-2 sum-check
  arguments at ≤ `2⁻¹²⁰` per invocation over the ~200-round envelope,
  keeping the aggregate within the protocol's 120-bit target the same way
  the single-chunk step 4 does (see `docs/security.md`).

## 7. What is next (the H2 roadmap mapping)

| Roadmap item | Status in this layer |
|---|---|
| Instrument the cost model | `aggregate_stats` — layers + the batching win measured |
| Claim aggregation | the merged commitment + one step-5 (this layer) |
| Tree-IVC schedule | the associative merge + the ordered-transcript constraint documented; the distributed partial-message path is the `magnetar_sumcheck` lockstep API |
| Verifier-guest wrap | **next**: prove the aggregate's verification in-circuit (the RV64I interpreter already executes the guest; SHA-256-in-RV64I is the engineering item) — collapses the `O(k)` core layer to the wrap size |
| Dual-rate | the chunk proofs keep the high rate; the wrap parameterizes the low rate (both are `PcsParams`) |
| Streaming within chunks | the prover materializes per-chunk Val tables; streaming trades space for pass count and does not change the protocol |

## 8. Testing

The layer ships 16 tests (in `ivc.rs`, on top of the workspace's 77 others): the cross-chunk RAM + register
carryover (a value that only exists through the carried image), the
public-input binding, single-chunk compatibility, a five-chunk
fibonacci, two write-log consistency regressions (the boundary reads vs
the shadow evolution; the padding's write one-hot vs its increments),
the measured cost model, and ten tamper classes: the continuity values,
a false chunk-1 entry image, a false chunk-0 entry image (the public
input pin), a forged exit register, a forged exit micro-PC, a broken
boundary chain, the commitment root, an identity round, an identity
terminal, a corrupted interior-chunk core, and the Mosaic terminal.
