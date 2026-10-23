# Performance Rules — parallelism, SIMD, streaming, batching, caching

These are the **mandatory engineering rules** for every module in the
Magnetar prover stack. They encode the industry practices used by Jolt,
Binius, and Plonky3-class provers, specialized to our field
`K = GF(2^128)` and the Fiat–Shamir architecture. Every PR that touches a
prover-side hot path is reviewed against this document (checklist at the
end).

## R1 — Determinism: parallel reductions must be order-free

All `K`-arithmetic reductions are order-independent: `+` is XOR and `×` is
commutative and associative, so **any** thread schedule produces the
bit-identical result. Therefore parallelize sums/products freely — the
Fiat–Shamir transcript stays deterministic.

What must **never** be parallelized or reordered:
- the **sequencing** of transcript absorbs/challenges (protocol order is
  security-critical), and
- any computation whose result feeds a transcript *before* all of its
  inputs are complete (compute-then-absorb, never absorb-while-computing).

Use `Gf128: Sum` / `par_dot_xor` for reductions; never collect into a
shared accumulator with fine-grained locking.

## R2 — Thresholds: no tiny parallel tasks

Dispatch work to rayon only above a measured threshold
(`PAR_THRESHOLD = 4096` field elements ≈ 64 KiB working set; hash and
encode kernels have their own). Below the threshold the sequential kernel
runs — task-dispatch overhead otherwise dominates. Thresholds live in
named constants, are referenced from this document, and may only change
with a benchmark in the PR description.

## R3 — Collapse structure before you parallelize it

A `2^JV` joint-space enumeration with one-hot or sparse factors is a bug,
not a workload. Before writing any `par_iter` over a joint cube:
1. **One-hot kill** — a one-hot selector collapses its dimension to a
   single per-column position (router join enumeration; the streaming
   Twist value sum-checks).
2. **Sparse support** — iterate only nonzero tensor entries
   (`nnz(R)` routing pairs, not `2^(o+u+h)`).
3. **Column streaming** — process per-cycle data in cycle order with
   O(state) memory, never materializing the `O(K·L·T)` history table.

Algorithmic collapse (often 100–1000×) always precedes parallelism
(2–16×). A parallel implementation of a blowup algorithm is still a
blowup.

## R4 — SIMD: branch-free field kernels on contiguous data

- Tables are `Vec<Gf128>` (16-byte elements, contiguous, alignment-friendly).
- Field multiplication goes through `mul` (PCLMULQDQ fast path,
  constant-time, portable fallback). Never hand-roll field arithmetic.
- Inner loops use the branch-free fold identities
  `A + X·(A + T1)` / `eq(x,b) = 1 + x + b` — no per-element `if`.
- Iterate forward over slices (hardware prefetch); never index with
  computed strides when a chunk iterator expresses the same access.

## R5 — Batching

- One `absorb_bytes` per round message (never per coefficient).
- Merkle trees hash level-by-level with one domain per level; leaf hashing
  is one parallel pass.
- Codeword encoding processes independent ε-blocks in one parallel pass.
- Challenge vectors are sampled with `challenge_field_vec`, not one call
  per element, whenever the protocol allows.

## R6 — Caching and memoization

Within a single `prove()` call, a table that depends only on
(statement, witness) is computed **once** and threaded through. Known
memoized artifacts: selector tables from the trace, X routing tables,
eq tables at shared points, the packed Mosaic witness. A second
computation of the same table is a defect.

## R7 — Streaming memory discipline

`O(T)`-per-cycle materializations are forbidden where a linear pass
suffices:
- witness generation uses **checkpoints** (state snapshots every `C`
  cycles) so any chunk regenerates on demand and in parallel;
- the Twist value histories are maintained **incrementally**
  (eq-weighted memory updates, O(L) per cycle);
- commitments pack and hash in streaming chunk order.

The only per-cycle storage kept for a whole proof run is per-column
scalars/vectors (`≤ L·T` field elements). Peak prover memory is a
tracked metric (docs/BENCHMARKS.md) — a regression is a bug.

## R8 — No false sharing, no shared mutable state

Each parallel task either writes a disjoint output range
(`par_chunks_mut`, output-buffer folds — the sequential in-place fold is
order-dependent and must not be parallelized in place) or reduces into a
task-local accumulator combined by `+` at the end. Zero mutexes on the
prover hot path.

## R9 — Structure of a parallel kernel

```text
if len < THRESHOLD { sequential kernel }
else {
    chunk = max(MIN_CHUNK, len / (threads * 4))   // multiple tasks per core
    out.par_chunks_mut(chunk)                     // disjoint writes
        .enumerate()
        .for_each(|(ci, chunk)| { ... })          // reads via absolute index
}
```
Chunks are power-of-two aligned and never split a pair-group; the
sequential and parallel kernels are the same body so tests can compare
them bit-for-bit (`par_bind_matches_seq`).

## R10 — Honest measurement

Benchmarks measure wall-clock prove/verify on fixed programs
(`benches/`, `docs/BENCHMARKS.md`), report cycles/s against the paper's
numbers, and include memory. Optimizations land with numbers, never with
"should be faster".

---

## Review checklist (tick all in PRs touching prover code)

- [ ] New joint-space loops: structure collapsed per R3 before
      parallelizing?
- [ ] All reductions order-free (R1); transcript order untouched?
- [ ] Thresholds respected (R2); parallel and sequential kernels
      bit-identical (tested)?
- [ ] Tables contiguous, branch-free folds, `mul`/`inv` only (R4)?
- [ ] Absorbs batched (R5); repeated tables memoized (R6)?
- [ ] No new `O(T)`-per-cycle materialization (R7); peak memory measured?
- [ ] Disjoint writes / local accumulators only (R8)?
- [ ] Benchmarks updated (R10)?
