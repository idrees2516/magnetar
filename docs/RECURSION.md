# Recursion — the dual-rate wrap, tree-IVC, and streaming design

This is the production design for the Magnetar paper's §6.2/§6.5
scaling path: *prove executions in 2^20–2^24-cycle chunks and
recursively compose their proofs by proving an implementation of the
verifier*. The paper ships the strategy in two sentences and its
parameter arithmetic; this document is the engineering version, with
every cost grounded in this repository's measured cost model
(`stats.rs`, `docs/BENCHMARKS.md`).

**Status: the chunk-side machinery (segments, aggregation, the
dual-rate profiles) is implemented; the wrap circuit is designed here
and not yet implemented** — tracked in `docs/STATUS.md`.

## 1. The strategy

Call the two artifacts the **chunk proof** and the **wrap proof**.

* Chunk proofs are internal: they never leave the prover's trust
  domain, so they tune purely for guest-verifier speed — the
  `FastChunk` PCS profile (rate 1/2, 40 queries): the fastest encode
  and Merkle phase per cycle.
* The wrap proof is the only public artifact: it proves the verifier
  circuit of the accumulated chunks, a witness vastly smaller than the
  application witness, at the `DeepWrap` profile (rate 1/16, 14
  queries). The dual-rate projection measured on this repository's
  proofs: 216 KiB at `FastChunk` → 128 KiB at `DeepWrap`
  (`magnetar profiles`); the paper's own projection for the composed
  system is 100–160 KiB final proofs.

The structural asymmetry that makes this the right bet (and that
Jolt-Akita cannot copy cheaply): the verifier is a hash-dominated
workload — exactly what the binary-field stack is world champion at.

## 2. What the chunk side already has (implemented)

* `segments.rs`: `execute_chunks` runs S×T cycles with explicit state
  handoff; the concatenated witness is the monolithic trace (tested
  bit-for-bit); the register/RAM Twist instances span the whole row
  space, so cross-chunk continuity is free.
* Claim aggregation: the Mosaic batching sum-check α-folds every
  component claim set into one ring-switch evaluation argument — the
  "verify a random linear fold of the k proofs' claims in one pass"
  step of the roadmap, already the shape of every proof this
  repository produces.
* The dual-rate profile dials and their measured size arithmetic.

## 3. The verifier-guest circuit (the wrap's witness)

What the Magnetar verifier does, from the guest's perspective, to
check one 2^22-cycle chunk proof (~600 KiB, verifying in 10–12 ms
natively in the paper's Table 6; this implementation's verify phase
measured in `docs/BENCHMARKS.md`):

1. **Transcript hashing** — the dominant cost. The Fiat–Shamir
   transcript is a SHA3-256 duplex stream (~600 KiB absorbed ≈ 9,600
   Keccak-f1600 permutations). Keccak-f1600 in binary-field
   arithmetization: ~25–30K witness bits per permutation including
   witness generation (the BinarySpartan paper measures 287K
   permutations/s CPU-side). Guest cost: ~2^28–2^29 bits.
2. **Merkle path checks** — ~40 correlated queries × 3 authentication
   paths of (n+k) digests: ~120 × 26 BLAKE3 invocations ≈ 3.1K hashes
   ≈ 2^24-ish bits.
3. **Field arithmetic** — the round-sum checks and the fold relations:
   a linear number of GF(2^128) multiplies, < 2^24 bits.

The wrap witness is therefore ≈ 2^28–2^29 bits — a **16–64× reduction**
from the chunk's ~2^32 committed bits, exactly the paper's arithmetic.
Proving it once at `DeepWrap` is sub-second on laptop-class hardware;
the public proof lands at the wrap's size (the measured 128 KiB
projection at this trace scale, 100–160 KiB at 2^22 per the paper's
model).

**Circuit shape**: one flat "verifier-as-guest" relation per
sub-check (the Jagged-PCS observation: a circuit whose size depends
only on total transcript area, not per-table structure), the SHA3/BLAKE3
cores as shared subcircuits, and the public input = the chunk
statement (the program, the trace length, the commitment root, the
I/O). The circuit is public and versioned with the verifier.

## 4. The IVC schedules

* **Sequential IVC**: chunk i's proof absorbs the accumulator ("all
  prior chunks verified") into chunk i+1's public input. Overhead:
  one verifier-guest proving pass per chunk — the paper's 30–120 ms
  against a 612 ms 2^22 chunk → **5–20% sequential overhead**;
  single-threaded, minimal total work.
* **Tree-IVC** (the default): pairs of chunk proofs combine in a
  binary tree of depth log₂k; each internal node **batches its
  children's evaluation claims** (the same α-fold Mosaic machinery,
  applied to the children's PCS claims) and proves the batch
  verification circuit once. Wall-clock parallelism at ~2× total
  verifier work; the final proof is independent of k; the schedule
  degrades gracefully when chunks arrive out of order — the property
  distributed proving needs.
* **Aggregation before the wrap** (the refinement that multiplies the
  saving): batch the k accumulated proofs' final evaluation claims by
  a random linear fold into one claim, verify the fold in one pass,
  then wrap once. Fiat–Shamir transcripts are sequential data, not
  sets — no multiset hash needed, no LtHash, no SIS: the hash-only
  property survives intact.

**Distributed Fiat–Shamir** (PCD shards): the coordinator-star for v1
— shards compute partial sum-check messages, one coordinator hashes
the round transcript in order. The Merkle-transcript is the scale-up
path. This is a throughput–latency tradeoff, not a security one.

## 5. Streaming within chunks (the small-space prover)

Today's prover materializes the witness matrix (q × m) and the packed
table; peak RSS measured at ~1.6 KiB/cycle-ish scale in
`docs/BENCHMARKS.md`. The streaming design: re-emit each witness
row-block's contribution per sum-check round, never holding more than
one row-block plus the running accumulators:

* The two SIMD relations already read one row-block at a time; the
  engine's 2^13-entry parallel threshold is the natural tile size.
* The Twist/Shout/Router sum-checks read per-cycle slices
  (address components, increments): stream them alongside.
* The PCS fold chain streams the codeword: encode → hash level 0 →
  fold → hash level 1 → …, one pass per level, O(2^n) space total for
  level 0 (halving each round).

The published small-space Jolt analysis (overhead "potentially well
below a factor of two", no recursion at all) sets the expected cost;
the payoff here is reach: traces of 2^26+ on one machine while the
recursion milestone matures. The two techniques compose — stream
within chunks, recurse across them.

## 6. Risk register (from the quantified roadmap)

* **Proven-soundness floor**: 100–160 KiB is the honest limit of the
  pure construction; sub-100 KiB needs Horizon-3's 64-bit challenges
  (a coupled verifier-hashing cost) or the hybrid lattice wrap.
* **Wrap-witness growth**: chunk proof size inflates the wrap witness
  — batch claims before wrapping and keep the chunk rate high.
* **QROM**: the composition argument concentrates at the wrap; pay the
  quantum debt early there.
* **Engineering**: streaming and tree-IVC touch the prover's core
  loops; the 2^13 tile granularity limits the blast radius.
