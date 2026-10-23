# Roadmap

Magnetar's development runs on a three-horizon schedule (grounded in the
competitive analysis versus Jolt-Akita: proof size 598 KiB today versus
91 KiB; proving throughput 6.9M cycles/s, 5–18× ahead), with the
**GPU-native proving target** as the next major milestone. Each item
below carries its status and the measured or modeled number that
defines "done".

## Horizon 1 — parameterize (months)

*Rate/query dials, ZK masking, small-domain arithmetic, Bolt-class
codes. Target: proofs of 240–350 KiB and 10–20M cycles/s on the same
12 CPU cores.*

| Item | Status | Notes |
|------|--------|-------|
| PCS rate/query profiles | **Done** | `PcsProfile::{FastChunk, Balanced, CompactWrap, DeepWrap}`; the dual-rate projection is measured by `magnetar profiles`: the same proof runs 216 KiB (rate 1/2) → 128 KiB (rate 1/16, 14 queries). Soundness accounting in `magnetar-pcs/src/eval.rs`. |
| Digest tuning | Analyzed | The 16-byte-digest option cuts path bytes ~45% but re-derives the collision budget at the 120-bit target; deferred until the wrap lands (its small trees make the tradeoff clean). |
| ZK masking | **Primitive done, wiring in progress** | `magnetar_sumcheck::zk`: per-round mask polynomials — perfect HVZK for the round messages, O(degree) randomness per round, no big masking tables (works over the routers' 2^27-entry joint spaces). Wired into the Mosaic batching discharge behind `KeyGen::enable_zk` (end-to-end tested). The remaining wiring sites are inventoried in `docs/STATUS.md`. |
| Small-domain sum-check arithmetic | Remaining | Bit/subfield arithmetic for the bit-coefficient rounds plus the eq-optimization: 1.3–2× on the sum-check phases (published instruments; the engine's Horner rewrite landed earlier). |
| Bolt-class sketched codes | Research | The encode phase is `(3+ε)N` adds + `(1+ε)N` hashes vs. the affine-substitution transform; a Ligerito-compatible drop-in by its authors' own design. Worth 3–4× on the commit phase — `docs/BENCHMARKS.md` shows commit is currently ~0.1% of prove time at small traces, so this lands with the GPU milestone where the commit phase dominates. |

## Horizon 2 — recurse (1–2 quarters)

*Verifier-guest + tree IVC + claim batching; low-rate wrap; streaming.
Target: final proofs of 100–160 KiB, throughput amortized within 5–20%
of the chunk prover, unbounded traces via chunks.*

| Item | Status | Notes |
|------|--------|-------|
| Claim aggregation | **Done, measured** | The Mosaic batching sum-check α-folds every component claim set into **one** ring-switch evaluation argument. `segments.rs::aggregation_report` measures the shared-commitment saving (one commitment, one evaluation argument, one transcript chain instead of S). |
| Segments share the commitment | **Done (v1)** | `magnetar-zkvm::segments`: `execute_chunks` runs S×T cycles with explicit state handoff; the concatenated witness is the monolithic trace (tested bit-for-bit); the Twist continuity spans chunk boundaries; one proof, one commitment. |
| Dual-rate wrap | Profiles done; **wrap circuit remaining** | The chunk/wrap PCS profiles exist and are measured (`FastChunk` vs `DeepWrap`). The **wrap** — an implementation of this verifier proved once at a low rate — needs the verifier-guest circuit below. |
| Verifier-guest circuit | Design complete, implementation remaining | The wrap circuit: SHA3-256 duplex transcripts + BLAKE3 Merkle paths + GF(2^128) arithmetic over the ~600 KiB chunk transcript, proved with BinarySpartan at `DeepWrap`. Full design + cost arithmetic in `docs/RECURSION.md`. |
| Tree-IVC schedule | Design complete | Binary tree over chunk proofs; internal nodes batch children's claims (the same Mosaic batching machinery); the coordinator-star distributed Fiat-Shamir keeps the hash-only property. `docs/RECURSION.md`. |
| Streaming within chunks | Design complete | Stream trace contributions through the sum-check rounds (small-space proving): the witness matrix is already produced row-block-wise; the 2^13-entry tile granularity of the engine's parallel threshold is the natural streaming unit. `docs/RECURSION.md`. |

## Horizon 3 — push (research)

64-bit challenge fields (fewer bytes per element, more queries — a
coupled verifier-hashing tradeoff), systematic-code sparsity exploitation
for the one-hot components (the hash-based shadow of Akita's pay-per-bit;
needs a repeated-subtree binding proof), and the hybrid lattice wrap as
the fallback if the pure wrap misses its 100–160 KiB projection.

## The GPU target — 150M cycles/s (the next milestone)

The program's stated goal: **at least 150 million cycles per second on
the least GPUs** — 6.9× beyond Airbender's 21.8M single-H100 baseline —
while holding proofs at 72–105 KiB through the accumulation layer.
The full study ("Magnetar at 150 Million Cycles per Second") grounds
every factor in a published, measured anchor; the build order:

| Phase | Months | Scope | Exit criterion |
|-------|--------|-------|----------------|
| 1. Single-GPU port | 0–3 | `cudarc` host plane; `cutile-rs` bit-sliced sum-check + Merkle kernels; ICICLE as reference | ≥ 2× over the CPU baseline on one H100 (≥ 15M c/s); the commitment kernel < 1 ms |
| 2. Fusion and tiling | 3–9 | Fused message+fold passes; 2^13-entry shared-memory tiles running 13 rounds on-chip; TMA loads; persistent CUDA graphs; the Bolt encoder | ≥ 40M c/s on one H100; ≤ 60 GiB effective traffic per chunk |
| 3. Sharding and accumulation | 9–15 | 2–4 GPU segment sharding (the `segments.rs` interface); Arc folds on the chunk tail; Tree-IVC + DeepFold batching; the FRI-Binius wrap | ≥ 100M on 2× H100 / 1× B300; proofs ≤ 105 KiB at 120-bit |
| 4. The tracer | 12–18 | GPU tracer (chunk-parallel, limb-packed) or a 96-core host pool; double-buffered upload | End-to-end ≥ 150M on 2× B300-class |
| 5. Research integration | 15–24 | AG-sumcheck constants; log-star rounds; tensor-core spike; the VEIL ZK layer | Stretch: 250M+ and/or ZK proofs at unchanged size |

The multiplicative stack behind the numbers:

* **Algorithms**: the algebraic-geometry sum-check (2.36× on the
  dominant round), the Flock zerocheck/lincheck playbook over the same
  Ligerito layer, log-star-round IOPs against the
  rounds-versus-hashing serialization, and Bolt for the encoder.
* **Kernels**: tower-field bit-slicing (one byte per base element,
  carry-free multiplies, PRMT tables — the binary field is worth *more*
  on a GPU than on any CPU), tiling and fusion worth 1.5–2× over the
  straight port.
* **Memory**: a B300's 8 TB/s HBM3e versus the H100's 3.35 — the
  binding kernel (tower-field sum-check over the ~6.4 GiB tables) is a
  streaming workload; 150M on one GPU is a bandwidth calculation, not a
  hope (30–60 GiB effective traffic per 2^22 chunk against a 4–8 ms
  roofline floor).
* **Sharding**: 1.85× on two GPUs; 4× H100 SXM as the air-cooled
  fallback (155–396M modeled).
* **The governor**: at 150M c/s the 2^22-chunk cadence drops to 28 ms,
  so witness generation (host-side ~22M c/s today) must move to the GPU
  tracer or overlap a large CPU pool — the Phase-4 risk with a
  measured precedent (ZisK's 1.5 GHz trace-generation claim).
* **Toolchain**: official Rust CUDA (September 2026): `cuda-oxide`
  (experimental), `cutile-rs` (production track), `cudarc`, ICICLE.
* **The proof-size co-target**: the accumulation layer (Arc-style folds
  on the chunk tail, never re-proving a verifier inside the guest) holds
  72–105 KB at 120-bit proven soundness — the recursion tax ~5× lighter
  than Airbender's 2.56×.

CPU-side prerequisites already in this repository that the GPU phases
build on: the join-enumeration router prover (the top CPU item,
`docs/BENCHMARKS.md`), the `segments.rs` chunk interface, the
measured dual-rate profile tables, and the phase instrumentation
(`MAGNETAR_TIME=1`) that keeps every kernel's CPU baseline honest.
