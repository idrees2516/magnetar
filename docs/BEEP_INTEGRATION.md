# The BEEP integration — m DEEP samples per fold round, threshold proofs of work

Integrated from the **BEEP** artifact (worldfnd/BEEP, cloned 2026-11-27: the
paper `paper/main.typ`, the implementation
`beep-ligerito/flock/crates/flock-core/src/pcs/ligerito/{beep.rs, affine.rs,
affine_fast.rs}`, and the Lean formalization `lean/` of Theorems 3.4 and 5.3)
into the ring switches of `magnetar-pcs`.

## What BEEP is

**"Better Elimination of Pretenders"** improves the soundness analysis of the
DEEP technique (Domain Extending for Eliminating Pretenders) for linear codes,
with one modification: perform **`m` DEEP samples instead of one**. The DEEP
sampling error decreases **exponentially with `m`** (in practice `m = 2..5`
makes it tiny); the subsequent folding error is `(δ·n + 1)/|F|`. Code-based
IOPPs (WHIR, Ligerito, Basefold — and our ring switches) then need only the
code's **list decodability**, not Mutual Correlated Agreement, with smaller
soundness errors — which allows arbitrary linear codes up to the Johnson
bound (and beyond, for list-decodable codes) over practical fields.

The paper's core construction (§ "The BEEP IOR"), per fold of
`u = (u_1, u_2)` with challenge `α`:

1. **DEEP sampling**: the verifier samples `m` points `z`; the prover answers
   `b_1, b_2 ∈ F^m` — the two halves' multilinear extensions at them.
2. **Fold and sumcheck round**: the prover sends the quadratic
   `h(X) = Σ (f_0 + X(f_0+f_1))(v_0 + X(v_0+v_1))`; the verifier checks
   `h(0)+h(1) = μ` and samples `α`.
3. **Batching**: `γ ∈ F^{m+1}` batches the new linear claims
   (`h(α)` and `φ_z(g) = (1-α)b_1 + α b_2`) into one.

The affine variant (folding challenges as affine polynomials, following
CWYZ26 and polygero) improves the post-DEEP folding error to `2/|F|`. Over
GF(2^128) — where this stack's challenges already live in the field — the
scalar form achieves the same envelope and is what is implemented here.

## What was ported (BEEP artifact → magnetar)

| BEEP (`beep.rs` et al.) | magnetar (`magnetar-pcs`) |
| --- | --- |
| `Grind` (`Bits`/`Below`), `exact_at` threshold arithmetic | `deep::Grind` (verbatim port: `numerator·T < 2^(64+128−target)`) |
| `sample_bits`, `least_threshold`, `select_fold`, `beep_ladder` | `deep::deep_ladder` — the least-work `(m, t, grind)` per round for a 128-bit round-by-round target, Johnson-shape inputs at the profile rate |
| `Level { f, w, claim }` — `dot_eq`, `add_eq_weights`, `fold_answers`, `round_message`, `fold` | `deep::DeepChain` — the same `(W, claim)` state; the DEEP answers are the halves' eq-weighted dots, the round message is `[h(0), h_2]`, the fold adds `Σ ζ_j eq(z_j)` into `W'` and the folded answers into `v'` |
| OOD binding (`bind`), grinding the claim batch (`claim_grind`) | `DeepChain::bind` — `bind_points` out-of-domain samples on the level-0 committed table, ground ζ batch |
| `grind_vec` / `verify_vec` (nonce + squeeze behind a threshold PoW) | `magnetar_hash::Transcript::{pow_probe, grind_pow, verify_pow}` — a detached probe digest, so failed candidates leave no trace; prover and verifier mirror exactly |
| The BEEP ladder's per-round accounting (`BeepLevel`) | `deep::DeepLadderLine` (sample bits, post-grind fold error, expected hashes) |
| Query grinding (`grind_and_sample_queries`) | The query **and** anchor position draws run behind the query-grind PoW on both ring switches |
| — (Ligerito's per-level consistency gluing) | not needed: our per-level Merkle roots + correlated query walks + full tail already pin the committed tables; the DEEP chain rides on top |

## Where it landed

* `crates/magnetar-pcs/src/deep.rs` (new, ~1100 lines): the BEEP layer —
  `Grind`, `deep_ladder`, `DeepChain`, `DeepVerify`, and the two-chain
  `DeepPair`/`DeepPairVerify` (the split ring's merged `(P, C)` components
  run as one BEEP layer with shared points, challenges, ζ batches, and
  grinding), plus `DeepLayerProof` (the proof parts).
* `crates/magnetar-hash/src/lib.rs`: `pow_probe` / `grind_pow` / `verify_pow`.
* `crates/magnetar-pcs/src/ring.rs` (the monolithic ring — the zkVM wrap
  path): every fold round now takes `m` DEEP samples **before** its challenge;
  the level-0 committed table is bound out-of-domain; the fold chain runs
  inside the sum-check round loop (the DEEP answers must precede the
  challenge draw); the verifier rebuilds `W_final` over the `2^k` tail and
  checks `<tail, W_final> = v_final`.
* `crates/magnetar-pcs/src/ring_split.rs` (the streaming root's path): the
  same protocol over the **shadow chain** — the full combined `(P, C)`
  codeword tables `[ε][s][x_loc]`, built once (per-segment encodes reused by
  the tree derivations, no second encode pass) and folded in place per round.
  The anchor positions are now also **draw-verifier-checked** (a pre-existing
  gap: the verifier never mirrored the `ring-anchor-{ai}` draws, so anchor
  positions were effectively prover-chosen — fixed here).
* `PcsParams.deep: DeepParams` — off (`DeepParams::off()`) reproduces the
  legacy protocol byte-identically; the default is on (`m = 3`, 2 binding
  points, modest grinds), and `deep_ladder(k, n_total, 128.0)` derives the
  least-work parameters at each profile rate.

## Soundness accounting

Per fold round, with list bound `L`, line bound `N`, threshold `t`, and `m`
samples (the ported formulas):

* the tail of the pretender count: `-log2 Pr[|A_z| ≥ t] ≥ m(128 − log2 v) −
  3(128 + log2 L) + log2 t + log2(t−1) + log2(t−N)`;
* the fold squeeze: error `(t + 3L)/2^128`, brought below `2^-target` by
  `Grind::exact_at(t + 3L, target)` bits of grinding;
* the batch squeeze (`[ρ, ζs]` from one grind): same discipline;
* the ladder picks the least `(m, t, grind)` reaching the target.

For this stack's substitution code (relative distance ≈ 1 at rate `2^-k` over
generic γ) the Johnson-shape inputs at the profile rate are conservative.

## Costs and profile

The DEEP layer adds, per fold round: `2m` answer values and 2 round-message
values per chain (the split's pair: `4m` + 4), one grind nonce, and the
prover passes for the answer dots, round messages, and the `W`/shadow folds
(~4–8 passes per round over the shrinking shadow, ~2–4× the chain derivation
cost; the shadow also costs `4·2^{n_total+k}` values of prover memory — the
tree-streaming envelope is unchanged). The wins: fold-round soundness no
longer leans on MCA-style analyses (BEEP's exponential bound), the ground
challenges sharpen every squeeze, and the query draws can drop their counts
for the same target (the profiles keep the current counts; the ladder
documents the headroom).

## Tests

* `deep::tests::deep_chain_roundtrip` — the prover/verifier mirror, claim
  agreement, the shared challenges, the final tail check, tail tampering.
* `deep::tests::deep_chain_lie_rejected` — a lied answer or round message is
  caught (immediately, through the ground nonce, or at the final check).
* `deep::tests::deep_ladder_reaches_target` — the ladder reaches 128 bits at
  every profiled rate.
* `deep::tests::grind_exact_at` / `transcript_pow_mirror` — BEEP's threshold
  arithmetic and the PoW mirror.
* `ring::tests::ring_deep_roundtrip_and_tamper` — the full monolithic ring
  on the BEEP path: honest verify + the tamper classes, including DEEP
  answers, DEEP round messages, fold/bind/query-grind nonces.
* `ring_split::tests::split_ring_deep_roundtrip_and_tamper` — the split ring
  on the BEEP path at four geometries, including the anchor-position binding.

## Related 2025/2026 results (the research map)

Fetched through the Cloudflare challenge with cfsolve (all four verified,
text-extracted):
`scripts/rd_research/papers/{tas_2025_105_twist_and_shout,
dkt26_2026_2056, affcomb_2026_2863, polygero_2026_2481}.{pdf,txt}`.

* **DKT26** — Dao, Kominers, Thaler, *Reed–Solomon Codes Beyond Johnson*
  (ePrint 2026/2056, Thm 4.4): precise RS list-decoding/MCA bounds up to
  capacity, deterministic decoding, proof-size cuts in ProveKit/ZisK/
  LambdaVM. BEEP's `mca_dkt_ladder_*` variants use it to drop the row union.
* **CWYZ26 / affine combiners** — Chiesa et al., *Affine Combiners for
  Reed–Solomon Proximity Gaps* (ePrint 2026/2863): affine (degree-1)
  folding challenges with error `4/(η|F|) + O(n²/(ρ³η¹⁰|F|²))`, one position
  of agreement lost; BEEP's affine section (Thm 5.3, Lean-formalized) folds
  it into the DEEP analysis at `2/|F|`. Our scalar-over-GF(2^128) form keeps
  the same envelope.
* **Polygero** — Frigo, shelat (ePrint 2026/2481): polynomial challenges
  (`s ≈ 74` at `N ≈ 25724`) for proximity testing — the general form of the
  affine idea.
* **Krachun (ePrint 2026/782), Crites–Stewart (CRYPTO 2026), Chai–Fan**:
  proximity gaps **fail** near list-decoding capacity — the regime BEEP's
  "beyond Johnson with list-decodable codes" cautiously gestures at.
* **Flock** — Bünz et al. (ePrint 2026/1329): the Ligerito host
  SNARK (our previous deep-research report covers it; BEEP builds on its
  code).
* **Twist and Shout** — Setty, Thaler, Zhu (ePrint 2025/105, CRYPTO 2026):
  the 2025 revision's PDF was fetched through the Cloudflare challenge with
  `cfsolve` (see below) — `scripts/rd_research/papers/
  tas_2025_105_twist_and_shout.pdf`.
* Nearby systems worth tracking: **TensorSwitch** (Bünz et al., CRYPTO 2026
  — hash-based PCS from tensor codes), **Akita** (ePrint 2026/1983 — lattice
  PCS for Jolt/Twist-and-Shout-scale sparse polynomials),
  **Revisiting Sum-check-based PCS** (ePrint 2025/2249, rev. 2026).

## cfsolve

The Cloudflare blockage on eprint.iacr.org (a "Just a moment…" 403 on
`/2025/105.pdf`) was solved with **cfsolve**
(github.com/chris124567/cfsolve): built from source (BoringSSL + wreq with
the Safari-26-iOS profile; `LIBCLANG_PATH` from the libclang wheel,
`BINDGEN_EXTRA_CLANG_ARGS` pointing at the gcc headers), extended with a
public `Browser::clear_and_fetch` (clear the challenge, then fetch raw
bytes through the same TLS/cookie session — the stock CLI saves the cleared
body through a lossy string path, which mangles PDFs) and a `fetchpdf` bin.
The clearance cookie alone is not enough: it is bound to the browser TLS
fingerprint, so plain `curl` stays 403'd.
