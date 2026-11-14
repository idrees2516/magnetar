# Oracle-Aided IVC — the Implementation Plan

**Reference:** Shafi Goldwasser, Yael Tauman Kalai, *Incrementally Verifiable
Oracle Aided Computation*, ePrint 2026/2824 (Nov 8, 2026).

**Status:** planning document only — nothing below is implemented yet. Per the
repo's ground rules, every future claim of implementation will be backed by
code and tests in `docs/STATUS.md`; this file is the *specification of intent*
against which that accounting will be made. Section numbers (§x, Def x, Thm x,
Claim x, Rmk x) cite the paper.

---

## 0. Why this paper lands on this stack

Magnetar proves **machine executions**: given a trace of RV64I virtual cycles,
Construction 3 certifies that the interpreter obeyed the ISA. Everything the
machine touches — registers, RAM, bytecode — is *inside* the proof. What the
stack has no story for is **external data**: answers that arrive at runtime
from a database, a service, or a human, which the computation consumes but
which the prover does not generate. That is exactly the paper's subject — the
*oracle-aided* Turing machine (§4): a machine with a query tape and an answer
tape, where upon entering a designated query state the query `u` is sent to an
oracle `O` and the answer tape is replaced by `O(u)`.

The paper delivers four constructions and one barrier, all under LWE-style
assumptions plus standard hashing/signatures:

| # | Paper | Setting | Verifier | Primitive |
|---|---|---|---|---|
| 1 | Thm 20 (§5) | robust computations, unsigned oracle | **private** key, O(λ/ε) oracle calls | hash + seBARG |
| 2 | Thm 27 (§7.1) | **signed** oracle | public, **zero** oracle calls | hash + seBARG + Sig |
| 3 | Thm 31 (§7.3) | signed oracle, deterministic | public, zero oracle calls, poly(λ, log T) | hash + seBARG + Sig |
| 4 | Thm 45 (§8.3) | signed oracle, **incremental** | public, zero oracle calls | hash + **UpBARG** + Sig |
| 5 | Thm 49 (§8.4) | robust, unsigned oracle, **incremental** | private key, O(λ/ε) oracle calls | hash + UpBARG + OrdIVC |
| — | Thm 24 (§6) | *impossibility*: no public short-crs verifier with o(Q) oracle queries for robust computations | — | executable counterexample |

Positioning against the existing tree-IVC line (`magnetar-ivc`): tree-IVC
splits one long *internal* computation into segments and chains boundary
claims through shared commitments. The paper's IVC maintains a short proof
**per computation step** — update in time poly(λ, S, log T) with no access to
the preceding execution — and additionally certifies **oracle answers**
(signatures checked inside the local relation) so a public verifier needs no
oracle access at all. The two lines are complementary, not competing; the
bridge options are in §8 below.

The application hook for this repo is the ERC-20/stateless-blockchain line:
externalize state reads to a **signed state oracle** (the "oracle" is the
state database signing each (query, answer) pair it serves) and the §8.3 IVC
produces a publicly verifiable, per-block-updatable proof whose verifier never
re-queries the state database — the certificate story an auditor can check
offline.

---

## 1. The paper in one page — the parts that become code

**The model (§4).** An oracle-aided TM `M` with time bound `T`, space bound
`S`, oracle-query bound `Q ≤ T`. Every configuration carries a **step counter**
and an **oracle-call counter** (Rmk 15; both required also in purported
executions; O(log T) bits, so a configuration is O(S + log T) bits). A
**purported execution** (Def 16) is a configuration sequence `τ = (C_0..C_T)`
that respects all ordinary transition rules but may use *any* answer at oracle
steps. `err_O(τ)` counts answers `a_j ≠ O(u_j)` along τ itself.
**ε-robustness** (Def 17): on a no-instance, *every* accepting purported
execution has `err_O(τ) ≥ ε·q(τ)`.

**The primitives.**
- *Hash family with local opening* (Def 4): `Gen/Hash/Open/Verify`, Merkle
  trees — **exists** in `magnetar-hash` (BLAKE3).
- *Updatable hash family* (Def 8): authenticate a **sequence of strings**
  `(z_1..z_k)` with `HT.Append(hk, v, z)` and `HT.Size(v)` — the Merkle-forest
  of [TO26]. **New.**
- *Verification of local updates* (Def 32): `HT.VerifyUpdate(hk, rt, rt′, j, b,
  b′, ρ)` — recompute the new root from one authentication path. **New.**
- *Blank-tape hashing* (Rmk 34): `Hash(hk, x∥0^{N−|x|})` in time
  poly(λ, |x|, log N) — implicit all-zero subtrees. **New.**
- *seBARG* for BatchIndexTMSAT (Def 11): a batch argument over `k` local
  statements with a hidden extraction index `i*` — completeness, efficiency
  (proof ≤ m·poly(λ)), extraction completeness, **index hiding**, **somewhere
  argument of knowledge**. Exists under LWE (Thm 14). **Trait + reference
  stand-in.**
- *UpBARG* (Def 41, §8.2, from [TO26]): the updatable analogue —
  `Gen/TGen(1^λ, j)/Update(hk, crs, v, π, z⁻, z, w)/Ver(hk, crs, v, π, z⁻)/
  Extract(hk, td, v, π)` with incremental completeness, index hiding,
  somewhere AoK, and **last-statement collision resistance**. Exists under LWE
  (Thm 43). **Trait + reference stand-in.**
- *Signed oracle* (Def 25): `O_sig(u) = (a, σ)` with `σ ← Sign(sk, (u, a))`;
  EUF-CMA even given O (Rmk 26), `κ_sig = poly(λ, log T)`. **New** (a
  hash-based signature, see W2).

**The constructions.** All four follow *hash-and-BARG with local extraction*:
hash the transcript, batch-argue the local transitions, extract neighbors to
telescope a potential function. The differences are what the local relation
does at oracle steps and who holds trapdoors:

- §5: transcript `τ = (C_0..C_T, D_1..D_Q)` with an explicit **oracle-call
  table** `D_j = (t_j, u_j, a_j)`; batch size `k = T + Q`; setup samples
  `r = ⌊λ/ε⌋` hidden indices `J ⊂ [Q]` and generates `r` seBARG components
  with extraction indices `(T+j)_{j∈J}`; the **private** verifier extracts
  those table entries and checks `a_j = O(u_j)` against the real oracle.
- §7.1: no table, no fixed Q; the local relation `R_sig` checks
  `Sig.Verify(vk, (u_i, a_i), σ_i)` *inside* the witness; **two** seBARG
  components generated with extraction index 1 and **trapdoors discarded** —
  public verification, zero oracle calls.
- §7.3: deterministic machines; **compressed configurations** (Def 35) carry
  the work tape only as a Merkle hash; `R_short` checks reads via
  `HT.Verify` and writes via `HT.VerifyUpdate`; `κ = λ` suffices
  (polynomial LWE) for poly T. Rmk 38: for long queries/answers, the oracle
  signs their *hashes*.
- §8.3: IVC via UpBARG over **transition records**
  `D_i = (i, vk, hk, vx, C_{i−1}, C_i)` with `vx = HT.Hash(hk, x)`, the
  oracle-free relation `R_sig` (Def at §8.3), the consistency predicate
  `Cons(D, D′) ⇔ i′ = i+1 ∧ same keys ∧ boundary configs match`, and **two**
  independent UpBARG instances; proof `π_t = (v_t, Π_{1,t}, Π_{2,t}, D_t)`.
- §8.4: robust IVC by **blocking** — one block per oracle call,
  `D_j = (A_{j−1}, B_j, A_j)`; the ordinary intervals are certified by an
  inner *ordinary-interval IVC* (OrdIVC, the §8.3 scheme restricted to
  ordinary transitions); the outer UpBARG aggregates the blocks at **dyadic
  scales** `h ∈ {0..H}`, `H = ⌈log₂ Q⌉`, `K_h = 2^h`, with `r = ⌈2λ/ε⌉`
  hidden samples per scale; the private verifier checks at scale
  `h* = ⌈log₂ q⌉` only.

**The barrier (§6).** For `C^O(x) = Maj(O(x,1)..O(x,N))`: any deterministic
public verifier with `ℓ ≤ N/8` oracle queries and `|crs| = o(n)` accepts a
false ¼-robust claim with probability ≥ (1−negl)/(4ℓ), via a
pairwise-independent row-marking attack. This is *executable* (W6) and is the
repo's documented justification for why the unsigned-oracle line must stay
privately verifiable.

---

## 2. Inventory: paper artifact → repo target → verdict

| Paper artifact | Target | Verdict |
|---|---|---|
| Def 4 hash w/ local opening | `magnetar-hash` (BLAKE3 Merkle) | **exists** — extend with fixed-length encodings |
| Def 8 updatable hash (Merkle forest, [TO26]) | `magnetar-hash::updatable` | **new, fully implementable** |
| Def 32 `VerifyUpdate` (local update check) | `magnetar-hash::updatable` | **new, fully implementable** |
| Rmk 34 blank-tape hashing (implicit zero subtrees) | `magnetar-hash::updatable` | **new, fully implementable** |
| §4 oracle-aided machine, counters, purported executions | `magnetar-oa::machine` | **new** (trait + tiny bit-TM + RV64I ECALL hook) |
| Def 16/17 ε-robustness, `err_O` | `magnetar-oa::machine` | **new** (metrics + exact DP for small graphs, Claim 23) |
| Def 25 signed oracle, Rmk 26/38 | `magnetar-oa::sig` + `magnetar-oa::oracle` | **new** (hash-based signature + oracle service) |
| Def 11 seBARG (LWE, Thm 14) | `magnetar-oa::barg` | **trait + reference stand-in** (see §3) |
| Def 41 UpBARG (LWE, Thm 43) | `magnetar-oa::barg` | **trait + reference stand-in** |
| §5 robust privately-verifiable SNARG (Thm 20) | `magnetar-oa::snarg_robust` | **new** (construction exact, primitive pluggable) |
| §6 impossibility (Thm 24) | `magnetar-oa::attack` | **executable counterexample, new** |
| §7.1 signed-oracle SNARG (Thm 27) | `magnetar-oa::snarg_signed` | **new** |
| §7.3 fully succinct variant (Thm 31, Def 32/35, Rmk 38) | `magnetar-oa::snarg_short` | **new** |
| §8.1 IVC definitions (Def 39/40) | `magnetar-oa::ivc` (trait layer) | **new** |
| §8.3 signed-oracle IVC (Thm 45) + compressed variant | `magnetar-oa::ivc_signed` | **new** |
| §8.4 robust IVC (Thm 49), OrdIVC, dyadic scales | `magnetar-oa::ivc_robust` | **new** |
| Soundness *proofs* (hybrids, telescoping, Claims 23/29/30/36/37/46/47) | — | **not code** — only the executable sub-parts (DP potentials, attack, tamper suites) are testable |
| Sub-exponential LWE, `T*`-security, κ = poly(λ,S,log T) asymptotics | — | **not implementable** — concrete κ (128/256) + documented gap (§7) |

---

## 3. The one hard dependency: seBARG / UpBARG — and the two-tier strategy

Both LWE instantiations (Thm 14 via CJJ22/KVZ21/KLVW23; Thm 43 via TO26) are
lattice constructions (lossy/somewhere-extractable batch arguments from LWE
sampling, trapdoors, Regev-style encryption). Implementing them from scratch
is a lattice-cryptography program, not a work package on this stack, and the
repo's philosophy is *standard hashes only*. Faking them silently would be
dishonest accounting. The plan therefore splits the world at a **trait
boundary**:

**Tier 1 — exact, real, measured.** Everything above the trait: the Merkle
forest, `VerifyUpdate`, blank-tape hashing, the signature layer, the
fixed-length encodings, every local relation (`R_loc`, `R_sig`, `R_short`,
`R_block`), the consistency predicates, the Gen/P/V orchestration of §5/§7/§8,
the OrdIVC blocking, the dyadic-scale bookkeeping, the private verification
paths, and the §6 attack. These are the paper's actual constructions and they
are hash-and-logic — fully real here.

**Tier 2 — the trait + a reference instantiation.**
`trait SeBarg` and `trait UpBarg` carry the paper's *exact* syntax and property
contracts (completeness / incremental completeness, efficiency bounds, index
hiding, somewhere AoK, last-statement collision resistance). The reference
instantiation `ReferenceBarg`:

- `Gen(1^κ)` → an empty (or seed-commitment) crs; `TGen(1^κ, j)` → identical
  crs distribution **plus** `td = j` (with an empty crs, index hiding is
  *perfect* — strictly stronger than the paper needs).
- The prover's `Update`/`P` appends `(z, w)` to a hash-chained bundle carried
  in the proof; `v` is the real Merkle-forest root from W1, so all hash
  binding is genuine BLAKE3.
- `Ver` operates in two modes:
  - **`batch-check` (default, sound):** recompute the forest from the bundle
    and check `R(z_i, w_i) = 1` and `Cons(z_i, z_{i+1}) = 1` for *every* i.
    This makes the composite construction *sound as a verification system*
    (any tamper the paper's verifier would catch — a broken local relation, a
    broken consistency link, a bad oracle answer, a forged signature — is
    caught here too), at the cost of O(k) verification — non-succinct, and
    documented as such.
  - **`measure` (for benches):** check the hash binding of the last statement
    only, so that per-step update latency and growth curves measure the
    *construction's* cost, not the stand-in's verifier.
- `Extract(hk, td, v, π)` returns the `td`-th pair with its opening —
  exercising the paper's extraction path byte-for-byte.

What the reference instantiation does **not** give: succinct proofs, and
computational soundness for adversaries that exploit the gap between
"all statements check" and "the extracted one checks". The doc trail
(STATUS.md entry, this file) states this in exactly those words. A future
swap-in — a real LWE UpBARG, or a BinarySpartan-compressed batch proof with a
designated-verifier extraction channel (prove "∃ records hashing to v, all in
L, all consistent" with the SNARK, and let the trapdoor decrypt one encrypted
witness slot) — changes nothing above the trait. That swap-in is explicitly
*out of scope* for the first implementation pass.

---

## 4. Work packages

Ordered by dependency; each package lists paper refs, the API surface, the
tests, and the benches. Line counts are engineering estimates, not promises.

### W1 — the updatable hash layer (`magnetar-hash::updatable`)

**Paper:** Def 8 (updatable hash family: `Hash/Open/Verify` over string
sequences, `Append`, `Size`), the [TO26] Merkle forest, Def 32
(`VerifyUpdate`), Rmk 34 (blank-tape hashing).

- `UpdatableForest`: authenticate `(z_1, …, z_k)`, each `z_i` an arbitrary
  byte string (leaf = domain-separated BLAKE3 over `(index, len, z_i)` — index
  binding is what makes `Extract`'s position checks meaningful). `append(&mut
  v, z)` without touching preceding strings — the Merkle *forest* (one tree
  per power-of-two block, roots folded), which is what gives O(|z| + log k)
  appends and a stable root per prefix.
- `size(v) -> k` encoded in the root commitment (the forest's top fold).
- `open(forest, j) -> ρ` / `verify(hk, v, j, z, ρ) -> bool` — the local
  opening across the forest structure.
- `verify_update(hk, rt, rt′, j, b, b′, ρ) -> bool` (Def 32): check the old
  root against ρ, re-derive the new root with the leaf bit flipped — the
  same authentication path, leaf `b → b′`.
- `hash_padded(hk, x, N)` (Rmk 34): `Hash(hk, x ∥ 0^{N−|x|})` in
  O(|x| + log N) via implicit all-zero subtree roots (precomputed per level).
- **Tests:** append/open/verify roundtrips at k = 1..2^12; append-completeness
  identity `Append(Hash(z), z_{k+1}) = Hash(z ‖ z_{k+1})`; size invariants;
  Def-32 update completeness; the Def-32 *update-soundness experiment* as a
  unit test (adversarial `(rt′, ρ)` tamper classes: wrong new root, wrong old
  bit, path forgery); blank-tape hash equality vs. explicit zero padding;
  collision-resistance-with-opening tamper suite (two openings, same position,
  different strings); forest-root stability under append (byte-identity of
  prefix roots).
- **Benches:** append throughput vs. k; open cost vs. k; `verify_update` cost.
  ~500–700 LOC.

### W2 — the signature layer (`magnetar-oa::sig`)

**Paper:** Def 25 (signed oracle), Rmk 26 (EUF-CMA under O-access,
`κ_sig = poly(λ, log T)`), Rmk 38 (sign *hashes* of long queries/answers).

Decision — the repo is post-quantum-from-standard-hashes, so the default
signature is **in-repo hash-based**, not a curve scheme:

- A Lamport-style one-time signature over BLAKE3-256 (128-bit security:
  128 key-pair rows, sig ≈ 8 KiB, verify ≈ 256 hash calls), wrapped in a
  Merkle tree of `2^h` OTS key pairs (an XMSS-lite) for EUF-CMA over many
  messages — `Gen / Sign / Verify`, stateful signer, public verifier.
- A `sig-ed25519` **optional feature** (`ed25519-dalek`) for interop and for
  quantifying what signature size costs in the local witnesses m (the paper's
  `m = poly(κ, S, log T)` hides |σ|; we will *measure* it).
- Rmk 38 mode: the oracle signs `(HT.Hash(hk, u), HT.Hash(hk, a))` with the
  *oracle's* `hk` — implemented as the default long-query mode of the signed
  oracle service (W3).
- **Tests:** OTS single-use enforcement (second signature on another message
  under the same OTS key must fail the Merkle path anyway); EUF-CMA tamper
  suite (forged σ, wrong (u,a) pair, cross-oracle vk); Rmk-38 hash-signature
  equivalence on tampered long queries.
- **Benches:** sign/verify cost, sig size, and the Merkle-tree height vs.
  number of oracle calls (the signer's key horizon). ~400–600 LOC.

### W3 — the oracle-aided machine model (`magnetar-oa::machine` + `::oracle`)

**Paper:** §4 (oracle-aided TM, Rmk 15 counters, Def 16 purported executions,
Def 17 ε-robustness, `err_O`), §8.1's use of `Reach^O_t(x)`.

- `trait OracleMachine`: `type Config; fn initial(n) -> Config; fn
  is_query_state(&Config) -> bool; fn ordinary_step(&Config, choice) ->
  Option<Config>; fn apply_answer(&Config, answer) -> Config; fn
  is_accepting(&Config) -> bool; fn time(&Config) / queries(&Config)`.
  Fixed-length configuration encoding (Rmk 21: positions in the transcript
  are publicly determined).
- Two instantiations:
  1. **`TinyTM`** — a 32-bit word machine with a work tape, query/answer
     registers, and a programmable instruction list; the unit-test workhorse
     (including the paper's own majority computation for W6).
  2. **RV64I + ORACLE ECALL** — extend `magnetar-cpu` with one ECALL class:
     the guest writes a query to a fixed MMIO page, the oracle responds into
     the answer page; the query state is the ECALL cycle. This is the ERC-20
     integration point (W11).
- `trait Oracle { fn call(&self, u) -> (a, Option<σ>) }` + `SignedOracle`
  (holds `sk`, returns `(a, σ)`, optionally Rmk-38 mode) + `PlainOracle`
  (returns bare answers — the §5/§8.4 line).
- **Robustness tooling:** `err_O(τ)` over an execution; the exact
  configuration-graph DP of Claim 23 — `d^x_t(C)` (min #incorrect answers to
  reach C at t), the bounded potential `Φ^x_t(C) = min{d^x_t(C), Q}`, and the
  ε-robust no-instance predicate — computable for the *small test machines*
  (poly(2^S)); used by the tamper suites of W5/W10 to verify that rejected
  attacks are rejected for the paper's reason (potential gap), not by
  accident.
- **Tests:** counter invariants (time/queries under purported executions with
  lying answers); purported-execution acceptance with wrong answers (Def 16
  sanity); DP potential values on hand-worked graphs; `err_O` counts.
  ~600–800 LOC.

### W4 — the BARG trait layer (`magnetar-oa::barg`)

**Paper:** Def 11 (seBARG: Gen/P/V/Extract + the five properties; Rmk 12
T*-security; Rmk 13 vector-of-indices), Def 41 / §8.2 (UpBARG: Gen/TGen/
Update/Ver/Extract + incremental completeness, efficiency, index hiding,
somewhere AoK, last-statement collision resistance), Thm 14/43 (existence —
cited, not implemented; see §3).

- `trait SeBarg` parameterized by the local relation closure: matches Def 11
  syntax exactly, including the vector-index `Gen` convention of Rmk 13 (the
  r components of §5 and the (H+1)·r components of §8.4 are just r or (H+1)·r
  trait objects with independent trapdoors).
- `trait UpBarg` parameterized by `(R, Cons)`: `Gen / TGen(1^κ, j) /
  Update(hk, crs, v, π, z⁻, z, w) / Ver(hk, crs, v, π, z⁻) / Extract(hk, td,
  v, π)` — signatures byte-compatible with Def 41, including the
  `(z, ρ, w, z_prev, ρ_prev)` extraction shape and the `Cons(⊥, z)` first-
  statement convention.
- `ReferenceBarg` (both traits, the §3 design): real W1 forest roots as `v`,
  perfect index hiding, `batch-check` / `measure` modes, genuine extraction
  openings. Property tests encode each Def-11/Def-41 property as an executable
  experiment against the reference (e.g., last-statement collision resistance:
  produce two different `z̃ ≠ z⁻` with valid last-position openings from an
  accepting state — must be impossible without a BLAKE3 collision).
- **Tests:** incremental completeness chains to 2^10 records; extraction
  completeness for every j in a small batch; index-hiding distribution test
  (crs bytes independent of j — trivially true for the reference, kept as a
  contract test for future instantiations); tamper classes per property.
  ~500–700 LOC.

### W5 — §5: the privately-verifiable robust SNARG (`magnetar-oa::snarg_robust`)

**Paper:** §5.1 construction, §5.2 analysis (Thm 20; the potential argument is
proof, but Claim 23's DP is executable via W3).

- `Gen(1^λ, n, ε)`: `hk ← HT.Gen(1^κ)`; `r = ⌊λ/ε⌋`; sample
  `j_1..j_r ← [Q]`; `r` seBARG components `Gen(1^κ, n_loc, 1^m, k, T + j_i)`
  with `k = T + Q`; `crs = (hk, crs_seBARG)`, `td = (J, td_seBARG)`.
- `P^O(crs, x, w)`: run the accepting execution; transcript
  `τ = (C_0..C_T, D_1..D_Q)` with `D_j = (t_j, u_j, a_j)` (fixed-length
  encodings); `rt = HT.Hash(hk, τ)`; `z = (hk, x, rt)`; build the `T + Q`
  local witnesses of `R_loc` — the three witness kinds exactly as §5.1 step 4:
  ordinary step `w_i = (C_{i−1}, C_i, ρ)` with counter-preservation check;
  oracle step `w_i = (C_{i−1}, C_i, D_j, ρ)` with the `j = query(C_i) =
  query(C_{i−1}) + 1` linkage; table entry `w_{T+j} = (C_{t_j−1}, C_{t_j},
  D_j, ρ)` with the both-sides linkage. One seBARG proof per component;
  `π = (rt, π_seBARG)`.
- `V^O(crs, td, x, π)`: the public seBARG check; extract
  `(w_{T+j})_{j∈J}`; check `R_loc(z, T+j, w)` for each; check `a_j = O(u_j)`
  — exactly `r` oracle calls.
- `n_loc = n + |hk| + |rt| + log(T+Q)`, `m = poly(S, λ, log T, log(1/ε))`
  (§5.1 footnote 5) — both become concrete constants in the code and are
  reported by the bench.
- **Tests:** roundtrip on TinyTM workloads (several T/Q/ε shapes); tamper
  catalogue — (a) invalid ordinary transition, (b) counter break, (c) table
  entry disagreeing with the transcript, (d) *incorrect oracle answer at a
  hidden-sampled index* (the core case: an execution with few wrong answers
  must pass when none is sampled — verify the *statistical* behavior: with
  err = εQ the rejection probability matches the (1−ε)^r curve), (e) opening
  forgery, (f) extraction of a non-table index. The DP potential of W3
  cross-checks that rejected runs are rejected at a genuine potential gap.
- **Benches:** prover cost split (execution, transcript hashing — vs. the
  existing Merkle GB/s line, witness construction, BARG); verifier cost vs.
  r; proof size vs. (T, Q, ε). ~500–700 LOC.

### W6 — §6: the impossibility counterexample, executable (`magnetar-oa::attack`)

**Paper:** Thm 24, §6 proof (the majority machine, the pairwise-independent
row marking, the 1/(4ℓ) bound).

- The machine: `C^O(x) = Maj(O(x,1), …, O(x,N))`, N odd; a TinyTM program
  making the N non-adaptive queries.
- The attack generator, parameterized by a victim "SNARG-like" verifier
  interface `(crs, x, π) → accept` with `ℓ ≤ N/8` oracle queries:
  1. embed every candidate crs as `x_crs = crs ∥ 1 ∥ 0^{n−|crs|−1}`;
  2. run the honest prover under the all-ones oracle with fixed tape r;
  3. choose `k = log₂(2ℓ)`, a pairwise-independent `h(x) = Ax + b` over F₂;
  4. define `O_{h,r}` marking rows `h(x) = 0^k` (ones only at the verifier's
     queried positions `I_{crs,r}`);
  5. the cheating prover outputs `(x_crs, π_{crs,r})` — it never queries the
     real oracle.
- Empirical check: acceptance probability ≥ (1 − negl)/(4ℓ) over (h, r, crs);
  the marked row holds ≤ ℓ ≤ N/8 ones, so the false claim is ¼-robust — the
  attack *demonstrates* the lower bound, and doubles as the regression test
  that any future "public robust verifier" PR must fail to beat.
- Also encodes the two corollaries the repo cares about operationally: the
  same attack rules out public IVC with these parameters (the terminal proof
  is a SNARG), and it justifies the design split (signatures or private
  keys — no third option).
- **Tests:** the acceptance-probability measurement (seeded, tolerance-banded
  around 1/(4ℓ)); robustness verification of the marked oracle via the W3 DP.
  ~250–400 LOC.

### W7 — §7.1: the publicly-verifiable signed-oracle SNARG (`magnetar-oa::snarg_signed`)

**Paper:** §7.1 construction, §7.2 analysis (Thm 27; Rmk 28's two components).

- `Gen(1^λ, n, vk)`: `hk`; **two** independent components
  `Gen(1^κ, n_loc, 1^m, T, 1)` — extraction index 1, **trapdoors discarded**
  (this is what makes verification public); `crs = (vk, hk, crs¹, crs²)`.
- `P^{O_sig}(crs, x, w)`: run the accepting execution, collecting signatures
  per query; `τ = (C_0..C_T)`; `rt = HT.Hash(hk, τ)`; `z = (vk, hk, x, rt)`;
  local witnesses `w_i = (C_{i−1}, C_i, ρ_i, σ_i)` of `R_sig` — opening check;
  ordinary step: valid transition, `σ_i = ⊥`; oracle step: valid purported
  transition **and** `Sig.Verify(vk, (u_i, a_i), σ_i)`; `i = 1` initial, `i =
  T` accepting; two proofs; `π = (rt, π¹, π²)`.
- `V(crs, x, π)`: parse, `z`, and accept iff **both** seBARG checks pass. No
  oracle calls, no trapdoor, no extraction — extraction exists only in the
  soundness proof, which is why the somewhere-AoK property (not plain
  soundness) is the required primitive strength (§7.1's emphasized paragraph).
- **Tests:** roundtrip (TinyTM with the signed oracle; several T, Q); tamper
  catalogue — invalid transition; wrong answer *with a reused valid signature
  from a different query* (must reject: Sig binds (u,a) as a pair); wrong
  answer with a forged signature (rejects at the signature layer — the
  forgery-experiment case of §7.2); boundary tampers (fake initial config,
  non-accepting tail); single-component proof (must reject). Adaptive
  completeness smoke: x chosen after crs.
- **Benches:** proof size vs. T and |σ| (hash-sig vs. ed25519 feature);
  verifier time (2 BARG checks + 0 oracle calls — the headline "public,
  oracle-free" number). ~350–500 LOC.

### W8 — §7.3: the fully succinct deterministic variant (`magnetar-oa::snarg_short`)

**Paper:** §7.3 (Thm 31), Def 32 (used per-step), Def 35 (compressed
configurations), Rmk 34 (blank-tape init), Rmk 38 (hash-signed long
queries/answers).

- **Compressed configuration** `C̃`: machine state, head positions,
  oracle-call counter, the *full* query and answer tapes (ℓ cells each,
  ℓ = poly(λ, log T)), and the work tape only as its Merkle hash. Initial
  blank-tape hash computed implicitly (Rmk 34).
- `R_short` witnesses `w_i = (C̃_{i−1}, C̃_i, ρ_i, η_i)`:
  - ordinary step: `η_i = (b, ρ)` — the work-tape bit read plus opening,
    `HT.Verify(hk, rt^{i−1}_work, j, b, ρ)`; explicit fields follow the
    prescribed transition; on write, `HT.VerifyUpdate(hk, rt^{i−1}, rt^i, j,
    b, b′, ρ)`; on no-write, `rt^{i−1} = rt^i`;
  - oracle step: work-tape hash unchanged, explicit fields follow the
    prescribed transition, `Sig.Verify(vk, (u_i, a_i), σ_i)` (Rmk-38 mode:
    the signature covers the query/answer tape *roots*).
- Gen/P/V structurally identical to W7 (two components, index 1, trapdoors
  discarded), over `M_short` and the compressed transcript.
- **Tests:** roundtrip on deterministic TinyTM workloads with a work tape
  large enough that compression is meaningful (witness m independent of S —
  assert it); tamper catalogue — the Claim-36 cases as executable tests:
  (a) read-tape lie (wrong b with valid-looking ρ → opening collision),
  (b) write-tape lie (wrong new root → update-soundness violation), (c)
  oracle departure with signature (forgery), (d) explicit-field departure;
  blank-init tamper; Rmk-38 long-query mode roundtrip + tamper.
- **Benches:** proof size vs. log T at fixed S (the fully-succinct claim made
  concrete); prover cost vs. the W7 S-dependent variant at the same workload
  — the measured price/savings of compression. ~450–650 LOC.

### W9 — §8.3: the signed-oracle IVC (`magnetar-oa::ivc_signed`)

**Paper:** §8.1 Def 39 (definitions), §8.3 construction (Thm 45), Rmk 44
(hashing the input), the §8.3 tail (the fully succinct compressed variant).

- **Records and relations:** `D_i = (i, vk, hk, vx, C_{i−1}, C_i)` with
  `vx = HT.Hash(hk, x)`; `R_sig(D_i, η_i)` — ordinary: nondeterministic
  choice + input-bit opening against `vx`; oracle: the signature; `Cons(D,
  D′) ⇔ i′ = i+1 ∧ (vk′, hk′, v′_x) = (vk, hk, v_x) ∧ C′′ = C′`; first
  record `Cons(⊥, D_1)`: `i = 1` and initial configuration.
- `IVC.Gen(1^λ, n, T, vk)`: `hk ← HT.Gen(1^κ)`, `κ = poly(λ, S, log T)`;
  **two** independent `UpBARG.Gen` crs strings; `crs = (vk, hk, crs¹, crs²)`.
- `IVC.Update^{O_sig}(crs, x, t, C_t, π_t, w_{t+1})`: the t=0 initialization;
  the state check `HT.Size(v_t) = t ∧ D_t` well-formed; compute `C_{t+1}`
  (oracle step: `(a, σ) ← O_sig(u)`); build `D_{t+1}` and `η_{t+1}`; two
  `UpBARG.Update` calls; `v_{t+1} = HT.Append(hk, v_t, D_{t+1})` (assert both
  components agree — the construction's cross-check);
  `π_{t+1} = (v_{t+1}, Π¹, Π², D_{t+1})`.
- `IVC.V(crs, x, t, C, π_t)`: t=0 case; else the size/D_t checks and both
  `UpBARG.Ver` — **no oracle calls, no trapdoor**.
- **The compressed variant** (§8.3 tail): records over `C̃_i`; the prover
  maintains a *dynamic Merkle tree* of the work tape alongside the execution
  (implicit blank init, O(log T) authentication nodes touched per read/write,
  poly(λ, log T) per step — the auxiliary structure is explicitly *not* part
  of the proof); the updater verifies local openings and checks proposed
  roots via `HT.VerifyUpdate`; κ = λ for polynomially bounded T.
- **Tests:** long chains (t to 2^14 and a sampling beyond) with intermediate
  verification at every step (adaptive completeness in the executable sense);
  the tamper catalogue — Claim 46/47's cases: (a) ordinary-transition lie
  against the authenticated input (opening collision), (b) oracle answer lie
  with signature (forgery), (c) consistency break (skipped index, key swap,
  boundary mismatch), (d) last-statement lie (the last-statement collision
  case), (e) size lie, (f) two-component disagreement; chain-fork tests
  (update from a stale π); the compressed variant's full suite (the W8 cases
  re-run incrementally) + the "auxiliary tree eviction" test (drop the
  prover's dynamic tree, rebuild from a checkpoint, continue byte-identically
  — the streaming analogue of the existing `magnetar-ivc::streaming` test
  convention).
- **Benches:** **update latency per step** (ordinary vs. oracle step) — the
  headline IVC number, split into record build / witness build / forest
  append / UpBARG update / signature verify; proof size vs. t (flat — the
  point of IVC) and vs. S; verifier time vs. t (flat); the compressed variant
  vs. the full-config variant at matched workloads. ~600–800 LOC.

### W10 — §8.4: the privately-verifiable robust IVC (`magnetar-oa::ivc_robust`)

**Paper:** §8.4 (Thm 49), the ordinary-interval relation `R^x_ord`, the block
relation `R_block`, the dyadic-scale setup (Rmk 48), the update/verify
algorithms, Eq. (7)–(9).

- **OrdIVC** — the §8.3 scheme restricted to ordinary transitions, with the
  interval generalization of §8.4 footnote 19 (records carry the interval's
  start configuration A): proves `(A, B, ℓ) ∈ L^x_ord` — B reachable from A
  in exactly ℓ ordinary transitions on input x (input reads authenticated
  against `vx`).
- **Blocks:** `D_j = (A_{j−1}, B_j, A_j)`, `ℓ_j = time(B_j) − time(A_{j−1})`;
  statement `Z_j = (θ, D_j)`, `θ = (crs_ord, hk, vx)`; witness = the
  OrdIVC proof of the interval; `R_block`: well-formedness and bounds;
  `0 ≤ ℓ ≤ T`; `queries(A⁻) = queries(B)`; OrdIVC verification; B is an
  oracle-query configuration and A is B plus the oracle transition
  (`time(A) = time(B)+1`, `queries(A) = queries(B)+1`). **`R_block` never
  queries O.**
- **Setup** `IVC.Gen(1^λ, n, T, ε)`: `r = ⌈2λ/ε⌉`, `H = ⌈log₂ Q⌉`; for every
  scale `h ∈ {0..H}`: `K_h = 2^h`; sample `j_{h,1..r} ← [K_h]`; **(H+1)·r**
  `UpBARG.TGen` components; `crs` public, `td` holds the per-scale indices
  and trapdoors.
- **Maintained proof** `π_t = (v, (Π_{h,i}), D_last, η)`: v = forest root of
  `Z_1..Z_q`; per-scale-per-component UpBARG proofs; `D_last = D_q`; `η` =
  OrdIVC proof of the trailing ordinary interval `(A_q, C_t, time(C_t) −
  time(A_q))`.
- **Update:** ordinary step → only `η` updates (the forest and all Π
  untouched — the cheap path); oracle step → close the block (η *is* the
  witness), **(H+1)·r** UpBARG updates, reset `η` to the zero-length interval
  at `A_{q+1}`.
- **Verify:** the public checks 1–11 (bounds, counters, `HT.Size(v) = q`,
  `D_last` shape, all UpBARG.Ver at every scale, trailing OrdIVC), then the
  **private** checks 12–15 at scale `h* = ⌈log₂ q⌉` only: for each component
  i, skip if `j > q`; extract `(Ẑ, ρ, π̂, Ẑ_prev, ρ_prev)`; alignment
  (`queries(U) = queries(B) = j−1`, `queries(A) = j`); openings at j and
  j−1; `R_block`; `Cons`; `j = q ⇒ Ẑ = Z_last`; finally query the real
  oracle: `a = O(u)?` — at most r oracle calls.
- **Tests:** multi-block workloads with several oracle-call counts *not* at
  powers of two (to exercise the scale selection `h* = ⌈log₂ q⌉` and the
  `j > q` skips); the tamper catalogue — wrong answer at a hidden-sampled
  block (statistical rejection curve (1 − ε/2)^r per Eq. 8, banded);
  interval lie (OrdIVC tamper inherited); block-boundary lie (Cons break);
  scale-confusion lie (a proof whose q-claim and forest size disagree);
  trailing-interval lie; the DP potential cross-check on small machines;
  update-from-stale-proof.
- **Benches:** ordinary-step update latency (should be ≈ one OrdIVC step —
  the W9 number) vs. oracle-step update latency (the (H+1)·r cost, measured
  vs. Q and ε); proof size vs. (Q, ε) — the poly(λ, S, log T, 1/ε) bound
  made concrete; verifier cost and the oracle-call count vs. (λ, ε) —
  empirically O(λ/ε); the setup cost itself ((H+1)·r TGen calls — honest
  number, it is large). ~700–900 LOC.

### W11 — ERC-20 signed-state-oracle integration + the benchmark line

- The application wiring: the ERC-20 pooled workload (`magnetar-cli::
  magnetar-bench-erc20-pool`) gains an oracle-aided mode — balance reads and
  the block's state accesses issued as signed-oracle queries (the external
  state-DB signs each (query, answer) pair it serves; Rmk-38 mode for
  account-sized answers), the transfers executed by the RV64I guest via the
  W3 ECALL, the whole block incrementally certified by the W9 IVC.
- `magnetar-oa::bin/oracle-bench` — the benchmark harness: ladders over (T,
  Q, S, ε, λ), reporting per-step update latency, proof sizes, verifier time,
  oracle-call counts, and the primitive micro-costs (forest append, sig
  verify, DP potential on the small machines) — following the
  `docs/BENCHMARKS.md` format with a paper-comparison table (the paper gives
  asymptotics; the table quotes them alongside the measured constants).
- CI-scale smoke + the full ladder in the release bench profile, as with the
  existing lines.

### W12 — documentation and honest accounting

- `docs/STATUS.md`: the new "Oracle-aided line" section in the
  implemented/not-implemented format, with the §3 tier split stated in plain
  words: *the constructions, hash layer, and signature layer are real; the
  seBARG/UpBARG layer is a reference stand-in that is sound-but-not-succinct
  in `batch-check` mode and not sound in `measure` mode; no LWE lattice code
  exists in this repo*.
- `docs/SECURITY.md` / threat-model notes for the signed-oracle line (the
  signing key is the trust root; EUF-CMA under O-access; the verifier's
  oracle-freeness claim).
- README architecture-table row for `magnetar-oa` + this file linked.

---

## 5. Phasing, dependencies, sizes

```
Phase A (foundations):     W1 (updatable hash)   W2 (signatures)   W4 (BARG traits + reference)
Phase B (model + barrier): W3 (machine model)   W6 (§6 attack)            [needs W1 for nothing; W6 needs W3]
Phase C (SNARGs):          W5 (§5)   W7 (§7.1)   W8 (§7.3)                [need W1–W4]
Phase D (IVC):             W9 (§8.3)  →  W10 (§8.4)                        [W10 needs W9's OrdIVC]
Phase E (integration):     W11 (ERC-20 + benches)  W12 (docs)             [needs W9; W10 optional for the
                                                                            robust/unsigned ERC-20 variant]
```

Estimated new code: ≈ 4.5–6.5 K LOC of library + ≈ 1.5–2 K LOC of tests +
≈ 0.5 K LOC of benches/CLI. New crate `magnetar-oa`; `magnetar-hash` and
`magnetar-cpu` gain one module / one ECALL class respectively; no changes to
the existing proof-system crates.

Suggested commit cadence (matching the repo's release-note style): one commit
series per work package, each closing with its tamper suite green; the §6
attack generator lands early (Phase B) precisely because it is small,
self-contained, and sets the design constraint the rest of the plan obeys.

---

## 6. Testing strategy — the tamper-class catalogue (consolidated)

Per the repo convention (roundtrip + named tamper classes + byte-identity
where two paths exist):

1. **Hash layer (W1):** opening forgery; update-soundness experiment (Def 32)
   — wrong-new-root / wrong-old-bit / path forgery; append-completeness
   identity; prefix-root stability.
2. **Signature layer (W2):** forgery; message-pair substitution ((u,a′) under
   σ for (u,a)); OTS reuse; Rmk-38 hash-pair substitution.
3. **BARG layer (W4):** each Def-11/Def-41 property as an executable
   experiment; last-statement collision resistance.
4. **§5 (W5):** transition lie; counter break; table/transcript
   disagreement; sampled-index wrong answer (statistical band on the
   (1−ε)^{⌊λ/ε⌋} rejection curve); opening forgery.
5. **§6 (W6):** the attack's acceptance probability ≥ 1/(4ℓ) band; the marked
   oracle's ¼-robustness via the DP.
6. **§7 (W7/W8):** signature-forgery departure; read/write tape lies (the
   Claim-36/37 departure classes); boundary lies; single-component proof.
7. **§8 (W9/W10):** consistency breaks (index skip, key swap, boundary
   mismatch); last-statement lie; stale-proof update; chain fork; scale
   confusion; trailing-interval lie; the Eq.-8 rejection band; the compressed
   variant's auxiliary-tree eviction/rebuild byte-identity.
8. **Adaptive completeness smoke** everywhere: instance chosen after crs
   (the definitions' adaptive flavor, executable as: generate crs, then pick
   x, then prove — must verify).

## 7. What is explicitly NOT implemented — the honest caveats

1. **The lattice layer.** No LWE sampling, no Regev encryption, no lossy
   trapdoors; Thm 14 and Thm 43 are *cited*, and the trait boundary is the
   deliverable. The reference BARG is sound-but-not-succinct (`batch-check`)
   or a measuring instrument (`measure`), and is labeled as such in code,
   docs, and CLI output.
2. **Asymptotic security parameters.** The paper's `κ = poly(λ, S, log T,
   log(1/ε))` with sub-exponential LWE hardness (T*-security, Rmk 5/12/42)
   has no concrete instantiation here; we run κ ∈ {128, 256} and say so.
3. **The soundness theorems.** Thms 20/27/31/45/49 are proofs about
   polynomial-size adversaries and hybrid telescoping; what is executable is
   their *component* structure (extraction validity, consistency propagation,
   potential DP, the rejection curves) and the tamper suites — not the
   proofs themselves.
4. **Thm 24's statement** is a lower bound; only its counterexample machine
   and attack are executable (W6).
5. **The `measure` mode is not a proof system.** Any number produced under it
   is a performance measurement of the construction skeleton, never a
   security claim.

## 8. Relations to the existing lines

- **tree-IVC (`magnetar-ivc`):** unchanged. The paper's IVC is per-step and
  oracle-aware; tree-IVC is per-segment for one internal computation. Future
  bridge (out of scope): use §8.3 transition records as the boundary
  certificates between tree-IVC segments so that a segment's *external reads*
  are signed-oracle-certified.
- **Mikros / BinarySpartan:** the optional future BARG swap-in (§3) would
  prove the batch statement ("records hash to v, all in L, all consistent")
  with BinarySpartan over the packed R1CS — the natural successor to the
  reference stand-in, and the point where this line meets the Mikros stack.
- **ERC-20 pool (W11):** the signed-state-oracle mode is the "stateless
  verifier" story — a public verifier that checks a block proof without
  re-downloading state, because the state reads carry signatures that the
  IVC's local relation checked.
- **The SP1/guest-wrap and witness-compression line** from the earlier
  research sessions is orthogonal to this plan and remains tracked separately.

## 9. Open decisions (to settle at implementation kickoff)

1. **Signature default:** in-repo hash-based (recommended — keeps the
   standard-hashes-only philosophy, post-quantum, zero new dependencies)
   vs. `ed25519-dalek` as default with hash-based behind a feature. Plan
   assumes hash-based default + ed25519 feature for comparison benches.
2. **`magnetar-oa` crate vs. splitting into `magnetar-oa` + extending
   `magnetar-hash` only.** Plan assumes one new crate + the W1 module inside
   `magnetar-hash` (the forest is a hash primitive; the rest is not).
3. **RV64I oracle ECALL encoding** (MMIO page vs. dedicated instruction vs.
   ECALL subcode) — decided in W3 against the existing ECALL machinery of
   `magnetar-cpu`.
4. **Bench ladder extents** for CI vs. release profiles (the §8.4 setup cost
   at (H+1)·r components is large; the release ladder should include at
   least one ε < 1/4 point to make the 1/ε dependence visible).
5. **Whether Phase E includes the robust/unsigned ERC-20 variant** (W10
   integration) or ships the signed line only.
