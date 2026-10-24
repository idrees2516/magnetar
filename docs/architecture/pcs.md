# The polynomial commitment scheme and the dual-rate schedule

*Magnetar-pcs — the hash-based multilinear PCS (paper §3.6, Definition 6,
instantiated in Ligerito's role), plus ring switching (§5.2, [13]).*

## The code

For a table `T: {0,1}^n -> K` the codeword is the **affine-substitution
extension**

```text
C(x, eps) = T_hat(x + L(eps))   over {0,1}^n x {0,1}^k,
L(eps) = sum_j eps_j gamma_j,   gamma_j in K^n hash-derived.
```

The commitment is the BLAKE3 Merkle root of the codeword; the `(x, 0)`
slice is exactly the original table (the code is *systematic*). Encoding
is the delta-shifted Mobius transform in `O(n . 2^{n+k})` (`code.rs`):
values to monomial coefficients by the upward zeta (an involution mod 2),
a downward weighted pass per variable shifts by delta, and the same zeta
writes the cube values — per epsilon block.

**Distance.** Because the evaluation points `x + L(eps)` are generic
K-points (Schwartz–Zippel over the seed-derived gammas), any nonzero
codeword difference — a nonzero multilinear — vanishes on at most an
`O(n/2^128)` fraction of the `2^{n+k}` positions: relative distance is
approximately 1 *for every k >= 1*. This single fact drives the dual-rate
schedule below and is the honest difference from Reed–Solomon-based FRI,
where the distance is `1 - rho` and the rate genuinely moves the query
count.

## The evaluation argument (`eval.rs`)

For a claim `T_hat(r) = v`:

1. **Sum-check** over the n table variables with the public `eq(r, .)`
   factor: binds a random `r*` and ends at the terminal `T_hat(r*)`.
2. **Fold chain**: the codeword folded at the *same* challenges, one Merkle
   tree per level. Folding bit `i` at challenge `c` *is* the substitution
   `y_i -> y_i + c` on the multilinear, so the family is closed under
   folding, and after n folds the tail is `D_n(eps) = T_hat(L(eps) + r*)`
   — the `eps = 0` entry is the claimed `T_hat(r*)`.
3. **Correlated queries**: q random level-0 positions, each walked up the
   chain with Merkle-authenticated openings; every fold relation is
   checked on the way.
4. **Tail**: all `2^k` final entries revealed with paths; entry 0 must
   equal the sum-check's terminal.

Soundness follows the FRI/Ligerito analysis: the fold family is closed,
the level-0 tree is the commitment, correlated walks catch fold
inconsistencies, and the code's distance rules out concentrated
codeword-level deviations, with the unique-decoding-radius proximity-gap
argument (Ben-Sasson et al., TR25-169) covering the general committed
string — the standard the paper itself adopts (§6.1).

## Ring switching (`ring.rs`)

The Mosaic commitment is over **packed** words (128 witness bits per K
element), but the auxiliary arguments produce claims about *bit* tables.
The ring switch reduces a claim on the bit vector `w(u, b)` to a claim on
the packed word: define the c-folded table
`C_w(u) = phi_c(P(u))` with `c = eq(r_B, .)`, where `phi_c` is the
bit-extraction linear map — and prove the fold chain of `C_w`'s codeword,
with the level-0 `(x, eps=0)` slice anchored to the *main* commitment by
exact `phi_c` bit-extraction openings at random positions. The prover's
work is one extra fold chain (linear); the verifier's is the anchors.

## The dual-rate schedule (roadmap H1/H2, measured by `cost-model.md`)

`PcsParams` now names two profiles:

* **`PcsParams::chunk()`** — `k = 1` (rate 1/2). Internal, per-chunk
  artifacts: the codeword is `2^{n+1}` entries (half the encode/commit
  work of the default) and every Merkle path is one level shorter.
  Chunk proofs never leave the prover's trust domain, so they tune
  purely for prover speed.
* **`PcsParams::wrap()`** — the conservative default `k = 2` (rate 1/4),
  used for the *public* artifact: one more extension variable buys
  distance margin in the proximity argument at the cost of one level of
  path per query — a few percent of the artifact.

Because the affine-substitution code's distance is ~1 at every `k >= 1`,
the query schedule stays `q = 40` (the 120-bit constant of the existing
analysis) on both profiles; the rate here tunes *work and path depth*,
not query count. The roadmap's *low-rate wrap with fewer queries* is a
Reed–Solomon-coupled statement: it belongs to the recursion milestone,
where the wrap instance is the small verifier-guest witness and a lower
rate buys queries against an RS proximity analysis. Until that milestone
lands, the honest dual-rate lever is the chunk/wrap split above — and the
CLI prints both profiles' measured sizes.

## Production notes

* The `gamma` derivation is domain-separated and position-bound
  (`blake3_domain`), so two instances with different `(n, k)` never share
  substitution points.
* Every Merkle leaf is domain-separated per level (`.../level/i`), which
  is what lets the correlated walk re-derive leaf digests on the verifier
  side without transmitting indices.
* All field arithmetic is constant-time; the fold chain never branches
  on secret data (there are no secrets — the prover is honest-by-default
  and the soundness argument covers deviations).
