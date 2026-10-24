# Binary fields, packing, and the `eq` polynomial

*Magnetar-field — the arithmetic substrate of every layer above it.*

## The tower

Magnetar's base field is **B = F2** (one bit) and its challenge field is
**K = GF(2^128)**, the GCM polynomial
`x^128 + x^7 + x^2 + x + 1`. The choice is not stylistic:

* **Verifier challenges and evaluation claims must live in an extension.**
  A sum-check over a Boolean cube with challenges drawn from B collapses
  (a degree-1 polynomial over F2 is determined by two points, and the
  round messages carry no entropy), so the paper draws every challenge in
  K while keeping the *witness* in B. This is the entire reason the
  "bit-witness, field-challenge" discipline exists, and it is why the
  circuit is a *dual* relation: one matrix set over B, one over K.
* **GF(2^128) is the fastest large field on commodity hardware.** The
  implementation (`gf128.rs`) carries elements as a 2x64-bit pair with a
  PCLMULQDQ inline path and a portable Karatsuba fallback; both are
  constant-time because carry-free arithmetic has no data-dependent
  branches. Addition is XOR, subtraction is addition, and the
  per-operation cost is a single carry-less multiply plus a reduction.

## Multilinear extensions over the cube

Every table in the system is a multilinear `f: {0,1}^n -> K`, indexed
little-endian (bit 0 = the LSB of the index). Its unique multilinear
extension is evaluated off-cube by the standard product form
(`multilinear.rs`), and the **eq** tensor
`eq(r, x) = prod_i (1 + r_i + x_i)` (characteristic two!) is the
workhorse of every claim-binding argument in the paper: a claim
`f_hat(r) = v` is interchangeable with the sum `sum_x eq(r, x) f(x) = v`,
which is exactly what a sum-check proves. `EqBits::build(r)` materializes
the 2^n eq values in one downward sweep; every "bind this claim at that
point" idiom in the protocol is an eq-fold in disguise.

## Packing: B-witnesses into K-elements

The witness is *bit-valued*; the commitment layer commits *K-valued*
words. `pack.rs` packs 128 witness bits into one K element by the
`beta`-basis: `P(u) = sum_b beta_b w(u, b)` with `beta_b = 2^b`, so
`P(u)`'s bit b is exactly `w(u, b)`. This is the representation MosaicPCS
commits to, and the ring switch (see `pcs.md`) is the argument that
moves a claim on the packed word back down to claims on the individual
bits. The per-guest-cycle cost economics the paper cares about flow from
this packing: one 16-byte field element carries 128 witness bits, and
Merkle-path amortization is measured on packed elements, not raw bits.

## Invariants other layers rely on

1. **Char-2 linearity**: `1 + 1 = 0`, so `a + a = 0`, so subtraction and
   addition coincide. The sum-check engine exploits this for the
   replicated-factor terminal rule (see `sumcheck.md`), and the protocol
   exploits it for claim composition (the stage-select's
   `Q_lo = ROM + STAGE` split composes *additively*).
2. **The involution transform**: over F2, the values/monomial-coefficient
   zeta transform is its own inverse (`M^2 = I`), which the code encoder
   uses to move between evaluation and coefficient forms with the same
   sweep in both directions (`pcs.md`).
3. **Frobenius is squaring**: `r -> r^2` is F2-linear, which the memory
   arguments' refined-point resampling (`r' = r^3 / c` with
   `c = 1 + r + r^2`) relies on to keep every resampled point inside K
   while breaking the algebraic dependence the soundness argument forbids.

## Where to look

* `magnetar-field/src/gf128.rs` — the field, inv, carry-less multiply.
* `magnetar-field/src/multilinear.rs` — MLEs.
* `magnetar-field/src/eq.rs` — the eq tensor and EqBits.
* `magnetar-field/src/pack.rs` — pack/unpack, the beta basis.
