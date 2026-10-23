# MosaicPCS: one commitment for everything, and the index dimension

*Magnetar-zkvm/mosaic.rs — the packing layer (paper §5.2).*

## The problem it solves

By the end of the auxiliary arguments, the verifier holds evaluation
claims about dozens of bit-valued polynomials of wildly different sizes:
the direct witness segments, the one-hot address components of two
BinaryTwist instances and one BinaryShout, the register and RAM
increments. Naively each needs its own commitment and its own evaluation
argument, and with a hash-based PCS *every* evaluation argument carries
Merkle paths — the byte bill multiplies by the number of components.

## The aligned-subcube layout

Every component has power-of-two size, so `MosaicLayout` places them on
disjoint, **aligned subcubes** of one larger cube `{0,1}^ell`
(declared order, offsets rounded up to each component's size). The
committed polynomial `G` agrees with component `p_i` on its subcube and
is zero outside the union — and `commit_mosaic` never materializes the
zero padding as a cost: the packing sweep visits each subcube once.

The point of *aligned* is the **lift** (paper Equation 4): a claim
`p_hat_i(r) = v` becomes `G_hat(r || s_i) = v` by appending the fixed
Boolean suffix `s_i` that selects the subcube. No adapter sum-check, no
height-balancing (the JaggedPCS alternative the paper explicitly
compares against in Remark 1). The verifier appends coordinates; nothing
else happens.

## The discharge

One sum-check over the ell variables with the public
`SumEqFactor` (the alpha-batched eq combination of all lifted claims)
against the dense G table — binding the top 7 variables first
(`HighBitsFirst { high: 7 }`), so the terminal point splits into a
`(block, bit)` pair: the block coordinate lands on the packed word
`P(u)`, and the bit coordinate selects which packed bit the claim is
about. That terminal is exactly the input of the ring switch (`pcs.md`),
which anchors it to the single commitment. One commitment, one batching
sum-check, one evaluation argument — the "Mosaic" is the layout, not a
new cryptographic primitive.

## The segment-index dimension: claim aggregation (roadmap H2)

The aggregation layer adds one observation: *the layout is already the
right shape to share a commitment across proofs*. For `K = 2^idx_bits`
segments, every component's subcube gains the index dimension:

```text
component vars: lv + n            ->  lv + n + idx_bits
component table: 2^(lv+n) bits    ->  K copies concatenated,
                                      index = local | (s << local_vars)
claim lift:       point || suffix ->  (point || s-bits) || suffix
```

Each segment's claims are produced by that segment's own steps 3–6
against the shared commitment; `lift_claim_with_segment` appends the
segment's Boolean index bits; and the *same* single Mosaic discharge
covers every segment's claims — the alpha-batching mixes claims across
segments exactly as it mixes claims across components.

**What this buys**: the evaluation argument — the byte-dominant part of
every hash-based proof (62% at the test scale, ~84% at the paper's
production scale per `cost-model.md`) — is paid **once**, independent of
K. Only the small per-segment sum-check transcripts scale with K. The
measured ledger at K=4 prints a 3.46x saving against the standalone
counterfactual (four independent proofs), and the saving grows with K
because the counterfactual's dominant term is K evaluation arguments
while the aggregate's is one.

**What this does not yet buy**: cross-segment *state* continuity. The
segments here are independent `(input, output)` pairs of one program —
the rollup/batch flavor. Chaining segments of one long execution
(segment i+1's initial RAM = segment i's final RAM) is the tree-IVC
milestone on the roadmap: the instruments (this layout, the dual-rate
profiles, and the deferred single discharge) are exactly the pieces the
roadmap says that milestone reuses, which is why it was built in this
order.

## Production notes

* `ell >= 7` is asserted: the packing sweep needs 7 bit-coordinates.
* The component list is built once in `keygen` and shared by prover and
  verifier — the layout is a *transparent* function of the program.
* The multi-segment transcript stays strictly sequential (coordinator
  order): shards can compute their segment bodies in parallel, but the
  Fiat–Shamir challenges of segment i+1 depend on segment i's messages.
  The roadmap's sharded-prover section documents the coordinator-star
  topology this enables without ever invoking anything stronger than
  standard hashes.
