# Architecture

Magnetar is a transparent, post-quantum zkVM for RV64I, built from standard
hashes (BLAKE3 for Merkle commitments, SHA3-256 for Fiat–Shamir transcripts)
per eprint 2026-2423. This directory is the engineering companion to the
paper: one deep-dive per component, in the order data flows through the
prover.

```text
                 ┌────────────────────────────────────────────────┐
                 │                magnetar-zkvm                   │
                 │   protocol (Construction 3) · Mosaic · stats   │
                 └───────────────┬────────────────────────────────┘
        ┌────────────┬───────────┼───────────────┬───────────────┐
   ┌────▼─────┐ ┌────▼─────┐ ┌───▼──────────┐ ┌───▼───────────┐ ┌──▼────────┐
   │ spartan  │ │  memory  │ │    router    │ │      cpu      │ │  sumcheck │
   │ (VS, R1CS)│ │(Twist,   │ │ (R_sh/R_st, │ │ (RV64I decode,│ │  engine   │
   │          │ │ Shout, Ψ)│ │  stage-sel) │ │  micro-expand)│ │           │
   └────┬─────┘ └────┬─────┘ └───┬──────────┘ └───┬───────────┘ └───┬───────┘
        │            │           │                │               │
   ┌────▼────────────▼───────────▼────────────────▼───────────────▼───────┐
   │                     magnetar-pcs · magnetar-field                    │
   │      (affine-subspace code, fold chain, ring switch)  (GF(2^128))     │
   └──────────────────────────────────┬───────────────────────────────────┘
                                      │
                              magnetar-hash (BLAKE3 / SHA3-256)
```

The layering rule (the [LambdaVM][lambda] school): each crate depends only
on the layer below it, exposes a *sum-check-shaped* interface (a claim in,
a set of smaller claims out), and can be replaced by any other construction
that satisfies the same claim contract. The protocol crate is a pure
orchestrator — every cryptographic decision lives one layer down.

| Document | Covers | Paper |
|---|---|---|
| [binary-fields.md](binary-fields.md) | the tower B = F2 -> K = F2^128, packing, `eq` | §3 |
| [sumcheck.md](sumcheck.md) | the engine, bind orders, public factors, the sparse prover, the char-2 idle-variable rule | §3.2, §5 |
| [pcs.md](pcs.md) | the affine-subspace code, the fold chain, correlated queries, ring switching, the dual-rate schedule | §3.6, §5.2, §6.1 |
| [mosaic.md](mosaic.md) | aligned-subcube layout, the lift, alpha-batching, the segment-index dimension (claim aggregation) | §5.2 |
| [spartan.md](spartan.md) | SIMD R1CS, Virtual Spartan (Construction 2), Equation (1) | §4 |
| [memory.md](memory.md) | BinaryTwist, BinaryShout, the Psi-sum-checks, the bus discharge | §5.1, §5.3, §5.5 |
| [router.md](router.md) | the routing identity, R_sh / R_st, the register-shift stage-select extension | §5.4 |
| [cpu.md](cpu.md) | RV64I, micro-expansion, the trash-cell stage chain, the dual relation, the zero-extension semantics | §2, §4.2 |
| [protocol.md](protocol.md) | Construction 3 end to end, transcript discipline, multi-segment aggregation | §5.5, §6.2 |
| [cost-model.md](cost-model.md) | the measured ledger: committed bits, byte breakdown, amortization | §6, roadmap §3 |

[lambda]: https://github.com/lambdaclass/lambda_vm

## Reading order

If you are reviewing soundness: `binary-fields` -> `sumcheck` -> `pcs` ->
`mosaic`, then `spartan` -> `memory` -> `router`, and finally `protocol` for
the composition. If you are adding a feature: start at `protocol.md`, find
the step your feature touches, and follow the claim chain downward. If you
are benchmarking: `cost-model.md` is the map of what the instrumented CLI
prints.
