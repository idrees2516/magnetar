# The measured cost model

*Magnetar-zkvm/stats.rs + the instrumented CLI — the roadmap's build
order item #1: "measure the committed-bit inventory and the
query/path/digest split of the proof directly, so every subsequent lever
lands on measured constants."*

## What is measured (not estimated)

`proof_stats` walks a finished proof — the actual wire structure, not a
model of it — and reports:

* **The committed-bit inventory**: per Mosaic component (the one-hot
  address components dominate: 2 x 8-bit-address x T each for RAM,
  6-bit x T x 3 for registers), plus the virtual bits the protocol
  never commits (the Virtual-Spartan saving, ~22% at fib scale).
* **The PCS geometry**: packed-cube variables, rate, codeword length,
  Merkle depth, query count, digest size.
* **The byte breakdown** per protocol component: field elements at 16
  bytes, digests at 32, per segment and per shared argument.

At the paper's production scale the breakdown concentrates exactly where
the paper says it does: the single evaluation argument (Merkle paths)
dominates — ~84% of proof bytes — with the entire interactive
machinery (all the sum-check transcripts of steps 3–6) at ~16% and one
commitment root. At CI scale the ratio is gentler (62% at K=4) because
the sum-check transcripts do not shrink with the instance while the
paths do; both numbers are printed, and neither is extrapolated.

## The aggregation ledger

For a claim-aggregated proof the report adds the three-line ledger the
Horizon-2 program asked the cost model to produce:

```text
aggregation ledger: per-segment sum-checks 126144 x 4 = 504576 bytes;
                     ONE eval argument 203040 bytes (62% of the proof)
standalone counterfactual (4 independent chunk-rate proofs, analytic): 1138048 bytes
aggregation saving: 3.46x
```

* The **per-segment sum-check bytes** are measured from the proof.
* The **ONE evaluation argument** bytes are measured from the proof.
* The **standalone counterfactual** is the analytic byte model of K
  independent chunk-rate proofs: K times the measured per-segment share
  plus K times `estimate_pcs_bytes(n_chunk, k, q)` — the deterministic
  geometry of one evaluation argument (level roots, correlated query
  paths, tail, anchors) at the chunk rate. It equals what measuring K
  such proofs would print, without paying to build them.

The saving grows with K because the counterfactual's dominant term is
K evaluation arguments while the aggregate's is one; at production
scale, where the evaluation argument is ~84% of a chunk proof, the
saving approaches K x 0.84 + 0.16 bytes-of-standalone against
0.84 + K x 0.16 bytes-of-aggregate.

## The rate schedule

`PcsParams::chunk()` (rate 1/2) and `PcsParams::wrap()` (rate 1/4) are
measured by proving the same workload at both profiles
(`magnetar agg N chunk|wrap`): the chunk profile halves the codeword
(the commit work) and shortens each path by one level; the wrap profile
adds the conservative distance margin for the public artifact. The
honest analysis — our affine-subspace code's distance is ~1 at every
`k >= 1`, so the rate moves work and paths, not the query count — is
documented in `pcs.md` and visible in the printed geometry.

## What the model does not model

* **Prover time** is not part of `proof_stats` (the CLI prints it
  inline); the roadmap's throughput work (GPU kernels, the sum-check
  instrument ledger) is measured against wall-clock benchmarks, not
  byte counts.
* **The wrap's future RS-coupled saving** (fewer queries at lower rate)
  belongs to the recursion milestone and is *not* claimed here — the
  printed schedule is the honest present-day one.
