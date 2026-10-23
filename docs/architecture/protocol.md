# The protocol: Construction 3 end to end, and aggregation

*Magnetar-zkvm/protocol.rs — paper §5.5, plus the claim-aggregation layer
of roadmap Horizon 2.*

## The online phase

**KeyGen** is transparent: from the public program it derives the
expanded ROM, the dual R1CS, the witness-segment layout, the routing
tensors, the Mosaic layout (with the optional segment-index dimension),
and the ROM tables (the bus, the four selector one-hot columns). No
trusted setup exists anywhere in the system.

The transcript is initialized with the domain tag, the program words,
and every segment's public I/O — then, in strict order:

1. **Execute** — the interpreter runs the expanded program per segment;
   `execute_padded` fills the witness cube exactly; the output binding
   is asserted before proving.
2. **Commit** — the concatenated component tables (all segments) pack
   into ONE Mosaic commitment; the root is absorbed. This is the
   protocol's only commitment.
3. **Virtual Spartan** (per segment, lockstep over B and K, the shared
   `r_inst`) — produces the segment claims and Equation (1)'s split;
   the virtual claims are *forwarded*, the direct ones held for the
   Mosaic.
4. **Reduce the virtual claims**:
   * the register BinaryTwist (+ the three register-index Psi
     sum-checks, each checked against its bus-slice claim);
   * the RAM BinaryTwist (+ two address Psi sum-checks, each checked
     against the `ram-addr` segment claim — exact equality);
   * the verifier-sampled RAM Inc point and claim (the store router's
     output claim);
   * the shift router (the stage-conditional amount selector) and the
     store router (the X split into the two copy segments).
5. **The bytecode Shout** — ONE joint sum-check authenticating the
   fetched bus, the register-index slices, and the pc Psi against the
   committed fetch component; then the stage-selector sum-check and the
   four selector checks (`sel_stage`, `sel_lo` with the composed claim
   `q_lo + q_stage`, `sel_hi`, `sel_st`).
6. **The Mosaic discharge** — every segment's claims, lifted through
   the index dimension, alpha-batched into ONE sum-check whose terminal
   the ring switch proves against the ONE commitment.

**Verify** mirrors this exactly: every helper returns claims rather
than booleans, so the verifier's state is the same claim set the
prover's is, checked in the same transcript order.

## Transcript discipline

The Fiat–Shamir transcript is the protocol's backbone, and the rule is:
**the prover and the verifier perform the same absorptions in the same
order, with the same values** — the only prover-side extras are the
dense tables (which never enter the transcript except through round
messages and claimed values). Every bug class this codebase has hit in
the wiring (the lockstep double-absorb, the claim absorb-order desync)
was a violation of this rule; `CONTRIBUTING.md` pins the pattern, and
the tamper tests enforce it structurally: any desynchronization fails
*every* proof, not just crafted ones.

## Claim aggregation (`prove_multi` / `verify_multi`)

The multi-segment shape is the single-segment shape with
`idx_bits = log2(K)`:

* one shared commitment over the concatenated cube;
* K independent segment bodies — each with its own public input, its own
  challenges derived from the shared transcript, its own claims;
* ONE Mosaic discharge over all lifted claims.

The single-segment `prove`/`verify` are the K=1 wrappers (identical
transcripts). What scales with K: the per-segment sum-checks (~38% of
bytes at K=4 test scale, shrinking as the eval argument dominates).
What does not: the commitment, the batching sum-check's per-claim cost,
and — the whole point — the evaluation argument. The measured ledger
lives in `cost-model.md`.

**Continuity is deliberately not claimed.** The segments are independent
`(input, output)` pairs — the batch/rollup flavor. Cross-segment state
chaining is the tree-IVC milestone (see `ROADMAP.md`): the layout, the
dual-rate profiles, and this discharge are the pieces it composes.

## The dual-rate deployment

The public aggregated artifact commits at the wrap profile; standalone
per-chunk proving uses the chunk profile internally (halved commit work,
shorter paths — internal artifacts never leave the prover). The
`agg [segments] [chunk|wrap]` CLI mode proves at either rate so the cost
model's rate schedule is a printed number, not a design claim.

## Soundness ledger

The composition's soundness budget (docs/security.md carries the full
accounting): ~200 sum-check rounds over K at <= 2^-120 per round
(Schwartz–Zippel on multilinear identities at random K-points) plus the
fold-chain/Merkle layer at ~2^-128, meeting the paper's 120-bit target
at the unique-decoding standard. Every step-3–6 argument contributes
its claims as *verified values*; the Mosaic discharge is the only place
claims convert to commitment openings, and it does so once for the whole
proof — which is exactly why aggregating before it is the byte-optimal
shape the roadmap prescribed.
