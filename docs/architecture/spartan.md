# Spartan: SIMD R1CS and the virtual-witness construction

*Magnetar-spartan — paper §4 (Constructions 1 and 2).*

## SIMD R1CS

A Spartan-style R1CS relation is `A z . B z = C z` with z the joint
`(1, x, W)` vector. Magnetar's twist is that the relation is **dual**:
the same constraint set is expressed over B (the bit-matrices
`a_bin/b_bin/c_bin`) *and* over K (`a_fld/b_fld/c_fld`), because
witness bits are naturally B-valued but the R1CS identity must also
hold when the packed, K-valued view carries the same bits (the packing
is a ring homomorphism on the Boolean cube). The sparse matrices are
built once in the CPU crate (`circuit.rs`) and applied per cycle by a
block-diagonal `apply_simd` — the SIMD in the name is the per-cycle
replication, so the outer sum-check's instance dimension *is* the cycle
index.

## Virtual Spartan (Construction 2)

The paper's key economy: **witness segments are claimed, not committed.**
The joint W splits into virtual segments (register values, loaded words,
shift outputs, the PC, the fetch bus — owned by the memory arguments)
and direct segments (copies, flags, addresses — committed by Mosaic).
Virtual Spartan proves the R1CS satisfaction with only *evaluation
claims* on the virtual segments, which the memory arguments then
discharge against the one commitment:

1. **Two lockstep outer sum-checks** (one per relation, over B and K)
   over `(instance || constraint-row)`, with the instance rounds sharing
   challenges — the paper's shared `r_inst` — via `HighBitsFirst`
   binding the instance coordinates first and a `round_external`
   transcript pattern that absorbs both messages before sampling one
   challenge.
2. **Six terminals** alpha-batched by transcript challenges.
3. **One inner sum-check** over the column variables with the public
   `M_combo` table (the verifier rebuilds it in `O(nnz)` — never
   transmitted) against the eq(r_inst)-folded witness matrix.
4. **Equation (1)**: the inner terminal splits on the top column bit —
   the public half folds linearly (the verifier evaluates it), the
   witness half decomposes as `sum_i eq(q_i, b_i) v_i`, the per-segment
   values the prover sends. Those become the **segment claims** that
   every downstream argument consumes: the shared `r_val = r_local(rs1)`
   and `r_cycle = r_inst` used by the Twists, the per-segment VS claims
   the Mosaic discharge finally proves against the commitment.

The segment layout (`r1cs.rs`, `SegmentLayout`) is declared-order with
length-aligned offsets, the same discipline as the Mosaic layout — a
segment's claim point is its local coordinates, and every consumer knows
exactly which sub-cube it lives on.

## Why this is the throughput layer

The outer sum-checks run over the constraint rows (log of the padded
row count), not the trace; the inner runs over the columns. Prover work
is dominated by the matrix applications, which are sparse and
per-cycle-block-diagonal — linear in trace length with tiny constants.
Nothing in the construction touches a hash except the alpha and
instance challenges; the entire hash budget of the protocol concentrates
in the single commitment, which is precisely the cost anatomy the
roadmap's optimization program targets (`cost-model.md`).

## The measured saving

The virtual segments are never committed: at the paper's scale that is
the ~22% of witness bits the stats instrument prints as "virtual
(uncommitted)" — the difference between committing ~2^32 bits and
~2^33 bits per 2^22-cycle chunk is the difference between the paper's
throughput claims and a naively-laid-out system's.
