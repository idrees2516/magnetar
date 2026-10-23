# The router and the register-shift stage-select extension

*Magnetar-router + the protocol's step 5/6 wiring — paper §5.4.*

## The routing identity

Some RV64 operations have no compact R1CS encoding because each output
bit is a *copy or zero* of an input bit at a position the instruction
fixes: shifts and sub-word stores. The paper's observation is that both
are **bit routings**:

```text
Y(o, j) = sum_{u,h} X(u, j) . Q(h, j) . R(o, u, h)
```

* `X` — the candidate source bits (direct copy segments of W);
* `Q(., j)` — a one-hot (or zero) selector naming the routing in use at
  cycle j, stored in the bytecode row as two shorter one-hot fields
  (`Q_lo` the amount, `Q_hi` the kind) to keep h narrow;
* `R` — the public, sparse routing tensor fixed at preprocessing
  (under any one routing, each output bit is fed by at most two sources).

Given a claim about `Y` (from Virtual Spartan for shifts; from the
verifier-sampled Inc point for stores), ONE sum-check over the joint
`(o, u, h, j)` space — the **sparse prover**, because the joint
materialization would be 2^25 at production parameters — reduces it to
claims on `X` (which lift to the copy segments by construction: the shift
u-space *is* the bit index, the store u-space splits `(old, src)` by the
top coordinate) and on the `Q` factors, which the selector checks then
authenticate against the ROM. The verifier evaluates R-hat itself.

## The two instances

* **R_sh (shifts, Eq. 5)**: `u` is the source word's bit position; the
  zero-fill is `R = 0` and the SRA sign-fill routes from bit 63. The
  selector is `(amount, kind)`, both ROM fields for *immediate* shifts.
* **R_st (stores, Eq. 6)**: sources `(old word, store source)` with
  `u = bit + 64.half`; inside a written byte the increment bit is the XOR
  of old and source — expressed by routing the output from *both* source
  bits, whose B-sum is the XOR. The byte offset `a in B^3` enters as
  three extra Boolean *summed* variables linked to the witness offset
  bits by `eq(a_k, hat a_k(j))` factors, and the offset-bit claims lift
  to the `off` segment at Boolean unit points.

## The register-shift stage-select (the documented extension, now implemented)

The paper expands a register-specified shift ("which does not have a
static routing") into a short fixed sequence of virtual instructions
"each of which is a static routing" — and leaves the conditional inside
each stage as engineering. This is that engineering, in four pieces:

**1. The trash-cell chain (`cpu/expand.rs`).** Stage rows re-target the
public ROM fields so the *existing* memory arguments carry the
accumulator: stage 0 reads the instruction's rs1, later stages read
cell 32 (the trash cell the `rd == 0` redirect already used), and
intermediate stages write cell 32 back. No scratch register exists,
`x0` stays hardwired to zero, and the register Twist's
one-write-per-cycle model authenticates the chain exactly as it
authenticates every other write.

**2. The ROM's `sel_stage` one-hot (`zkvm/rom.rs`).** Stage rows carry a
one-hot over the 6 stages in a new ROM column, authenticated through
the fetch one-hot by a selector check exactly like `sel_lo`/`sel_hi`/
`sel_st`. The *kind* stays a static ROM field (`sel_hi`); the *amount*
becomes the conditional pair `{identity, shift-by-2^k}`.

**3. The conditional selector.** The router's amount table `Q_lo` on a
stage row is the witness-conditional one-hot `(1+b)` at amount 0 and
`b` at amount `2^k`, where `b` is a committed flag bit (`F_SHAMT_BIT`)
and `k` comes from the fetched row. The **stage-selector sum-check**
(`memory.md`, `discharge.rs`) binds `b` to bit `k` of the amount
register (the `store-src` copy segment's bits — the R1CS binds the copy
to rs2 in every cycle) and produces the claim `q_stage` for the
structured part.

**4. The additive composition.** The `sel_lo` selector check runs with
claim `q_lo + q_stage` (characteristic two: the difference is the
sum), which forces the router's witness selector to equal
`ROM(sel_lo) + STAGE` at the router's random point — a multilinear
identity at a random K-point, hence everywhere with overwhelming
probability. Soundness composition: the ROM part is fetch-authenticated,
the stage part is bound to committed bits with its structure forced by
the sum-check's polynomial identity, and the union is one-hot per cycle
by construction.

**Why additive composition is safe.** Every check in the chain binds a
*claim value* that is itself verified: the router's `q_lo` by the
routing sum-check, `q_stage` by the stage sum-check against discharged
dense claims (the fetched `sel_stage` via its selector check, the flag
and the amount byte via the Mosaic). There is no "prover says trust me"
edge: tamper classes 1–5 in the e2e test each break exactly one link
and are all rejected.

## Scope note

The *decode-vs-flag correlation* (binding the witness flag bits —
`is_arith`, `is_shift`, load widths — to the fetched bus bits by
constraints, rather than trusting them as witness) remains the separate
documented extension it always was; the stage-select argument here
consumes the flags the same way the rest of the circuit does, and its
soundness does not depend on that correlation being constrained.
