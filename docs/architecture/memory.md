# Memory: BinaryTwist, BinaryShout, and the discharge arguments

*Magnetar-memory — paper §5.1, §5.3, §5.5 steps 3–4 and 6.*

## The memory-checking pattern

All three arguments implement the same paper pattern — offline memory
checking over a table of L cells across T cycles:

* an **address component** `ra_i: {0,1}^{a_i} x {0,1}^n -> B` — the
  one-hot "port p touched address i at cycle j" table;
* an **increment** vector `Inc: {0,1}^{a x ell} x {0,1}^n -> B` — the
  XOR of the old and new contents of the touched cell;
* the running **last-touch** recurrence, in characteristic two
  `LT_s = (1 - a_{s-1}) b_{s-1} + (1 + a_{s-1} + b_{s-1}) LT_{s-1}` on
  a linearized time axis, with the strict/inclusive variants fixing
  read-vs-write ordering.

The **Frobenius refinement** samples refined points
`r' = r^3 / c` (with `c = 1 + r + r^2`, `C = prod c_h`) so the one-hot
check runs at points algebraically independent of the value points — the
Delta* argument — and every component's claims are placed at exactly
the points the verifier can re-derive.

## BinaryTwist (`twist.rs`)

The read/write consistency argument for the register file (6-bit
addresses, one write port per cycle at the ROM row's redirected
destination — `rd == 0` maps to the trash cell 32) and for RAM (8-bit
word addresses, the load/store log). The prover materializes the memory
table incrementally (`materialize_val`, `O(K L T)`) and runs the
value sum-checks over the joint `(k, o, j)` space with padded factors;
the terminal claims land on the committed one-hot components and
increments, and the **reconstruction** emits the per-component claims
that pin the read values to the VS segment claims (`rv_claims`) and the
write value to the write-mux claim (`wv_claim`) — this is where the
virtual segments the Spartan layer only *claimed* become bound to the
commitment.

## BinaryShout (`shout.rs`)

The read-only ROM argument: a table is selected by a public ROM row and
the claim is that the fetched values equal the ROM's. The
**Phi-combination sum-check** proves the combination over the address
cube with the alpha_0 sum-forcing trick (the non-emptiness term), the
full-address `F(k) = prod G_i(k^{(i)})` dense factor, and the
**Delta\* one-hot sumcheck** (`[eq x A^3] + [C x eq x A']`, degree 4)
certifying the fetch one-hot components. Address virtualization turns
the terminal claim into the product of per-component claims, which is
what lets one committed `fetch-ra` component serve the bus, the pc, and
all four selector checks.

## The discharge helpers (`discharge.rs`)

Three sum-check families that bind claims produced elsewhere to
committed components:

* **Psi-sum-checks**: an address claim `c` (an eq-weighted fold of the
  address bits at the value point) is bound by
  `c = sum_{k,j} eq(r_cycle, j) Psi(k) prod_i ra_i(k^{(i)}, j)` with the
  public weight table `Psi(k) = sum_t w_t k_t`. The register-index
  claims (rs1/rs2/rd slices of the bus) and the RAM address claims are
  pinned this way, and checked against the *segment* claims (exact
  equality, no per-bit leak).
* **The joint bytecode Shout**: ONE sum-check over the fetch component
  `(k, o, j)` space checking the bus bits, the register-index slices,
  and the folded non-emptiness at once — the alpha_0 term forces the
  one-hot columns to be live, the gamma-weighted slices check the bus
  slices against the register Psi claims.
* **The selector checks**: per selector factor (the routers' one-hot
  fields), a sum-check over `(k, h_f, j)` with the fetch components as
  dense factors and the ROM's selector columns as public tables —
  binding the router's terminal selector claims to the committed fetch
  one-hot. Per-factor runs because of the char-2 idle-variable rule.

## The stage-selector sum-check (the register-shift stage-select)

`discharge.rs`'s newest family. A register-shift stage row offers the
shift router a *conditional* amount pair `{identity, shift-by-2^k}`,
selected by bit `k` of the amount register. The sum-check proves, over
the joint `(k', j)` space,

```text
q_stage = sum eq(r_j, j) . oh(k', j) . [(1+b(j)) E0 + b(j) E(k')]
        + gamma . sum eq(r_j, j) . oh(k', j) . (b(j) + amt(k', j))
```

with `oh` the fetched `sel_stage` one-hot (discharged against the ROM by
the caller's selector check), `b` the committed shamt flag, `amt` the
amount register's low byte on the `store-src` copy segment, `E0` and
`E(k')` the identity/shift structure constants at the router's amount
point, and the gamma-folded second sum the bit-binding identity
(`b = bit_k(amount)`, honest sum zero). The caller composes the claims
additively in characteristic two: the router proves its witness amount
selector's terminal `q_lo`, this argument proves `q_stage` for the
*structured* stage part, and the `sel_lo` selector check runs with claim
`q_lo + q_stage` — forcing `Q_lo = ROM(sel_lo) + STAGE` at the router's
random point, hence (Schwartz–Zippel over the multilinear extensions)
everywhere. See `router.md` for the full flow.
