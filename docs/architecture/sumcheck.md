# The sum-check engine

*Magnetar-sumcheck — the only interactive-proof primitive in the system.
Everything in the paper's protocol is a sum-check; every component differs
only in the shape of its summand.*

## The contract

A [`SumcheckSpec`] fixes the summand as a set of **terms**, each a constant
coefficient times a product of *dense factors* (prover-held tables) and
*public factors* (verifier-evaluable multilinears, `public.rs`). The
prover folds a term at round `i` if and only if one of its factors depends
on variable `i`; the round message is the degree-`d` univariate in that
variable (degree = the number of factors in the term that own it). The
engine's round messages, challenge absorption (Fiat–Shamir via the shared
transcript), and terminal identity are generic over all of it.

Two bind orders cover every protocol use:

* `BindOrder::Increasing` — variable 0 first. The default for joint spaces
  where the low bits are "small" dimensions (addresses, selectors).
* `BindOrder::HighBitsFirst { high }` — the top `high` variables first,
  in order. Virtual Spartan uses it to run the two outer sum-checks in
  *lockstep* over shared instance coordinates; MosaicPCS uses it to bind
  the 7 packing variables first so the terminal splits cleanly into a
  (block, bit) pair for the ring switch.

## The sparse prover (`sparse.rs`)

The dense engine materializes every factor over the *joint* space. For the
routers that is wasteful bordering on impossible: the joint
`(o, u, h, j)` space at production parameters is 2^20+ entries *per
factor*, while the factors themselves live on sub-cubes (the routing
tensor on `(o, u, h)`, the selectors on `(h, j)`). The sparse prover keeps
every factor in its **own variable space** and, at each round, expands only
the elementary products that involve the round's variable — the paper's
"prover work linear in T" claim is realized here as "linear in the sum of
factor sizes", which for the routers is ~2^20 instead of ~2^25.

## The characteristic-two rules

These two identities are load-bearing for correctness, are documented in
`CONTRIBUTING.md`, and have each caused exactly one bug in this
codebase's history — both are now pinned by tests:

1. **The idle-variable rule.** A term whose factors *all* skip the round's
   variable contributes `2^1 . v = 0` to the Boolean-cube sum — it
   vanishes. Consequently, *every term must cover every joint variable
   through at least one factor*. The router selector checks run
   per-factor sum-checks precisely because a combined check would leave
   idle h-slices whose contributions collapse to zero, silently breaking
   the very identity being proven.
2. **The replicated-factor terminal rule.** Folding a factor that is
   constant along the round's variable leaves it unchanged
   (`(1+c)v + cv = v` in characteristic two, since `1 + c + c = 1`).
   Therefore a factor replicated over idle variables produces a terminal
   claim *exactly at its own sub-point* — no `prod(1+c)` correction term.
   The stage-selector sum-check relies on this to place the shamt-flag
   claim on the flags segment at the cycle point directly.

## Public factors (`public.rs`)

* `EqFactor` — `eq(r, .)`; the claim-binding workhorse.
* `SumEqFactor` — `sum_i alpha_i eq(p_i, .)`; MosaicPCS's alpha-batched
  lift of many claims into one sum-check.
* `TableFactor` — any public table both parties materialize (the R1CS
  matrix combination in the inner sum-check).
* `JointTableFactor` (memory crate) — a public table over a *subset* of a
  joint space (ROM columns over `(h, k-row)`, eq tensors over slices),
  which is how the discharge arguments keep their factors sparse while
  the verifier still evaluates them at the sampled point.

## Soundness discipline

Round messages are absorbed challenge-by-challenge; the terminal identity
is checked against the *claimed* dense evaluations and the *verifier's own*
evaluation of the public factors — never against prover-supplied "check
this holds" bytes. Any new component built on the engine inherits this:
the only way to introduce a soundness bug at this layer is to misstate the
summand (see the idle-variable rule) or to desynchronize the transcript
order between prover and verifier (see `protocol.md` on transcript
discipline).
