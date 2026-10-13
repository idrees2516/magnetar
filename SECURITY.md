# Security Policy

## Supported versions

The `main` branch carries all fixes; there are no long-lived release
branches yet.

## Reporting a vulnerability

**Do not open a public issue for anything you believe is a soundness or
security problem.**

Report privately to the maintainers via GitHub's *Report a vulnerability*
flow (the *Security* tab → *Report a vulnerability*), which routes
encrypted to the repository admins. Include:

1. The component (crate, function, proof structure) you believe is
   unsound;
2. The attack sketch — what a malicious prover would do, and which
   identity or bound it evades;
3. The fix idea, if you have one.

We respond within 7 days and credit reporters unless anonymity is
requested.

## Scope

Magnetar is a research-grade proof system implementation. The security
properties that matter are, in order:

1. **Knowledge soundness of every reduction** (sum-check round checks,
   address virtualization product identities, the Δ\* one-hotness,
   the Φ-combination, the routing identities, the ring-switch anchors,
   and the PCS fold chain);
2. **Constant-time arithmetic** in `magnetar-field` (no secret-dependent
   branches or indices);
3. **Fiat–Shamir transcript hygiene** (domain separation, length
   binding, and the exact prover/verifier absorb order — a divergence
   between the two sides is a soundness-relevant bug even when the
   honest prover still verifies).

Out of scope: denial-of-service through extreme trace lengths, and side
channels on prover-local memory.
