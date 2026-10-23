# Contributing to Magnetar

Thank you for improving a post-quantum proof system. Code here is held to
a higher bar than usual, because **a silent soundness bug invalidates
every proof ever produced**. Please read this document before your first
PR.

## Ground rules

1. **Every protocol change maps to the paper.** The PR description states
   which construction, equation, or lemma it touches (or that it is
   purely engineering). Implementations that deviate from the paper must
   explain the deviation and its soundness in `docs/security.md`.
2. **Prover/verifier symmetry is tested by construction.** A reduction
   that adds a transcript absorb on one side must add the same absorb at
   the same point on the other — desyncs are exactly the class of bug
   that honest proofs hide.
3. **Characteristic-two arithmetic discipline.** A factor replicated
   over idle variables sums to zero over the Boolean cube; every
   sum-check term must cover all joint variables. If you add a sum-check,
   check the idle-variable coverage first.
4. **Tamper tests accompany every new argument.** At least one negative
   test per transmitted claim and per round-message family.
5. `#![forbid(unsafe_code)]` stays everywhere except the audited
   PCLMULQDQ module.

## Workflow

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
cargo run --release --bin magnetar -- fib 2   # end-to-end smoke
```

- Rust: stable (MSRV = latest stable toolchain).
- Layout: one crate per paper component; see the README's architecture
  table.
- Bugs → issues with the templates; soundness concerns → the private
  channel in SECURITY.md.

## Commit style

Conventional prefixes (`feat:`, `fix:`, `proof:`, `docs:`, `perf:`,
`test:`), imperative subject, paper reference in the body when relevant.
