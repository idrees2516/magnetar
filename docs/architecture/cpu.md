# The CPU: RV64I, micro-expansion, and the dual circuit

*Magnetar-cpu — paper §2 (the machine) and §4.2 (the circuit).*

## The machine

A Harvard RV64I core: 32 architectural registers (x0 hardwired; cells
32..64 are trash/padding — the write port's `rd == 0` redirect to cell 32
makes the one-write-per-cycle model total), 2^8 words of byte-addressed
RAM (byte-lane semantics: `LB/LBU` loads byte `off`, `LH/LHU` the pair
`off, off+1 mod 8`, `LW/LD` the word), and a micro-PC over the *expanded*
program. Branches resolve in micro-PC units (the byte immediate over 4,
landing on the target's first virtual instruction); ECALL/EBREAK latch
the halt with `out = x10`.

`interp.rs` is the single source of truth for semantics; every witness
column is a field of its trace records, and the full-circuit
**satisfiability tests** (`witness_sat_tests`) evaluate both relations on
every row of a trace — the regression harness that caught both latent
bugs below.

## Micro-expansion (`expand.rs`)

Every instruction becomes one or more virtual instructions forming the
public ROM:

* **register shifts** -> 6 static routing stages (see `router.md`): the
  trash-cell chain, the bus carrying the stage index, and the ROM's
  `sel_stage` one-hot;
* everything else stays single (immediate shifts are already static
  routings with a ROM-constant amount).

The expansion is a public, deterministic function of the program —
KeyGen derives the ROM tables from it, and the trace-length padding
(`execute_padded`) fills the witness cube exactly to the proving bound
with real no-op cycles.

## The dual SIMD R1CS circuit (`circuit.rs`)

2048 columns per cycle: 18 witness segments (7 virtual, 11 direct — the
copies, the flags, the offsets, the bus) plus the public `(1, x)` half.
The constraint families, all degree <= 4 in the witness bits:

1. **Booleanity** over every used column.
2. **The adder**: t2's subtraction mask, the carry recurrence
   `(t1_i + c_i)(t2_i + c_i) = c_{i+1} + c_i`, and the masked sum
   `B = is_arith` so non-arith cycles leave the ALU column free.
3. **Copy bindings**: `shift-src = rs1`, `store-old = load-val`,
   `store-src = rs2` — the router's X sources, bound to the virtual
   segments the memory arguments own.
4. **The address binding**: `ram_addr = alu[3..11]` gated on
   `is_load + is_store` — the effective address's word bits on memory
   cycles only.
5. **The eq-chain byte-lane muxes** for sub-word loads
   (`PA/EQA/PB/EQB` aux products, the `CA/CH/CH2` conditionals, the
   M0/M1 packed-byte muxes).
6. **Sub-word-load zero-extension**: byte loads zero `rd` bits 8..63,
   half loads bits 16..63 — masked by `ld_byte`/`ld_half` only.
7. **The write mux** `WNL . (sum beta alu + sum beta rd) = 0` (over K),
   the shift select `is_shift . (sum beta alu + sum beta shift_out)`,
   the word-load copy `ld_word . (sum beta rd + sum beta load_val)`,
   and the **halt-latch I/O binding**
   `halt . (sum beta rd + x_out) = 0`.

The layout is positional (offsets are compile-time constants); the
`F_SHAMT_BIT` flag (the register-shift stage condition) lives in a spare
flags slot, is witness-trusted exactly like the other flags, and is
bound *outside* the R1CS by the stage-selector sum-check — the one
argument in the system whose binding lives entirely in the
sum-check layer.

## The two latent bugs the satisfiability harness caught

Both were invisible at the old test scale and both are fixed:

1. **Trace padding**: `execute` padded to the natural next-power-of-two,
   not the proving bound — any program whose cycle count rounded below
   `trace_len` left most witness rows zero, violating per-cycle
   constraints (the one-column, the I/O binding). Every old test
   program coincidentally matched its bound. `execute_padded` now
   fills the cube exactly, and the protocol asserts it.
2. **Inverted zero-extension masks**: the sub-word-load zero-extension
   used `(1 + ld_word) . rd_b` — zeroing `rd`'s high bits on every
   *non-word-load* cycle, which rejects any honest ALU result with high
   bits set (a negative ADDI was unprovable). The masks are now
   `ld_byte + ld_half` (bits 16..63) and `ld_byte` (bits 8..15), so
   only genuine sub-word loads zero-extend. All old tests passed with
   small non-negative values — the classic coincidence trap.

## Scope notes (paper-faithful)

* The cross-step PC constraints are omitted exactly as the paper's
  Remark 4 omits them from the formal protocol.
* The decode-vs-flag correlation remains the documented engineering
  extension (flags are witness-trusted; the fetched bus is
  authenticated by the Shout).
* Test traces are 2^3–2^5 cycles (CI scale); the paper's production
  envelope is 2^22-cycle segments at 6.5–7.7M cycles/s.
