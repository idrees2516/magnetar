//! The **linear-in-T provers** for the memory-checking sum-checks (the
//! paper's §4.4 prover-work claim: "the prover's cost is linear in T").
//!
//! The dense engine materializes the joint `(k, o, j)` cube —
//! `O(2^{a+ell+n})` words per factor — which dominated the reference
//! prover's time and memory (docs/BENCHMARKS.md item 1). These provers
//! compute the **same round messages** (byte-identical, cross-validated
//! by tests) by enumerating the trace **cycle by cycle** and exploiting
//! the two structural facts of the memory-checking summands:
//!
//! 1. **The one-hot kill.** The address factors are one-hot per cycle:
//!    `ra(k, j) = Π_i e_{q_i(j)}(k^{(i)})`, so the MLE of the full
//!    address at a partially bound point is the per-cycle product
//!    `Π_β (1 + c_β + q_β(j))` — and summing over the unbound address
//!    bits selects the addressed cell. Every address round costs `O(1)`
//!    per cycle (two cell lookups: the round bit at 0/1), never `O(2^a)`.
//! 2. **The prefix-XOR cumulative sum.** The reconstruction identity
//!    `Val = M0 + Σ_{j'<j} LT(j',j)·wfa·Inc` couples cycles only
//!    through the *same-address* running XOR of the increments — an
//!    `O(T)` scan with a per-address-suffix accumulator, never the
//!    `O(T²)` pair enumeration and never the `K·L·T` history table.
//!
//! ## The port sum-check, phase by phase
//!
//! The statement (paper §4.4, the read/write value identities):
//! `Σ_{k,o,j} eq(r_val,o)·eq(r_cycle,j)·ra(k,j)·Val(k,o,j) = claim`,
//! with the joint layout `k | o | j` bound in increasing order, where
//! `Val`'s MLE is given by the algebraic identity above (valid at every
//! point, not just Boolean ones, because the summed result is
//! multilinear in the free variables).
//!
//! * **k-rounds** — at `k_t = X` the one-hot kill leaves, per cycle,
//!   `eq_c(j)·RAEQ(j)·((1+q_t(j))+X)·V(X,j)` where `V` is `Val` at the
//!   (mixed) addressed cell: its `M0` part is two lookups into the
//!   address-folded, value-pre-folded initial memory `M0v`; its `Inc`
//!   part is the same-address prefix sum of the value-pre-folded
//!   increments `INCv`, keyed by the address bits not yet bound.
//! * **o-rounds** — `k` is bound: `RAw(j)` and `WFAw(j')` are per-cycle
//!   scalars; the `LT`-coupled sum becomes the suffix sum
//!   `LTW(j') = Σ_j LT(j',j)·W(j)` (computed once); each round folds the
//!   small value tables and the increment columns.
//! * **j-rounds** — everything is a `T`-vector (`eq_c`, `RAw`, the
//!   prefix-XOR `CUM`); the rounds fold them pairwise.
//!
//! The emitted round messages, terminal claims, and transcript schedule
//! are identical to the dense prover's — the verifier is unchanged.

use crate::twist::PortRun;
use crate::{AddressComponent, TwistInstance};
use magnetar_field::{bind_var, mul, par_bind_var, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;

/// `eq(c, b) = 1 + c + b` for a Boolean `b` (branch-free, char 2).
#[inline]
fn eqb(c: Gf128, bit: bool) -> Gf128 {
    c + Gf128(bit as u128) + ONE
}

/// The LENIENT per-cycle address trace: cycles with zero or multiple
/// hot bits (a malformed component — the tamper tests' shape) resolve to
/// the first hot address or 0, so provers never panic on bad witnesses;
/// the soundness layers reject the resulting proofs.
pub fn addr_bits_lenient(comps: &[AddressComponent]) -> Vec<usize> {
    let ad = comps[0].addr_vars;
    let n = comps[0].cycle_vars;
    let cycles = 1usize << n;
    let mut out = vec![0usize; cycles];
    for (i, comp) in comps.iter().enumerate() {
        let av = comp.addr_vars;
        for j in 0..cycles {
            let base = j << av;
            for k in 0..(1usize << av) {
                if comp.table[base + k] == ONE {
                    out[j] |= k << (i * ad);
                    break;
                }
            }
        }
    }
    out
}

/// A per-cycle address trace: the full address bits of one port,
/// extracted from the one-hot components (`O(K^{1/d}·T)`).
pub fn addr_bits(comps: &[AddressComponent]) -> Vec<usize> {
    let ad = comps[0].addr_vars;
    let n = comps[0].cycle_vars;
    let cycles = 1usize << n;
    let mut out = vec![0usize; cycles];
    for (i, comp) in comps.iter().enumerate() {
        let av = comp.addr_vars;
        for j in 0..cycles {
            let base = j << av;
            let mut found = None;
            for k in 0..(1usize << av) {
                if comp.table[base + k] == ONE {
                    found = Some(k);
                    break;
                }
            }
            let k = found.expect("honest witness: one-hot address component");
            out[j] |= k << (i * ad);
        }
    }
    out
}

/// The shared per-instance precomputation of one BinaryTwist instance:
/// the value-pre-folded increments and initial memory, the write address
/// trace (every port's `Val` references the same write address), and the
/// cycle-point eq table.
pub struct TwistShared {
    /// `INCv(j) = Înc(r_val, j)` — the increments value-pre-folded.
    pub inc_v: Vec<Gf128>,
    /// `M0v(k) = M̂0(k, r_val)` — the initial memory value-pre-folded.
    pub m0_v: Vec<Gf128>,
    /// The write address bits per cycle.
    pub w_addr: Vec<usize>,
    /// `eq(r_cycle, j)`.
    pub eq_c: Vec<Gf128>,
}

impl TwistShared {
    pub fn new(inst: &TwistInstance, w_addr: Vec<usize>, inc: &[Gf128]) -> Self {
        let (a, ell, n) = (inst.a, inst.ell, inst.n);
        let cycles = 1usize << n;
        let mut inc_v = inc.to_vec();
        // inc layout: bit low (ell), cycle high (n) — fold the ell low
        // bits at r_val (the fold position stays 0 as bits slide down).
        for b in 0..ell {
            par_bind_var(&mut inc_v, 0, inst.r_val[b]);
        }
        let mut m0_v = inst.m0.clone();
        // m0 layout: address low (a), bit high (ell) — the o-bit b sits
        // at position a (sliding down after each fold).
        for b in 0..ell {
            par_bind_var(&mut m0_v, a, inst.r_val[b]);
        }
        debug_assert_eq!(inc_v.len(), cycles);
        debug_assert_eq!(m0_v.len(), 1 << a);
        TwistShared {
            inc_v,
            m0_v,
            w_addr,
            eq_c: magnetar_field::EqBits::build(&inst.r_cycle).values,
        }
    }
}

/// The suffix sums `Σ_{j>jp} v[j]` (strict) or `Σ_{j≥jp} v[j]`
/// (inclusive) — the `LT`-weighted row of the less-than table in `O(T)`.
pub fn suffix_sums(v: &[Gf128], inclusive: bool) -> Vec<Gf128> {
    let n = v.len();
    let mut out = vec![ZERO; n];
    let mut run = ZERO;
    for jp in (0..n).rev() {
        if inclusive {
            run += v[jp];
        }
        out[jp] = run;
        if !inclusive {
            run += v[jp];
        }
    }
    out
}

/// The linear port sum-check (read or write shape). Emits round messages
/// byte-identical to the dense engine's for the same statement.
///
/// * `read_comps` — the port's address components (the `ra` factor).
/// * `inclusive` — `false` for reads (`Val`, strict `LT`), `true` for
///   the write (`Val + Inc`, inclusive `LT`).
#[allow(clippy::too_many_arguments)]
pub fn prove_port_linear(
    inst: &TwistInstance,
    shared: &TwistShared,
    inc: &[Gf128],
    read_comps: &[AddressComponent],
    claim: Gf128,
    inclusive: bool,
    label: &str,
    t: &mut Transcript,
) -> PortRun {
    let (a, ell, n) = (inst.a, inst.ell, inst.n);
    let cycles = 1usize << n;
    let joint_vars = a + ell + n;
    let r_addr = addr_bits(read_comps);
    let eq_c = &shared.eq_c;
    let inc_v = &shared.inc_v;
    let mut m0_v = shared.m0_v.clone();
    // Per-cycle address weights (start at ONE; each k-round folds).
    let mut raeq = vec![ONE; cycles];
    let mut wfaeq = vec![ONE; cycles];
    let mut rounds: Vec<Vec<Gf128>> = Vec::with_capacity(joint_vars);
    let mut challenges: Vec<Gf128> = Vec::with_capacity(joint_vars);

    let absorb = |t: &mut Transcript, coeffs: &[Gf128; 4]| -> Gf128 {
        let mut buf = Vec::with_capacity(4 * 16);
        for c in coeffs {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        t.challenge_field(label)
    };

    // ---- k-rounds: the one-hot kill + the same-address prefix sums. ----
    struct Pt {
        label: &'static str,
        start: std::time::Instant,
    }
    impl Pt {
        fn new(label: &'static str) -> Self {
            Pt {
                label,
                start: std::time::Instant::now(),
            }
        }
    }
    impl Drop for Pt {
        fn drop(&mut self) {
            if std::env::var("MAGNETAR_TIME").is_ok() {
                eprintln!(
                    "[time] {:<24} {:>9.1} ms",
                    self.label,
                    self.start.elapsed().as_secs_f64() * 1e3
                );
            }
        }
    }
    let _pt_k = Pt::new("port-k-rounds");
    for bit in 0..a {
        let shift = bit + 1;
        let suff_space = 1usize << (a - shift);
        let mut acc0 = vec![ZERO; suff_space];
        let mut acc1 = vec![ZERO; suff_space];
        let mut coeffs = [ZERO; 4];
        for u in 0..cycles {
            let qu = r_addr[u];
            let wu = shared.w_addr[u];
            // Insert the current cycle's write contribution:
            // C(j') = WFAEQ·((1+wq_bit) + X)·INCv → (C0, C1).
            let insert = |acc0: &mut Vec<Gf128>, acc1: &mut Vec<Gf128>| {
                let key = wu >> shift;
                let base = mul(wfaeq[u], inc_v[u]);
                acc0[key] += mul(ONE + Gf128(((wu >> bit) & 1) as u128), base);
                acc1[key] += base;
            };
            if inclusive {
                insert(&mut acc0, &mut acc1);
            }
            let s0 = acc0[qu >> shift];
            let s1 = acc1[qu >> shift];
            if !inclusive {
                insert(&mut acc0, &mut acc1);
            }
            // RA's round factor: ((1+q_bit) + X)·eq_c·RAEQ.
            let a_j = mul(
                mul(eq_c[u], raeq[u]),
                ONE + Gf128(((qu >> bit) & 1) as u128),
            );
            let b_j = mul(eq_c[u], raeq[u]);
            // Val's round factor at the addressed (mixed) cell:
            // V(X) = (m00 + X·(m00+m01)) + (s0 + X·s1).
            let m00 = m0_v[(qu >> shift) << 1];
            let m01 = m0_v[((qu >> shift) << 1) | 1];
            let v0 = m00 + s0;
            let v1 = (m00 + m01) + s1;
            coeffs[0] += mul(a_j, v0);
            coeffs[1] += mul(a_j, v1) + mul(b_j, v0);
            coeffs[2] += mul(b_j, v1);
        }
        if rounds.is_empty() {
            debug_assert_eq!(
                coeffs[1] + coeffs[2] + coeffs[3],
                claim,
                "linear port prover round 0 must sum to the claim"
            );
        }
        let c = absorb(&mut *t, &coeffs);
        rounds.push(coeffs.to_vec());
        challenges.push(c);
        // Fold the per-cycle weights and the M0 view.
        for u in 0..cycles {
            raeq[u] = mul(raeq[u], eqb(c, (r_addr[u] >> bit) & 1 == 1));
            wfaeq[u] = mul(wfaeq[u], eqb(c, (shared.w_addr[u] >> bit) & 1 == 1));
        }
        bind_var(&mut m0_v, 0, c);
    }

    drop(_pt_k);
    // ---- the post-k per-cycle state. ----
    let raw = raeq.clone(); // RA-MLE(r_k, j)
    let w: Vec<Gf128> = eq_c
        .iter()
        .zip(raw.iter())
        .map(|(&e, &r)| mul(e, r))
        .collect();
    let sj: Gf128 = w.iter().copied().sum();
    // M0k = M̂0(r_k, o): fold the full initial memory's address part.
    let mut m0k = inst.m0.clone();
    for &c in &challenges[..a] {
        par_bind_var(&mut m0k, 0, c);
    }
    debug_assert_eq!(m0k.len(), 1 << ell);
    // LTW(j') = Σ_j LT(j',j)·W(j) — the suffix sums of W.
    let ltw = suffix_sums(&w, inclusive);
    // The increment columns, o-folded at the round challenges.
    let mut inc_b = inc.to_vec();
    let mut eqv = ONE;

    // ---- o-rounds: fold the value dimension. ----
    let _pt_o = Pt::new("port-o-rounds");
    for b in 0..ell {
        let rem = ell - b; // remaining o variables (incl. the round's)
        let eq_rest = magnetar_field::EqBits::build(&inst.r_val[b + 1..]).values;
        // s0/s1: the eq-sums of m0k at o_b = 0/1.
        let mut s0 = ZERO;
        let mut s1 = ZERO;
        for (i, &e) in eq_rest.iter().enumerate() {
            s0 += mul(e, m0k[i << 1]);
            s1 += mul(e, m0k[(i << 1) | 1]);
        }
        // I0/I1: the eq-sums of inc_b's o_b = 0/1 slices per cycle.
        let mut i0 = vec![ZERO; cycles];
        let mut i1 = vec![ZERO; cycles];
        for u in 0..cycles {
            let base = u << rem;
            let mut a0 = ZERO;
            let mut a1 = ZERO;
            for (i, &e) in eq_rest.iter().enumerate() {
                a0 += mul(e, inc_b[base | (i << 1)]);
                a1 += mul(e, inc_b[base | (i << 1) | 1]);
            }
            i0[u] = a0;
            i1[u] = a1;
        }
        // C = SJ·s0 + Σ WI·LTW·I0 (the o_b=0 value); D = SJ·(s0+s1) +
        // Σ WI·LTW·(I0+I1) (the MLE's X coefficient — value(0)+value(1)).
        let mut cval = mul(sj, s0);
        let mut dval = mul(sj, s0 + s1);
        for u in 0..cycles {
            cval += mul(mul(wfaeq[u], ltw[u]), i0[u]);
            dval += mul(mul(wfaeq[u], ltw[u]), i0[u] + i1[u]);
        }
        let a_const = ONE + inst.r_val[b];
        let coeffs = [
            mul(eqv, mul(a_const, cval)),
            mul(eqv, mul(a_const, dval) + cval),
            mul(eqv, dval),
            ZERO,
        ];
        let d = absorb(&mut *t, &coeffs);
        rounds.push(coeffs.to_vec());
        challenges.push(d);
        eqv = mul(eqv, ONE + inst.r_val[b] + d);
        bind_var(&mut m0k, 0, d);
        par_bind_var(&mut inc_b, 0, d);
    }

    drop(_pt_o);
    // ---- j-rounds: everything is a T-vector. ----
    let _pt_j = Pt::new("port-j-rounds");
    let eqf = eqv; // = eq(r_val, r_v)
    let m0s = m0k[0]; // M̂0(r_k, r_v)
                      // CUM(j) = M̂0(r_k,r_v) + Σ_{j'<j(≤)} WFAw(j')·Înc(r_v, j').
    let mut cum = vec![ZERO; cycles];
    {
        let mut run = ZERO;
        for u in 0..cycles {
            let wiv = mul(wfaeq[u], inc_b[u]);
            if inclusive {
                run += wiv;
            }
            cum[u] = m0s + run;
            if !inclusive {
                run += wiv;
            }
        }
    }
    let mut eqt = eq_c.clone();
    let mut rawt = raw;
    let mut cumt = cum;
    for _ in 0..n {
        let half = eqt.len() / 2;
        let mut coeffs = [ZERO; 4];
        for i in 0..half {
            // Each factor's MLE in the round variable: v0 + X·(v0+v1).
            let e0 = eqt[i << 1];
            let e1 = eqt[(i << 1) | 1];
            let es = e0 + e1;
            let r0 = rawt[i << 1];
            let r1 = rawt[(i << 1) | 1];
            let rs = r0 + r1;
            let v0 = cumt[i << 1];
            let v1 = cumt[(i << 1) | 1];
            let vs = v0 + v1;
            coeffs[0] += mul(mul(e0, r0), v0);
            coeffs[1] += mul(mul(es, r0), v0) + mul(mul(e0, rs), v0) + mul(mul(e0, r0), vs);
            coeffs[2] += mul(mul(es, rs), v0) + mul(mul(es, r0), vs) + mul(mul(e0, rs), vs);
            coeffs[3] += mul(mul(es, rs), vs);
        }
        for k in 0..4 {
            coeffs[k] = mul(eqf, coeffs[k]);
        }
        let e = absorb(&mut *t, &coeffs);
        rounds.push(coeffs.to_vec());
        challenges.push(e);
        bind_var(&mut eqt, 0, e);
        bind_var(&mut rawt, 0, e);
        bind_var(&mut cumt, 0, e);
    }

    drop(_pt_j);
    PortRun {
        rounds,
        r_joint: challenges,
        ra_claim: rawt[0],
        val_claim: cumt[0],
    }
}

// ---------------------------------------------------------------------------
// The linear bytecode-Shout Φ sum-check (discharge::prove_bus_shout's core).
// ---------------------------------------------------------------------------

use crate::claims::PcsPoint;
use crate::discharge::BusShoutSpec;

/// The linear Φ sum-check: `Σ_{k,o,j} Πcomps(k,j)·eq_v(o)·eq_c(j)·(α₀ +
/// w(o)·bus(o,k)) = claim` over the joint `(k | o | j)` layout, bound in
/// increasing order. `d = 1` (the single fetch component); the round
/// messages are byte-identical to the dense engine's.
///
/// Returns `(rounds, r_joint, component_claim)` where the component claim
/// is `comp-MLE(r_k, r_j)`.
pub fn prove_bus_phi_linear(
    spec: &BusShoutSpec,
    comp: &AddressComponent,
    alpha0: Gf128,
    gamma_bus: Gf128,
    gammas: &[Gf128],
    claim: Gf128,
    label: &str,
    t: &mut Transcript,
) -> (Vec<Vec<Gf128>>, Vec<Gf128>, PcsPoint) {
    let (a, ell, n) = (spec.a, spec.ell, spec.n);
    let cycles = 1usize << n;
    debug_assert_eq!(comp.addr_vars, a);
    debug_assert_eq!(comp.cycle_vars, n);
    let q = addr_bits(std::slice::from_ref(comp));
    let eq_c = magnetar_field::EqBits::build(&spec.r_cycle).values;
    let eq_v = magnetar_field::EqBits::build(&spec.r_val).values;
    // The bus weight table w(o).
    let mut w = vec![gamma_bus; 1usize << ell];
    for ((pos, width, _), &g) in spec.index_claims.iter().zip(gammas.iter()) {
        for o in *pos..(*pos + *width) {
            w[o] += g;
        }
    }
    // EQW(k) = Σ_o eq_v(o)·w(o)·bus(o, k) — the value-and-weight-folded
    // bus row, folded at the address challenges round by round.
    let mut eqw: Vec<Gf128> = (0..(1usize << a))
        .map(|k| {
            let mut acc = ZERO;
            for o in 0..(1usize << ell) {
                acc += mul(mul(eq_v[o], w[o]), spec.rom_bus[o | (k << ell)]);
            }
            acc
        })
        .collect();
    // The per-cycle address weight.
    let mut raeq = vec![ONE; cycles];
    let mut rounds: Vec<Vec<Gf128>> = Vec::with_capacity(a + ell + n);
    let mut challenges: Vec<Gf128> = Vec::with_capacity(a + ell + n);
    let ncoef = 6; // d + 5 with d = 1
    let absorb = |t: &mut Transcript, coeffs: &[Gf128; 6]| -> Gf128 {
        let mut buf = Vec::with_capacity(ncoef * 16);
        for c in coeffs {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        t.challenge_field(label)
    };

    // ---- k-rounds: the one-hot kill against the folded bus row. ----
    for bit in 0..a {
        let shift = bit + 1;
        let mut coeffs = [ZERO; 6];
        for u in 0..cycles {
            let qu = q[u];
            let a_j = mul(
                mul(eq_c[u], raeq[u]),
                ONE + Gf128(((qu >> bit) & 1) as u128),
            );
            let b_j = mul(eq_c[u], raeq[u]);
            let bw0 = eqw[(qu >> shift) << 1];
            let bw1 = eqw[((qu >> shift) << 1) | 1];
            let bws = bw0 + bw1;
            // (A + X·B)·(bw0 + X·bws) + α₀·(A + X·B).
            coeffs[0] += mul(a_j, bw0) + mul(alpha0, a_j);
            coeffs[1] += mul(a_j, bws) + mul(b_j, bw0) + mul(alpha0, b_j);
            coeffs[2] += mul(b_j, bws);
        }
        if rounds.is_empty() {
            debug_assert_eq!(
                coeffs[1] + coeffs[2] + coeffs[3] + coeffs[4] + coeffs[5],
                claim,
                "bus phi round 0 must sum to the claim"
            );
        }
        let c = absorb(&mut *t, &coeffs);
        rounds.push(coeffs.to_vec());
        challenges.push(c);
        for u in 0..cycles {
            raeq[u] = mul(raeq[u], eqb(c, (q[u] >> bit) & 1 == 1));
        }
        bind_var(&mut eqw, 0, c);
    }

    // ---- o-rounds: fold the (w, bus-r_k) pair against eq(r_val, ·). ----
    // busr(o) = bus-MLE(o, r_k); wt(o) = w(o).
    let mut busr = spec.rom_bus.clone();
    for &c in &challenges[..a] {
        bind_var(&mut busr, ell, c);
    }
    debug_assert_eq!(busr.len(), 1usize << ell);
    let raw = raeq.clone();
    let sj: Gf128 = eq_c.iter().zip(raw.iter()).map(|(&e, &r)| mul(e, r)).sum();
    let mut wv = w;
    let mut eqv = ONE;
    for b in 0..ell {
        let eq_rest = magnetar_field::EqBits::build(&spec.r_val[b + 1..]).values;
        // P0/P1/P2: the eq-weighted sums of w·bus at o_b = 0/1 and of the
        // pairwise X-coefficients (degree 3 in X: eq_v × w × bus).
        let mut p0 = ZERO;
        let mut p1 = ZERO;
        let mut p2 = ZERO;
        for (i, &e) in eq_rest.iter().enumerate() {
            let w0 = wv[i << 1];
            let w1 = wv[(i << 1) | 1];
            let b0 = busr[i << 1];
            let b1 = busr[(i << 1) | 1];
            let ws = w0 + w1;
            let bs = b0 + b1;
            p0 += mul(e, mul(w0, b0));
            p1 += mul(e, mul(w0, bs) + mul(ws, b0));
            p2 += mul(e, mul(ws, bs));
        }
        // α₀ term: eqv·(A + X)·SJ (Σ eq_rest = 1).
        let a_const = ONE + spec.r_val[b];
        let coeffs = [
            mul(
                eqv,
                mul(a_const, mul(sj, p0)) + mul(alpha0, mul(a_const, sj)),
            ),
            mul(
                eqv,
                mul(sj, p0) + mul(a_const, mul(sj, p1)) + mul(alpha0, sj),
            ),
            mul(eqv, mul(sj, p1) + mul(a_const, mul(sj, p2))),
            mul(eqv, mul(sj, p2)),
            ZERO,
            ZERO,
        ];
        let d = absorb(&mut *t, &coeffs);
        rounds.push(coeffs.to_vec());
        challenges.push(d);
        eqv = mul(eqv, ONE + spec.r_val[b] + d);
        bind_var(&mut wv, 0, d);
        bind_var(&mut busr, 0, d);
    }

    // ---- j-rounds: (e + X·es)(r + X·rs)·G. ----
    let eqf = eqv;
    let g_const = mul(eqf, alpha0 + mul(wv[0], busr[0]));
    let mut eqt = eq_c.clone();
    let mut rawt = raw;
    for _ in 0..n {
        let half = eqt.len() / 2;
        let mut coeffs = [ZERO; 6];
        for i in 0..half {
            let e0 = eqt[i << 1];
            let es = e0 + eqt[(i << 1) | 1];
            let r0 = rawt[i << 1];
            let rs = r0 + rawt[(i << 1) | 1];
            coeffs[0] += mul(mul(e0, r0), g_const);
            coeffs[1] += mul(mul(e0, rs) + mul(es, r0), g_const);
            coeffs[2] += mul(mul(es, rs), g_const);
        }
        let e = absorb(&mut *t, &coeffs);
        rounds.push(coeffs.to_vec());
        challenges.push(e);
        bind_var(&mut eqt, 0, e);
        bind_var(&mut rawt, 0, e);
    }

    // The component claim at (r_k, r_j).
    let mut pt = challenges[..a].to_vec();
    pt.extend_from_slice(&challenges[a + ell..]);
    let comp_claim = PcsPoint::new(pt, rawt[0]);
    (rounds, challenges, comp_claim)
}

// ---------------------------------------------------------------------------
// The linear Ψ-sum-check (discharge::prove_psi_claim).
// ---------------------------------------------------------------------------

use crate::discharge::PsiProof;

/// The linear Ψ-sum-check: `Σ_{k,j} Ψ(k)·eq(r_cycle,j)·Π_i comp_i(k^{(i)},j)
/// = claim` over the joint `(k | j)` layout. Returns the honest claim and
/// the proof (round messages byte-identical to the dense engine's).
pub fn prove_psi_linear(
    components: &[AddressComponent],
    r_cycle: &[Gf128],
    weights: &[Gf128],
    label: &str,
    t: &mut Transcript,
) -> (Gf128, PsiProof) {
    let d = components.len();
    let a = components[0].addr_vars * d;
    let n = components[0].cycle_vars;
    let cycles = 1usize << n;
    let q = addr_bits(components);
    let eq_c = magnetar_field::EqBits::build(r_cycle).values;
    // Ψ(k) = Σ_t w_t·k_t.
    let psi: Vec<Gf128> = (0..(1usize << a))
        .map(|k| {
            let mut acc = ZERO;
            for (tbit, &wt) in weights.iter().enumerate() {
                if (k >> tbit) & 1 == 1 {
                    acc += wt;
                }
            }
            acc
        })
        .collect();
    // The honest claim (the one-hot kill: Ψ at the addressed cell).
    let claim: Gf128 = (0..cycles).map(|u| mul(eq_c[u], psi[q[u]])).sum();
    t.absorb_field(&format!("{label}-claim"), &[claim]);
    let mut psik = psi;
    let mut raeq = vec![ONE; cycles];
    let ncoef = d + 3;
    let mut rounds: Vec<Vec<Gf128>> = Vec::with_capacity(a + n);
    let mut challenges: Vec<Gf128> = Vec::with_capacity(a + n);
    let absorb = |t: &mut Transcript, coeffs: &[Gf128], label: &str| -> Gf128 {
        let mut buf = Vec::with_capacity(coeffs.len() * 16);
        for c in coeffs {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        t.challenge_field(label)
    };
    // ---- k-rounds. ----
    for bit in 0..a {
        let shift = bit + 1;
        let mut coeffs = vec![ZERO; ncoef];
        for u in 0..cycles {
            let a_j = mul(
                mul(eq_c[u], raeq[u]),
                ONE + Gf128(((q[u] >> bit) & 1) as u128),
            );
            let b_j = mul(eq_c[u], raeq[u]);
            let p0 = psik[(q[u] >> shift) << 1];
            let p1 = psik[((q[u] >> shift) << 1) | 1];
            coeffs[0] += mul(a_j, p0);
            coeffs[1] += mul(a_j, p0 + p1) + mul(b_j, p0);
            coeffs[2] += mul(b_j, p0 + p1);
        }
        if rounds.is_empty() {
            debug_assert_eq!(
                coeffs[1..].iter().copied().sum::<Gf128>(),
                claim,
                "psi round 0 must sum to the claim"
            );
        }
        let c = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs);
        challenges.push(c);
        for u in 0..cycles {
            raeq[u] = mul(raeq[u], eqb(c, (q[u] >> bit) & 1 == 1));
        }
        bind_var(&mut psik, 0, c);
    }
    // ---- j-rounds. ----
    // Per-component per-cycle weights (the slice-local products).
    let mut compw: Vec<Vec<Gf128>> = components
        .iter()
        .enumerate()
        .map(|(i, comp)| {
            let ad = comp.addr_vars;
            (0..cycles)
                .map(|u| {
                    let mut acc = ONE;
                    for b in 0..ad {
                        acc = mul(
                            acc,
                            eqb(challenges[i * ad + b], (q[u] >> (i * ad + b)) & 1 == 1),
                        );
                    }
                    acc
                })
                .collect()
        })
        .collect();
    let psi_final = psik[0];
    let mut eqt = eq_c;
    for _ in 0..n {
        let half = eqt.len() / 2;
        let mut coeffs = vec![ZERO; ncoef];
        for i in 0..half {
            let e0 = eqt[i << 1];
            let es = e0 + eqt[(i << 1) | 1];
            // Π_f (f0 + X·fs)·psi — degree d+1.
            let mut poly = vec![psi_final];
            for f in compw.iter_mut() {
                let f0 = f[i << 1];
                let fs = f0 + f[(i << 1) | 1];
                let mut next = vec![ZERO; poly.len() + 1];
                for (k, &c) in poly.iter().enumerate() {
                    next[k] += mul(c, f0);
                    next[k + 1] += mul(c, fs);
                }
                poly = next;
            }
            for (k, &c) in poly.iter().enumerate().take(ncoef) {
                coeffs[k] += mul(e0, c);
                if k + 1 < ncoef {
                    coeffs[k + 1] += mul(es, c);
                }
            }
        }
        let e = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs);
        challenges.push(e);
        bind_var(&mut eqt, 0, e);
        for f in compw.iter_mut() {
            bind_var(f, 0, e);
        }
    }
    // The per-component claims at (r_k^{(i)}, r_j).
    let component_claims: Vec<PcsPoint> = (0..d)
        .map(|i| {
            let ad = components[i].addr_vars;
            let mut pt = challenges[i * ad..(i + 1) * ad].to_vec();
            pt.extend_from_slice(&challenges[a..]);
            PcsPoint::new(pt, compw[i][0])
        })
        .collect();
    for c in &component_claims {
        t.absorb_field(&format!("{label}-comp"), &[c.value]);
    }
    (
        claim,
        PsiProof {
            rounds,
            component_claims,
        },
    )
}

// ---------------------------------------------------------------------------
// The linear selector check (discharge::prove_selector_shout).
// ---------------------------------------------------------------------------

use crate::discharge::{SelectorCheckSpec, SelectorShoutSpec};

/// One selector factor's linear run: the round messages and the component
/// claims (the same shape the dense prover emits).
pub struct SelectorLinearRun {
    pub rounds: Vec<Vec<Gf128>>,
    pub component_claims: Vec<PcsPoint>,
}

/// The linear selector sum-check: `Σ_{k,h,j} Πcomps(k,j)·eq(r_j,j)·
/// eq(r_h,h)·sel(h,k) = claim` over the joint `(k | h | j)` layout.
/// `d = 1` (the fetch component).
pub fn prove_selector_linear(
    spec: &SelectorShoutSpec,
    f: &SelectorCheckSpec,
    comp: &AddressComponent,
    label: &str,
    t: &mut Transcript,
) -> SelectorLinearRun {
    let (a, n) = (spec.a, spec.n);
    let hv = f.h_vars;
    let cycles = 1usize << n;
    let q = addr_bits(std::slice::from_ref(comp));
    let eq_c = magnetar_field::EqBits::build(&spec.r_j).values;
    // SELH(k) = Σ_h eq(r_h, h)·sel(h, k) — folded at the challenges.
    let eq_h = magnetar_field::EqBits::build(&f.r_h).values;
    let mut selh: Vec<Gf128> = (0..(1usize << a))
        .map(|k| {
            let mut acc = ZERO;
            for h in 0..(1usize << hv) {
                acc += mul(eq_h[h], f.rom_sel[h | (k << hv)]);
            }
            acc
        })
        .collect();
    let mut raeq = vec![ONE; cycles];
    let jv = a + hv + n;
    let mut rounds: Vec<Vec<Gf128>> = Vec::with_capacity(jv);
    let mut challenges: Vec<Gf128> = Vec::with_capacity(jv);
    let absorb = |t: &mut Transcript, coeffs: &[Gf128; 5], label: &str| -> Gf128 {
        let mut buf = Vec::with_capacity(coeffs.len() * 16);
        for c in coeffs {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        t.challenge_field(label)
    };
    // ---- k-rounds. ----
    for bit in 0..a {
        let shift = bit + 1;
        let mut coeffs = [ZERO; 5];
        for u in 0..cycles {
            let a_j = mul(
                mul(eq_c[u], raeq[u]),
                ONE + Gf128(((q[u] >> bit) & 1) as u128),
            );
            let b_j = mul(eq_c[u], raeq[u]);
            let s0 = selh[(q[u] >> shift) << 1];
            let s1 = selh[((q[u] >> shift) << 1) | 1];
            coeffs[0] += mul(a_j, s0);
            coeffs[1] += mul(a_j, s0 + s1) + mul(b_j, s0);
            coeffs[2] += mul(b_j, s0 + s1);
        }
        if rounds.is_empty() {
            debug_assert_eq!(
                coeffs[1] + coeffs[2] + coeffs[3] + coeffs[4],
                f.claim,
                "selector round 0 must sum to the claim"
            );
        }
        let c = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs.to_vec());
        challenges.push(c);
        for u in 0..cycles {
            raeq[u] = mul(raeq[u], eqb(c, (q[u] >> bit) & 1 == 1));
        }
        bind_var(&mut selh, 0, c);
    }
    // ---- h-rounds. ----
    let raw = raeq.clone();
    let sj: Gf128 = eq_c.iter().zip(raw.iter()).map(|(&e, &r)| mul(e, r)).sum();
    // selr(h) = sel-MLE(h, r_k): fold the sel table's k-dim (high bits).
    let mut selr = f.rom_sel.clone();
    for &c in &challenges[..a] {
        bind_var(&mut selr, hv, c);
    }
    debug_assert_eq!(selr.len(), 1usize << hv);
    let mut eqhv = ONE;
    for b in 0..hv {
        let eq_rest = magnetar_field::EqBits::build(&f.r_h[b + 1..]).values;
        let mut s0 = ZERO;
        let mut s1 = ZERO;
        for (i, &e) in eq_rest.iter().enumerate() {
            s0 += mul(e, selr[i << 1]);
            s1 += mul(e, selr[(i << 1) | 1]);
        }
        let a_const = ONE + f.r_h[b];
        // (A + X)·(S0 + X·(S0+S1)) — expanded directly.
        let d0 = mul(sj, s0);
        let d1 = mul(sj, s0 + s1);
        let coeffs = [
            mul(eqhv, mul(a_const, d0)),
            mul(eqhv, d0 + mul(a_const, d1)),
            mul(eqhv, d1),
            ZERO,
            ZERO,
        ];
        let d = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs.to_vec());
        challenges.push(d);
        eqhv = mul(eqhv, ONE + f.r_h[b] + d);
        bind_var(&mut selr, 0, d);
    }
    // ---- j-rounds. ----
    let const_h = mul(eqhv, selr[0]);
    let mut eqt = eq_c;
    let mut rawt = raw;
    for _ in 0..n {
        let half = eqt.len() / 2;
        let mut coeffs = [ZERO; 5];
        for i in 0..half {
            let e0 = eqt[i << 1];
            let es = e0 + eqt[(i << 1) | 1];
            let r0 = rawt[i << 1];
            let rs = r0 + rawt[(i << 1) | 1];
            coeffs[0] += mul(mul(e0, r0), const_h);
            coeffs[1] += mul(mul(e0, rs) + mul(es, r0), const_h);
            coeffs[2] += mul(mul(es, rs), const_h);
        }
        let e = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs.to_vec());
        challenges.push(e);
        bind_var(&mut eqt, 0, e);
        bind_var(&mut rawt, 0, e);
    }
    let mut pt = challenges[..a].to_vec();
    pt.extend_from_slice(&challenges[a + hv..]);
    SelectorLinearRun {
        rounds,
        component_claims: vec![PcsPoint::new(pt, rawt[0])],
    }
}

// ---------------------------------------------------------------------------
// The linear cross-step transition argument (cross_step::prove_cross_step's
// S1 sum-check).
// ---------------------------------------------------------------------------

use crate::cross_step::CrossStepSpec;

/// The linear S1 sum-check: `Σ_{k,k',t,h,j} A(k,j)·ASH(k',j)·L_t(t,j)·
/// L_h(h,j)·eq(r_j,j)·(α₀ + γ·TT(k,k',t,h)) = α₀ + γ` over the joint
/// `(k | k' | t | h | j)` layout, bound in increasing order. Round messages
/// byte-identical to the sparse prover's; every round is `O(T)` via the
/// one-hot kills (A over `k`, ASH over `k'`) and the tiny folded TT table.
///
/// Returns `(rounds, challenges, [a_claim, ash_claim, lt_claim, lh_claim])`.
#[allow(clippy::too_many_arguments)]
pub fn prove_cross_step_s1_linear(
    spec: &CrossStepSpec,
    fetch: &AddressComponent,
    taken: &[Gf128],
    halt: &[Gf128],
    alpha0: Gf128,
    gamma: Gf128,
    r_j: &[Gf128],
    label: &str,
    t: &mut Transcript,
) -> (Vec<Vec<Gf128>>, Vec<Gf128>, [Gf128; 4]) {
    let a = spec.a;
    let n = spec.n;
    let cycles = 1usize << n;
    let q = addr_bits(std::slice::from_ref(fetch));
    // The shifted address trace (ASH's one-hot per cycle).
    let nq: Vec<usize> = (0..cycles)
        .map(|j| {
            if j + 1 < cycles {
                q[j + 1]
            } else {
                match spec.next_pc_public {
                    Some(pc) => pc,
                    None => q[j],
                }
            }
        })
        .collect();
    let eq_c = magnetar_field::EqBits::build(r_j).values;
    let mut tt = spec.tt.clone();
    let mut raeq = vec![ONE; cycles];
    let mut asheq = vec![ONE; cycles];
    let jv = 2 * a + 2 + n;
    let ncoef = 7; // 4 dense + 2 public (the engine's degree cap)
    let mut rounds: Vec<Vec<Gf128>> = Vec::with_capacity(jv);
    let mut challenges: Vec<Gf128> = Vec::with_capacity(jv);
    let absorb = |t: &mut Transcript, coeffs: &[Gf128], label: &str| -> Gf128 {
        let mut buf = Vec::with_capacity(coeffs.len() * 16);
        for c in coeffs {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        t.challenge_field(label)
    };
    // The (t, h)-summed TT value with the link weights, at an explicit
    // low-part index and bit layout: `idx = low | (t << t_off) |
    // (h << (t_off + 1))`.
    let tt_sum = |tt: &[Gf128], low: usize, t_off: usize, j: usize| -> Gf128 {
        let mut acc = ZERO;
        for h in 0..2usize {
            for tb in 0..2usize {
                let lt = ONE + Gf128(tb as u128) + taken[j];
                let lh = ONE + Gf128(h as u128) + halt[j];
                acc += mul(mul(lt, lh), tt[low | (tb << t_off) | (h << (t_off + 1))]);
            }
        }
        acc
    };

    // ---- k-rounds (the A kill; the k'-sum kills ASH inline). ----
    for bit in 0..a {
        let krem_bits = a - bit; // TT's remaining k width (incl. round bit)
        let mut coeffs = vec![ZERO; ncoef];
        for u in 0..cycles {
            let a_j = mul(
                mul(eq_c[u], raeq[u]),
                ONE + Gf128(((q[u] >> bit) & 1) as u128),
            );
            let b_j = mul(eq_c[u], raeq[u]);
            let base = ((q[u] >> (bit + 1)) << 1) | (nq[u] << krem_bits);
            let t_off = krem_bits + a;
            let t0 = tt_sum(&tt, base, t_off, u);
            let t1 = tt_sum(&tt, base | 1, t_off, u);
            coeffs[0] += mul(a_j, alpha0 + mul(gamma, t0));
            coeffs[1] += mul(a_j, mul(gamma, t0 + t1)) + mul(b_j, alpha0 + mul(gamma, t0));
            coeffs[2] += mul(b_j, mul(gamma, t0 + t1));
        }
        // NOTE: no round-0 self-check here — the tamper tests call this
        // prover with deliberately inconsistent inputs (the corrupted
        // one-hot), where the round sum legitimately misses the claim;
        // the verifier's rejection is the test. (A debug_assert here
        // fired on `cross_step_tamper` inputs — fixed by removal.)
        let c = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs);
        challenges.push(c);
        for u in 0..cycles {
            raeq[u] = mul(raeq[u], eqb(c, (q[u] >> bit) & 1 == 1));
        }
        bind_var(&mut tt, 0, c);
    }
    let a_row = raeq; // A-MLE(r_k, j)
                      // ---- k'-rounds (the ASH kill). ----
    for bit in 0..a {
        let krem_bits = a - bit;
        let mut coeffs = vec![ZERO; ncoef];
        for u in 0..cycles {
            let a_j = mul(
                mul(mul(eq_c[u], a_row[u]), asheq[u]),
                ONE + Gf128(((nq[u] >> bit) & 1) as u128),
            );
            let b_j = mul(mul(eq_c[u], a_row[u]), asheq[u]);
            let base = (nq[u] >> (bit + 1)) << 1;
            let t0 = tt_sum(&tt, base, krem_bits, u);
            let t1 = tt_sum(&tt, base | 1, krem_bits, u);
            coeffs[0] += mul(a_j, alpha0 + mul(gamma, t0));
            coeffs[1] += mul(a_j, mul(gamma, t0 + t1)) + mul(b_j, alpha0 + mul(gamma, t0));
            coeffs[2] += mul(b_j, mul(gamma, t0 + t1));
        }
        let c = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs);
        challenges.push(c);
        for u in 0..cycles {
            asheq[u] = mul(asheq[u], eqb(c, (nq[u] >> bit) & 1 == 1));
        }
        bind_var(&mut tt, 0, c);
    }
    let ash_row = asheq; // ASH-MLE(r_k', j)
                         // ---- the t-round. ----
    let lt_row;
    {
        // Only h is summed here (t is the round variable): the TT slice
        // at t = β weighted by L_h.
        let hsum = |tt: &[Gf128], low: usize, j: usize| -> Gf128 {
            let mut acc = ZERO;
            for h in 0..2usize {
                let lh = ONE + Gf128(h as u128) + halt[j];
                acc += mul(lh, tt[low | (h << 1)]);
            }
            acc
        };
        let mut coeffs = vec![ZERO; ncoef];
        for u in 0..cycles {
            // L_t's MLE in t: (1 + taken) + X.
            let a_j = mul(mul(mul(eq_c[u], a_row[u]), ash_row[u]), ONE + taken[u]);
            let b_j = mul(mul(mul(eq_c[u], a_row[u]), ash_row[u]), ONE);
            let t0 = hsum(&tt, 0, u);
            let t1 = hsum(&tt, 1, u);
            coeffs[0] += mul(a_j, alpha0 + mul(gamma, t0));
            coeffs[1] += mul(a_j, mul(gamma, t0 + t1)) + mul(b_j, alpha0 + mul(gamma, t0));
            coeffs[2] += mul(b_j, mul(gamma, t0 + t1));
        }
        let c = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs);
        challenges.push(c);
        bind_var(&mut tt, 0, c);
        lt_row = (0..cycles).map(|u| ONE + c + taken[u]).collect::<Vec<_>>();
    }
    // ---- the h-round. ----
    let lh_row;
    {
        let mut coeffs = vec![ZERO; ncoef];
        for u in 0..cycles {
            let w = mul(mul(mul(eq_c[u], a_row[u]), ash_row[u]), lt_row[u]);
            // L_h's MLE in h: (1 + halt) + X.
            let a_j = mul(w, ONE + halt[u]);
            let b_j = w;
            let t0 = tt[0];
            let t1 = tt[1];
            coeffs[0] += mul(a_j, alpha0 + mul(gamma, t0));
            coeffs[1] += mul(a_j, mul(gamma, t0 + t1)) + mul(b_j, alpha0 + mul(gamma, t0));
            coeffs[2] += mul(b_j, mul(gamma, t0 + t1));
        }
        let c = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs);
        challenges.push(c);
        bind_var(&mut tt, 0, c);
        lh_row = (0..cycles).map(|u| ONE + c + halt[u]).collect::<Vec<_>>();
    }
    let ttval = tt[0];
    // ---- j-rounds: eq_c × the four per-cycle factor vectors × const. ----
    let const_h = alpha0 + mul(gamma, ttval);
    let mut eqt = eq_c;
    let mut aw = a_row;
    let mut sw = ash_row;
    let mut ltw = lt_row;
    let mut lhw = lh_row;
    for _ in 0..n {
        let half = eqt.len() / 2;
        let mut coeffs = vec![ZERO; ncoef];
        for i in 0..half {
            let e0 = eqt[i << 1];
            let es = e0 + eqt[(i << 1) | 1];
            // Π (f0 + X·fs) over the four factor vectors.
            let mut poly = [ONE, ZERO, ZERO, ZERO, ZERO];
            for f in [&aw, &sw, &ltw, &lhw] {
                let f0 = f[i << 1];
                let fs = f0 + f[(i << 1) | 1];
                let mut next = [ZERO; 5];
                for (e, &pv) in poly.iter().enumerate() {
                    if pv == ZERO {
                        continue;
                    }
                    next[e] += mul(pv, f0);
                    next[e + 1] += mul(pv, fs);
                }
                poly = next;
            }
            coeffs[0] += mul(mul(e0, poly[0]), const_h);
            coeffs[1] += mul(mul(e0, poly[1]) + mul(es, poly[0]), const_h);
            coeffs[2] += mul(mul(e0, poly[2]) + mul(es, poly[1]), const_h);
            coeffs[3] += mul(mul(e0, poly[3]) + mul(es, poly[2]), const_h);
            coeffs[4] += mul(mul(e0, poly[4]) + mul(es, poly[3]), const_h);
            coeffs[5] += mul(mul(es, poly[4]), const_h);
        }
        let e = absorb(&mut *t, &coeffs, label);
        rounds.push(coeffs);
        challenges.push(e);
        bind_var(&mut eqt, 0, e);
        bind_var(&mut aw, 0, e);
        bind_var(&mut sw, 0, e);
        bind_var(&mut ltw, 0, e);
        bind_var(&mut lhw, 0, e);
    }
    (rounds, challenges, [aw[0], sw[0], ltw[0], lhw[0]])
}

// ---------------------------------------------------------------------------
// The linear sum-to-one prover (the twist st1 sum-check, one-hot-kill form).
// ---------------------------------------------------------------------------

/// The **linear sum-to-one prover**: proves
/// `Σ_{k,j} eq(ρ, j)·comp(k, j) = 1` over the joint `(k | j)` cube
/// (address bits low, cycle bits high, bound in increasing order) for a
/// component that is **one-hot per cycle** (`comp(k, j) = [k == addr(j)]`).
///
/// The k-rounds use the one-hot kill: summing over the address cube picks
/// the addressed cell per cycle, so each round is an O(T) scan — the
/// unbound address bits collapse to `Σ_b eq(X_b, q_b) = 1` — and the
/// j-rounds run over T-length vectors. Never a `2^{ad+n}` materialization.
///
/// The round messages are byte-identical to the generic dense engine's
/// (cross-validated by `linear_st1_matches_engine`).
///
/// Returns `(rounds, point, component_claim)` where the claim is the
/// component's MLE at the full point `(r_k, r_j)`.
pub fn prove_st1_linear(
    addr: &[usize],
    ad: usize,
    rho: &[Gf128],
    t: &mut Transcript,
    label: &str,
) -> (Vec<Vec<Gf128>>, Vec<Gf128>, Gf128) {
    let n = rho.len();
    let cycles = addr.len();
    debug_assert_eq!(cycles, 1usize << n);
    let mut rounds: Vec<Vec<Gf128>> = Vec::with_capacity(ad + n);
    let mut challenges: Vec<Gf128> = Vec::with_capacity(ad + n);
    let mut eqt = magnetar_field::EqBits::build(rho).values; // eq(ρ, ·)
    let mut w = vec![ONE; cycles]; // Π_{bound k-bits} eq(r_b, q_b(j))
    let absorb = |t: &mut Transcript, coeffs: &[Gf128]| -> Gf128 {
        let mut buf = Vec::with_capacity(coeffs.len() * 16);
        for c in coeffs {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        t.challenge_field(label)
    };
    // ---- k-rounds: the one-hot kill. ----
    // The summand is eq(ρ,j)·comp(k,j): at a k-round the public factor is
    // constant (its pair coefficient is zero), so the engine's message is
    // [S0, S0+S1, 0] with S_b = Σ_{q_bit=b} eq·w — degree-2 vector shape,
    // byte-identical to the generic engine's.
    for bit in 0..ad {
        let mut s0 = ZERO;
        let mut s1 = ZERO;
        for u in 0..cycles {
            let v = mul(eqt[u], w[u]);
            if (addr[u] >> bit) & 1 == 0 {
                s0 += v;
            } else {
                s1 += v;
            }
        }
        let coeffs = [s0, s0 + s1, ZERO];
        if rounds.is_empty() {
            debug_assert_eq!(
                coeffs[1] + coeffs[2],
                ONE,
                "st1 round 0 must sum to the claim"
            );
        }
        let c = absorb(t, &coeffs);
        rounds.push(coeffs.to_vec());
        challenges.push(c);
        // Fold the per-cycle address product at the round challenge.
        for u in 0..cycles {
            w[u] = mul(w[u], eqb(c, (addr[u] >> bit) & 1 == 1));
        }
    }
    // ---- j-rounds: two T-vectors (eq(ρ,·) and the comp row at r_k). ----
    // Pair coefficient convention (a, a+t1): c0 = e0·w0, c1 = e0·ws +
    // es·w0, c2 = es·ws with es = e0+e1, ws = w0+w1.
    for _ in 0..n {
        let half = eqt.len() / 2;
        let mut coeffs = [ZERO; 3];
        for i in 0..half {
            let e0 = eqt[i << 1];
            let es = e0 + eqt[(i << 1) | 1];
            let w0 = w[i << 1];
            let ws = w0 + w[(i << 1) | 1];
            coeffs[0] += mul(e0, w0);
            coeffs[1] += mul(e0, ws) + mul(es, w0);
            coeffs[2] += mul(es, ws);
        }
        let c = absorb(t, &coeffs);
        rounds.push(coeffs.to_vec());
        challenges.push(c);
        bind_var(&mut eqt, 0, c);
        bind_var(&mut w, 0, c);
    }
    (rounds, challenges, w[0])
}

#[cfg(test)]
mod st1_tests {
    use super::*;
    use magnetar_sumcheck::{BindOrder, SumcheckProver, SumcheckSpec, Term};

    /// The linear st1 prover emits **byte-identical** round messages and
    /// challenges to the generic dense engine's for the same statement,
    /// with the same terminal component claim — the cross-validation that
    /// gates its use in the protocol.
    #[test]
    fn linear_st1_matches_engine() {
        use rand::RngExt;
        let mut rng = rand::rng();
        for &(ad, n) in &[(3usize, 4usize), (6, 5), (8, 4)] {
            let cycles = 1usize << n;
            // A random one-hot component (exactly one set bit per cycle).
            let mut b = vec![false; (1usize << ad) * cycles];
            for j in 0..cycles {
                b[rng.random_range(0..(1usize << ad)) + j * (1usize << ad)] = true;
            }
            let comp = crate::AddressComponent::from_bits(ad, n, &b);
            let addr = addr_bits(std::slice::from_ref(&comp));
            // Two identically-seeded transcripts: rho sampled the same way.
            let rho = {
                let mut t = magnetar_hash::Transcript::new("st1", &[b"test"]);
                t.challenge_field_vec("rho", n)
            };
            // ---- the generic dense engine (the old path). ----
            let (e_rounds, e_pt, e_claim) = {
                let eq_rho = magnetar_field::EqBits::build(&rho).values;
                let kj_vars = ad + n;
                let mut pubf = vec![ZERO; 1usize << kj_vars];
                for idx in 0..pubf.len() {
                    let j = idx >> ad;
                    pubf[idx] = eq_rho[j];
                }
                let spec = SumcheckSpec {
                    num_vars: kj_vars,
                    dense: vec![comp.table.clone()],
                    public: vec![Box::new(magnetar_sumcheck::TableFactor::new(pubf))],
                    terms: vec![Term {
                        coeff: ONE,
                        dense: vec![0],
                        public: vec![0],
                    }],
                    claim: ONE,
                };
                let mut pr = SumcheckProver::new(&spec, BindOrder::Increasing);
                let mut rounds = Vec::new();
                let mut t = magnetar_hash::Transcript::new("st1", &[b"test"]);
                t.challenge_field_vec("rho", n); // consume rho identically
                for _ in 0..kj_vars {
                    let msg = pr.message();
                    rounds.push(msg.0.clone());
                    pr.absorb_and_fold(&mut t, "st1");
                }
                let pt = pr.challenges();
                let (claims, _) = pr.finish();
                (rounds, pt, claims[0])
            };
            // ---- the linear prover. ----
            let mut t = magnetar_hash::Transcript::new("st1", &[b"test"]);
            t.challenge_field_vec("rho", n);
            let (rounds, pt, claim) = prove_st1_linear(&addr, ad, &rho, &mut t, "st1");
            assert_eq!(rounds.len(), e_rounds.len(), "round count (ad={ad}, n={n})");
            for (r, e) in rounds.iter().zip(e_rounds.iter()) {
                assert_eq!(r, e, "round message (ad={ad}, n={n})");
            }
            assert_eq!(pt, e_pt, "challenge point (ad={ad}, n={n})");
            assert_eq!(claim, e_claim, "component claim (ad={ad}, n={n})");
        }
    }
}
