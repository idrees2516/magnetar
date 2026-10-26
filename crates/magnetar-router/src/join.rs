//! The **join-enumeration routing prover** (paper §5.4's linear-in-T
//! prover work): `O(c·T)` field operations per routing sum-check, replacing
//! the sparse prover's `O(2^{joint_vars})` round messages.
//!
//! The joint variable layout binds **`a | o | h | u | j`** (the cycle index
//! last). Every round message is computed by enumerating the trace
//! **column by column** (one column per cycle), never the joint cube:
//!
//! * **a-rounds** (stores only) — the offset-link factors are affine in
//!   `a`; the prover keeps a per-column `τ` vector over the remaining `a`
//!   bits (the eq-weighted, one-hot-collapsed `R` contribution) and folds
//!   it algebraically, mirroring `R`'s own fold over `a`.
//! * **o-rounds** — iterate the *sparse support pairs* of the selector
//!   position's `R`-slice (a routing matrix has `O(2^o)` nonzeros per
//!   position). Each pair carries its a-folded coefficient, an eq-prefix
//!   product bound at previous challenges, and the round's suffix eq
//!   product; the message interpolates the pair's `o_k` bit against the
//!   eq factor's round value.
//! * **h-rounds** — the one-hot selector *kills* the `h` dimension: each
//!   column contributes one interpolated `R` slice at its selector
//!   position's remaining bits.
//! * **u-rounds** — the global (o,a,h)-folded `R` vector dots against each
//!   column's `X` slice; both fold in lockstep (the `X` table folds as a
//!   single `(u, j)` table across all columns).
//! * **j-rounds** — per-cycle factor vectors (the eq, the `X` j-part, the
//!   selector j-parts and the link j-parts) pair-fold with per-column
//!   polynomial accumulation.
//!
//! Zero-selector columns are inert (they contribute no message in the a/o
//! rounds) but their selector factors are identically zero, so their
//! selector-claim vectors carry ZERO and every later contribution
//! vanishes; their `X` slice still folds — the terminal `X` claim is the
//! folded MLE over **all** cycles.
//!
//! All phases are parallel over column chunks with per-chunk coefficient
//! accumulators combined by `+` (docs/PERFORMANCE.md R1/R3/R8) — the round
//! messages and terminal claims are bit-identical to the sparse prover's
//! at the same layout (cross-validated by `join_matches_sparse_prover`).

use crate::{RouterProof, RoutingLayout};
use magnetar_field::{bind_var, mul, par_bind_var, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use rayon::prelude::*;

/// The public routing tensor `R` in its own variable space (little-endian
/// `o | u<<o | h<<(o+u) | a<<(o+u+h)`), as built by the tensor builders.
#[derive(Clone, Debug)]
pub struct RoutingTensor {
    pub values: Vec<Gf128>,
    pub o_vars: usize,
    pub u_vars: usize,
    pub h_vars: usize,
    pub a_vars: usize,
}

impl RoutingTensor {
    /// Construct from the builders' `(values, o, u, h, a)` tuple shape.
    pub fn from_parts(values: Vec<Gf128>, o: usize, u: usize, h: usize, a: usize) -> Self {
        RoutingTensor {
            values,
            o_vars: o,
            u_vars: u,
            h_vars: h,
            a_vars: a,
        }
    }

    /// The nonzero support grouped per `(h, a)` position:
    /// `support[h | a<<h_vars] = [(o, u, coef)]`.
    fn support_by_ha(&self) -> Vec<Vec<(usize, usize, Gf128)>> {
        let (o_v, u_v, h_v, a_v) = (self.o_vars, self.u_vars, self.h_vars, self.a_vars);
        let mut support = vec![Vec::new(); 1usize << (h_v + a_v)];
        for idx in 0..self.values.len() {
            let v = self.values[idx];
            if v == ZERO {
                continue;
            }
            let o = idx & ((1usize << o_v) - 1);
            let u = (idx >> o_v) & ((1usize << u_v) - 1);
            let h = (idx >> (o_v + u_v)) & ((1usize << h_v) - 1);
            let a = idx >> (o_v + u_v + h_v);
            support[h | (a << h_v)].push((o, u, v));
        }
        support
    }
}

/// One one-hot selector column: the table over `(h_f, j)`
/// (`table[h | j<<h_vars]`), at bit-offset `h_off` inside the joint `h`
/// block.
#[derive(Clone, Debug)]
pub struct SelCol {
    pub table: Vec<Gf128>,
    pub h_off: usize,
    pub h_vars: usize,
}

/// The join-enumeration witness.
#[derive(Clone, Debug)]
pub struct JoinWitness {
    /// `X(u, j)` over `(u, j)`: `x[u | j<<u_vars]`.
    pub x: Vec<Gf128>,
    /// The one-hot selector columns (factor order).
    pub selectors: Vec<SelCol>,
    /// The offset bits `â_k(j)` per offset variable (stores only).
    pub offset_bits: Vec<Vec<bool>>,
}

/// Prover-side per-column state.
struct Column {
    /// The cycle index (stable across chunking).
    j: usize,
    /// All selectors one-hot at this cycle.
    active: bool,
    /// The combined one-hot h position.
    h_pos: usize,
    /// Bound link product `Π_m (1 + â_m + c_{a,m})` (stores).
    lam: Gf128,
    /// Bound selector prefix products, per factor (the q-claim vectors).
    lamh_f: Vec<Gf128>,
    /// Product of `lamh_f` (ZERO for inert columns).
    lamh: Gf128,
    /// a-phase: the eq-weighted, h-collapsed R contribution over the
    /// remaining a bits (empty for shifts).
    tau: Vec<Gf128>,
    /// o-phase: the sparse pairs at this column's selector position
    /// (the generic path; the binary-tensor fast path replaces them
    /// with the per-o `xo2` dots).
    pairs: Vec<Pair>,
    /// The column's X slice packed as bits (`(u_v + 63) / 64` words) —
    /// during the h-rounds the X table is still raw bits, so the row
    /// contraction is a popcount-driven select, never a multiply.
    x_bits: Vec<u64>,
    /// The binary-tensor fast path's per-o X-selected sums:
    /// `xo2[o] = Σ_{pairs at o} coef·X(u)` — one pass over the support
    /// (the X entries are bits during the (a, o, h)-rounds), replacing
    /// the per-pair per-round products of the generic path.
    xo2: Vec<Gf128>,
    /// The nonzero entries of `xo2` (the per-round dots skip zeros).
    xo_nz: Vec<u32>,
}

struct Pair {
    o: usize,
    u: usize,
    /// The a-folded coefficient.
    coef: Gf128,
    /// The eq-prefix product `eq(c_{o,<k}, o_{<k})`.
    pref: Gf128,
}

/// Round-message coefficient cap (the largest routing term degree in this
/// stack is 9; all messages are padded to `degree + 1` coefficients).
const MAXC: usize = 12;

type Coeffs = [Gf128; MAXC];

/// A tiny `MAGNETAR_TIME`-gated phase timer (the same pattern as the
/// zkvm crate's `PhaseTimer`, local to this crate to keep the dependency
/// direction one-way).
pub(crate) struct JoinTimer(Option<(std::time::Instant, &'static str)>);

impl JoinTimer {
    pub(crate) fn new(label: &'static str) -> Self {
        JoinTimer(
            std::env::var_os("MAGNETAR_TIME")
                .map(|_| (std::time::Instant::now(), label)),
        )
    }
}

impl Drop for JoinTimer {
    fn drop(&mut self) {
        if let Some((t0, label)) = self.0.take() {
            eprintln!("[time] {label:<24} {:7.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
    }
}

#[inline]
fn zero_coeffs() -> Coeffs {
    [ZERO; MAXC]
}

fn combine(a: Coeffs, b: Coeffs) -> Coeffs {
    let mut c = a;
    for k in 0..MAXC {
        c[k] += b[k];
    }
    c
}

/// The coefficients of `(1 + r + t)·(v0 + (v0 + v1)·t)·s` — the generic
/// two-factor round contribution (an affine factor times an affine
/// factor, both in the round variable `t`), scaled by `s`.
#[inline]
fn quad_term(r: Gf128, v0: Gf128, v1: Gf128, s: Gf128) -> [Gf128; 3] {
    let w0 = mul(s, v0);
    let w1 = mul(s, v1);
    let sum = w0 + w1;
    let one_r = ONE + r;
    [mul(one_r, w0), mul(one_r, sum) + w0, sum]
}

/// `eq(x, b)` for a Boolean `b` — the branch-free form `1 + x + b` (R4).
#[inline]
fn eqb(x: Gf128, b: bool) -> Gf128 {
    x + Gf128(b as u128) + ONE
}

/// The round-0 self-check: the first round message must sum to the stated
/// claim. In characteristic two `m(0) + m(1) = Σ_{j≥1} m_j` — the constant
/// coefficient cancels — matching the engine's round check
/// (`Σ_{j≥1} m_{i,j} = running`).
#[inline]
fn round0_check(rounds: &[Vec<Gf128>], coeffs: &Coeffs, degree: usize, claim: Gf128) {
    if rounds.is_empty() {
        debug_assert_eq!(
            coeffs[1..degree + 1].iter().copied().sum::<Gf128>(),
            claim,
            "join prover round 0 must sum to the claim"
        );
    }
}

fn absorb_round(t: &mut Transcript, coeffs: &Coeffs, degree: usize) -> Gf128 {
    let mut buf = Vec::with_capacity((degree + 1) * 16);
    for c in &coeffs[..degree + 1] {
        buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
    }
    t.absorb_bytes("router", &buf);
    t.challenge_field("router")
}

/// Column-chunk size for the parallel phases (R2).
fn chunk_size(n: usize) -> usize {
    if n < magnetar_field::PAR_THRESHOLD {
        n.max(1)
    } else {
        (n / (rayon::current_num_threads().max(1) * 4)).max(1)
    }
}

/// The join-enumeration routing prover. Proves the same statement as the
/// sparse prover at the same joint layout — identical round messages and
/// terminal claims.
#[allow(clippy::too_many_arguments)]
pub fn prove_routing_join(
    layout: &RoutingLayout,
    claim: Gf128,
    r_o: &[Gf128],
    r_j: &[Gf128],
    tensor: &RoutingTensor,
    witness: &JoinWitness,
    t: &mut Transcript,
) -> RouterProof {
    let (a_v, o_v, h_v, u_v, n) = (
        layout.a_vars,
        layout.o_vars,
        layout.h_vars,
        layout.u_vars,
        layout.j_vars,
    );
    let cycles = 1usize << n;
    debug_assert_eq!(witness.x.len(), cycles << u_v);
    debug_assert_eq!(r_o.len(), o_v);
    debug_assert_eq!(r_j.len(), n);
    let n_sel = witness.selectors.len();
    let n_off = witness.offset_bits.len();
    let degree = 1 + n_sel + n_off + 3;
    let mut rounds: Vec<Vec<Gf128>> = Vec::with_capacity(layout.joint_vars());
    let mut challenges: Vec<Gf128> = Vec::with_capacity(layout.joint_vars());

    // ---- Shared tables (R6: computed once). ----
    let eq_o = magnetar_field::EqBits::build(r_o).values;
    let w: Vec<Gf128> = magnetar_field::EqBits::build(r_j).values;
    let support = tensor.support_by_ha();
    // The binary-tensor fast path: the routing tensors of this stack are
    // 0/1-valued, so every X entry select is a bit test — the extraction's
    // tau sums become XOR-selects of the GLOBAL eq_o[o] (no CLMUL), and
    // the o-rounds become per-o dots of the per-column `xo2` vectors
    // against per-round GLOBAL weight vectors (the per-pair pref/suffix
    // products collapse into the fold history). Non-binary tensors keep
    // the generic per-pair path (byte-identical, cross-validated).
    let tensor_binary = tensor.values.iter().all(|&v| v == ZERO || v == ONE);

    // ---- Per-column extraction (one pass). ----
    let _tj_extract = JoinTimer::new("jr-extract");
    let words_per_col = (1usize << u_v).div_ceil(64); // 2^u_v slots per column
    let col_len = 1usize << u_v;
    let mut cols: Vec<Column> = Vec::with_capacity(cycles);
    {
        let x = &witness.x;
        for j in 0..cycles {
            let mut active = true;
            let mut h_pos = 0usize;
            let mut lamh_f = Vec::with_capacity(n_sel);
            for sel in &witness.selectors {
                let hv = sel.h_vars;
                let mut pos = None;
                for h in 0..(1usize << hv) {
                    if sel.table[h | (j << hv)] != ZERO {
                        pos = Some(h);
                        break;
                    }
                }
                match pos {
                    Some(p) => {
                        h_pos |= p << sel.h_off;
                        lamh_f.push(ONE);
                    }
                    None => {
                        active = false;
                        lamh_f.push(ZERO);
                    }
                }
            }
            let x_col = &x[(j << u_v)..((j + 1) << u_v)];
            // Pack the column's X bits FIRST — the extraction's fast path
            // and the h-rounds both read the words.
            let mut x_bits = vec![0u64; words_per_col];
            debug_assert!(x_col.iter().all(|&v| v == ZERO || v == ONE));
            for (w, slot) in x_bits.iter_mut().enumerate() {
                for b in 0..64usize {
                    let in_col = (w << 6) + b;
                    if in_col < col_len && x_col[in_col] == ONE {
                        *slot |= 1u64 << b;
                    }
                }
            }
            let bit_at = |u: usize| -> bool { (x_bits[u >> 6] >> (u & 63)) & 1 == 1 };
            let mut tau = Vec::new();
            if a_v > 0 {
                tau = vec![ZERO; 1usize << a_v];
                if active {
                    for a in 0..(1usize << a_v) {
                        let list = &support[h_pos | (a << h_v)];
                        let mut acc = ZERO;
                        if tensor_binary {
                            // coef = 1: the pair's product is the bit
                            // selecting the GLOBAL eq_o[o] — XOR only.
                            for &(po, pu, _) in list {
                                if bit_at(pu) {
                                    acc += eq_o[po];
                                }
                            }
                        } else {
                            for &(po, pu, coef) in list {
                                acc += mul(coef, mul(eq_o[po], x_col[pu]));
                            }
                        }
                        tau[a] = acc;
                    }
                }
            }
            cols.push(Column {
                j,
                active,
                h_pos,
                lam: ONE,
                lamh_f,
                lamh: if active { ONE } else { ZERO },
                tau,
                pairs: Vec::new(),
                x_bits,
                xo2: Vec::new(),
                xo_nz: Vec::new(),
            });
        }
    }
    drop(_tj_extract);
    let mut x_work = witness.x.clone();

    // ---- a-rounds: τ algebraic folding (stores only). ----
    let _tj_a = JoinTimer::new("jr-a");
    for k in 0..a_v {
        let obits = &witness.offset_bits[k];
        let coeffs = cols
            .par_chunks(chunk_size(cycles))
            .map(|chunk| {
                let mut local = zero_coeffs();
                for col in chunk.iter() {
                    if !col.active || col.tau.is_empty() {
                        continue;
                    }
                    let j = col.j;
                    let a_hat = Gf128(obits[j] as u128);
                    // P_b = Σ_{a_{>k}} Π_{m>k} link_m(a_m, j)·τ[b | a_{>k}<<1].
                    let rest = 1usize << (a_v - k - 1);
                    let mut p = [ZERO; 2];
                    for r in 0..rest {
                        let mut l = ONE;
                        for m in (k + 1)..a_v {
                            l = mul(
                                l,
                                eqb(
                                    Gf128(witness.offset_bits[m][j] as u128),
                                    (r >> (m - k - 1)) & 1 == 1,
                                ),
                            );
                        }
                        p[0] += mul(l, col.tau[r << 1]);
                        p[1] += mul(l, col.tau[(r << 1) | 1]);
                    }
                    let q = quad_term(a_hat, p[0], p[1], mul(w[j], col.lam));
                    local[0] += q[0];
                    local[1] += q[1];
                    local[2] += q[2];
                }
                local
            })
            .reduce(zero_coeffs, combine);
        round0_check(&rounds, &coeffs, degree, claim);
        let c = absorb_round(t, &coeffs, degree);
        rounds.push(coeffs[..degree + 1].to_vec());
        challenges.push(c);
        // λ *= (1 + â_k + c); τ folds bit 0 at c (mirroring R's fold).
        cols.par_chunks_mut(chunk_size(cycles)).for_each(|chunk| {
            for col in chunk.iter_mut() {
                if !col.tau.is_empty() {
                    let a_hat = Gf128(obits[col.j] as u128);
                    col.lam = mul(col.lam, ONE + a_hat + c);
                    bind_var(&mut col.tau, 0, c);
                }
            }
        });
    }
    drop(_tj_a);
    let _tj_xo2 = JoinTimer::new("jr-xo2");
    // The a-folded support per h (the o-phase pairs).
    let support_a: Vec<Vec<(usize, usize, Gf128)>> = if a_v > 0 {
        let ca = &challenges[..a_v];
        let eq_ca: Vec<Gf128> = (0..(1usize << a_v))
            .map(|a| {
                let mut v = ONE;
                for m in 0..a_v {
                    v = mul(v, eqb(ca[m], (a >> m) & 1 == 1));
                }
                v
            })
            .collect();
        (0..(1usize << h_v))
            .map(|h| {
                let mut out = Vec::new();
                for a in 0..(1usize << a_v) {
                    for &(o, u, coef) in &support[h | (a << h_v)] {
                        let folded = mul(coef, eq_ca[a]);
                        if folded != ZERO {
                            out.push((o, u, folded));
                        }
                    }
                }
                out
            })
            .collect()
    } else {
        Vec::new()
    };
    if tensor_binary {
        // The per-o X-selected sums — ONE pass over each column's support
        // (the X entries are still bits): `xo2[o] = Σ_{pairs at o} coef·bit`.
        // The o-rounds below dot these against GLOBAL per-round weight
        // vectors — the per-pair pref/suffix products are gone entirely.
        let obits = 1usize << o_v;
        cols.par_chunks_mut(chunk_size(cycles)).for_each(|chunk| {
            for col in chunk.iter_mut() {
                let mut xo2 = vec![ZERO; obits];
                if col.active {
                    let list: &[(usize, usize, Gf128)] = if a_v > 0 {
                        &support_a[col.h_pos]
                    } else {
                        &support[col.h_pos]
                    };
                    for &(o, u, coef) in list {
                        if (col.x_bits[u >> 6] >> (u & 63)) & 1 == 1 {
                            xo2[o] += coef;
                        }
                    }
                }
                let xo_nz = xo2
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v != ZERO)
                    .map(|(i, _)| i as u32)
                    .collect();
                col.xo2 = xo2;
                col.xo_nz = xo_nz;
            }
        });
    } else {
        // The generic path: per-pair state, initialized per column.
        if a_v > 0 {
            for col in cols.iter_mut() {
                if col.active {
                    col.pairs = support_a[col.h_pos]
                        .iter()
                        .map(|&(o, u, coef)| Pair {
                            o,
                            u,
                            coef,
                            pref: ONE,
                        })
                        .collect();
                }
            }
        } else {
            for col in cols.iter_mut() {
                if col.active {
                    col.pairs = support[col.h_pos]
                        .iter()
                        .map(|&(o, u, coef)| Pair {
                            o,
                            u,
                            coef,
                            pref: ONE,
                        })
                        .collect();
                }
            }
        }
        for col in cols.iter_mut() {
            for p in col.pairs.iter_mut() {
                p.pref = p.coef;
            }
        }
    }

    // ---- o-rounds: the o-dimension of R against the eq claim point. ----
    drop(_tj_xo2);
    let _tj_o = JoinTimer::new("jr-o");
    // The high-bit suffix `Π_{i>k} eq(r_o[i], o_i)` depends only on
    // `o >> (k+1)` and the FIXED claim point `r_o` — one table per round
    // (≈ 2^{o_v} entries total) instead of a product per pair per round.
    let suffix_tabs: Vec<Vec<Gf128>> = (0..o_v)
        .map(|k| {
            let width = 1usize << (o_v - k - 1);
            (0..width)
                .map(|hi| {
                    let mut v = ONE;
                    for i in (k + 1)..o_v {
                        v = mul(v, eqb(r_o[i], (hi >> (i - k - 1)) & 1 == 1));
                    }
                    v
                })
                .collect()
        })
        .collect();
    let mut e_o = ONE; // eq(r_o, c_{o,<k}) — multiplies every later round.
    if tensor_binary {
        // The fast path: per column, the pre-computed per-o sums `xo2`
        // dot against the GLOBAL round weight
        // `g[o] = hist[o]·suffix_k[o >> (k+1)]` (the o-fold challenge
        // history `hist` is global — it depends on no column data), one
        // multiply per NONZERO o per round instead of two per PAIR.
        let obits = 1usize << o_v;
        let mut hist = vec![ONE; obits];
        for k in 0..o_v {
            let rk = r_o[k];
            let suffix = &suffix_tabs[k];
            let g: Vec<Gf128> = (0..obits)
                .map(|o| mul(hist[o], suffix[o >> (k + 1)]))
                .collect();
            let gt = &g;
            let coeffs = cols
                .par_chunks(chunk_size(cycles))
                .map(|chunk| {
                    let mut local = zero_coeffs();
                    for col in chunk.iter() {
                        if !col.active {
                            continue;
                        }
                        let mut v = [ZERO; 2];
                        for &oi in &col.xo_nz {
                            let o = oi as usize;
                            v[(o >> k) & 1] += mul(col.xo2[o], gt[o]);
                        }
                        let wlam = mul(w[col.j], col.lam);
                        let q = quad_term(rk, v[0], v[1], wlam);
                        local[0] += mul(e_o, q[0]);
                        local[1] += mul(e_o, q[1]);
                        local[2] += mul(e_o, q[2]);
                    }
                    local
                })
                .reduce(zero_coeffs, combine);
            round0_check(&rounds, &coeffs, degree, claim);
            let c = absorb_round(t, &coeffs, degree);
            rounds.push(coeffs[..degree + 1].to_vec());
            challenges.push(c);
            e_o = mul(e_o, ONE + rk + c);
            let ec = ONE + c;
            for o in 0..obits {
                hist[o] = mul(hist[o], if (o >> k) & 1 == 1 { c } else { ec });
            }
        }
    } else {
        // The generic path: per-pair eq-prefix products, folded per
        // round (each round costs 2 muls + 1 lookup per pair).
        for k in 0..o_v {
            let rk = r_o[k];
            let suffix = &suffix_tabs[k];
            let coeffs = cols
                .par_chunks(chunk_size(cycles))
                .map(|chunk| {
                    let mut local = zero_coeffs();
                    for col in chunk.iter() {
                        if !col.active {
                            continue;
                        }
                        let j = col.j;
                        let base = j << u_v;
                        let wlam = mul(w[j], col.lam);
                        let mut v = [ZERO; 2];
                        for p in &col.pairs {
                            let wgt = mul(p.pref, suffix[p.o >> (k + 1)]);
                            let xv = x_work[base + p.u];
                            if (p.o >> k) & 1 == 0 {
                                v[0] += mul(wgt, xv);
                            } else {
                                v[1] += mul(wgt, xv);
                            }
                        }
                        let q = quad_term(rk, v[0], v[1], wlam);
                        local[0] += mul(e_o, q[0]);
                        local[1] += mul(e_o, q[1]);
                        local[2] += mul(e_o, q[2]);
                    }
                    local
                })
                .reduce(zero_coeffs, combine);
            round0_check(&rounds, &coeffs, degree, claim);
            let c = absorb_round(t, &coeffs, degree);
            rounds.push(coeffs[..degree + 1].to_vec());
            challenges.push(c);
            e_o = mul(e_o, ONE + rk + c);
            let ec = ONE + c;
            cols.par_chunks_mut(chunk_size(cycles)).for_each(|chunk| {
                for col in chunk.iter_mut() {
                    for p in col.pairs.iter_mut() {
                        p.pref = mul(p.pref, if (p.o >> k) & 1 == 1 { c } else { ec });
                    }
                }
            });
        }
    }

    // ---- The (o, a)-folded R over (u, h): one dense global table. ----
    let co = &challenges[a_v..a_v + o_v];
    let eq_co: Vec<Gf128> = (0..(1usize << o_v))
        .map(|o| {
            let mut v = ONE;
            for i in 0..o_v {
                v = mul(v, eqb(co[i], (o >> i) & 1 == 1));
            }
            v
        })
        .collect();
    let mut r_uh = vec![ZERO; (1usize << u_v) << h_v];
    let support_oa: &Vec<Vec<(usize, usize, Gf128)>> = if a_v > 0 { &support_a } else { &support };
    for h in 0..(1usize << h_v) {
        for &(o, u, coef) in &support_oa[h] {
            r_uh[u | (h << u_v)] += mul(coef, eq_co[o]);
        }
    }
    // Which selector factor owns each h bit.
    let owner: Vec<usize> = {
        let mut own = vec![usize::MAX; h_v];
        for (fi, sel) in witness.selectors.iter().enumerate() {
            for b in 0..sel.h_vars {
                own[sel.h_off + b] = fi;
            }
        }
        own
    };

    // ---- h-rounds: the one-hot kill. ----
    drop(_tj_o);
    let _tj_h = JoinTimer::new("jr-h");
    for k in 0..h_v {
        let coeffs = {
            let rtab = &r_uh;
            cols.par_chunks(chunk_size(cycles))
                .map(|chunk| {
                    let mut local = zero_coeffs();
                    for col in chunk.iter() {
                        if !col.active {
                            continue;
                        }
                        let j = col.j;
                        let h_lo = (col.h_pos >> (k + 1)) << 1;
                        let row0 = h_lo << u_v;
                        let row1 = (h_lo | 1) << u_v;
                        let mut ab = [ZERO; 2];
                        // The X column is still bits: the contraction is
                        // a select — one add per SET bit, no multiply.
                        for (w, &word) in col.x_bits.iter().enumerate() {
                            let mut bits = word;
                            while bits != 0 {
                                let u = bits.trailing_zeros() as usize;
                                ab[0] += rtab[row0 + (w << 6) + u];
                                ab[1] += rtab[row1 + (w << 6) + u];
                                bits &= bits - 1;
                            }
                        }
                        let s = mul(mul(mul(w[j], col.lam), col.lamh), e_o);
                        let hb = Gf128(((col.h_pos >> k) & 1) as u128);
                        let q = quad_term(hb, ab[0], ab[1], s);
                        local[0] += q[0];
                        local[1] += q[1];
                        local[2] += q[2];
                    }
                    local
                })
                .reduce(zero_coeffs, combine)
        };
        round0_check(&rounds, &coeffs, degree, claim);
        let c = absorb_round(t, &coeffs, degree);
        rounds.push(coeffs[..degree + 1].to_vec());
        challenges.push(c);
        let f0 = ONE + c;
        let f1 = c;
        cols.par_chunks_mut(chunk_size(cycles)).for_each(|chunk| {
            for col in chunk.iter_mut() {
                if !col.active {
                    continue;
                }
                let f = if (col.h_pos >> k) & 1 == 1 { f1 } else { f0 };
                col.lamh = mul(col.lamh, f);
                let fi = owner[k];
                if fi != usize::MAX {
                    col.lamh_f[fi] = mul(col.lamh_f[fi], f);
                }
            }
        });
        bind_var(&mut r_uh, u_v, c);
    }

    // ---- u-rounds: the R vector × per-column X slices. ----
    drop(_tj_h);
    let _tj_u = JoinTimer::new("jr-u");
    let mut r_u = r_uh; // h fully folded: a vector over u.
    for k in 0..u_v {
        let half = 1usize << (u_v - k - 1);
        // Bit 0 of the working tables is the round variable u_k (each
        // round folds bit 0), so the line pairs are **strided**:
        // r0[i] = r_u[2i] (u_k=0), r1[i] = r_u[2i+1] (u_k=1), with `i`
        // enumerating the remaining high bits.
        let (r0, r1): (Vec<Gf128>, Vec<Gf128>) = {
            let mut a = vec![ZERO; half];
            let mut b = vec![ZERO; half];
            for i in 0..half {
                a[i] = r_u[i << 1];
                b[i] = r_u[(i << 1) | 1];
            }
            (a, b)
        };
        let coeffs = {
            let x = &x_work;
            let wt = &w;
            cols.par_chunks(chunk_size(cycles))
                .map(|chunk| {
                    let mut local = zero_coeffs();
                    for col in chunk.iter() {
                        if !col.active {
                            continue;
                        }
                        let xc = &x[(col.j << (u_v - k))..((col.j + 1) << (u_v - k))];
                        let mut d = [ZERO; 4];
                        for i in 0..half {
                            let x0 = xc[i << 1];
                            let x1 = xc[(i << 1) | 1];
                            d[0] += mul(r0[i], x0);
                            d[1] += mul(r0[i], x1);
                            d[2] += mul(r1[i], x0);
                            d[3] += mul(r1[i], x1);
                        }
                        let s = mul(mul(mul(wt[col.j], col.lam), col.lamh), e_o);
                        local[0] += mul(s, d[0]);
                        local[1] += mul(s, d[1] + d[2]);
                        local[2] += mul(s, d[0] + d[1] + d[2] + d[3]);
                    }
                    local
                })
                .reduce(zero_coeffs, combine)
        };
        round0_check(&rounds, &coeffs, degree, claim);
        let c = absorb_round(t, &coeffs, degree);
        rounds.push(coeffs[..degree + 1].to_vec());
        challenges.push(c);
        par_bind_var(&mut x_work, 0, c); // every column folds u_k
        bind_var(&mut r_u, 0, c);
    }

    // ---- j-rounds: per-cycle factor vectors, pair-folded. ----
    drop(_tj_u);
    let _tj_j = JoinTimer::new("jr-j");
    // The fully folded R and eq_o constants scale every j-round message
    // (they depend on no j variable).
    let s_fin = mul(e_o, r_u[0]);
    let mut xq = x_work; // the X j-part (cycles entries).
    let mut wv = w; // eq(r_j, ·).
    let qv: Vec<Vec<Gf128>> = (0..n_sel)
        .map(|f| cols.iter().map(|c| c.lamh_f[f]).collect())
        .collect();
    let lv: Vec<Vec<Gf128>> = (0..n_off)
        .map(|m| {
            (0..cycles)
                .map(|j| ONE + Gf128(witness.offset_bits[m][j] as u128) + challenges[m])
                .collect()
        })
        .collect();
    let mut qv = qv;
    let mut lv = lv;
    for k in 0..n {
        let mut factor_refs: Vec<&Vec<Gf128>> = Vec::with_capacity(2 + n_sel + n_off);
        factor_refs.push(&wv);
        factor_refs.push(&xq);
        for q in &qv {
            factor_refs.push(q);
        }
        for l in &lv {
            factor_refs.push(l);
        }
        let len = 1usize << (n - k);
        // One task per even-index pair group (bit 0 = the round variable
        // j_k; each pair (2m, 2m+1) must be visited exactly once — twice
        // would self-cancel in characteristic two).
        let pairs = len / 2;
        let coeffs = (0..pairs)
            .into_par_iter()
            .chunks(chunk_size(pairs.max(1)))
            .map(|idxs| {
                let mut local = zero_coeffs();
                for m in idxs {
                    let i = m << 1; // the pair base (j_k = 0 / 1).
                                    // Per column: s_fin · Π_f (f0 + (f0 + f1)·t).
                    let mut poly = [ZERO; MAXC];
                    poly[0] = s_fin;
                    let mut deg = 0usize;
                    for f in &factor_refs {
                        let f0 = f[i];
                        let d = f0 + f[i | 1];
                        let mut next = [ZERO; MAXC];
                        for e in 0..=deg {
                            next[e] += mul(poly[e], f0);
                            next[e + 1] += mul(poly[e], d);
                        }
                        poly = next;
                        deg += 1;
                    }
                    for (e, &pv) in poly.iter().enumerate().take(deg + 1) {
                        local[e] += pv;
                    }
                }
                local
            })
            .reduce(zero_coeffs, combine);
        round0_check(&rounds, &coeffs, degree, claim);
        let c = absorb_round(t, &coeffs, degree);
        rounds.push(coeffs[..degree + 1].to_vec());
        challenges.push(c);
        par_bind_var(&mut wv, 0, c);
        par_bind_var(&mut xq, 0, c);
        for q in qv.iter_mut() {
            par_bind_var(q, 0, c);
        }
        for l in lv.iter_mut() {
            par_bind_var(l, 0, c);
        }
    }

    RouterProof {
        rounds,
        x_value: xq[0],
        q_values: qv.iter().map(|q| q[0]).collect(),
        offset_values: lv.iter().map(|l| l[0]).collect(),
        terminal: challenges,
    }
}
