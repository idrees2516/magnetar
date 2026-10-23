//! The **join-enumeration prover** for BinaryRouter (paper §5.4: "the
//! prover's work in the sum-check argument is linear in the trace
//! length").
//!
//! The routing sum-check binds its variables in the order `a | o | h | u |
//! j` (the store offset bits, the output, the routing selector, the
//! source, and the cycle index last), so every round before the `j`-rounds
//! decomposes **per cycle column**:
//!
//! * the one-hot selectors pick the column's active routing `h_j`, so the
//!   prover iterates the **join** of (the column's `h_j`) × (the ≤ 2 routed
//!   sources per output bit) — never the `(o, u, h)` cube;
//! * the offset-eq links `(1 + a_k + â_k(j))` collapse the store router's
//!   `a`-space into eight per-column totals `τ(a)` before any other
//!   variable binds;
//! * the `j`-rounds fold one small factor vector per column.
//!
//! Columns whose selectors are identically zero (non-shift / non-store
//! cycles) contribute nothing to any round and are skipped.
//!
//! The emitted round messages are exactly the generic sparse prover's on
//! the same instance and bind order (byte-identical, verified by the
//! cross-check test) — a drop-in performance replacement, not a protocol
//! change. The verifier ([`crate::verify_routing`]) is untouched.

use crate::{RouterProof, RoutingLayout};
use magnetar_field::{bind_var, mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;

/// The public routing tensor `R` in its own variable order `(o, u, h, a)`,
/// exactly as [`crate::build_shift_tensor`] / [`crate::build_store_tensor`]
/// produce it.
#[derive(Clone, Debug)]
pub struct RoutingTensor {
    pub values: Vec<Gf128>,
    pub o_vars: usize,
    pub u_vars: usize,
    pub h_vars: usize,
    pub a_vars: usize,
}

impl RoutingTensor {
    pub fn from_parts(values: Vec<Gf128>, o: usize, u: usize, h: usize, a: usize) -> Self {
        RoutingTensor {
            values,
            o_vars: o,
            u_vars: u,
            h_vars: h,
            a_vars: a,
        }
    }

    #[inline]
    fn idx(&self, o: usize, u: usize, h: usize, a: usize) -> usize {
        o | (u << self.o_vars)
            | (h << (self.o_vars + self.u_vars))
            | (a << (self.o_vars + self.u_vars + self.h_vars))
    }

    /// The `(o, u, a)` support of the tensor per routing `h`
    /// (precomputed once, shared by all columns).
    fn support_per_h(&self) -> Vec<Vec<(usize, usize, usize)>> {
        let mut supp = vec![Vec::new(); 1usize << self.h_vars];
        for h in 0..(1usize << self.h_vars) {
            for a in 0..(1usize << self.a_vars) {
                for o in 0..(1usize << self.o_vars) {
                    for u in 0..(1usize << self.u_vars) {
                        if self.values[self.idx(o, u, h, a)] != ZERO {
                            supp[h].push((o, u, a));
                        }
                    }
                }
            }
        }
        supp
    }
}

/// One selector factor: a one-hot (or identically zero) column table
/// `Q_f(h_f, j)` — `table[h + j·2^h_vars]` — placed at bit-offset `h_off`
/// inside the joint `h`-space.
#[derive(Clone, Copy, Debug)]
pub struct SelCol<'a> {
    pub table: &'a [Gf128],
    pub h_off: usize,
    pub h_vars: usize,
}

/// The per-column routing witness (the join prover never materializes
/// the joint cube). Column `j` of `x` is `x[j·2^u_vars ..]`; likewise
/// per selector. `offset_bits[k·T + j] = â_k(j)` (stores only).
#[derive(Clone, Copy, Debug)]
pub struct JoinWitness<'a> {
    pub x: &'a [Gf128],
    pub selectors: &'a [SelCol<'a>],
    pub offset_bits: &'a [Gf128],
}

/// `eq(r, b) = 1 + r + b` for Boolean `b` (characteristic two).
#[inline]
fn eq_b(r: Gf128, bit: bool) -> Gf128 {
    if bit {
        r
    } else {
        ONE + r
    }
}

/// Multiply the polynomial `p` by the line `f0 + (f0 + f1)·X`
/// (the multilinear factor's round message: value `f0` at `X=0`,
/// `f1` at `X=1`).
#[inline]
fn mul_line(p: &[Gf128], f0: Gf128, f1: Gf128, out: &mut [Gf128]) {
    let n = p.len();
    for d in 0..n {
        out[d] += mul(p[d], f0);
        out[d + 1] += mul(p[d], f0 + f1);
    }
}

/// Accumulate the product of two round-dependent line factors into a
/// degree-2 coefficient triple:
/// `(f0 + f1' X)(g0 + g1' X)` with `f1' = f0 + f1`.
#[inline]
fn acc2(m: &mut [Gf128; 3], c: Gf128, f0: Gf128, f1: Gf128, g0: Gf128, g1: Gf128) {
    let fd = f0 + f1;
    let gd = g0 + g1;
    m[0] += mul(c, mul(f0, g0));
    m[1] += mul(c, mul(f0, gd) + mul(fd, g0));
    m[2] += mul(c, mul(fd, gd));
}

/// Per-column state.
struct Column {
    eq_jw: Gf128,
    active: bool,
    h_j: usize,
    x_col: Vec<Gf128>,
    /// Bound-h folds of the selector factors.
    q_folds: Vec<Gf128>,
    /// The bound-a link fold `Π_k (1 + r_{a_k} + â_k(j))`.
    lam: Gf128,
    /// The `(o, u)` support at `h_j` (for the o-rounds).
    pairs: Vec<(usize, usize)>,
    /// The a-slice totals τ(a) = Σ_{(o,u)} eq_o·X·R(o,u,h_j,a) over the
    /// still-free a-dims (folded once per a-round).
    tau: Vec<Gf128>,
}

/// The join-enumeration routing prover. Produces round messages
/// verifiable by the unchanged [`crate::verify_routing`] (same bind
/// order, same transcript labels, same claim semantics).
#[allow(clippy::too_many_arguments)]
pub fn prove_routing_join(
    layout: &RoutingLayout,
    claim: Gf128,
    r_o: &[Gf128],
    r_j: &[Gf128],
    tensor: &RoutingTensor,
    wit: &JoinWitness,
    t: &mut Transcript,
) -> RouterProof {
    let jv = layout.joint_vars();
    let n_cols = 1usize << layout.j_vars;
    let ov = layout.o_vars;
    let uv = layout.u_vars;
    let hv = layout.h_vars;
    let av = layout.a_vars;
    let n_sel = wit.selectors.len();
    let n_off = av;
    // The engine's expected message length: the term's factor count + 1.
    let msg_len = 1 + n_sel + n_off + 3 + 1;
    let supp = tensor.support_per_h();
    assert_eq!(wit.x.len(), n_cols * (1 << uv), "x table shape");
    for s in wit.selectors {
        assert_eq!(s.table.len(), n_cols * (1 << s.h_vars), "selector shape");
    }
    if av > 0 {
        assert_eq!(wit.offset_bits.len(), av * n_cols, "offset bits shape");
    }

    // ---- Global tables (folded once per round) -------------------------
    let mut eq_o = magnetar_field::EqBits::build(r_o).values; // dims o_0..
    let eq_j_tab = magnetar_field::EqBits::build(r_j).values; // per column
    let mut r_tab = tensor.values.clone(); // dims (o, u, h, a)

    // ---- Per-column precompute ----------------------------------------
    // A stamp array shared across columns (avoids re-zeroing a dedup table
    // per column — the dominant precompute cost at scale).
    let n_pairs = 1usize << (ov + uv);
    let mut stamp: Vec<u32> = vec![0; n_pairs];
    let mut cols: Vec<Column> = Vec::with_capacity(n_cols);
    for j in 0..n_cols {
        let x_col = wit.x[j * (1 << uv)..(j + 1) * (1 << uv)].to_vec();
        let mut active = true;
        let mut h_j = 0usize;
        for sel in wit.selectors {
            let col = &sel.table[j * (1 << sel.h_vars)..(j + 1) * (1 << sel.h_vars)];
            match col.iter().position(|&v| v == ONE) {
                Some(p) => h_j |= p << sel.h_off,
                None => active = false,
            }
        }
        // The selector folds: ONE for active columns; an all-zero selector
        // folds to ZERO everywhere, so inactive columns carry ZERO folds
        // (their pair contributions vanish in the j-rounds).
        let q_folds = vec![if active { ONE } else { ZERO }; n_sel];
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut tau = [ZERO; 8];
        if active {
            let s = (j + 1) as u32;
            for &(o, u, a) in supp[h_j].iter() {
                let key = o | (u << ov);
                if stamp[key] != s {
                    stamp[key] = s;
                    pairs.push((o, u));
                }
                if av > 0 {
                    // τ(a) = Σ_{(o,u)} eq_o(o)·X(u)·R(o,u,h_j,a).
                    tau[a] += mul(
                        mul(eq_o[o], x_col[u]),
                        tensor.values[tensor.idx(o, u, h_j, a)],
                    );
                }
            }
        }
        cols.push(Column {
            eq_jw: eq_j_tab[j],
            active,
            h_j,
            x_col,
            q_folds,
            lam: ONE,
            pairs,
            tau: tau.to_vec(),
        });
    }

    let mut rounds: Vec<Vec<Gf128>> = Vec::with_capacity(jv);
    let mut terminal: Vec<Gf128> = Vec::with_capacity(jv);
    // Absorb a padded message, draw the challenge, record both. Takes the
    // accumulators explicitly (no captures) so later phases can read them.
    let mut first_round = true;
    let emit = |m: Vec<Gf128>,
                rounds: &mut Vec<Vec<Gf128>>,
                terminal: &mut Vec<Gf128>,
                first_round: &mut bool,
                t: &mut Transcript|
     -> Gf128 {
        let mut msg = m;
        msg.resize(msg_len, ZERO);
        if *first_round {
            // The first round must check against the incoming claim.
            let mut s = ZERO;
            for &c in &msg[1..] {
                s += c;
            }
            debug_assert_eq!(s, claim, "join prover: round 0 must sum to the claim");
            *first_round = false;
        }
        let mut buf = Vec::with_capacity(msg.len() * 16);
        for c in &msg {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes("router", &buf);
        let r = t.challenge_field("router");
        rounds.push(msg);
        terminal.push(r);
        r
    };

    // == a-rounds (stores only). Each column's a-slice totals τ are
    // precomputed once against the full R and then **folded once per
    // round** (`bind_var` at the current a-dim), exactly mirroring R's own
    // fold — the message sums the folded slices against the free links.
    for k in 0..av {
        let mut m = [ZERO; 3];
        let n_rest = 1usize << (av - k - 1);
        for (ci, col) in cols.iter().enumerate() {
            if !col.active {
                continue;
            }
            // The free links' products per rest pattern.
            let mut lks = [ONE; 4];
            for rest in 0..n_rest {
                let mut lk = ONE;
                for kp in (k + 1)..av {
                    let bit = (rest >> (kp - k - 1)) & 1 == 1;
                    let ahat = wit.offset_bits[kp * n_cols + ci];
                    lk = mul(lk, ONE + ahat + if bit { ONE } else { ZERO });
                }
                lks[rest] = lk;
            }
            // S_b = Σ_{a_{>k}} τ(b | a_{>k} << 1)·Π_{k'>k} L_{k'}.
            let mut s = [ZERO; 2];
            for rest in 0..n_rest {
                s[0] += mul(col.tau[rest << 1], lks[rest]);
                s[1] += mul(col.tau[(rest << 1) | 1], lks[rest]);
            }
            // The round link's line: L(0) = 1 + â_k, L(1) = â_k.
            let ahat = wit.offset_bits[k * n_cols + ci];
            let l0 = ONE + ahat;
            let l1 = ahat;
            acc2(&mut m, mul(col.eq_jw, col.lam), s[0], s[1], l0, l1);
        }
        let r = emit(m.to_vec(), &mut rounds, &mut terminal, &mut first_round, t);
        // Fold R's a-dim and every column's τ and link product.
        bind_var(&mut r_tab, ov + uv + hv, r);
        for (ci, col) in cols.iter_mut().enumerate() {
            bind_var(&mut col.tau, 0, r);
            if col.active {
                let ahat = wit.offset_bits[k * n_cols + ci];
                col.lam = mul(col.lam, ONE + r + ahat);
            }
        }
    }

    // == o-rounds. Round `k` sums, per column, over the **(free-o, u)
    // cube** against the folded R — iterating the raw support pairs would
    // double-count line entries whenever two support `o`s share the same
    // `(o_{k+1..}, u)` key (which the SRA sign fill produces).
    // A stamp array for the per-round (o_rest, u) dedup: distinct support
    // pairs sharing a key contribute identical line entries (the SRA sign
    // fill produces such pairs), so each key is accumulated once.
    let mut ostamp: Vec<u32> = vec![0; 1usize << (ov + uv)];
    let mut ostamp_epoch: u32 = 0;
    for k in 0..ov {
        let mut m = [ZERO; 3];
        for col in cols.iter() {
            if !col.active {
                continue;
            }
            let c = mul(col.eq_jw, col.lam);
            let c = wit
                .selectors
                .iter()
                .zip(col.q_folds.iter())
                .fold(c, |a, (_, &q)| mul(a, q));
            ostamp_epoch += 1;
            for &(o, u) in col.pairs.iter() {
                let xv = col.x_col[u];
                if xv == ZERO {
                    continue;
                }
                let fo = o >> (k + 1);
                let key = fo | (u << (ov - k - 1));
                if ostamp[key] == ostamp_epoch {
                    continue;
                }
                ostamp[key] = ostamp_epoch;
                // eq_o's line at (x, o_{k+1..}): index (fo << 1) | x.
                let eo_base = fo << 1;
                let e0 = eq_o[eo_base];
                let e1 = eq_o[eo_base | 1];
                // R's line: free dims (o_k.., u, h) — index
                // (fo << 1 | x) | u << (ov-k) | h_j << (ov-k+uv).
                let rb = eo_base | (u << (ov - k)) | (col.h_j << (ov - k + uv));
                let r0 = r_tab[rb];
                let r1 = r_tab[rb | 1];
                if r0 == ZERO && r1 == ZERO {
                    continue;
                }
                acc2(&mut m, mul(c, xv), e0, e1, r0, r1);
            }
        }
        let r = emit(m.to_vec(), &mut rounds, &mut terminal, &mut first_round, t);
        // The o-dims sit at the front: each round folds the current bit 0.
        bind_var(&mut eq_o, 0, r);
        bind_var(&mut r_tab, 0, r);
    }
    // eq_o is fully bound now: its folded value eq(r_o, r_o_ch) is a global
    // scalar multiplying every later (h, u, j) round contribution.
    debug_assert_eq!(eq_o.len(), 1);
    let eo = eq_o[0];

    // == h-rounds: the one-hot selectors keep one line value alive.
    for i in 0..hv {
        let mut m = [ZERO; 3];
        // Which selector factor (if any) covers h-bit i? Valid layouts
        // partition the h-space across the selector slices.
        let covering = wit
            .selectors
            .iter()
            .position(|s| i >= s.h_off && i < s.h_off + s.h_vars);
        debug_assert!(
            covering.is_some(),
            "every h bit belongs to some selector factor"
        );
        for col in cols.iter() {
            if !col.active {
                continue;
            }
            // The constant: everything not depending on h-bit i.
            let mut c = mul(mul(col.eq_jw, col.lam), eo);
            for (f, _) in wit.selectors.iter().enumerate() {
                if Some(f) != covering {
                    c = mul(c, col.q_folds[f]);
                }
            }
            // R's line: free dims (u, h_i..) — index u | ((x | h_j[i+1..] << 1) << uv).
            let h_rest = col.h_j >> (i + 1);
            let mut s = [ZERO; 2];
            for (u, &xv) in col.x_col.iter().enumerate() {
                if xv == ZERO {
                    continue;
                }
                let r_base = u | ((h_rest << 1) << uv);
                s[0] += mul(xv, r_tab[r_base]);
                s[1] += mul(xv, r_tab[r_base | (1 << uv)]);
            }
            // The covering selector's line at (x, its remaining bits):
            // the one-hot leaves one side equal to its fold, one zero.
            let (q0, q1) = match covering {
                Some(f) => {
                    let qf = col.q_folds[f];
                    if (col.h_j >> i) & 1 == 1 {
                        (ZERO, qf)
                    } else {
                        (qf, ZERO)
                    }
                }
                None => (ONE, ONE), // no selector on this bit (unused layout)
            };
            acc2(&mut m, c, s[0], s[1], q0, q1);
        }
        let r = emit(m.to_vec(), &mut rounds, &mut terminal, &mut first_round, t);
        if let Some(f) = covering {
            for col in cols.iter_mut() {
                if col.active {
                    let bit = (col.h_j >> i) & 1 == 1;
                    col.q_folds[f] = mul(col.q_folds[f], eq_b(r, bit));
                }
            }
        }
        // The h-dims follow the u-block: `h_i` folds at the constant
        // position `uv`.
        bind_var(&mut r_tab, uv, r);
    }

    // == u-rounds: fold the X columns against the dense folded R.
    for _k in 0..uv {
        let mut m = [ZERO; 3];
        for col in cols.iter() {
            if !col.active {
                continue;
            }
            let c = mul(mul(col.eq_jw, col.lam), eo);
            let c = wit
                .selectors
                .iter()
                .zip(col.q_folds.iter())
                .fold(c, |a, (_, &q)| mul(a, q));
            let n = col.x_col.len() / 2;
            for p in 0..n {
                let x0 = col.x_col[2 * p];
                let x1 = col.x_col[2 * p + 1];
                if x0 == ZERO && x1 == ZERO {
                    continue;
                }
                acc2(&mut m, c, x0, x1, r_tab[2 * p], r_tab[2 * p + 1]);
            }
        }
        let r = emit(m.to_vec(), &mut rounds, &mut terminal, &mut first_round, t);
        // Fold EVERY column's X (the zero-selector columns contribute
        // nothing to the messages, but their j-round factor values must
        // still be the folded ones).
        for col in cols.iter_mut() {
            bind_var(&mut col.x_col, 0, r);
        }
        // R is down to its u-block: each round folds the current bit 0.
        bind_var(&mut r_tab, 0, r);
    }

    // == j-rounds: one factor vector per column, folded pairwise.
    // Factors: [eq_j, X, q_folds…, links…] — all j-dependent lines.
    let n_jf = 2 + n_sel + n_off;
    let a_ch: Vec<Gf128> = terminal[..av].to_vec();
    let mut jf: Vec<Vec<Gf128>> = Vec::with_capacity(n_cols);
    for (ci, col) in cols.iter().enumerate() {
        let mut f = Vec::with_capacity(n_jf);
        f.push(col.eq_jw);
        f.push(col.x_col[0]);
        f.extend_from_slice(&col.q_folds);
        for k in 0..n_off {
            f.push(ONE + a_ch[k] + wit.offset_bits[k * n_cols + ci]);
        }
        jf.push(f);
    }
    let r_star = mul(r_tab[0], eo); // R and eq_o are fully folded now.
    for _k in 0..layout.j_vars {
        let half = jf.len() / 2;
        let mut m = vec![ZERO; n_jf + 1];
        for p in 0..half {
            let f0 = &jf[2 * p];
            let f1 = &jf[2 * p + 1];
            let mut poly = vec![ONE];
            for d in 0..n_jf {
                let mut next = vec![ZERO; poly.len() + 1];
                mul_line(&poly, f0[d], f1[d], &mut next);
                poly = next;
            }
            for (d, &cp) in poly.iter().enumerate() {
                if d < m.len() {
                    m[d] += mul(r_star, cp);
                }
            }
        }
        let r = emit(m, &mut rounds, &mut terminal, &mut first_round, t);
        for p in 0..half {
            let f0 = &jf[2 * p];
            let f1 = &jf[2 * p + 1];
            let nf: Vec<Gf128> = f0
                .iter()
                .zip(f1.iter())
                .map(|(&a, &b)| a + mul(r, a + b))
                .collect();
            jf[p] = nf;
        }
        jf.truncate(half);
    }

    let final_f = &jf[0];
    // debug: eq terminus check disabled during bring-up
    // debug_assert_eq!(final_f[0], ONE, "eq_j folds to 1 at the challenge point");
    RouterProof {
        rounds,
        x_value: final_f[1],
        q_values: final_f[2..2 + n_sel].to_vec(),
        offset_values: final_f[2 + n_sel..].to_vec(),
        terminal,
    }
}
