//! # magnetar-router
//!
//! BinaryRouter (paper §5.4): a family of sum-check arguments for
//! computations expressible as **bit routings**, instantiated for shifts
//! (`R_sh`, Equation 5) and sub-word stores (`R_st`, Equation 6).
//!
//! The routing identity:
//! `Y(o, j) = Σ_{u,h} X(u, j)·Q(h, j)·R(o, u, h)`,
//! where `X` holds candidate source bits, `Q(·, j)` is a one-hot (or zero)
//! routing selector, and `R` is a public sparse tensor fixed at
//! preprocessing.
//!
//! Given a claim `Ŷ(r_o, r_j) = v`, one sum-check over the **joint
//! `(o, j, u, h)` space** (plus the summed offset variables `a ∈ B³` for
//! stores) reduces it to evaluation claims on `X` and on the selector
//! factors `Q`. The verifier evaluates `R̂` itself.
//!
//! * Shifts (Eq. 5): the `u` space **is the source word's bit positions**
//!   (`u = o'` selects source bit `o'`): a source segment claim lifts to
//!   MosaicPCS directly. Zero-fill is expressed by `R = 0` (no source); the
//!   arithmetic-right-shift sign fill routes from bit `w−1`. The selector is
//!   two one-hot ROM fields — `Q_lo` over the shift amount (low `h` bits)
//!   and `Q_hi` over the kind (high `h` bits) — which BinaryShout
//!   authenticates against the bytecode.
//! * Sub-word stores (Eq. 6): `R_st(o, u, h, a)` takes the byte offset
//!   `a ∈ B³` as three additional Boolean **summed** variables, linked to
//!   the witness offset bits via `eq(a_k, â_k(j))` factors; inside a
//!   written byte the increment bit is the XOR of the old-RAM bit and the
//!   store-source bit, expressed by routing the output from *both* source
//!   bits (their sum over `B` is the XOR). The sources `X` are the pair
//!   `(old word, store source)` with `u = o + w·s` (`s` the source half),
//!   so the X claim splits into the two segment claims at the same bit
//!   point. On non-store cycles the selector is identically zero, which
//!   forces `Inc = 0`.
//!
//! Joint-space layout (little-endian, low to high): `o | j | u | h | a`.

#![forbid(unsafe_code)]
// Proof-system style: loops indexed over Boolean-cube coordinates and
// multi-input reductions are the domain's normal shape.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use magnetar_field::{Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_sumcheck::{
    engine::SumcheckVerifier,
    sparse::{SparseFactor, SparseProver, SparseSpec},
    BindOrder, RoundMsg, Term,
};

/// A public table over a subset of the joint variables, padded across the
/// rest (padding leaves the MLE unchanged: the eq folds over idle variables
/// sum to one).
#[derive(Clone, Debug)]
pub struct PaddedTableFactor {
    pub table: Vec<Gf128>,
    /// Which joint bit positions (low to high order of the table's
    /// variables) are active.
    pub active: Vec<usize>,
    pub joint_vars: usize,
}

impl PaddedTableFactor {
    pub fn new(table: Vec<Gf128>, active: Vec<usize>, joint_vars: usize) -> Self {
        let table_vars = table.len().trailing_zeros() as usize;
        assert_eq!(active.len(), table_vars);
        PaddedTableFactor {
            table,
            active,
            joint_vars,
        }
    }
}

impl magnetar_sumcheck::public::PublicFactor for PaddedTableFactor {
    fn num_vars(&self) -> usize {
        self.joint_vars
    }
    #[allow(dead_code)] // used by the routing tests via the dense engine
    fn materialize(&self) -> Vec<Gf128> {
        let mut out = vec![ZERO; 1usize << self.joint_vars];
        for (idx, o) in out.iter_mut().enumerate() {
            let mut ti = 0usize;
            for (tv, &av) in self.active.iter().enumerate() {
                if (idx >> av) & 1 == 1 {
                    ti |= 1usize << tv;
                }
            }
            *o = self.table[ti];
        }
        out
    }
    fn eval_at(&self, point: &[Gf128]) -> Gf128 {
        let sub: Vec<Gf128> = self.active.iter().map(|&a| point[a]).collect();
        magnetar_field::eval_mle(&self.table, &sub)
    }
}

/// A dense (witness) table over a subset of joint variables, padded.
/// Materialized fully for the prover's folding at the test scale.
#[derive(Clone, Debug)]
pub struct PaddedDense {
    pub table: Vec<Gf128>,
    pub active: Vec<usize>,
    pub joint_vars: usize,
}

impl PaddedDense {
    pub fn new(table: Vec<Gf128>, active: Vec<usize>, joint_vars: usize) -> Self {
        assert_eq!(active.len(), table.len().trailing_zeros() as usize);
        PaddedDense {
            table,
            active,
            joint_vars,
        }
    }
    #[allow(dead_code)] // used by the routing tests via the dense engine
    fn materialize(&self) -> Vec<Gf128> {
        let mut out = vec![ZERO; 1usize << self.joint_vars];
        for (idx, o) in out.iter_mut().enumerate() {
            let mut ti = 0usize;
            for (tv, &av) in self.active.iter().enumerate() {
                if (idx >> av) & 1 == 1 {
                    ti |= 1usize << tv;
                }
            }
            *o = self.table[ti];
        }
        out
    }
}

/// The joint-space layout of a routing instance.
#[derive(Clone, Debug)]
pub struct RoutingLayout {
    pub o_vars: usize,
    pub j_vars: usize,
    pub u_vars: usize,
    pub h_vars: usize,
    /// Extra summed variables (the store offset bits); 0 for shifts.
    pub a_vars: usize,
}

impl RoutingLayout {
    pub fn joint_vars(&self) -> usize {
        self.o_vars + self.j_vars + self.u_vars + self.h_vars + self.a_vars
    }
    /// Bit ranges (low to high): o | j | u | h | a.
    pub fn offsets(&self) -> [usize; 5] {
        let o = 0;
        let j = o + self.o_vars;
        let u = j + self.j_vars;
        let h = u + self.u_vars;
        let a = h + self.h_vars;
        [o, j, u, h, a]
    }
}

/// Output claims of a routing reduction: the terminal values plus the full
/// terminal point (original coordinates); accessors slice it per factor.
#[derive(Clone, Debug)]
pub struct RoutingClaims {
    /// The full terminal point in original joint coordinates.
    pub r_joint: Vec<Gf128>,
    /// The X terminal claim.
    pub x_value: Gf128,
    /// The selector factors' terminal claims (in factor order).
    pub q_values: Vec<Gf128>,
    /// (Stores) the offset-link terminal claims, one per offset bit.
    pub offset_values: Vec<Gf128>,
}

impl RoutingClaims {
    /// The X claim point: `(r_u, r_j)` — the u-slice then the j-slice.
    pub fn x_point(&self, layout: &RoutingLayout) -> Vec<Gf128> {
        let [_, j_off, u_off, _, _] = layout.offsets();
        let mut pt = self.r_joint[u_off..u_off + layout.u_vars].to_vec();
        pt.extend_from_slice(&self.r_joint[j_off..j_off + layout.j_vars]);
        pt
    }
    /// One selector factor's claim point: `(h-slice, j-slice)`.
    pub fn q_point(&self, layout: &RoutingLayout, h_offset: usize, h_vars: usize) -> Vec<Gf128> {
        let [_, j_off, _, h_off, _] = layout.offsets();
        let mut pt = self.r_joint[h_off + h_offset..h_off + h_offset + h_vars].to_vec();
        pt.extend_from_slice(&self.r_joint[j_off..j_off + layout.j_vars]);
        pt
    }
    /// The k-th offset bit's claim: the witness `â_k` at `r_j`, solved from
    /// the link factor's terminal `1 + r_{a_k} + â_k(r_j)`.
    pub fn offset_bit_value(&self, layout: &RoutingLayout, k: usize) -> Gf128 {
        let [_, _j_off, _, _, a_off] = layout.offsets();
        let r_ak = self.r_joint[a_off + k];
        // terminal = 1 + r_ak + â_k(r_j)  →  â_k = terminal + 1 + r_ak.
        self.offset_values[k] + ONE + r_ak
    }
}

/// The routing proof.
#[derive(Clone, Debug)]
pub struct RouterProof {
    pub rounds: Vec<Vec<Gf128>>,
    /// The X terminal claim.
    pub x_value: Gf128,
    /// The selector terminal claims.
    pub q_values: Vec<Gf128>,
    /// (Stores) the offset-link terminal claims.
    pub offset_values: Vec<Gf128>,
    /// The terminal point in original joint coordinates (prover-side
    /// bookkeeping for placing the segment claims; the verifier derives
    /// its own point by replaying the rounds and never reads this).
    pub terminal: Vec<Gf128>,
}

/// Prove a generic routing reduction with the **sparse** sum-check prover
/// (the paper's linear-in-T prover work: every factor stays in its own
/// variable space and only folds when a round binds one of its own
/// variables).
///
/// Dense factor order: `X` first, then `n_selectors` selector factors, then
/// the store offset link factors (if any). `dense` and `r_public` carry
/// each factor's **own** table plus its active joint bit positions.
#[allow(clippy::too_many_arguments)]
pub fn prove_routing(
    layout: &RoutingLayout,
    claim: Gf128,
    r_o: &[Gf128],
    r_j: &[Gf128],
    r_public: &PaddedTableFactor,
    dense: &[PaddedDense],
    n_selectors: usize,
    t: &mut Transcript,
) -> RouterProof {
    let jv = layout.joint_vars();
    let [_, j_off, _, _, _] = layout.offsets();
    let eq_o = magnetar_field::EqBits::build(r_o).values;
    let eq_j = magnetar_field::EqBits::build(r_j).values;
    let spec = SparseSpec {
        num_vars: jv,
        dense: dense
            .iter()
            .map(|d| SparseFactor::new(d.table.clone(), d.active.clone()))
            .collect(),
        public: vec![
            SparseFactor::new(eq_o, (0..layout.o_vars).collect()),
            SparseFactor::new(eq_j, (j_off..j_off + layout.j_vars).collect()),
            SparseFactor::new(r_public.table.clone(), r_public.active.clone()),
        ],
        terms: vec![Term {
            coeff: ONE,
            dense: (0..dense.len()).collect(),
            public: vec![0, 1, 2],
        }],
        claim,
    };
    let mut prover = SparseProver::new(spec, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(jv);
    for _ in 0..jv {
        let msg = prover.message();
        rounds.push(msg.0.clone());
        prover.absorb_and_fold(t, "router");
    }
    let challenges = prover.challenges();
    let (claims, _) = prover.finish();
    let terminal = BindOrder::Increasing.point(jv, &challenges);
    RouterProof {
        rounds,
        x_value: claims[0],
        q_values: claims[1..1 + n_selectors].to_vec(),
        offset_values: claims[1 + n_selectors..].to_vec(),
        terminal,
    }
}

/// Verify a routing reduction (mirrors the prover's sumcheck). Returns the
/// accepted terminal claims at the jointly-derived random point.
#[allow(clippy::too_many_arguments)]
pub fn verify_routing(
    layout: &RoutingLayout,
    claim: Gf128,
    r_o: &[Gf128],
    r_j: &[Gf128],
    r_public: &PaddedTableFactor,
    n_selectors: usize,
    n_offsets: usize,
    proof: &RouterProof,
    t: &mut Transcript,
) -> Result<RoutingClaims, &'static str> {
    let jv = layout.joint_vars();
    let [_, j_off, _, _, _] = layout.offsets();
    let eq_o = magnetar_field::EqBits::build(r_o).values;
    let eq_j = magnetar_field::EqBits::build(r_j).values;
    let eq_o_factor = PaddedTableFactor::new(eq_o, (0..layout.o_vars).collect(), jv);
    let eq_j_factor = PaddedTableFactor::new(eq_j, (j_off..j_off + layout.j_vars).collect(), jv);
    let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> = vec![
        Box::new(eq_o_factor),
        Box::new(eq_j_factor),
        Box::new(r_public.clone()),
    ];
    let terms = vec![Term {
        coeff: ONE,
        dense: (0..1 + n_selectors + n_offsets).collect(),
        public: vec![0, 1, 2],
    }];
    let mut v = SumcheckVerifier::new(jv, claim, &public, terms, BindOrder::Increasing);
    let mut r_joint = Vec::with_capacity(jv);
    for msg in &proof.rounds {
        r_joint.push(v.round_capturing(&RoundMsg(msg.clone()), t, "router")?);
    }
    let mut all = vec![proof.x_value];
    all.extend_from_slice(&proof.q_values);
    all.extend_from_slice(&proof.offset_values);
    v.finish(&all)?;
    Ok(RoutingClaims {
        r_joint,
        x_value: proof.x_value,
        q_values: proof.q_values.clone(),
        offset_values: proof.offset_values.clone(),
    })
}

// ---------------------------------------------------------------------------
// The two Magnetar instances (paper §5.4).
// ---------------------------------------------------------------------------

/// Build the shift routing tensor `R_sh(o, u, h)` (Equation 5).
///
/// The `u` space is the source word's bit positions (`u < w`): the X table
/// **is** the source segment's bit table, so the X claim lifts to the
/// commitment directly. Routings `h = (kind, amount)` with the amount in
/// the low `amount_bits` bits and the kind in the high bits:
/// kind 0 = SLL, 1 = SRL, 2 = SRA (padded to four).
/// * SLL: `Y[o] = X[o − amount]` for `o ≥ amount`, else 0 (no source).
/// * SRL: `Y[o] = X[o + amount]` for `o + amount < w`, else 0.
/// * SRA: like SRL but the fill routes from the sign bit `w − 1`.
///
/// Returns `(tensor, o_vars, u_vars, h_vars)` with the tensor indexed
/// `o | u << o_vars | h << (o_vars + u_vars)`.
pub fn build_shift_tensor(w: usize, amount_bits: usize) -> (Vec<Gf128>, usize, usize, usize) {
    let o_vars = w.trailing_zeros() as usize;
    let u_vars = o_vars; // u = the source bit position
    let h_count = 4 * (1usize << amount_bits);
    let h_vars = h_count.trailing_zeros() as usize;
    let mut r = vec![ZERO; (1usize << o_vars) * (1usize << u_vars) * (1usize << h_vars)];
    for kind in 0..4usize {
        for amount in 0..(1usize << amount_bits) {
            let h = kind * (1usize << amount_bits) + amount;
            for o in 0..w {
                let src: Option<usize> = match kind {
                    0 => {
                        // SLL: source bit o - amount, or none (zero fill).
                        if o >= amount {
                            Some(o - amount)
                        } else {
                            None
                        }
                    }
                    1 => {
                        // SRL: source bit o + amount, or none (zero fill).
                        if o + amount < w {
                            Some(o + amount)
                        } else {
                            None
                        }
                    }
                    2 => {
                        // SRA: source bit o + amount, else the sign bit.
                        if o + amount < w {
                            Some(o + amount)
                        } else {
                            Some(w - 1)
                        }
                    }
                    _ => None, // padding kind: no routing
                };
                if let Some(u) = src {
                    let idx = o | (u << o_vars) | (h << (o_vars + u_vars));
                    r[idx] = ONE;
                }
            }
        }
    }
    (r, o_vars, u_vars, h_vars)
}

/// The shift selector layout: two one-hot ROM fields — `Q_lo` over the
/// amount (the low `amount_bits` bits of `h`) and `Q_hi` over the kind (the
/// next 2 bits). Returns `(amount_h_offset, amount_vars, kind_h_offset,
/// kind_vars)`.
pub fn shift_selector_layout(amount_bits: usize) -> (usize, usize, usize, usize) {
    (0, amount_bits, amount_bits, 2)
}

/// Build the sub-word-store tensor `R_st(o, u, h, a)` (Equation 6).
///
/// Sources `u = o' + w·s`: `s = 0` the old-RAM word's bit `o'`, `s = 1` the
/// store-source bit `o'`. Routings `h`: one-hot over the four store widths
/// {byte, half, word, doubleword}; the selector is identically zero on
/// non-store cycles (which forces `Inc = 0`). For output bit `o` with byte
/// offset `a`: inside the written lanes the increment is the XOR of the old
/// and store bits — both sources routed (their sum over `B` is the XOR);
/// outside, no source is routed (the increment is zero).
///
/// Returns `(tensor, o_vars, u_vars, h_vars, a_vars)`.
pub fn build_store_tensor(w: usize) -> (Vec<Gf128>, usize, usize, usize, usize) {
    let u_count = 2 * w;
    let o_vars = w.trailing_zeros() as usize;
    let u_vars = u_count.next_power_of_two().trailing_zeros() as usize;
    let h_vars = 2; // 4 widths
    let a_vars = 3;
    let mut r =
        vec![
            ZERO;
            (1usize << o_vars) * (1usize << u_vars) * (1usize << h_vars) * (1usize << a_vars)
        ];
    for h in 0..4usize {
        // The write width in bytes per routing.
        let width = match h {
            0 => 1, // SB
            1 => 2, // SH
            2 => 4, // SW
            _ => 8, // SD
        };
        for a in 0..(1usize << a_vars) {
            for o in 0..w {
                let byte = o / 8;
                let inside = byte >= a && byte < a + width;
                if inside {
                    // Route from both the old and the store source of bit o.
                    for s in 0..2usize {
                        let u = o + w * s;
                        let idx = o
                            | (u << o_vars)
                            | (h << (o_vars + u_vars))
                            | (a << (o_vars + u_vars + h_vars));
                        r[idx] = ONE;
                    }
                }
            }
        }
    }
    (r, o_vars, u_vars, h_vars, a_vars)
}

/// Split a store X claim into the two source-segment claims:
/// `X̂(r_u, r_j) = (1 + r_u[hi])·Ô(r_bits, r_j) + r_u[hi]·Ŝ(r_bits, r_j)`
/// where `hi` is the source-half variable (the top bit of the u space) and
/// `r_bits` the remaining u coordinates. Returns the two point/value
/// pairs' coefficients: `((w0, v_old_expected_coef), ...)`.
pub fn store_x_split(r_u: &[Gf128], _x_claim: Gf128) -> ((Gf128, Vec<Gf128>), (Gf128, Vec<Gf128>)) {
    let hi = r_u.len() - 1;
    let bits = r_u[..hi].to_vec();
    let w_old = ONE + r_u[hi];
    let w_src = r_u[hi];
    ((w_old, bits.clone()), (w_src, bits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_field::mul;

    fn rg(rng: &mut dyn rand::RngCore) -> Gf128 {
        Gf128(rand::Rng::gen::<u128>(rng))
    }

    #[test]
    fn shift_tensor_shape() {
        // w = 16 bits, 2 amount bits.
        let (r, o_vars, u_vars, h_vars) = build_shift_tensor(16, 2);
        assert_eq!(o_vars, 4);
        assert_eq!(u_vars, 4);
        assert_eq!(h_vars, 4);
        // SLL by 1 (kind 0, amount 1, h = 1): output 5 fed by source bit 4.
        let idx = 5 | (4 << o_vars) | (1 << (o_vars + u_vars));
        assert_eq!(r[idx], ONE);
        // SLL by 1: output 0 has no source (zero fill).
        for u in 0..16usize {
            let idx = (u << o_vars) | (1 << (o_vars + u_vars));
            assert_eq!(r[idx], ZERO);
        }
        // SRA (kind 2) by 16 (h = 2*4 + 3 = 11... amount 3 = shift 3):
        // every output o with o+3 >= 16 routes from bit 15.
        let h_sra_amount3 = 2 * 4 + 3;
        for o in 13..16usize {
            let idx = o | (15 << o_vars) | (h_sra_amount3 << (o_vars + u_vars));
            assert_eq!(r[idx], ONE, "o={o}");
        }
    }

    #[test]
    fn store_tensor_shape() {
        let (r, o_vars, u_vars, h_vars, a_vars) = build_store_tensor(16);
        assert_eq!(a_vars, 3);
        assert_eq!(h_vars, 2);
        // Byte store (h=0) at offset a=1, bit o=9 (byte 1): inside [8,16).
        for s in 0..2usize {
            let u = 9 + 16 * s;
            let idx =
                9 | (u << o_vars) | (0 << (o_vars + u_vars)) | (1 << (o_vars + u_vars + h_vars));
            assert_eq!(r[idx], ONE);
        }
        // Bit o=1 (byte 0): outside [8,16) for a=1 — not routed.
        for s in 0..2usize {
            let u = 1 + 16 * s;
            let idx =
                1 | (u << o_vars) | (0 << (o_vars + u_vars)) | (1 << (o_vars + u_vars + h_vars));
            assert_eq!(r[idx], ZERO);
        }
    }

    #[test]
    fn shift_routing_roundtrip() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let w = 4usize;
        let amount_bits = 2usize;
        let (tensor, o_vars, u_vars, h_vars) = build_shift_tensor(w, amount_bits);
        let j_vars = 3usize;
        let layout = RoutingLayout {
            o_vars,
            j_vars,
            u_vars,
            h_vars,
            a_vars: 0,
        };
        let jv = layout.joint_vars();
        let [_, j_off, u_off, h_off, _] = layout.offsets();
        let cycles = 1usize << j_vars;
        // X: the source word bits (u = bit index).
        let mut x = vec![ZERO; (1usize << u_vars) * cycles];
        for j in 0..cycles {
            for b in 0..w {
                x[b + j * (1usize << u_vars)] = Gf128(rng.gen::<bool>() as u128);
            }
        }
        // Selectors: one-hot (or zero) per cycle.
        let (a_off_h, a_hv, k_off_h, k_hv) = shift_selector_layout(amount_bits);
        let mut q_lo = vec![ZERO; (1usize << a_hv) * cycles];
        let mut q_hi = vec![ZERO; (1usize << k_hv) * cycles];
        let mut y = vec![ZERO; (1usize << o_vars) * cycles];
        for j in 0..cycles {
            if rng.gen::<bool>() {
                let amount = rng.gen_range(0..(1usize << amount_bits));
                let kind = rng.gen_range(0..3usize);
                q_lo[amount + j * (1usize << a_hv)] = ONE;
                q_hi[kind + j * (1usize << k_hv)] = ONE;
                for o in 0..w {
                    let mut acc = ZERO;
                    for u in 0..w {
                        for hl in 0..(1usize << a_hv) {
                            for hh in 0..(1usize << k_hv) {
                                let h = hh * (1usize << amount_bits) + hl;
                                let ridx = o | (u << o_vars) | (h << (o_vars + u_vars));
                                acc += mul(
                                    mul(
                                        x[u + j * (1usize << u_vars)],
                                        mul(
                                            q_lo[hl + j * (1usize << a_hv)],
                                            q_hi[hh + j * (1usize << k_hv)],
                                        ),
                                    ),
                                    tensor[ridx],
                                );
                            }
                        }
                    }
                    y[o + j * (1usize << o_vars)] = acc;
                }
            }
        }
        let r_o: Vec<Gf128> = (0..o_vars).map(|_| rg(&mut rng)).collect();
        let r_j: Vec<Gf128> = (0..j_vars).map(|_| rg(&mut rng)).collect();
        let y_mle = magnetar_field::eval_mle(&y, &[r_o.clone(), r_j.clone()].concat());
        // Joint tables.
        let x_joint: Vec<Gf128> = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let u = (idx >> u_off) & ((1usize << u_vars) - 1);
                let j = (idx >> j_off) & (cycles - 1);
                t[idx] = x[u + j * (1usize << u_vars)];
            }
            t
        };
        let q_lo_joint: Vec<Gf128> = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let hl = (idx >> (h_off + a_off_h)) & ((1usize << a_hv) - 1);
                let j = (idx >> j_off) & (cycles - 1);
                t[idx] = q_lo[hl + j * (1usize << a_hv)];
            }
            t
        };
        let q_hi_joint: Vec<Gf128> = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let hh = (idx >> (h_off + k_off_h)) & ((1usize << k_hv) - 1);
                let j = (idx >> j_off) & (cycles - 1);
                t[idx] = q_hi[hh + j * (1usize << k_hv)];
            }
            t
        };
        let r_factor: Vec<Gf128> = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let o = idx & ((1usize << o_vars) - 1);
                let u = (idx >> u_off) & ((1usize << u_vars) - 1);
                let h = (idx >> h_off) & ((1usize << h_vars) - 1);
                let ridx = o | (u << o_vars) | (h << (o_vars + u_vars));
                t[idx] = tensor[ridx];
            }
            t
        };
        let dense = vec![
            PaddedDense::new(x_joint, (0..jv).collect(), jv),
            PaddedDense::new(q_lo_joint, (0..jv).collect(), jv),
            PaddedDense::new(q_hi_joint, (0..jv).collect(), jv),
        ];
        let r_pub = PaddedTableFactor::new(r_factor, (0..jv).collect(), jv);
        let mut t = Transcript::new("router", &[b"t"]);
        let proof = prove_routing(&layout, y_mle, &r_o, &r_j, &r_pub, &dense, 2, &mut t);
        let mut tv = Transcript::new("router", &[b"t"]);
        let claims = verify_routing(&layout, y_mle, &r_o, &r_j, &r_pub, 2, 0, &proof, &mut tv)
            .expect("router sumcheck verifies");
        // The X claim must equal the X table's MLE at the claim point.
        let x_pt = claims.x_point(&layout);
        let want_x = magnetar_field::eval_mle(&x, &x_pt);
        assert_eq!(claims.x_value, want_x);
        // The selector claims match their tables' MLEs.
        let qlo_pt = claims.q_point(&layout, a_off_h, a_hv);
        let want_qlo = magnetar_field::eval_mle(&q_lo, &qlo_pt);
        assert_eq!(claims.q_values[0], want_qlo);
        let qhi_pt = claims.q_point(&layout, k_off_h, k_hv);
        let want_qhi = magnetar_field::eval_mle(&q_hi, &qhi_pt);
        assert_eq!(claims.q_values[1], want_qhi);
        // Tamper: the X claim.
        let mut p = proof.clone();
        p.x_value += ONE;
        let mut tv = Transcript::new("router", &[b"t"]);
        assert!(verify_routing(&layout, y_mle, &r_o, &r_j, &r_pub, 2, 0, &p, &mut tv).is_err());
        // Tamper: a round.
        let mut p = proof.clone();
        p.rounds[0][1] += ONE;
        let mut tv = Transcript::new("router", &[b"t"]);
        assert!(verify_routing(&layout, y_mle, &r_o, &r_j, &r_pub, 2, 0, &p, &mut tv).is_err());
    }

    #[test]
    fn store_routing_roundtrip() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let w = 8usize;
        let (tensor, o_vars, u_vars, h_vars, a_vars) = build_store_tensor(w);
        let j_vars = 3usize;
        let layout = RoutingLayout {
            o_vars,
            j_vars,
            u_vars,
            h_vars,
            a_vars,
        };
        let jv = layout.joint_vars();
        let [_, j_off, u_off, h_off, a_off] = layout.offsets();
        let cycles = 1usize << j_vars;
        // The sources: old and store words per cycle; the offset bits; the
        // selector; the increment (the output).
        let mut old = vec![ZERO; w * cycles];
        let mut src = vec![ZERO; w * cycles];
        let mut off_bits = vec![ZERO; 3 * cycles];
        let mut q_st = vec![ZERO; 4 * cycles];
        let mut inc = vec![ZERO; w * cycles];
        for j in 0..cycles {
            let o_word: u64 = rng.gen_range(0..(1u64 << w));
            let s_word: u64 = rng.gen_range(0..(1u64 << w));
            for b in 0..w {
                old[b + j * w] = Gf128(((o_word >> b) & 1) as u128);
                src[b + j * w] = Gf128(((s_word >> b) & 1) as u128);
            }
            if rng.gen::<bool>() {
                let hsel = rng.gen_range(0..4usize);
                let a = rng.gen_range(0..8usize);
                q_st[hsel + j * 4] = ONE;
                off_bits[j * 3] = Gf128((a & 1) as u128);
                off_bits[1 + j * 3] = Gf128(((a >> 1) & 1) as u128);
                off_bits[2 + j * 3] = Gf128(((a >> 2) & 1) as u128);
                let width = 1usize << hsel;
                for b in 0..w {
                    let byte = b / 8;
                    if byte >= a && byte < a + width {
                        inc[b + j * w] = old[b + j * w] + src[b + j * w];
                    }
                }
            }
        }
        // The claim: Inc at a random point.
        let r_o: Vec<Gf128> = (0..o_vars).map(|_| rg(&mut rng)).collect();
        let r_j: Vec<Gf128> = (0..j_vars).map(|_| rg(&mut rng)).collect();
        let y_mle = magnetar_field::eval_mle(&inc, &[r_o.clone(), r_j.clone()].concat());
        // Joint tables: X (over (u, j)), Q_st, the three eq-links, R.
        let x_joint: Vec<Gf128> = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let u = (idx >> u_off) & ((1usize << u_vars) - 1);
                let j = (idx >> j_off) & (cycles - 1);
                t[idx] = if u < w {
                    old[u + j * w]
                } else if u < 2 * w {
                    src[(u - w) + j * w]
                } else {
                    ZERO
                };
            }
            t
        };
        let q_joint: Vec<Gf128> = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let h = (idx >> h_off) & 3;
                let j = (idx >> j_off) & (cycles - 1);
                t[idx] = q_st[h + j * 4];
            }
            t
        };
        let links: Vec<Vec<Gf128>> = (0..3)
            .map(|k| {
                let mut t = vec![ZERO; 1usize << jv];
                for idx in 0..t.len() {
                    let a_k = (idx >> (a_off + k)) & 1;
                    let j = (idx >> j_off) & (cycles - 1);
                    t[idx] = ONE + Gf128(a_k as u128) + off_bits[k + j * 3];
                }
                t
            })
            .collect();
        let r_factor: Vec<Gf128> = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let o = idx & ((1usize << o_vars) - 1);
                let u = (idx >> u_off) & ((1usize << u_vars) - 1);
                let h = (idx >> h_off) & 3;
                let a = (idx >> a_off) & 7;
                let ridx = o
                    | (u << o_vars)
                    | (h << (o_vars + u_vars))
                    | (a << (o_vars + u_vars + h_vars));
                t[idx] = tensor[ridx];
            }
            t
        };
        let mut dense = vec![
            PaddedDense::new(x_joint, (0..jv).collect(), jv),
            PaddedDense::new(q_joint, (0..jv).collect(), jv),
        ];
        for lk in &links {
            dense.push(PaddedDense::new(lk.clone(), (0..jv).collect(), jv));
        }
        let r_pub = PaddedTableFactor::new(r_factor, (0..jv).collect(), jv);
        let mut t = Transcript::new("router2", &[b"t"]);
        let proof = prove_routing(&layout, y_mle, &r_o, &r_j, &r_pub, &dense, 1, &mut t);
        let mut tv = Transcript::new("router2", &[b"t"]);
        let claims = verify_routing(&layout, y_mle, &r_o, &r_j, &r_pub, 1, 3, &proof, &mut tv)
            .expect("store router verifies");
        // The offset-bit claims recover â_k at the *terminal* cycle slice
        // (the router's own random point, not the claim's r_j).
        let [_, j_off, _, _, _] = layout.offsets();
        let r_j_term: Vec<Gf128> = claims.r_joint[j_off..j_off + layout.j_vars].to_vec();
        for k in 0..3 {
            let got = claims.offset_bit_value(&layout, k);
            let col: Vec<Gf128> = (0..cycles).map(|j| off_bits[k + j * 3]).collect();
            let want = magnetar_field::eval_mle(&col, &r_j_term);
            assert_eq!(got, want, "offset bit {k}");
        }
        // The X claim splits into the two segment claims.
        let x_pt = claims.x_point(&layout);
        let (w_old, r_bits) = {
            let ((w0, b), _) = store_x_split(&x_pt[..u_vars], claims.x_value);
            (w0, b)
        };
        let _ = (w_old, r_bits);
        // Direct: X̂(r_u, r_j) over the (u, j) table.
        let x_table: Vec<Gf128> = {
            let mut t2 = vec![ZERO; (1usize << u_vars) * cycles];
            for j in 0..cycles {
                for u in 0..(1usize << u_vars) {
                    t2[u + j * (1usize << u_vars)] = if u < w {
                        old[u + j * w]
                    } else if u < 2 * w {
                        src[(u - w) + j * w]
                    } else {
                        ZERO
                    };
                }
            }
            t2
        };
        let want_x = magnetar_field::eval_mle(&x_table, &x_pt);
        assert_eq!(claims.x_value, want_x);
        // Tamper: the offset link claim.
        let mut p = proof.clone();
        p.offset_values[0] += ONE;
        let mut tv = Transcript::new("router2", &[b"t"]);
        assert!(verify_routing(&layout, y_mle, &r_o, &r_j, &r_pub, 1, 3, &p, &mut tv).is_err());
    }
}
