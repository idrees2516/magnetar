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
//! Given a claim `Ŷ(r_o, r_j) = v` (from Virtual Spartan), one sum-check
//! over the **joint (o, j, u, h) space** reduces it to evaluation claims on
//! `X` (a direct witness segment) and on the selector factors `Q`
//! (bytecode fields, discharged by BinaryShout). The verifier evaluates
//! `R̂` itself.
//!
//! * Shifts (Eq. 5): `Q_sh(h, j) = Q_lo(h_lo, j)·Q_hi(h_hi, j)` — the
//!   routing index is split into two shorter one-hot fields; the sources
//!   include a constant-zero and the sign bit for arithmetic right shifts.
//! * Sub-word stores (Eq. 6): `R_st(o, u, h, a)` takes the byte offset
//!   `a ∈ B³` as three additional Boolean **summed** variables, linked to
//!   the witness offset bits via `eq(a_k, â_k(j))` factors; inside a
//!   written byte the increment bit is the XOR of the old-RAM bit and the
//!   store-source bit, expressed by routing the output from *both* source
//!   bits. The output claim is an evaluation of `Inc` at a random
//!   verifier-chosen point (the paper's Step 4 tie-in), and the reduction
//!   outputs claims on the three offset bits (a direct segment).
//!
//! Joint-space layout (little-endian, low to high): `o | j | u | h | a`.

#![forbid(unsafe_code)]

use magnetar_field::{mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::TableFactor,
    BindOrder, RoundMsg, SumcheckSpec, Term,
};

/// A public table over a subset of the joint variables, padded across the
/// rest (padding leaves the MLE unchanged: the eq factors sum to 1).
#[derive(Clone, Debug)]
pub struct PaddedTableFactor {
    pub table: Vec<Gf128>,
    /// Number of variables the table actually depends on.
    pub table_vars: usize,
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
            table_vars,
            active,
            joint_vars,
        }
    }
}

impl magnetar_sumcheck::public::PublicFactor for PaddedTableFactor {
    fn num_vars(&self) -> usize {
        self.joint_vars
    }
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

/// Output claims of a routing reduction.
#[derive(Clone, Debug)]
pub struct RoutingClaims {
    /// The X claim at `(r_u, r_j)`.
    pub x_point: Vec<Gf128>,
    pub x_value: Gf128,
    /// The selector claims at `(r_h, r_j)` — one per selector factor.
    pub q_points: Vec<Vec<Gf128>>,
    pub q_values: Vec<Gf128>,
    /// (Stores only) the offset-bit claims at `r_j`.
    pub offset_points: Vec<Vec<Gf128>>,
    pub offset_values: Vec<Gf128>,
}

/// The routing proof.
#[derive(Clone, Debug)]
pub struct RouterProof {
    pub rounds: Vec<Vec<Gf128>>,
    /// The X terminal claim.
    pub x_value: Gf128,
    /// The selector terminal claims.
    pub q_values: Vec<Gf128>,
    /// (Stores) the offset-bit terminal claims.
    pub offset_values: Vec<Gf128>,
}

/// Prove a generic routing reduction.
///
/// Dense factor order: `X` first, then `n_selectors` selector factors
/// (`Q_lo`, `Q_hi` for shifts; `Q_st` for stores), then the store offset
/// link factors (if any).
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
    let eq_o_factor = PaddedTableFactor::new(eq_o, (0..layout.o_vars).collect(), jv);
    let eq_j_factor = PaddedTableFactor::new(eq_j, (j_off..j_off + layout.j_vars).collect(), jv);
    let dense_tables: Vec<Vec<Gf128>> = dense.iter().map(|d| d.materialize()).collect();
    let terms = vec![Term {
        coeff: ONE,
        dense: (0..dense.len()).collect(),
        public: vec![0, 1, 2],
    }];
    let spec = SumcheckSpec {
        num_vars: jv,
        dense: dense_tables,
        public: vec![
            Box::new(eq_o_factor),
            Box::new(eq_j_factor),
            Box::new(r_public.clone()),
        ],
        terms,
        claim,
    };
    let mut prover = SumcheckProver::new(&spec, BindOrder::Increasing);
    let mut rounds = Vec::with_capacity(jv);
    for _ in 0..jv {
        let msg = prover.message();
        rounds.push(msg.0.clone());
        prover.absorb_and_fold(t, "router");
    }
    let r_joint = prover.challenges();
    let (claims, _) = prover.finish();
    let _ = r_joint;
    RouterProof {
        rounds,
        x_value: claims[0],
        q_values: claims[1..1 + n_selectors].to_vec(),
        offset_values: claims[1 + n_selectors..].to_vec(),
    }
}

/// Verify a routing reduction (mirrors the prover's sumcheck; the caller
/// supplies the same factor structure). Returns the accepted terminal
/// claims (X, selectors, offsets) at the jointly-derived random point.
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
    let [_, j_off, u_off, h_off, a_off] = layout.offsets();
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
    // Split the joint point into per-factor points.
    let r_u: Vec<Gf128> = r_joint[u_off..u_off + layout.u_vars].to_vec();
    let r_h: Vec<Gf128> = r_joint[h_off..h_off + layout.h_vars].to_vec();
    let j_pt: Vec<Gf128> = r_joint[j_off..j_off + layout.j_vars].to_vec();
    let a_pt: Vec<Gf128> = r_joint[a_off..a_off + layout.a_vars].to_vec();
    Ok(RoutingClaims {
        x_point: [r_u.clone(), j_pt.clone()].concat(),
        x_value: proof.x_value,
        q_points: vec![[r_h.clone(), j_pt.clone()].concat()],
        q_values: proof.q_values.clone(),
        offset_points: (0..layout.a_vars)
            .map(|k| [vec![r_joint[a_off + k]], j_pt.clone()].concat())
            .collect(),
        offset_values: proof.offset_values.clone(),
    })
}

/// Build the shift routing tensor `R_sh(o, u, h)` (Equation 5).
///
/// Sources `u`: `[0] = constant zero, [1..=w] = the w source bits
/// (bit u−1), [w+1] = the sign bit (source bit w−1)`. Routings
/// `h = (kind, amount)` with kind ∈ {sll, srl, sra} and amount < 2^{bits}:
/// * sll: `Y[o] = X[o − amount]` for `o ≥ amount`, else 0.
/// * srl: `Y[o] = X[o + amount]` for `o + amount < w`, else 0.
/// * sra: like srl but fill with the sign bit.
pub fn build_shift_tensor(w: usize, amount_bits: usize) -> (Vec<Gf128>, usize, usize, usize) {
    // u layout: 1 (zero) + w (bits) + 1 (sign) = w + 2 sources.
    let u_count = w + 2;
    let h_count = 3 * (1usize << amount_bits);
    let o_vars = w.trailing_zeros() as usize;
    let u_vars = (u_count as usize).next_power_of_two().trailing_zeros() as usize;
    let h_vars = (h_count as usize).next_power_of_two().trailing_zeros() as usize;
    let mut r = vec![ZERO; (1usize << o_vars) * (1usize << u_vars) * (1usize << h_vars)];
    for kind in 0..3usize {
        for amount in 0..(1usize << amount_bits) {
            let h = kind * (1usize << amount_bits) + amount;
            for o in 0..w {
                let src = match kind {
                    0 => {
                        // sll: source u = o - amount + 1 (bit o-amount), or
                        // the zero source.
                        if o >= amount {
                            Some(o - amount + 1)
                        } else {
                            Some(0) // zero
                        }
                    }
                    1 => {
                        // srl: bit o + amount, or zero.
                        if o + amount < w {
                            Some(o + amount + 1)
                        } else {
                            Some(0)
                        }
                    }
                    _ => {
                        // sra: bit o + amount, or the sign source (w+1).
                        if o + amount < w {
                            Some(o + amount + 1)
                        } else {
                            Some(w + 1)
                        }
                    }
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

/// Build the sub-word-store tensor `R_st(o, u, h, a)` (Equation 6).
///
/// Sources `u`: `[0] = old-RAM word bits, [1] = store-source bits` — each
/// of width `w`. Routings `h` ∈ {none, byte, half, word, doubleword} (the
/// one-hot selector width). For output bit `o` with byte offset `a`:
/// * outside the written lanes: the increment bit is 0 (route from a
///   virtual zero — expressed by routing from *both* sources, whose XOR
///   cancels); inside: the increment = old XOR store (route from both).
///
/// We model `X` as the *pair* (old, store): `X(2·b + s, j)` gives old-bit
/// `b` (s=0) or store-bit `b` (s=1). `R_st` feeds each output from both
/// sources of the relevant bit: `R = 1` at `(o, u = old_o, h, a)` and
/// `(o, u = store_o, h, a)` when bit `o` is inside the written region for
/// `(h, a)`; outside, `R = 1` at both as well but the sources carry equal
/// bits, so the XOR cancels to zero — this expresses the identity with the
/// paper's "feeding that output bit from both source bits".
pub fn build_store_tensor(w: usize) -> (Vec<Gf128>, usize, usize, usize, usize) {
    // u layout: 2w sources (old bit b => u = 2b, store bit b => u = 2b+1).
    let u_count = 2 * w;
    let h_count = 5; // none, byte, half, word, doubleword (one-hot)
    let a_count = 8; // 3 offset bits
    let o_vars = w.trailing_zeros() as usize;
    let u_vars = (u_count as usize).next_power_of_two().trailing_zeros() as usize;
    let h_vars = (h_count as usize).next_power_of_two().trailing_zeros() as usize;
    let a_vars = 3usize;
    let mut r =
        vec![
            ZERO;
            (1usize << o_vars) * (1usize << u_vars) * (1usize << h_vars) * (1usize << a_vars)
        ];
    for h in 0..h_count {
        // The write width in bytes per routing.
        let width = match h {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 4,
            _ => 8,
        };
        for a in 0..a_count {
            for o in 0..w {
                // Is bit o inside [8a, 8a + 8·width)?
                let byte = o / 8;
                let inside = byte >= a && byte < a + width;
                // Route from both the old and store source of bit o —
                // outside the written region the sources are equal by
                // construction of X, so the XOR is 0.
                if inside || h == 0 {
                    for s in 0..2usize {
                        let u = 2 * o + s;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut dyn rand::RngCore) -> Gf128 {
        Gf128(rand::Rng::gen::<u128>(rng))
    }

    #[test]
    fn shift_tensor_shape() {
        let (r, o_vars, u_vars, h_vars) = build_shift_tensor(16, 2);
        assert_eq!(o_vars, 4);
        assert_eq!(u_vars, 5usize); // w=16 -> 18 sources -> 32 slots -> 5 vars
        assert_eq!(h_vars, 4usize);
        // sll by 1: output 5 fed by source bit 4.
        let idx = 5 | ((4 + 1) << o_vars) | ((0 * 4 + 1) << (o_vars + u_vars));
        assert_eq!(r[idx], ONE);
        // sll by 1: output 0 fed by the zero source.
        let idx0 = 0 | (0 << o_vars) | (1 << (o_vars + u_vars));
        assert_eq!(r[idx0], ONE);
    }

    #[test]
    fn store_tensor_shape() {
        let (r, o_vars, u_vars, h_vars, a_vars) = build_store_tensor(16);
        assert_eq!(a_vars, 3);
        // Byte store (h=1) at offset a=1, bit o=9 (byte 1): inside [8,16).
        for s in 0..2usize {
            let u = 2 * 9 + s;
            let idx =
                9 | (u << o_vars) | (1 << (o_vars + u_vars)) | (1 << (o_vars + u_vars + h_vars));
            assert_eq!(r[idx], ONE);
        }
        // Bit o=1 (byte 0): outside [8,16) for a=1 — not routed.
        for s in 0..2usize {
            let u = 2 * 1 + s;
            let idx =
                1 | (u << o_vars) | (1 << (o_vars + u_vars)) | (1 << (o_vars + u_vars + h_vars));
            assert_eq!(r[idx], ZERO);
        }
        let _ = (h_vars, a_vars);
    }

    #[test]
    fn routing_sumcheck_roundtrip() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // Small shift instance: w = 4 bits, 2 amount bits.
        let w = 4usize;
        let (tensor, o_vars, u_vars, h_vars) = build_shift_tensor(w, 2);
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
        // X table: per cycle, the source bits (w bits + sign).
        let cycles = 1usize << j_vars;
        let u_count = w + 2;
        let mut x = vec![ZERO; (1usize << u_vars) * cycles];
        for j in 0..cycles {
            for b in 0..w {
                let bit = rng.gen::<bool>();
                x[(b + 1) + j * (1usize << u_vars)] = Gf128(bit as u128);
            }
            x[0 + j * (1usize << u_vars)] = ZERO; // zero source
            x[(w + 1) + j * (1usize << u_vars)] = x[(w) + j * (1usize << u_vars)];
        }
        // Selectors: Q_lo (amount, 2 vars) and Q_hi (kind, 2 vars) — one-hot
        // per cycle (or all-zero for non-shift cycles).
        let h_lo_vars = 2usize;
        let h_hi_vars = 2usize;
        let mut q_lo = vec![ZERO; (1usize << h_lo_vars) * cycles];
        let mut q_hi = vec![ZERO; (1usize << h_hi_vars) * cycles];
        let mut y = vec![ZERO; (1usize << o_vars) * cycles];
        for j in 0..cycles {
            if rng.gen::<bool>() {
                let amount = rng.gen_range(0..4usize);
                let kind = rng.gen_range(0..3usize);
                q_lo[amount + j * 4] = ONE;
                q_hi[kind + j * 4] = ONE;
                // Compute Y(o, j) from the identity.
                for o in 0..w {
                    let mut acc = ZERO;
                    for u in 0..u_count {
                        for hl in 0..4usize {
                            for hh in 0..4usize {
                                let h = hh * 4 + hl;
                                if h < 12 {
                                    let ridx = o | (u << o_vars) | (h << (o_vars + u_vars));
                                    acc = acc
                                        + mul(
                                            mul(
                                                x[u + j * (1usize << u_vars)],
                                                mul(q_lo[hl + j * 4], q_hi[hh + j * 4]),
                                            ),
                                            tensor[ridx],
                                        );
                                }
                            }
                        }
                    }
                    y[o + j * (1usize << o_vars)] = acc;
                }
            }
        }
        // Claim point.
        let r_o: Vec<Gf128> = (0..o_vars).map(|_| rg(&mut rng)).collect();
        let r_j: Vec<Gf128> = (0..j_vars).map(|_| rg(&mut rng)).collect();
        let y_mle = {
            // Y as a joint (o, j) table (o low, j high).
            let mut joint = vec![ZERO; (1usize << o_vars) * cycles];
            for o in 0..(1usize << o_vars) {
                for j in 0..cycles {
                    joint[o + j * (1usize << o_vars)] = y[o + j * (1usize << o_vars)];
                }
            }
            magnetar_field::eval_mle(&joint, &[r_o.clone(), r_j.clone()].concat())
        };
        // Dense factors: X (over u|j), Q_lo (over h_lo|j), Q_hi (over h_hi|j).
        let x_joint = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let u = (idx >> u_off) & ((1usize << u_vars) - 1);
                let j = (idx >> j_off) & (cycles - 1);
                t[idx] = if u < u_count {
                    x[u + j * (1usize << u_vars)]
                } else {
                    ZERO
                };
            }
            t
        };
        let q_lo_joint = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let hl = (idx >> h_off) & 3;
                let j = (idx >> j_off) & (cycles - 1);
                t[idx] = q_lo[hl + j * 4];
            }
            t
        };
        let q_hi_joint = {
            let mut t = vec![ZERO; 1usize << jv];
            for idx in 0..t.len() {
                let hh = (idx >> (h_off + h_lo_vars)) & 3;
                let j = (idx >> j_off) & (cycles - 1);
                t[idx] = q_hi[hh + j * 4];
            }
            t
        };
        // The R tensor padded over the joint space (o, u, h active; j idle).
        let r_factor = {
            // The tensor's layout: o | u | h — map to joint bits o | u | h
            // (skipping j). Build the padded table directly.
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
        // Verify: replay the sumcheck with the same structure.
        let eq_o = magnetar_field::EqBits::build(&r_o).values;
        let eq_j = magnetar_field::EqBits::build(&r_j).values;
        let eq_o_f = PaddedTableFactor::new(eq_o, (0..o_vars).collect(), jv);
        let eq_j_f = PaddedTableFactor::new(eq_j, (j_off..j_off + j_vars).collect(), jv);
        // The public R factor for the verifier: evaluate directly from the
        // sparse tensor (padded eval).
        let r_verifier = SparseRTensor {
            tensor: tensor.clone(),
            o_vars,
            u_vars,
            h_vars,
            jv,
            u_off,
            h_off,
        };
        let public: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
            vec![Box::new(eq_o_f), Box::new(eq_j_f), Box::new(r_verifier)];
        let terms = vec![Term {
            coeff: ONE,
            dense: vec![0, 1, 2],
            public: vec![0, 1, 2],
        }];
        let mut tv = Transcript::new("router", &[b"t"]);
        let mut v = SumcheckVerifier::new(jv, y_mle, &public, terms, BindOrder::Increasing);
        for msg in &proof.rounds {
            v.round_capturing(&RoundMsg(msg.clone()), &mut tv, "router")
                .unwrap();
        }
        let claims = vec![proof.x_value, proof.q_values[0], proof.q_values[1]];
        v.finish(&claims).expect("router sumcheck verifies");
    }

    /// Verifier-side sparse R evaluation (O(nnz) MLE eval via the padded
    /// point extraction).
    #[derive(Debug)]
    struct SparseRTensor {
        tensor: Vec<Gf128>,
        o_vars: usize,
        u_vars: usize,
        h_vars: usize,
        jv: usize,
        u_off: usize,
        h_off: usize,
    }

    impl magnetar_sumcheck::public::PublicFactor for SparseRTensor {
        fn num_vars(&self) -> usize {
            self.jv
        }
        fn materialize(&self) -> Vec<Gf128> {
            // Test-scale materialization.
            let mut t = vec![ZERO; 1usize << self.jv];
            for idx in 0..t.len() {
                let o = idx & ((1usize << self.o_vars) - 1);
                let u = (idx >> self.u_off) & ((1usize << self.u_vars) - 1);
                let h = (idx >> self.h_off) & ((1usize << self.h_vars) - 1);
                t[idx] = self.tensor[o | (u << self.o_vars) | (h << (self.o_vars + self.u_vars))];
            }
            t
        }
        fn eval_at(&self, point: &[Gf128]) -> Gf128 {
            let o: Vec<Gf128> = point[..self.o_vars].to_vec();
            let u: Vec<Gf128> = point[self.u_off..self.u_off + self.u_vars].to_vec();
            let h: Vec<Gf128> = point[self.h_off..self.h_off + self.h_vars].to_vec();
            magnetar_field::eval_mle(&self.tensor, &[o, u, h].concat())
        }
    }
}
