//! The sparse sum-check prover (the paper's "the prover's work is linear
//! in the trace length"): every factor is kept as its **own** table over
//! only the variables it depends on, and a round folds a factor only when
//! the round's variable is one of its own. Round messages are computed by
//! accumulating, per remaining-cube assignment, the product of the factors
//! evaluated through their active-variable slices — building the round
//! polynomial through per-assignment elementary products.
//!
//! **Sparse anchors.** A factor whose table is mostly zero (the routing
//! tensors: one source per output bit, ~1/8–1/32 dense) anchors the
//! enumeration on its nonzero list: the per-round work becomes
//! `nnz(anchor) × 2^(remaining vars the anchor does not cover)` instead of
//! the full remaining joint cube. Zero anchor entries contribute nothing
//! to the sum, so the anchored message is identical to the dense
//! enumeration's — for the store routing at a 2^10-cycle trace this is the
//! difference between 2^24 and 2^13 assignments per round.
//!
//! The verifier is unchanged (the standard engine: round checks are `O(d)`
//! per round and the terminal evaluates the public factors at the point),
//! so a sparse proof verifies with the regular [`SumcheckVerifier`] against
//! the full-space public factors.

use crate::{BindOrder, RoundMsg, Term};
use magnetar_field::{mul, Gf128, ZERO};
use magnetar_hash::Transcript;

/// A prover-side factor: a table over its own variables (in the table's
/// variable order) plus the joint bit position of each of its variables.
#[derive(Clone, Debug)]
pub struct SparseFactor {
    pub table: Vec<Gf128>,
    /// `active[t]` = the joint bit position of the table's variable `t`.
    pub active: Vec<usize>,
    /// Marked automatically when the table is mostly zero: the prover
    /// anchors its enumeration on the nonzero list (rebuilt after every
    /// fold) instead of the full remaining cube.
    sparse: bool,
    /// The nonzero entries `(index, value)`, maintained when `sparse`.
    nnz: Vec<(usize, Gf128)>,
}

impl SparseFactor {
    pub fn new(table: Vec<Gf128>, active: Vec<usize>) -> Self {
        assert_eq!(table.len(), 1usize << active.len());
        let sparse = Self::sparsity_hint(&table);
        let nnz = if sparse {
            table
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != ZERO)
                .map(|(i, &v)| (i, v))
                .collect()
        } else {
            Vec::new()
        };
        SparseFactor {
            table,
            active,
            sparse,
            nnz,
        }
    }

    /// A table is worth anchoring on when its nonzero share is low. The
    /// check samples 64 positions so construction stays cheap even for
    /// large tensors.
    fn sparsity_hint(table: &[Gf128]) -> bool {
        if table.len() < 64 {
            return false;
        }
        let stride = table.len() / 64;
        let mut zeros = 0usize;
        for i in (0..table.len()).step_by(stride) {
            if table[i] == ZERO {
                zeros += 1;
            }
        }
        zeros > 48 // more than 3/4 of the sample is zero
    }

    /// Rebuild the nonzero list after a fold (the table halves).
    fn rebuild_nnz(&mut self) {
        if self.sparse {
            self.nnz = self
                .table
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != ZERO)
                .map(|(i, &v)| (i, v))
                .collect();
        }
    }

    fn depends_on(&self, joint_var: usize) -> bool {
        self.active.contains(&joint_var)
    }

    /// The local position of a joint variable, if active.
    fn local_pos(&self, joint_var: usize) -> Option<usize> {
        self.active.iter().position(|&x| x == joint_var)
    }

    /// The "savings" of anchoring on this factor: the joint variables
    /// covered per nonzero entry (higher is better; `MIN` when not
    /// sparse).
    fn anchor_savings(&self) -> f64 {
        if !self.sparse || self.nnz.is_empty() {
            return f64::MIN;
        }
        self.active.len() as f64 - (self.nnz.len() as f64).log2()
    }
}

/// A sparse sum-check specification: the factors in their own spaces, the
/// terms referencing them by index, and the claimed sum over the joint
/// cube.
#[derive(Debug)]
pub struct SparseSpec {
    pub num_vars: usize,
    /// Dense (prover) factors — their terminals become the output claims.
    pub dense: Vec<SparseFactor>,
    /// Public (verifier-evaluable) factors, materialized by the caller.
    pub public: Vec<SparseFactor>,
    pub terms: Vec<Term>,
    pub claim: Gf128,
}

/// The sparse prover. Produces round messages verifiable by the standard
/// `SumcheckVerifier` (with the public factors in full-space form and the
/// same terms).
pub struct SparseProver {
    num_vars: usize,
    dense: Vec<SparseFactor>,
    public: Vec<SparseFactor>,
    terms: Vec<Term>,
    running: Gf128,
    order: BindOrder,
    round: usize,
    challenges: Vec<Gf128>,
    /// `bound[j]` = the joint variable `j` is already bound.
    bound: Vec<bool>,
}

impl SparseProver {
    pub fn new(spec: SparseSpec, order: BindOrder) -> Self {
        for t in &spec.terms {
            for &d in &t.dense {
                assert!(d < spec.dense.len(), "dense index out of range");
            }
            for &p in &t.public {
                assert!(p < spec.public.len(), "public index out of range");
            }
        }
        SparseProver {
            num_vars: spec.num_vars,
            dense: spec.dense,
            public: spec.public,
            terms: spec.terms,
            running: spec.claim,
            order,
            round: 0,
            challenges: Vec::new(),
            bound: vec![false; spec.num_vars],
        }
    }

    fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|t| t.dense.len() + t.public.len())
            .max()
            .unwrap_or(0)
    }

    /// The remaining (unbound, non-round) joint variables, ascending.
    fn remaining(&self, v: usize) -> Vec<usize> {
        (0..self.num_vars)
            .filter(|&x| !self.bound[x] && x != v)
            .collect()
    }

    /// The current round's message.
    pub fn message(&self) -> RoundMsg {
        let v = self.order.round_var(self.num_vars, self.round);
        let degree = self.degree();
        let mut coeffs = vec![ZERO; degree + 1];
        // Pick the best sparse anchor: the factor covering the most joint
        // variables per nonzero entry.
        let mut anchor: Option<(&SparseFactor, bool)> = None;
        let mut best = 0.0f64;
        for f in self.dense.iter().chain(self.public.iter()) {
            let s = f.anchor_savings();
            if s > best {
                best = s;
                let has_v = f.local_pos(v).is_some();
                anchor = Some((f, has_v));
            }
        }
        // The fixed-part multiplication order, computed once per round:
        // round-inactive factors, sparse ones first.
        let anchor_ptr = anchor.map(|(f, _)| f as *const SparseFactor);
        let mut fixed_order: Vec<&SparseFactor> = self
            .dense
            .iter()
            .chain(self.public.iter())
            .filter(|f| !f.depends_on(v) && anchor_ptr != Some(*f as *const SparseFactor))
            .collect();
        fixed_order.sort_by_key(|f| !f.sparse);
        match anchor {
            Some((f, true)) => self.message_anchored_round(v, f, &fixed_order, &mut coeffs),
            Some((f, false)) => self.message_anchored_fixed(v, f, &fixed_order, &mut coeffs),
            None => self.message_dense(v, &fixed_order, &mut coeffs),
        }
        RoundMsg(coeffs)
    }

    /// The classic full-cube enumeration (no anchor): `r` enumerates the
    /// remaining variables in `u`-order.
    fn message_dense(&self, v: usize, fixed_order: &[&SparseFactor], coeffs: &mut [Gf128]) {
        let u = self.remaining(v);
        let mut bit_index = [0usize; 64];
        for (h, &x) in u.iter().enumerate() {
            bit_index[x] = h;
        }
        let n_rem = 1usize << u.len();
        for r in 0..n_rem {
            self.accumulate(v, &bit_index, r, None, fixed_order, false, coeffs);
        }
    }

    /// Anchored enumeration when the anchor does **not** contain the
    /// round variable: the anchor's value is a fixed multiplier per
    /// nonzero entry, and the remaining variables outside the anchor are
    /// enumerated.
    fn message_anchored_fixed(
        &self,
        v: usize,
        anchor: &SparseFactor,
        fixed_order: &[&SparseFactor],
        coeffs: &mut [Gf128],
    ) {
        let u = self.remaining(v);
        // `r` enumerates the remaining variables the anchor does not
        // cover; the anchor's own variables come from the nonzero entry.
        let rest: Vec<usize> = u
            .iter()
            .copied()
            .filter(|x| !anchor.active.contains(x))
            .collect();
        let mut bit_index = [0usize; 64];
        for (h, &x) in rest.iter().enumerate() {
            bit_index[x] = h;
        }
        let n_rest = 1usize << rest.len();
        for &(idx, aval) in anchor.nnz.iter() {
            for r in 0..n_rest {
                self.accumulate(
                    v,
                    &bit_index,
                    r,
                    Some((anchor, idx, aval)),
                    fixed_order,
                    true,
                    coeffs,
                );
            }
        }
    }

    /// Anchored enumeration when the anchor **contains** the round
    /// variable `v`: iterate the anchor's nonzero **bases** — the pairs
    /// `(A, D) = (table[base], table[base] + table[base | bit])` for every
    /// base with at least one nonzero child — deduplicated across the two
    /// children's entries.
    fn message_anchored_round(
        &self,
        v: usize,
        anchor: &SparseFactor,
        fixed_order: &[&SparseFactor],
        coeffs: &mut [Gf128],
    ) {
        let pos = anchor.local_pos(v).expect("anchor contains the round var");
        let u = self.remaining(v);
        // `r` enumerates the remaining variables the anchor does not
        // cover (the anchor's other variables are fixed by each base).
        let rest: Vec<usize> = u
            .iter()
            .copied()
            .filter(|x| !anchor.active.contains(x))
            .collect();
        let mut bit_index = [0usize; 64];
        for (h, &x) in rest.iter().enumerate() {
            bit_index[x] = h;
        }
        let bit = 1usize << pos;
        // The distinct bases with a nonzero child, in first-seen order.
        let mut seen = std::collections::HashSet::new();
        let mut bases: Vec<usize> = Vec::with_capacity(anchor.nnz.len());
        for &(idx, _) in anchor.nnz.iter() {
            let base = idx & !bit;
            if seen.insert(base) {
                bases.push(base);
            }
        }
        let n_rest = 1usize << rest.len();
        for &base in bases.iter() {
            // The (A, D) pair at this base is read from the anchor's
            // table inside `accumulate` (the base determines both
            // children); the entry value is unused on this path.
            let a = anchor.table[base];
            for r in 0..n_rest {
                self.accumulate(
                    v,
                    &bit_index,
                    r,
                    Some((anchor, base, a)),
                    fixed_order,
                    false,
                    coeffs,
                );
            }
        }
    }

    /// Accumulate one remaining-cube assignment's contribution into the
    /// round polynomial coefficients. When an anchor entry is supplied,
    /// the assignment is *completed by* the anchor's index bits: the
    /// anchor's own factor value/pair comes from that entry (never from a
    /// full-table lookup of the anchor).
    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        &self,
        v: usize,
        bit_index: &[usize; 64],
        r: usize,
        anchor: Option<(&SparseFactor, usize, Gf128)>,
        fixed_order: &[&SparseFactor],
        anchor_is_fixed: bool,
        coeffs: &mut [Gf128],
    ) {
        let (anchor, a_idx, a_val) = match anchor {
            Some((f, idx, val)) => (Some(f), idx, val),
            None => (None, 0usize, ZERO),
        };
        let anchor_ptr = anchor.map(|f| f as *const SparseFactor);
        // The per-assignment bit reader: the anchor's bits come from its
        // entry index; every other remaining variable's bit comes from
        // `r` through `bit_index` (built by the enumeration driver over
        // exactly the variables `r` spans).
        let bit_of = |var: usize| -> bool {
            if let Some(f) = anchor {
                if let Some(p) = f.local_pos(var) {
                    return (a_idx >> p) & 1 == 1;
                }
            }
            (r >> bit_index[var]) & 1 == 1
        };
        for term in &self.terms {
            // Fixed part: every round-inactive factor, sparse ones first
            // (a zero one-hot selector short-circuits the assignment
            // before any dense table lookup). The order is computed once
            // per round by the driver.
            let mut fixed = term.coeff;
            let mut zero = false;
            if anchor_is_fixed {
                fixed = mul(fixed, a_val);
                if fixed == ZERO {
                    zero = true;
                }
            }
            if !zero {
                for f in fixed_order {
                    let mut sub = 0usize;
                    for (t, &av) in f.active.iter().enumerate() {
                        if bit_of(av) {
                            sub |= 1usize << t;
                        }
                    }
                    fixed = mul(fixed, f.table[sub]);
                    if fixed == ZERO {
                        zero = true;
                        break;
                    }
                }
            }
            if zero {
                continue;
            }
            // The (A, D) pairs of the round-active factors. The anchor's
            // pair comes from its base entry (both children determined);
            // the others from their sub-index lookups.
            let mut ads: Vec<(Gf128, Gf128)> = Vec::new();
            for f in self.dense.iter().chain(self.public.iter()) {
                if let Some(pos) = f.local_pos(v) {
                    let (a, d) = if anchor_ptr == Some(f as *const SparseFactor) {
                        let base = a_idx & !(1usize << pos);
                        (
                            f.table[base],
                            f.table[base] + f.table[base | (1usize << pos)],
                        )
                    } else {
                        let mut sub = 0usize;
                        for (t, &av) in f.active.iter().enumerate() {
                            if av == v {
                                continue;
                            }
                            if bit_of(av) {
                                sub |= 1usize << t;
                            }
                        }
                        let base = sub & !(1usize << pos);
                        (
                            f.table[base],
                            f.table[base] + f.table[base | (1usize << pos)],
                        )
                    };
                    ads.push((a, d));
                }
            }
            // fixed · Π (A + X·D) — accumulate the coefficient vector.
            let mut poly = vec![fixed];
            for (a, d) in ads {
                let mut next = vec![ZERO; poly.len() + 1];
                for (k, &c) in poly.iter().enumerate() {
                    next[k] += mul(c, a);
                    next[k + 1] += mul(c, d);
                }
                poly = next;
            }
            for (k, &c) in poly.iter().enumerate() {
                if k < coeffs.len() {
                    coeffs[k] += c;
                }
            }
        }
    }

    /// Absorb the message, sample the challenge, fold, and advance.
    pub fn absorb_and_fold(&mut self, t: &mut Transcript, label: &str) -> Gf128 {
        let msg = self.message();
        let mut buf = Vec::with_capacity(msg.0.len() * 16);
        for c in &msg.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(label, &buf);
        let r = t.challenge_field(label);
        self.fold(r);
        r
    }

    /// Fold at the challenge and advance (the running claim becomes
    /// `m(r)`).
    pub fn fold(&mut self, r: Gf128) {
        let msg = self.message();
        let mut mr = ZERO;
        let mut xp = Gf128(1);
        for &c in &msg.0 {
            mr += mul(c, xp);
            xp = mul(xp, r);
        }
        self.running = mr;
        let v = self.order.round_var(self.num_vars, self.round);
        // Fold only the factors that depend on v.
        let fold_factor = |f: &mut SparseFactor| {
            if let Some(pos) = f.local_pos(v) {
                magnetar_field::bind_var(&mut f.table, pos, r);
                f.active.remove(pos);
                f.rebuild_nnz();
            }
        };
        for f in self.dense.iter_mut() {
            fold_factor(f);
        }
        for f in self.public.iter_mut() {
            fold_factor(f);
        }
        self.bound[v] = true;
        self.challenges.push(r);
        self.round += 1;
    }

    /// The dense factors' terminal claims (their folded single entries —
    /// every factor is fully folded once all its variables are bound).
    pub fn finish(self) -> (Vec<Gf128>, Gf128) {
        debug_assert_eq!(self.round, self.num_vars);
        let claims = self
            .dense
            .iter()
            .map(|f| {
                debug_assert_eq!(f.table.len(), 1);
                f.table[0]
            })
            .collect();
        (claims, self.running)
    }

    /// The challenges so far (binding order).
    pub fn challenges(&self) -> Vec<Gf128> {
        self.challenges.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public::PublicFactor;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rng.gen::<u128>())
    }

    /// A random sparse instance cross-checked against the dense engine:
    /// the sparse prover's proof must verify under the standard
    /// `SumcheckVerifier` with full-space public factors.
    #[test]
    fn sparse_matches_dense_engine() {
        let mut rng = rand::thread_rng();
        // Joint layout: a | b | c | d (4 vars). Factors: X over (a, d),
        // Y over (b, d), public P over (a, b, c).
        let m = 4;
        let active_x = vec![0usize, 3];
        let active_y = vec![1usize, 3];
        let active_p = vec![0usize, 1, 2];
        let tx: Vec<Gf128> = (0..4).map(|_| rg(&mut rng)).collect();
        let ty: Vec<Gf128> = (0..4).map(|_| rg(&mut rng)).collect();
        let tp: Vec<Gf128> = (0..8).map(|_| rg(&mut rng)).collect();
        // The claim: Σ_{joint} P·X·Y.
        let claim = {
            let mut acc = ZERO;
            for idx in 0..(1usize << m) {
                let a = idx & 1;
                let b = (idx >> 1) & 1;
                let c = (idx >> 2) & 1;
                let d = (idx >> 3) & 1;
                let x = tx[a | (d << 1)];
                let y = ty[b | (d << 1)];
                let p = tp[a | (b << 1) | (c << 2)];
                acc += mul(mul(p, x), y);
            }
            acc
        };
        let spec = SparseSpec {
            num_vars: m,
            dense: vec![
                SparseFactor::new(tx.clone(), active_x.clone()),
                SparseFactor::new(ty.clone(), active_y.clone()),
            ],
            public: vec![SparseFactor::new(tp.clone(), active_p.clone())],
            terms: vec![Term {
                coeff: Gf128(1),
                dense: vec![0, 1],
                public: vec![0],
            }],
            claim,
        };
        let mut t = Transcript::new("sparse", &[b"t"]);
        let mut prover = SparseProver::new(spec, BindOrder::Increasing);
        let mut rounds = Vec::new();
        for _ in 0..m {
            let msg = prover.message();
            rounds.push(msg.0.clone());
            prover.absorb_and_fold(&mut t, "sp");
        }
        let (claims, _) = prover.finish();
        // Verify with the standard engine: the public factor in full
        // space.
        let full_p: Vec<Gf128> = {
            let mut v = vec![ZERO; 1usize << m];
            for idx in 0..v.len() {
                let sub = (idx & 1) | ((idx >> 1) & 1) << 1 | ((idx >> 2) & 1) << 2;
                v[idx] = tp[sub];
            }
            v
        };
        let pub_full: Vec<Box<dyn PublicFactor + Send + Sync>> =
            vec![Box::new(crate::public::TableFactor::new(full_p))];
        let terms = vec![Term {
            coeff: Gf128(1),
            dense: vec![0, 1],
            public: vec![0],
        }];
        let mut tv = Transcript::new("sparse", &[b"t"]);
        let mut v =
            crate::engine::SumcheckVerifier::new(m, claim, &pub_full, terms, BindOrder::Increasing);
        for msg in &rounds {
            v.round_capturing(&RoundMsg(msg.clone()), &mut tv, "sp")
                .unwrap();
        }
        v.finish(&claims).expect("sparse proof must verify");
    }

    /// A factor with no active variables (a constant) contributes
    /// `2^remaining` copies — exercise it.
    #[test]
    fn sparse_constant_factor() {
        let mut rng = rand::thread_rng();
        let m = 3;
        // Summand: C · Z(x) with C constant, Z over all vars.
        let c0 = rg(&mut rng);
        let z: Vec<Gf128> = (0..8).map(|_| rg(&mut rng)).collect();
        let claim = {
            let sz: Gf128 = z.iter().fold(ZERO, |a, b| a + *b);
            mul(c0, sz)
        };
        let spec = SparseSpec {
            num_vars: m,
            dense: vec![SparseFactor::new(z.clone(), vec![0, 1, 2])],
            public: vec![SparseFactor::new(vec![c0], vec![])],
            terms: vec![Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![0],
            }],
            claim,
        };
        let mut t = Transcript::new("sparse2", &[b"t"]);
        let mut prover = SparseProver::new(spec, BindOrder::Increasing);
        let mut rounds = Vec::new();
        for _ in 0..m {
            let msg = prover.message();
            rounds.push(msg.0.clone());
            prover.absorb_and_fold(&mut t, "sp2");
        }
        let (claims, _) = prover.finish();
        // The verifier: the public factor = the constant table over the
        // full space.
        let full_c: Vec<Gf128> = vec![c0; 8];
        let pub_full: Vec<Box<dyn PublicFactor + Send + Sync>> =
            vec![Box::new(crate::public::TableFactor::new(full_c))];
        let terms = vec![Term {
            coeff: Gf128(1),
            dense: vec![0],
            public: vec![0],
        }];
        let mut tv = Transcript::new("sparse2", &[b"t"]);
        let mut v =
            crate::engine::SumcheckVerifier::new(m, claim, &pub_full, terms, BindOrder::Increasing);
        for msg in &rounds {
            v.round_capturing(&RoundMsg(msg.clone()), &mut tv, "sp2")
                .unwrap();
        }
        v.finish(&claims)
            .expect("constant-factor proof must verify");
    }

    /// The anchored path: a mostly-zero routing-tensor-like factor with
    /// the round variable inside and outside it. Both paths must produce
    /// proofs the dense verifier accepts, and the round messages must
    /// match the dense enumeration bit for bit.
    #[test]
    fn anchored_matches_dense_enumeration() {
        let mut rng = rand::thread_rng();
        // Joint layout: o | j | u | h (9 vars). The tensor over
        // (o, u, h) is one-hot per (h, o): 8 × 4 entries with one
        // nonzero each — a routing tensor shape.
        let o_vars = 3usize;
        let j_vars = 2usize;
        let u_vars = 3usize;
        let h_vars = 1usize;
        let m = o_vars + j_vars + u_vars + h_vars;
        let mut tensor = vec![ZERO; 1usize << (o_vars + u_vars + h_vars)];
        for h in 0..(1usize << h_vars) {
            for o in 0..(1usize << o_vars) {
                let u = (o + h) % (1usize << u_vars);
                let idx = o | (u << o_vars) | (h << (o_vars + u_vars));
                tensor[idx] = rg(&mut rng);
            }
        }
        let x_table: Vec<Gf128> = (0..(1usize << (u_vars + j_vars)))
            .map(|_| rg(&mut rng))
            .collect();
        let r_o: Vec<Gf128> = (0..o_vars).map(|_| rg(&mut rng)).collect();
        let r_j: Vec<Gf128> = (0..j_vars).map(|_| rg(&mut rng)).collect();
        let eq_o = magnetar_field::EqBits::build(&r_o).values;
        let eq_j = magnetar_field::EqBits::build(&r_j).values;
        // The claim: Σ eq_o·eq_j·tensor·X.
        let claim = {
            let mut acc = ZERO;
            for idx in 0..(1usize << m) {
                let o = idx & ((1 << o_vars) - 1);
                let j = (idx >> o_vars) & ((1 << j_vars) - 1);
                let u = (idx >> (o_vars + j_vars)) & ((1 << u_vars) - 1);
                let h = idx >> (o_vars + j_vars + u_vars);
                let t_idx = o | (u << o_vars) | (h << (o_vars + u_vars));
                let x_idx = u | (j << u_vars);
                acc += mul(mul(mul(eq_o[o], eq_j[j]), tensor[t_idx]), x_table[x_idx]);
            }
            acc
        };
        let tensor_active: Vec<usize> = {
            let mut a: Vec<usize> = (0..o_vars).collect(); // o
            a.extend(o_vars + j_vars..o_vars + j_vars + u_vars); // u
            a.push(o_vars + j_vars + u_vars); // h
            a
        };
        let x_active: Vec<usize> = {
            // The table is indexed `u | (j << u_vars)`: u low, j high.
            let mut a: Vec<usize> = (o_vars + j_vars..o_vars + j_vars + u_vars).collect();
            a.extend(o_vars..o_vars + j_vars);
            a
        };
        let spec = SparseSpec {
            num_vars: m,
            dense: vec![SparseFactor::new(x_table.clone(), x_active.clone())],
            public: vec![
                SparseFactor::new(eq_o.clone(), (0..o_vars).collect()),
                SparseFactor::new(eq_j.clone(), (o_vars..o_vars + j_vars).collect()),
                SparseFactor::new(tensor.clone(), tensor_active.clone()),
            ],
            terms: vec![Term {
                coeff: Gf128(1),
                dense: vec![0],
                public: vec![0, 1, 2],
            }],
            claim,
        };
        // The anchored path must agree with the forced-dense path on
        // every round message.
        let run = |anchor: bool| -> (Vec<Vec<Gf128>>, Vec<Gf128>, Gf128) {
            let mut spec2 = SparseSpec {
                num_vars: spec.num_vars,
                dense: spec
                    .dense
                    .iter()
                    .map(|f| SparseFactor::new(f.table.clone(), f.active.clone()))
                    .collect(),
                public: spec
                    .public
                    .iter()
                    .map(|f| {
                        let mut f2 = SparseFactor::new(f.table.clone(), f.active.clone());
                        if !anchor {
                            // Force the dense path.
                            f2.sparse = false;
                            f2.nnz.clear();
                        }
                        f2
                    })
                    .collect(),
                terms: spec.terms.clone(),
                claim: spec.claim,
            };
            let _ = &mut spec2;
            let mut t = Transcript::new("sparse3", &[b"t"]);
            let mut prover = SparseProver::new(spec2, BindOrder::Increasing);
            let mut rounds = Vec::new();
            for _ in 0..m {
                let msg = prover.message();
                rounds.push(msg.0.clone());
                prover.absorb_and_fold(&mut t, "sp3");
            }
            let (claims, running) = prover.finish();
            (rounds, claims, running)
        };
        let (rounds_dense, claims_dense, _) = run(false);
        let (rounds_anchored, claims_anchored, _) = run(true);
        assert_eq!(rounds_dense, rounds_anchored, "round messages diverge");
        assert_eq!(claims_dense, claims_anchored, "terminal claims diverge");
    }
}
