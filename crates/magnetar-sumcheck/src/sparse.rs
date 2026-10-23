//! The sparse sum-check prover (the paper's "the prover's work is linear
//! in the trace length"): every factor is kept as its **own** table over
//! only the variables it depends on, and a round folds a factor only when
//! the round's variable is one of its own. Round messages are computed by
//! enumerating the remaining cube once per round, evaluating each factor
//! via its active-variable slice, and accumulating the polynomial in the
//! round variable through per-assignment elementary products.
//!
//! The verifier is unchanged (the standard engine: round checks are
//! `O(d)` per round and the terminal evaluates the public factors at the
//! point), so a sparse proof verifies with the regular
//! [`SumcheckVerifier``] against the full-space public factors.

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
}

impl SparseFactor {
    pub fn new(table: Vec<Gf128>, active: Vec<usize>) -> Self {
        assert_eq!(table.len(), 1usize << active.len());
        SparseFactor { table, active }
    }

    fn depends_on(&self, joint_var: usize) -> bool {
        self.active.contains(&joint_var)
    }

    /// The local position of a joint variable, if active.
    fn local_pos(&self, joint_var: usize) -> Option<usize> {
        self.active.iter().position(|&x| x == joint_var)
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

    /// The sub-index of a factor at a remaining assignment `r` (the bits
    /// of `r` mapped through the factor's active positions; the round
    /// variable's bit handled separately by the caller).
    fn sub_index(f: &SparseFactor, v: usize, u_pos: &[usize; 64], r: usize) -> usize {
        let mut sub = 0usize;
        for (t, &a) in f.active.iter().enumerate() {
            if a == v {
                continue;
            }
            if (r >> u_pos[a]) & 1 == 1 {
                sub |= 1usize << t;
            }
        }
        sub
    }

    /// The current round's message.
    pub fn message(&self) -> RoundMsg {
        let v = self.order.round_var(self.num_vars, self.round);
        let u = self.remaining(v);
        let mut u_pos = [0usize; 64];
        for (h, &x) in u.iter().enumerate() {
            u_pos[x] = h;
        }
        let degree = self.degree();
        let mut coeffs = vec![ZERO; degree + 1];
        // Per remaining assignment: the fixed product and the (A, D) pairs
        // of the round-active factors.
        let n_rem = 1usize << u.len();
        for r in 0..n_rem {
            for term in &self.terms {
                // Fixed part.
                let mut fixed = term.coeff;
                let mut zero = false;
                for &d in &term.dense {
                    if !self.dense[d].depends_on(v) {
                        let sub = Self::sub_index(&self.dense[d], v, &u_pos, r);
                        fixed = mul(fixed, self.dense[d].table[sub]);
                        if fixed == ZERO {
                            zero = true;
                            break;
                        }
                    }
                }
                if !zero {
                    for &p in &term.public {
                        if !self.public[p].depends_on(v) {
                            let sub = Self::sub_index(&self.public[p], v, &u_pos, r);
                            fixed = mul(fixed, self.public[p].table[sub]);
                            if fixed == ZERO {
                                zero = true;
                                break;
                            }
                        }
                    }
                }
                if zero {
                    continue;
                }
                // (A, D) pairs of the round-active factors.
                let mut ads: Vec<(Gf128, Gf128)> = Vec::new();
                let collect = |f: &SparseFactor, ads: &mut Vec<(Gf128, Gf128)>| {
                    if let Some(pos) = f.local_pos(v) {
                        let sub = Self::sub_index(f, v, &u_pos, r);
                        let base = sub & !(1usize << pos);
                        let a = f.table[base];
                        let d = a + f.table[base | (1usize << pos)];
                        ads.push((a, d));
                    }
                };
                for &d in &term.dense {
                    collect(&self.dense[d], &mut ads);
                }
                for &p in &term.public {
                    collect(&self.public[p], &mut ads);
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
        RoundMsg(coeffs)
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
        Gf128(rand::RngExt::random::<u128>(rng))
    }

    /// A random sparse instance cross-checked against the dense engine:
    /// the sparse prover's proof must verify under the standard
    /// `SumcheckVerifier` with full-space public factors.
    #[test]
    fn sparse_matches_dense_engine() {
        let mut rng = rand::rng();
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
        let mut rng = rand::rng();
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
}
