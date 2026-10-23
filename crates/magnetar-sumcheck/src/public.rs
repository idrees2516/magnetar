//! Verifier-evaluable multilinear factors for the sum-check engine.
//!
//! A public factor is a multilinear the *verifier* can evaluate at the final
//! challenge point in sub-linear time (the paper's "the verifier evaluates
//! the multilinear extensions of A, B, C on its own", Remark 3, and the
//! BatchedEq factor of MosaicPCS §5.2). The prover materializes the same
//! factor as a dense table to fold during the rounds.

use magnetar_field::{mul, Gf128, ONE, ZERO};

/// A verifier-evaluable multilinear over a fixed number of variables.
pub trait PublicFactor: std::fmt::Debug {
    fn num_vars(&self) -> usize;

    /// Materialize the factor's values over the whole Boolean cube
    /// (little-endian, variable 0 = LSB) — used by the prover's folding.
    fn materialize(&self) -> Vec<Gf128>;

    /// Evaluate at an arbitrary point (little-endian variable order).
    fn eval_at(&self, point: &[Gf128]) -> Gf128;
}

/// `eq(r, ·)` — the multilinear equality factor (paper §3.1).
#[derive(Clone, Debug)]
pub struct EqFactor {
    pub point: Vec<Gf128>,
}

impl EqFactor {
    pub fn new(point: Vec<Gf128>) -> Self {
        EqFactor { point }
    }
}

impl PublicFactor for EqFactor {
    fn num_vars(&self) -> usize {
        self.point.len()
    }
    fn materialize(&self) -> Vec<Gf128> {
        magnetar_field::EqBits::build(&self.point).values
    }
    fn eval_at(&self, point: &[Gf128]) -> Gf128 {
        let mut acc = ONE;
        for (&r, &x) in self.point.iter().zip(point.iter()) {
            // eq(r, x) = 1 + r + x  (characteristic two).
            acc = mul(acc, ONE + r + x);
        }
        acc
    }
}

/// `Σ_i α_i · eq(p_i, ·)` — the MosaicPCS batching factor (paper §5.2):
/// the verifier's α-combination of the lifted claims' equality functions.
#[derive(Clone, Debug)]
pub struct SumEqFactor {
    /// (coefficient, point) pairs. All points have the same length.
    pub items: Vec<(Gf128, Vec<Gf128>)>,
}

impl SumEqFactor {
    pub fn new(items: Vec<(Gf128, Vec<Gf128>)>) -> Self {
        debug_assert!(items.windows(2).all(|w| w[0].1.len() == w[1].1.len()));
        SumEqFactor { items }
    }
}

impl PublicFactor for SumEqFactor {
    fn num_vars(&self) -> usize {
        self.items.first().map(|i| i.1.len()).unwrap_or(0)
    }
    fn materialize(&self) -> Vec<Gf128> {
        let nv = self.num_vars();
        let mut table = vec![ZERO; 1 << nv];
        for (alpha, point) in &self.items {
            let eq = magnetar_field::EqBits::build(point).values;
            for (t, e) in table.iter_mut().zip(eq.iter()) {
                *t += mul(*alpha, *e);
            }
        }
        table
    }
    fn eval_at(&self, point: &[Gf128]) -> Gf128 {
        let mut acc = ZERO;
        for (alpha, p) in &self.items {
            let mut e = ONE;
            for (&r, &x) in p.iter().zip(point.iter()) {
                e = mul(e, ONE + r + x);
            }
            acc += mul(*alpha, e);
        }
        acc
    }
}

/// A plain table both parties materialize (e.g. the α-combination of the
/// R1CS matrix polynomials in the inner sum-check, §4.1 step 5 — the
/// verifier builds it in `O(nnz)`).
#[derive(Clone, Debug)]
pub struct TableFactor {
    pub table: Vec<Gf128>,
    pub num_vars: usize,
}

impl TableFactor {
    pub fn new(table: Vec<Gf128>) -> Self {
        let num_vars = table.len().trailing_zeros() as usize;
        TableFactor { table, num_vars }
    }
}

impl PublicFactor for TableFactor {
    fn num_vars(&self) -> usize {
        self.num_vars
    }
    fn materialize(&self) -> Vec<Gf128> {
        self.table.clone()
    }
    fn eval_at(&self, point: &[Gf128]) -> Gf128 {
        magnetar_field::eval_mle(&self.table, point)
    }
}

/// A table over a **subset** of the joint variables, padded across the
/// rest: a public factor whose own variables live at `active` positions of
/// a larger joint space. Used whenever a sum-check's public factor (an eq
/// table, a routing tensor, a Ψ table) spans only part of the joint cube —
/// the verifier evaluates it at the point restricted to `active` in
/// `O(own vars)`, and the padded MLE equals the own-space MLE (padding
/// over idle variables leaves the sum unchanged).
#[derive(Clone, Debug)]
pub struct PaddedFactor {
    pub table: Vec<Gf128>,
    /// `active[t]` = the joint bit position of the table's variable `t`.
    pub active: Vec<usize>,
    pub joint_vars: usize,
}

impl PaddedFactor {
    pub fn new(table: Vec<Gf128>, active: Vec<usize>, joint_vars: usize) -> Self {
        let table_vars = table.len().trailing_zeros() as usize;
        assert_eq!(active.len(), table_vars);
        PaddedFactor {
            table,
            active,
            joint_vars,
        }
    }
}

impl PublicFactor for PaddedFactor {
    fn num_vars(&self) -> usize {
        self.joint_vars
    }
    fn materialize(&self) -> Vec<Gf128> {
        let mut out = vec![Gf128(0); 1usize << self.joint_vars];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_factor_materialize_matches_eval() {
        let mut rng = rand::thread_rng();
        fn rg(rng: &mut impl rand::Rng) -> Gf128 {
            Gf128(rng.gen::<u128>())
        }
        for nv in [1usize, 3, 5] {
            let point: Vec<Gf128> = (0..nv).map(|_| rg(&mut rng)).collect();
            let f = EqFactor::new(point.clone());
            let table = f.materialize();
            for (i, &v) in table.iter().enumerate() {
                let bits: Vec<Gf128> = (0..nv).map(|k| Gf128(((i >> k) & 1) as u128)).collect();
                assert_eq!(v, f.eval_at(&bits), "nv={nv} i={i}");
            }
            // Random point: eval = Π eq factors — cross-check with EqBits at
            // Boolean points only (MLE uniqueness implies consistency).
        }
    }

    #[test]
    fn sum_eq_matches_sum_of_eq() {
        let mut rng = rand::thread_rng();
        fn rg(rng: &mut impl rand::Rng) -> Gf128 {
            Gf128(rng.gen::<u128>())
        }
        let nv = 4;
        let items: Vec<(Gf128, Vec<Gf128>)> = (0..3)
            .map(|_| {
                let p: Vec<Gf128> = (0..nv).map(|_| rg(&mut rng)).collect();
                (rg(&mut rng), p)
            })
            .collect();
        let f = SumEqFactor::new(items.clone());
        let table = f.materialize();
        let singles: Vec<Vec<Gf128>> = items
            .iter()
            .map(|(_, p)| magnetar_field::EqBits::build(p).values)
            .collect();
        for i in 0..(1 << nv) {
            let mut want = ZERO;
            for ((alpha, _), s) in items.iter().zip(singles.iter()) {
                want += mul(*alpha, s[i]);
            }
            assert_eq!(table[i], want, "i = {i}");
        }
    }

    #[test]
    fn table_factor_eval() {
        let mut rng = rand::thread_rng();
        fn rg(rng: &mut impl rand::Rng) -> Gf128 {
            Gf128(rng.gen::<u128>())
        }
        let table: Vec<Gf128> = (0..16).map(|_| rg(&mut rng)).collect();
        let f = TableFactor::new(table.clone());
        let point: Vec<Gf128> = (0..4).map(|_| rg(&mut rng)).collect();
        assert_eq!(f.eval_at(&point), magnetar_field::eval_mle(&table, &point));
    }
}

