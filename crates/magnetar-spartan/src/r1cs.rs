//! Sparse matrices and SIMD R1CS structures (paper Definitions 3–4).

use magnetar_field::{mul, Gf128, ONE, ZERO};

/// A sparse matrix over `K` in (row, col, value) entry form.
///
/// Row/column indices use the little-endian bit convention of the
/// multilinear layer: the MLE of the matrix is
/// `M̂(r_row, r_col) = Σ_entries val·eq(r_row, row)·eq(r_col, col)`.
#[derive(Clone, Debug, Default)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    /// (row, col, value)
    pub entries: Vec<(usize, usize, Gf128)>,
}

/// A borrowed view (avoids clones in the dual-relation prover).
#[derive(Clone, Copy)]
pub struct SparseMatRef<'a> {
    pub rows: usize,
    pub cols: usize,
    pub entries: &'a [(usize, usize, Gf128)],
}

impl<'a> From<&'a SparseMatrix> for SparseMatRef<'a> {
    fn from(m: &'a SparseMatrix) -> Self {
        SparseMatRef {
            rows: m.rows,
            cols: m.cols,
            entries: &m.entries,
        }
    }
}

impl SparseMatrix {
    /// Identity-ish padding helper: an empty matrix of the given shape.
    pub fn empty(rows: usize, cols: usize) -> Self {
        SparseMatrix {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    /// Matrix–vector product (dense z).
    pub fn apply(&self, z: &[Gf128]) -> Vec<Gf128> {
        debug_assert_eq!(z.len(), self.cols);
        let mut out = vec![ZERO; self.rows];
        for &(r, c, v) in &self.entries {
            out[r] += mul(v, z[c]);
        }
        out
    }

    /// The matrix MLE restricted to a Boolean column point but a general
    /// row point, as a **table over the column cube**:
    /// `table[y] = M̂(r_row, y)` for `y ∈ {0,1}^{log cols}`.
    ///
    /// For Boolean `y`, `eq(y, col) = [y == col]`, so this is a per-column
    /// accumulation of `val·eq(r_row, row)` — `O(nnz)`.
    pub fn eval_mle_table(&self, r_row: &[Gf128]) -> Vec<Gf128> {
        debug_assert_eq!(r_row.len(), self.rows.trailing_zeros() as usize);
        let lc = self.cols.trailing_zeros() as usize;
        // Precompute eq(r_row, row) for every row: fold the eq table.
        let eq_row = magnetar_field::EqBits::build(r_row).values;
        let mut table = vec![ZERO; self.cols];
        for &(r, c, v) in &self.entries {
            table[c] += mul(v, eq_row[r]);
        }
        let _ = lc;
        table
    }

    /// Full MLE evaluation `M̂(r_row, r_col)` at arbitrary points: `O(nnz)`.
    pub fn eval_mle_at(&self, r_row: &[Gf128], r_col: &[Gf128]) -> Gf128 {
        let eq_row = magnetar_field::EqBits::build(r_row).values;
        let eq_col = magnetar_field::EqBits::build(r_col).values;
        let mut acc = ZERO;
        for &(r, c, v) in &self.entries {
            acc += mul(v, mul(eq_row[r], eq_col[c]));
        }
        acc
    }

    /// Diagonal-block application for SIMD: for each instance `u`, apply to
    /// `z_u`, producing the joint table `Az(u, y)` (row-major layout, `y`
    /// low bits).
    pub fn apply_simd(&self, joint_z: &[Gf128], q: usize) -> Vec<Gf128> {
        // Each instance's z_u has `cols` entries; the output has `rows`.
        debug_assert_eq!(joint_z.len(), q * self.cols);
        let mut out = vec![ZERO; q * self.rows];
        for u in 0..q {
            let z_u = &joint_z[u * self.cols..(u + 1) * self.cols];
            let out_u = &mut out[u * self.rows..(u + 1) * self.rows];
            for &(r, c, v) in &self.entries {
                out_u[r] += mul(v, z_u[c]);
            }
        }
        out
    }
}

/// The joint SIMD assignment: `Z(u, y) = z_u[y]`, row-major (`y` low bits),
/// with `z_u = (w_u, 1, x_u)`.
#[derive(Clone, Debug)]
pub struct SimdInstance {
    /// Number of SIMD copies `q` (power of two).
    pub q: usize,
    /// Column count `m` (power of two, `m = 2·(|w| + …)`).
    pub m: usize,
    /// Public inputs per instance: `x_u` has `ℓ = m/2 − 1` entries.
    pub public: Vec<Vec<Gf128>>,
}

impl SimdInstance {
    /// Assemble the joint `Z` table from the per-instance witnesses.
    pub fn joint_z(&self, witnesses: &[Vec<Gf128>]) -> Vec<Gf128> {
        debug_assert_eq!(witnesses.len(), self.q);
        let half = self.m / 2;
        let mut z = vec![ZERO; self.q * self.m];
        for (u, w) in witnesses.iter().enumerate() {
            debug_assert_eq!(w.len(), half);
            let base = u * self.m;
            z[base..base + half].copy_from_slice(w);
            z[base + half] = ONE;
            let x = &self.public[u];
            debug_assert_eq!(x.len(), half - 1);
            z[base + half + 1..base + self.m].copy_from_slice(x);
        }
        z
    }

    /// Equation (1) public part: the folded `(1, x)` half at
    /// `(r_inst, r_low)`, computed by the verifier:
    /// `(1,x)_fold(r_inst, r_low) = Σ_u eq(r_inst, u)·(1,x_u)^(r_low-ish)`
    /// restricted to the high half of the columns.
    pub fn fold_public_part(&self, r_inst: &[Gf128], r_low: &[Gf128]) -> Gf128 {
        // The (1, x) half lives in columns [m/2, m): its MLE over the
        // column variables, restricted to r_low with the MSB of the column
        // index set to 1: (1 + x^{msb})-folded via eq_char2.
        let half = self.m / 2;
        let n_low = r_low.len();
        debug_assert_eq!(n_low, (half.trailing_zeros() as usize));
        let eq_inst = magnetar_field::EqBits::build(r_inst).values;
        let eq_low = magnetar_field::EqBits::build(r_low).values;
        // Column index within the (1,x) half: j' = col - m/2 ∈ [0, half).
        // (1, x_u) at column j': value = 1 if j' == 0 else x_u[j'-1].
        // Folded value = Σ_j' eq(r_low, j')·(that value, instance-folded).
        // = Σ_u eq(r_inst,u)·[ eq(r_low,0)·1 + Σ_{j'=1..} eq(r_low,j')·x_u[j'-1] ]
        let mut acc = ZERO;
        for (u, x) in self.public.iter().enumerate() {
            let w_u = eq_inst[u];
            // j' = 0 contributes eq(r_low, 0).
            let mut term = eq_low[0];
            for (j, &xv) in x.iter().enumerate() {
                term += mul(eq_low[j + 1], xv);
            }
            acc += mul(w_u, term);
        }
        acc
    }
}

/// The characteristic-two `eq` fold of a per-instance vector at `r_inst`.
pub fn fold_instances(per_instance: &[Gf128], r_inst: &[Gf128]) -> Gf128 {
    let eq = magnetar_field::EqBits::build(r_inst).values;
    let mut acc = ZERO;
    for (u, &v) in per_instance.iter().enumerate() {
        acc += mul(eq[u], v);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rand::RngExt::random::<u128>(rng))
    }

    #[test]
    fn apply_and_mle_agree() {
        let mut rng = rand::rng();
        // Random sparse matrix, random z: apply == MLE-based identity at
        // Boolean points is not direct; check eval_mle_table against
        // direct MLE at Boolean column points.
        let (rows, cols) = (4usize, 8usize);
        let mut m = SparseMatrix::empty(rows, cols);
        for _ in 0..12 {
            m.entries.push((
                rng.random_range(0..rows),
                rng.random_range(0..cols),
                rg(&mut rng),
            ));
        }
        let r_row: Vec<Gf128> = (0..2).map(|_| rg(&mut rng)).collect();
        let table = m.eval_mle_table(&r_row);
        for y in 0..cols {
            let y_pt: Vec<Gf128> = (0..3).map(|i| Gf128(((y >> i) & 1) as u128)).collect();
            let direct = m.eval_mle_at(&r_row, &y_pt);
            assert_eq!(table[y], direct, "y = {y}");
        }
        // apply() equals the dense product on Boolean structure.
        let z: Vec<Gf128> = (0..cols).map(|_| rg(&mut rng)).collect();
        let az = m.apply(&z);
        // Check a few entries via the definition.
        for r in 0..rows {
            let mut acc = ZERO;
            for &(er, ec, v) in &m.entries {
                if er == r {
                    acc += mul(v, z[ec]);
                }
            }
            assert_eq!(az[r], acc);
        }
    }

    #[test]
    fn simd_apply_matches_block_diagonal() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let (rows, cols) = (4usize, 4usize);
        let mut m = SparseMatrix::empty(rows, cols);
        for _ in 0..10 {
            m.entries.push((
                rng.random_range(0..rows),
                rng.random_range(0..cols),
                rg(&mut rng),
            ));
        }
        let q = 4;
        let joint: Vec<Gf128> = (0..q * rows).map(|_| rg(&mut rng)).collect();
        let out = m.apply_simd(&joint, q);
        for u in 0..q {
            let z_u = &joint[u * rows..(u + 1) * rows];
            let expect = m.apply(z_u);
            assert_eq!(&out[u * rows..(u + 1) * rows], &expect[..], "u = {u}");
        }
    }

    #[test]
    fn public_fold_matches_definition() {
        let mut rng = rand::rng();
        let q = 4;
        let m = 8;
        let inst = SimdInstance {
            q,
            m,
            public: (0..q).map(|_| vec![rg(&mut rng); m / 2 - 1]).collect(),
        };
        let r_inst: Vec<Gf128> = (0..2).map(|_| rg(&mut rng)).collect();
        let r_low: Vec<Gf128> = (0..2).map(|_| rg(&mut rng)).collect();
        // Direct: (1,x)-half MLE at (r_inst, msb=1, r_low).
        let eq_inst = magnetar_field::EqBits::build(&r_inst).values;
        let eq_low = magnetar_field::EqBits::build(&r_low).values;
        let mut want = ZERO;
        for (u, x) in inst.public.iter().enumerate() {
            for j2 in 0..(m / 2) {
                // column = m/2 + j2 → msb (bit 2) = 1, low bits = j2.
                let val = if j2 == 0 { ONE } else { x[j2 - 1] };
                want += mul(mul(eq_inst[u], eq_low[j2]), val);
            }
        }
        assert_eq!(inst.fold_public_part(&r_inst, &r_low), want);
    }
}
