//! Bit-level R1CS over `F = GF(2)` (the paper's Definition 2) and the packed
//! multi-instance layout of Algorithm 6 (§5.3).
//!
//! A constraint system is three sparse matrices `S_L, S_R, S_O ∈ F^{N×N}`
//! with O(N) nonzeros; an assignment `w ∈ F^N` satisfies
//! `(S_L w) ∘ (S_R w) = S_O w`.
//!
//! **Wiring taken public (the paper's Table-4 benchmark setting):** "Flock
//! does not commit to the wiring matrices, so in this batched setting both
//! systems take them as explicit public inputs and have the verifier
//! evaluate them directly, making the comparison fair." We implement exactly
//! that: the verifier evaluates the sparse-MLE claims directly in O(nnz).

use crate::planes::PlaneSet;
use magnetar_field::Gf128;
use magnetar_field::{mul, ONE, ZERO};

/// A sparse GF(2) matrix: `entries[row]` lists the nonzero columns.
#[derive(Clone, Debug, Default)]
pub struct SparseMatrix {
    pub n: usize,
    pub rows: Vec<Vec<u32>>,
}

impl SparseMatrix {
    pub fn new(n: usize) -> Self {
        SparseMatrix {
            n,
            rows: vec![Vec::new(); n],
        }
    }

    pub fn add(&mut self, row: usize, col: u32) {
        assert!(col < self.n as u32);
        self.rows[row].push(col);
    }

    pub fn nnz(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }

    /// `S · v` over F_2 (v: bits; a shorter witness is zero-padded to the
    /// padded matrix size).
    pub fn apply_bits(&self, v: &[bool]) -> Vec<bool> {
        let get = |i: usize| v.get(i).copied().unwrap_or(false);
        (0..self.n)
            .map(|r| {
                let mut acc = false;
                for &c in &self.rows[r] {
                    acc ^= get(c as usize);
                }
                acc
            })
            .collect()
    }

    /// Matrix MLE `Ŝ(ρ, σ) = Σ_{i,j} S[i][j]·ts(ρ)_i·ts(σ)_j` (verifier-side,
    /// O(nnz) with the eq tables).
    pub fn eval_mle(&self, rho: &[Gf128], sigma: &[Gf128]) -> Gf128 {
        let n = self.n;
        let nu = n.trailing_zeros() as usize;
        assert_eq!(rho.len(), nu);
        assert_eq!(sigma.len(), nu);
        let eq_r = eq_build(rho);
        let eq_s = eq_build(sigma);
        let mut acc = ZERO;
        for i in 0..n {
            if eq_r[i] == ZERO {
                continue;
            }
            for &j in &self.rows[i] {
                acc += mul(eq_r[i], eq_s[j as usize]);
            }
        }
        acc
    }
}

/// E-vector dot product (`O(n)` E-mults).
pub fn dot_gf(a: &[Gf128], b: &[Gf128]) -> Gf128 {
    let mut acc = ZERO;
    for i in 0..a.len() {
        acc += mul(a[i], b[i]);
    }
    acc
}

/// The eq (tensor) table of a Boolean point, little-endian variable order
/// (variable 0 = the LSB of the index) — matching this crate's row/factor
/// conventions.
pub fn eq_build(pt: &[Gf128]) -> Vec<Gf128> {
    let mut table = vec![ONE];
    for &p in pt.iter() {
        let prev = table.len();
        table.resize(prev * 2, ZERO);
        for i in 0..prev {
            // bit k of the new index selects the factor entry: 0 → (1+p),
            // 1 → p (char 2: ts factor (1−p, p) = (1+p, p))
            table[prev + i] = mul(table[i], p);
            table[i] = mul(table[i], ONE + p);
        }
    }
    table
}

/// A binary R1CS instance: the matrices plus the public-I/O slice layout.
pub struct BinaryR1cs {
    pub n_constraints: usize,
    pub n_witness: usize,
    pub s_l: SparseMatrix,
    pub s_r: SparseMatrix,
    pub s_o: SparseMatrix,
    /// Public-I/O positions in the witness vector (declared explicitly; the
    /// rest are private). The paper pads `w = (w_mul ‖ io)`; we keep an
    /// explicit index set for flexibility, defaulting to the suffix.
    pub io_positions: Vec<u32>,
}

impl BinaryR1cs {
    pub fn satisfied_by(&self, w: &[bool]) -> bool {
        let l = self.s_l.apply_bits(w);
        let r = self.s_r.apply_bits(w);
        let o = self.s_o.apply_bits(w);
        (0..self.n_constraints).all(|i| (l[i] & r[i]) == o[i])
    }

    /// The packed multi-instance matrix `W ∈ F^{N×κ}` in the *transposed
    /// packing layout*: plane c = witness position c, rows = the κ
    /// instances (`W[c][j]`). This is the layout the sparse application
    /// consumes word-wise.
    pub fn pack_instances_t(&self, witnesses: &[Vec<bool>]) -> PlaneSet {
        let k = witnesses.len();
        let n = self.s_l.n; // the padded matrix size
        let mut ps = PlaneSet::zeros(k.div_ceil(64) * 64, n);
        for (j, w) in witnesses.iter().enumerate() {
            assert_eq!(w.len(), self.n_witness);
            for (c, &bit) in w.iter().enumerate() {
                if bit {
                    ps.set_bit(c, j, true);
                }
            }
        }
        ps
    }
}

/// `S · W` over every instance, word-wise in the transposed layout:
/// `out.plane(r) = ⊕_{c ∈ row r} w.plane(c)` — plane = constraint row,
/// rows = instances.
pub fn apply_sparse_packed(s: &SparseMatrix, w: &PlaneSet) -> PlaneSet {
    let mut out = PlaneSet::zeros(w.m, s.n);
    for (r, cols) in s.rows.iter().enumerate() {
        for &c in cols {
            let src = w.plane(c as usize);
            let dst = out.plane_mut(r);
            for k in 0..w.words {
                dst[k] ^= src[k];
            }
        }
    }
    out
}

/// Transpose a packed plane set: `(plane p, row r) → (plane r, row p)` —
/// converts the sparse-application layout (plane = witness/constraint
/// position) into the Algorithm-2 layout (plane = the κ packing column,
/// rows = the m constraint positions) and back.
pub fn transpose_ps(ps: &PlaneSet) -> PlaneSet {
    assert!(
        ps.kappa.is_multiple_of(64),
        "kappa must be 64-aligned for the tile transpose"
    );
    let mut out = PlaneSet::zeros(ps.kappa, ps.m); // (m, kappa) swapped
    for rblk in (0..ps.m).step_by(64) {
        for pblk in (0..ps.kappa).step_by(64) {
            // tile[i] = the 64 rows rblk..rblk+64 of plane pblk+i (one word)
            let mut tile = [0u64; 64];
            for i in 0..64 {
                let p = pblk + i;
                if p < ps.kappa {
                    tile[i] = ps.plane(p)[rblk / 64];
                }
            }
            crate::planes::flip64(&mut tile);
            // after the flip: tile[j] = the i-bits (planes pblk..) of row rblk+j
            for j in 0..64 {
                let r = rblk + j;
                if r < ps.m {
                    out.plane_mut(r)[pblk / 64] = tile[j];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::KAPPA;

    #[test]
    fn sparse_apply_and_mle() {
        // S = [[1,1,0],[0,1,1]] as GF(2)
        let mut s = SparseMatrix::new(4);
        s.add(0, 0);
        s.add(0, 1);
        s.add(1, 1);
        s.add(1, 2);
        assert_eq!(
            s.apply_bits(&[true, true, false, true]),
            vec![false, true, false, false]
        );
        // the MLE at Boolean points must reproduce entries
        let rho = [Gf128(0x1234), Gf128(0x5678)];
        let sigma = [Gf128(0x9abc), Gf128(0xdef0)];
        let val = s.eval_mle(&rho, &sigma);
        // brute force
        let eqr = eq_build(&rho);
        let eqs = eq_build(&sigma);
        let mut want = ZERO;
        for i in 0..4 {
            for j in 0..4 {
                if s.rows.get(i).is_some_and(|r| r.contains(&(j as u32))) {
                    want += mul(eqr[i], eqs[j]);
                }
            }
        }
        assert_eq!(val, want);
    }

    #[test]
    fn pack_and_apply() {
        let mut cs = BinaryR1cs {
            n_constraints: 2,
            n_witness: 4,
            s_l: SparseMatrix::new(64),
            s_r: SparseMatrix::new(64),
            s_o: SparseMatrix::new(64),
            io_positions: vec![2, 3],
        };
        cs.s_l.add(0, 0);
        cs.s_l.add(0, 1);
        cs.s_r.add(0, 2);
        cs.s_o.add(0, 3); // x0·x1 = x3
        cs.s_l.add(1, 1);
        cs.s_l.add(1, 3);
        cs.s_r.add(1, 2); // (x1+x3)·x2 = 0 (empty S_O row)
        let w: Vec<Vec<bool>> = (0..KAPPA)
            .map(|j| {
                let x0 = j & 1 == 1;
                let x1 = (j >> 1) & 1 == 1;
                // row 0 is (x0+x1)·x2 = x3 with x2 = 0 ⇒ x3 must be 0
                vec![x0, x1, false, false]
            })
            .collect();
        for wj in &w {
            assert!(cs.satisfied_by(wj), "instance must satisfy");
        }
        let packed = cs.pack_instances_t(&w);
        let l_t = apply_sparse_packed(&cs.s_l, &packed);
        let r_t = apply_sparse_packed(&cs.s_r, &packed);
        let o_t = apply_sparse_packed(&cs.s_o, &packed);
        // transpose into the Algorithm-2 layout (plane = instance)
        let l = transpose_ps(&l_t);
        let r = transpose_ps(&r_t);
        let o = transpose_ps(&o_t);
        for j in 0..KAPPA {
            for i in 0..2 {
                let lv = l.bit(j, i);
                let rv = r.bit(j, i);
                let ov = o.bit(j, i);
                assert_eq!(lv & rv, ov, "packed constraint {i} instance {j}");
            }
        }
    }
}
