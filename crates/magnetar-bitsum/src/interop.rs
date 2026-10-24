//! Tower ↔ `K` interop: a verified isomorphism between this crate's tower
//! basis and the stack's `K = F_{2^128}` (GCM polynomial basis,
//! `magnetar-field`).
//!
//! The bitsliced kernel works in the **tower basis** (single-coordinate
//! extension constants keep the multiplication tree cheap); the rest of the
//! stack works in the **polynomial basis** (bit extraction is a plain bit
//! read, which ring switching exploits). Both are `F_{2^128}`, so the bridge
//! is the fixed linear map that re-expresses one basis in the other — and
//! because it is built from the field's own subfield structure it is a
//! *field isomorphism*, checked by tests here:
//!
//! ```text
//! to_gcm(a + b)  == to_gcm(a) + to_gcm(b)
//! to_gcm(a * b)  == magnetar_field::mul(to_gcm(a), to_gcm(b))
//! from_gcm(to_gcm(a)) == a == to_gcm(from_gcm(a))   (round trip)
//! ```
//!
//! # How the embedding is built (deterministic, no trusted setup)
//!
//! The tower is `F_2 ⊂ F_4 ⊂ … ⊂ F_{2^128}` with
//! `g_k^2 + g_k = NUS[k] = g_{k-1} · … · g_0`. To embed it into the
//! polynomial-basis field we solve, level by level, `z^2 + z = p_k` where
//! `p_k` is the image of `NUS[k]`. The map `z ↦ z^2 + z` is `F_2`-linear, so
//! each level is one GF(2) linear solve (Gaussian elimination over the 128
//! coordinate bits, run in a fixed order — a deterministic convention, not a
//! choice with cryptographic meaning). A solution always exists because
//! `p_k` lies in a proper subfield (even extension degree ⇒ trace 0), and it
//! lies *outside* that subfield (the tower constant has subfield-trace 1),
//! so the 128 images `B_i = Π_{j ⊆ i} g_j` form a basis. The inverse map is
//! one GF(2) matrix inversion.
//!
//! Tables in `K` enter the prover through [`prove_gcm`]; the proof's final
//! claim and evaluation point map back with [`to_gcm`], so the outer
//! polynomial-evaluation check can be run by the stack's own
//! [`magnetar_field::eval_mle`] (tested below against exactly that).

#![forbid(unsafe_code)]

use crate::field::F128;
use crate::sumcheck::{self, ProveError, ProveOutput};
use crate::transcript::Transcript;
use magnetar_field::{mul as gcm_mul, Gf128};
use std::sync::OnceLock;

/// The fixed tower↔polynomial-basis correspondence, built once per process
/// (a few hundred `K`-multiplications).
struct Interop {
    /// `basis_gcm[i]` = tower basis element `B_i` in polynomial coordinates
    /// (column `i` of the tower→`K` map).
    basis_gcm: [Gf128; 128],
    /// `gcm_to_tower[j]` = tower coordinates of the polynomial basis
    /// element `x^j` (column `j` of the `K`→tower map).
    gcm_to_tower: [u128; 128],
    /// Embedded generators `g_0 .. g_6` (kept for the relation tests).
    #[cfg(test)]
    gens: [Gf128; 7],
}

fn interop() -> &'static Interop {
    static I: OnceLock<Interop> = OnceLock::new();
    I.get_or_init(build)
}

/// Solves `z^2 + z = target` in `K` (returns `None` iff the target has
/// absolute trace 1, which never happens for the tower constants).
///
/// `z ↦ z^2 + z` is `F_2`-linear; the solve is Gaussian elimination over
/// the coordinate bits in a fixed order, so the returned root (of the root
/// pair `{z, z+1}`) is a deterministic convention.
fn solve_z2_z(target: Gf128) -> Option<Gf128> {
    // columns[j] = phi(e_j) = e_j^2 + e_j, with comb[j] tracking which
    // z-coordinates produce it.
    let mut basis_val = [0u128; 128];
    let mut basis_comb = [0u128; 128];
    let mut have = [false; 128];
    for j in 0..128usize {
        let e = Gf128(1u128 << j);
        let mut v = gcm_mul(e, e).0 ^ e.0;
        let mut c = 1u128 << j;
        while v != 0 {
            let b = 127 - v.leading_zeros() as usize;
            if !have[b] {
                have[b] = true;
                basis_val[b] = v;
                basis_comb[b] = c;
                v = 0;
            } else {
                v ^= basis_val[b];
                c ^= basis_comb[b];
            }
        }
    }
    let mut t = target.0;
    let mut z = 0u128;
    while t != 0 {
        let b = 127 - t.leading_zeros() as usize;
        if !have[b] {
            return None; // target outside the image: trace 1
        }
        t ^= basis_val[b];
        z ^= basis_comb[b];
    }
    Some(Gf128(z))
}

/// Inverts the 128×128 GF(2) matrix whose columns are the embedded tower
/// basis; the result maps polynomial coordinates back to tower coordinates.
fn invert_basis(basis: &[Gf128; 128]) -> [u128; 128] {
    // Row r of the tower→K matrix: bit i holds M[r][i] = bit r of basis[i].
    // Augment with the identity; reduce to RREF; the right half is M^-1.
    let mut left = [0u128; 128];
    let mut right = [0u128; 128];
    for (r, rhs) in right.iter_mut().enumerate() {
        *rhs = 1u128 << r;
    }
    for (i, &b) in basis.iter().enumerate() {
        for (r, row) in left.iter_mut().enumerate() {
            if (b.0 >> r) & 1 == 1 {
                *row |= 1u128 << i;
            }
        }
    }
    for c in 0..128usize {
        let mut pivot = None;
        for r in c..128 {
            if (left[r] >> c) & 1 == 1 {
                pivot = Some(r);
                break;
            }
        }
        let p = pivot.expect("embedded tower basis is linearly independent");
        left.swap(c, p);
        right.swap(c, p);
        for r in 0..128 {
            if r != c && (left[r] >> c) & 1 == 1 {
                left[r] ^= left[c];
                right[r] ^= right[c];
            }
        }
    }
    debug_assert!(left.iter().enumerate().all(|(c, &row)| row == 1u128 << c));
    // gcm_to_tower[r] = tower coordinates of x^r = column r of M^-1.
    let mut inv = [0u128; 128];
    for (c, &row) in right.iter().enumerate() {
        for r in 0..128 {
            if (row >> r) & 1 == 1 {
                inv[r] |= 1u128 << c;
            }
        }
    }
    inv
}

fn build() -> Interop {
    // 1. Embed the generators: g_k^2 + g_k = image of NUS[k], which is the
    //    product of the previously embedded generators.
    let mut gens = [Gf128(0); 7];
    let mut product = Gf128(1); // image of NUS[0] = 1
    for (k, g) in gens.iter_mut().enumerate() {
        let root =
            solve_z2_z(product).expect("tower level embeds: NUS[k] lies in a proper subfield");
        debug_assert_eq!(gcm_mul(root, root) + root, product, "level {k}");
        *g = root;
        product = gcm_mul(product, root); // image of NUS[k+1]
    }
    // 2. Tower basis: B_i = product of g_j over the set bits of i.
    let mut basis_gcm = [Gf128(0); 128];
    basis_gcm[0] = Gf128(1);
    for i in 1..128usize {
        let j = i.trailing_zeros() as usize;
        basis_gcm[i] = gcm_mul(basis_gcm[i ^ (1 << j)], gens[j]);
    }
    // 3. Inverse map.
    let gcm_to_tower = invert_basis(&basis_gcm);
    Interop {
        basis_gcm,
        gcm_to_tower,
        #[cfg(test)]
        gens,
    }
}

/// Maps a tower element to `K` (polynomial basis).
#[must_use]
pub fn to_gcm(x: F128) -> Gf128 {
    let basis = &interop().basis_gcm;
    let mut acc = 0u128;
    let mut v = x.0;
    while v != 0 {
        let i = v.trailing_zeros() as usize;
        acc ^= basis[i].0;
        v &= v - 1;
    }
    Gf128(acc)
}

/// Maps a `K` (polynomial basis) element to the tower basis.
#[must_use]
pub fn from_gcm(x: Gf128) -> F128 {
    let inv = &interop().gcm_to_tower;
    let mut acc = 0u128;
    let mut v = x.0;
    while v != 0 {
        let r = v.trailing_zeros() as usize;
        acc ^= inv[r];
        v &= v - 1;
    }
    F128(acc)
}

/// Converts a whole `K` table to tower coordinates.
#[must_use]
pub fn table_to_tower(table: &[Gf128]) -> Vec<F128> {
    table.iter().map(|&x| from_gcm(x)).collect()
}

/// Converts a whole tower table to `K` coordinates.
#[must_use]
pub fn table_to_gcm(table: &[F128]) -> Vec<Gf128> {
    table.iter().map(|&x| to_gcm(x)).collect()
}

/// Proves the multilinear sum-check over a table **in the stack's `K`**
/// (GCM polynomial basis): converts to the tower basis, proves, and returns
/// the standard output. The claimed sum and final evaluation are tower
/// images of the `K` values — map them back with [`to_gcm`] (or use
/// [`eval_mle_gcm`] for the outer check).
///
/// # Errors
///
/// [`ProveError::NotPowerOfTwo`] if the table length is not a power of two.
pub fn prove_gcm(table: &[Gf128], t: &mut Transcript) -> Result<ProveOutput, ProveError> {
    let tower = table_to_tower(table);
    sumcheck::prove(&tower, t)
}

/// Evaluates the multilinear extension of a `K` table at the sum-check's
/// evaluation point (challenges in binding order, as returned by the
/// verifier). Returns the value in `K` — the outer evaluation check is one
/// comparison.
#[must_use]
pub fn eval_mle_gcm(table: &[Gf128], challenges: &[F128]) -> Gf128 {
    let tower = table_to_tower(table);
    to_gcm(sumcheck::eval_mle(&tower, challenges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::TOWER_NUS;
    use crate::Rng;
    use magnetar_field::eval_mle as gcm_eval_mle;

    struct SplitMix(u64);
    impl crate::Rng for SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x09E3_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn roundtrip_both_directions() {
        let mut r = SplitMix(0x1337);
        for _ in 0..128 {
            let g = Gf128(r.next_u128());
            assert_eq!(to_gcm(from_gcm(g)), g);
            let f = F128(r.next_u128());
            assert_eq!(from_gcm(to_gcm(f)), f);
        }
        assert_eq!(to_gcm(F128::ZERO), Gf128(0));
        assert_eq!(to_gcm(F128::ONE), Gf128(1));
        assert_eq!(from_gcm(Gf128(0)), F128::ZERO);
        assert_eq!(from_gcm(Gf128(1)), F128::ONE);
    }

    #[test]
    fn field_isomorphism() {
        let mut r = SplitMix(0x1507);
        for _ in 0..64 {
            let a = F128(r.next_u128());
            let b = F128(r.next_u128());
            // addition (XOR) and multiplication commute with the map
            assert_eq!(to_gcm(a + b), to_gcm(a) + to_gcm(b));
            assert_eq!(to_gcm(a * b), gcm_mul(to_gcm(a), to_gcm(b)));
            let x = Gf128(r.next_u128());
            let y = Gf128(r.next_u128());
            assert_eq!(from_gcm(x + y), from_gcm(x) + from_gcm(y));
            assert_eq!(from_gcm(gcm_mul(x, y)), from_gcm(x) * from_gcm(y));
        }
    }

    #[test]
    fn generator_relations_hold() {
        let it = interop();
        let mut product = Gf128(1);
        for (k, &g) in it.gens.iter().enumerate() {
            // g_k^2 + g_k = NUS[k] = g_{k-1} * ... * g_0
            assert_eq!(gcm_mul(g, g) + g, product, "level {k}");
            // the embedded constant matches the tower constant's image
            assert_eq!(product, to_gcm(F128(TOWER_NUS[k])), "NUS[{k}] image");
            product = gcm_mul(product, g);
        }
    }

    #[test]
    fn e2e_proof_over_repo_field() {
        // Prove a sum-check over a K table and discharge the final claim
        // with the stack's OWN multilinear evaluator — the strongest form
        // of synchronicity this crate can demonstrate.
        let mut r = SplitMix(0xbee5);
        for m in [2usize, 3, 5] {
            let table: Vec<Gf128> = (0..(1usize << m)).map(|_| Gf128(r.next_u128())).collect();

            let mut t = Transcript::new("interop-e2e");
            let out = prove_gcm(&table, &mut t).expect("prove");

            // claimed sum is the field sum over the K table
            let mut want_sum = Gf128(0);
            for &v in &table {
                want_sum += v;
            }
            assert_eq!(to_gcm(out.claimed_sum), want_sum);

            let mut tv = Transcript::new("interop-e2e");
            let oc = sumcheck::verify(&out.proof, out.claimed_sum, &mut tv).expect("verify");

            // outer check with magnetar-field's evaluator: its convention is
            // little-endian (point[j] = index bit j); this crate binds the
            // most significant bit first, so point[j] = challenges[m-1-j].
            let ch = &oc.challenges;
            let point: Vec<Gf128> = (0..m).map(|j| to_gcm(ch[m - 1 - j])).collect();
            let want = gcm_eval_mle(&table, &point);

            // the proof's final claim (converted to K) equals the stack's
            // own evaluation of the table at the derived point
            assert_eq!(to_gcm(oc.final_claim), want, "m={m}");
            assert_eq!(to_gcm(out.final_eval), want, "m={m}");

            // and the one-call convenience agrees
            assert_eq!(eval_mle_gcm(&table, ch), want, "m={m}");
        }
    }
}
