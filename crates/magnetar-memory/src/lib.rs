//! # magnetar-memory
//!
//! BinaryTwist and BinaryShout (paper §4.4): Twist and Shout instantiated
//! over binary fields, with the paper's **refined one-hotness check**.
//!
//! ## Shared address representation
//!
//! `K = 2^a` cells, `T = 2^n` cycles, `d | a` address components
//! `ra_i: {0,1}^{a/d} × {0,1}^n → B`. The layout is row-major: the address
//! part occupies the **low** `a/d` bits, the cycle part the **high** `n`
//! bits. The full address on Boolean points is the product
//! `ra(k, j) = Π_i ra_i(k^{(i)}, j)`, and because both sides are multilinear
//! over the joint space, the identity extends to all points:
//! `r̂a(r_k, r_j) = Π_i r̂ai(r_k^{(i)}, r_j)` — this is *address
//! virtualization* (no separate sum-check needed).
//!
//! ## Refined one-hotness (Δ*)
//!
//! The verifier sets `s_h = r_h²` (Frobenius), computes
//! `c_h = 1 + r_h + r_h²`, resamples `r'_h = r_h³ / c_h` (resampling `r` if
//! any `c_h = 0`), `C(r) = Π_h c_h`, and checks
//! `Σ_j eq(ρ, j)·Δ*(j; r) = 0` with
//! `Δ*(j; r) = Â(r, j)³ + C(r)·Â(r', j)` —
//! the Frobenius collapses the collision polynomial to two evaluations
//! instead of three.
//!
//! ## BinaryShout (read-only)
//!
//! The Φ-combination check (α₀ sum-forcing included) over the address
//! space, exactly as §4.4: one identity checks every address-bit and value
//! claim at once, and forces `Σ_k ra(k, j) = 1`, which with the collision
//! check gives Hamming weight exactly one.
//!
//! ## BinaryTwist (mutable, p reads + one write)
//!
//! The three identities of §4.4: reconstruction
//! `V̂al = M̂₀ + Σ_{j'} LT(j',j)·ŵfa·Înc`, read values
//! `r̂v_s = Σ_k r̂a_s·V̂al`, and write value
//! `ŵv = Σ_k ŵfa·(V̂al + Înc)`, plus the separate sum-to-one argument (the identities are homogeneous,
//! so an all-zero address must be ruled out explicitly). The sum-checks run
//! over the joint (address, value-bit, cycle) spaces as in the paper.
//!
//! Implementation note: the prover materializes the `Val` history
//! incrementally (`O(K·L·T)` time, same as the paper's prover work; the
//! paper's production prover would stream it — a space tradeoff only, the
//! protocol and verifier are identical).

#![forbid(unsafe_code)]
// Proof-system style: loops indexed over Boolean-cube coordinates and
// multi-input reductions are the domain's normal shape.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod claims;
pub mod discharge;
pub mod shout;
pub mod twist;

pub use claims::{PcsPoint, SelectorPcsClaims, ShoutPcsClaims, TwistPcsClaims};
pub use discharge::{
    prove_bus_shout, prove_psi_claim, prove_selector_shout, verify_bus_shout, verify_psi_claim,
    verify_selector_shout, BusShoutProof, BusShoutSpec, PsiProof, SelectorCheckSpec,
    SelectorShoutProof, SelectorShoutSpec,
};
pub use shout::{prove_shout, verify_shout, ShoutInstance, ShoutProof};
pub use twist::{prove_twist, verify_twist, TwistInstance, TwistProof, TwistWitness};

use magnetar_field::Gf128;

/// One address component: a bit table `A: {0,1}^{a/d} × {0,1}^n → B`
/// (row-major: address low bits, cycle high bits), stored as `K`-embedded
/// values.
#[derive(Clone, Debug)]
pub struct AddressComponent {
    /// `a/d` address variables.
    pub addr_vars: usize,
    /// `n` cycle variables.
    pub cycle_vars: usize,
    /// `2^{a/d + n}` bit values (0/1 in K).
    pub table: Vec<Gf128>,
}

impl AddressComponent {
    pub fn from_bits(addr_vars: usize, cycle_vars: usize, bits: &[bool]) -> Self {
        debug_assert_eq!(bits.len(), 1 << (addr_vars + cycle_vars));
        AddressComponent {
            addr_vars,
            cycle_vars,
            table: bits.iter().map(|&b| Gf128(b as u128)).collect(),
        }
    }

    /// Row-major index: address part low, cycle part high.
    #[inline]
    pub fn index_with(&self, addr: usize, cycle: usize) -> usize {
        addr | (cycle << self.addr_vars)
    }

    /// The row table `R(j) = Â(r, j)` — the component's MLE with the
    /// address coordinates fixed at `r`.
    pub fn row_at(&self, r: &[Gf128]) -> Vec<Gf128> {
        debug_assert_eq!(r.len(), self.addr_vars);
        // eq(r, addr) table over the address cube, then weighted sum per cycle.
        let eq = magnetar_field::EqBits::build(r).values;
        let mut row = vec![magnetar_field::ZERO; 1 << self.cycle_vars];
        for cycle in 0..(1usize << self.cycle_vars) {
            let mut acc = magnetar_field::ZERO;
            for addr in 0..(1usize << self.addr_vars) {
                acc += magnetar_field::mul(eq[addr], self.table[self.index_with(addr, cycle)]);
            }
            row[cycle] = acc;
        }
        row
    }

    /// The column table `G(k) = Âi(k, r_cycle)` — the component's MLE with
    /// the cycle coordinates fixed at `r_cycle`.
    pub fn column_at(&self, r_cycle: &[Gf128]) -> Vec<Gf128> {
        debug_assert_eq!(r_cycle.len(), self.cycle_vars);
        let eq = magnetar_field::EqBits::build(r_cycle).values;
        let mut col = vec![magnetar_field::ZERO; 1 << self.addr_vars];
        for addr in 0..(1usize << self.addr_vars) {
            let mut acc = magnetar_field::ZERO;
            for cycle in 0..(1usize << self.cycle_vars) {
                acc += magnetar_field::mul(eq[cycle], self.table[self.index_with(addr, cycle)]);
            }
            col[addr] = acc;
        }
        col
    }

    /// Full MLE evaluation at (r_addr, r_cycle).
    pub fn eval_at(&self, r_addr: &[Gf128], r_cycle: &[Gf128]) -> Gf128 {
        let col = self.column_at(r_cycle);
        magnetar_field::eval_mle(&col, r_addr)
    }
}

/// The refined one-hotness check for one component (paper §4.4):
/// prove `Σ_j eq(ρ, j)·[Â(r, j)³ + C·Â(r', j)] = 0`.
///
/// Outputs the evaluation claims on `Â` at `(r, r*)` and `(r', r*)`, which
/// the caller sends to the commitment layer (they are component claims).
#[derive(Clone, Debug)]
pub struct OneHotProof {
    /// Sum-check round messages (degree 4: eq × Â³).
    pub rounds: Vec<Vec<Gf128>>,
    /// Claimed `Â(r, r*)`.
    pub v_at_r: Gf128,
    /// Claimed `Â(r', r*)`.
    pub v_at_rp: Gf128,
    /// The sampled address point `r` and resampled `r'` (public, verifier
    /// recomputes).
    pub r: Vec<Gf128>,
    pub r_prime: Vec<Gf128>,
    /// The product `C(r)`.
    pub c_product: Gf128,
}

/// Sample the Frobenius-refined points `(r, r')` and `C(r)`.
///
/// Resamples `r` (up to a bound) if some `c_h = 0`.
pub fn sample_refined_points(
    t: &mut magnetar_hash::Transcript,
    addr_vars: usize,
) -> (Vec<Gf128>, Vec<Gf128>, Gf128) {
    use magnetar_field::{inv, mul, square, ONE, ZERO};
    let (mut r, mut r_prime);
    let mut c_product;
    loop {
        r = t.challenge_field_vec("onehot-r", addr_vars);
        let mut ok = true;
        r_prime = Vec::with_capacity(addr_vars);
        c_product = ONE;
        for &rh in &r {
            let ch = ONE + rh + square(rh);
            if ch == ZERO {
                ok = false;
                break;
            }
            // r'_h = r_h^3 / c_h.
            let r3 = mul(square(rh), rh);
            r_prime.push(mul(r3, inv(ch)));
            c_product = mul(c_product, ch);
        }
        if ok {
            return (r, r_prime, c_product);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rng.gen::<u128>())
    }

    #[test]
    fn row_and_column_folds() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let (av, cv) = (3usize, 4usize);
        let bits: Vec<bool> = (0..(1 << (av + cv))).map(|_| rng.gen::<bool>()).collect();
        let comp = AddressComponent::from_bits(av, cv, &bits);
        let r: Vec<Gf128> = (0..av).map(|_| rg(&mut rng)).collect();
        let rc: Vec<Gf128> = (0..cv).map(|_| rg(&mut rng)).collect();
        // row_at(r)[j] must equal the MLE at (r, j_boolean).
        let row = comp.row_at(&r);
        for j in 0..(1usize << cv) {
            let mut point = r.clone();
            for b in 0..cv {
                point.push(Gf128(((j >> b) & 1) as u128));
            }
            assert_eq!(
                row[j],
                magnetar_field::eval_mle(&comp.table, &point),
                "j={j}"
            );
        }
        let col = comp.column_at(&rc);
        for k in 0..(1usize << av) {
            let mut point: Vec<Gf128> = (0..av).map(|b| Gf128(((k >> b) & 1) as u128)).collect();
            point.extend_from_slice(&rc);
            assert_eq!(
                col[k],
                magnetar_field::eval_mle(&comp.table, &point),
                "k={k}"
            );
        }
        assert_eq!(
            comp.eval_at(&r, &rc),
            magnetar_field::eval_mle(&comp.table, &[r.clone(), rc.clone()].concat())
        );
    }
}
