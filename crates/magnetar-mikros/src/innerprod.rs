//! Algorithm 1 — the packed inner-product proof (ePrint 2026/2646, §4.1).
//!
//! Reduces a length-`N = mκ` GF(2) inner-product claim
//! `c = 〈a, b〉 = ⊕_{i} a_i·b_i` to a length-`m` extension-field claim
//! through **monomial/reciprocal-monomial packing**:
//!
//! ```text
//! gA(X) = Σ_j a^(j) X^{j−1},   gB(X) = Σ_j b^(j) X^{−(j−1)}   (Laurent)
//! f(X)  = 〈gA(X), gB(X)〉  —  the constant coefficient is 〈a, b〉 = c,
//!         every cross-column term has a nonzero exponent:
//! f(X)  = c + X·h+(X) + X^{−1}·h−(X^{−1}),   deg h± ≤ κ−2   (Eq. 7)
//! ```
//!
//! The prover sends the `κ−1` GF(2) coefficients of each of `h+` and `h−`
//! (2·127 bits), the verifier samples `x ←$ E×` and both sides pack:
//!
//! ```text
//! ā = A·ρA = gA(x) ∈ E^m,  b̄ = B·ρB = gB(x) ∈ E^m,
//! ρA = (1, x, …, x^{κ−1}),  ρB = (1, x^{−1}, …, x^{−(κ−1)})
//! c̄ = c + x·h+(x) + x^{−1}·h−(x^{−1})                       (Eq. 8)
//! ```
//!
//! Fact 1 reduces `〈ā, b̄〉 = c̄` to the two terminal MLE claims
//! `v1 = ts(y)^⊤ AρA`, `v2 = ts(y)^⊤ BρB`, proved by the packed PCS
//! (Algorithm 4) with the *monomial* column weights — the one packing
//! route that does not go through the Lagrange basis.
//!
//! Prover cost: `O(Nκ)` bit-ops for the cross-correlations `f_τ` (κ²
//! plane AND+popcounts), `O(mκ)` E-mults for the packed encodings.

use crate::fact1::{self, Fact1Proof};
use crate::hadamard::encode_e_words;
use crate::interp::KAPPA;
use crate::planes::PlaneSet;
use magnetar_field::Gf128;
use magnetar_field::{mul, ONE, ZERO};
use magnetar_hash::Transcript;

/// The GF(2) inner product of two packed plane sets:
/// `〈a, b〉 = parity(⊕_j (a^(j) & b^(j)))` — the claim the caller fixes.
pub fn inner_product_f2(a: &PlaneSet, b: &PlaneSet) -> Gf128 {
    debug_assert_eq!(a.m, b.m);
    debug_assert_eq!(a.kappa, b.kappa);
    let w = a.words;
    let mut acc = vec![0u64; w];
    for j in 0..a.kappa {
        let pa = a.plane(j);
        let pb = b.plane(j);
        for k in 0..w {
            acc[k] ^= pa[k] & pb[k];
        }
    }
    let parity = acc.iter().map(|v| v.count_ones()).sum::<u32>() & 1;
    Gf128(parity as u128)
}

/// The cross-correlation `f_τ = Σ_{j−j'=τ} 〈a^(j), b^(j')〉 ∈ GF(2)`,
/// bitsliced: one plane-AND accumulation per τ, then a single popcount.
/// `τ ≠ 0`; positive `τ` shifts `a` against `b`, negative the other way.
fn f_cross(a: &PlaneSet, b: &PlaneSet, tau: isize) -> bool {
    let w = a.words;
    let mut acc = vec![0u64; w];
    if tau > 0 {
        let t = tau as usize;
        for j in t..a.kappa {
            let pa = a.plane(j);
            let pb = b.plane(j - t);
            for k in 0..w {
                acc[k] ^= pa[k] & pb[k];
            }
        }
    } else {
        let t = (-tau) as usize;
        for j in t..a.kappa {
            let pa = a.plane(j - t);
            let pb = b.plane(j);
            for k in 0..w {
                acc[k] ^= pa[k] & pb[k];
            }
        }
    }
    let parity = acc.iter().map(|v| v.count_ones()).sum::<u32>() & 1;
    parity == 1
}

/// A transcript challenge conditioned to be nonzero (`x ←$ E×` — the
/// reciprocal weights `ρB` need `x^{−1}`; the zero event has probability
/// 2^{−128}, handled by rejection to keep the protocol exact).
fn challenge_nonzero(t: &mut Transcript, label: &str) -> Gf128 {
    loop {
        let x = t.challenge_field(label);
        if x != ZERO {
            return x;
        }
    }
}

pub struct InnerProdProof {
    /// `h+` coefficients: `h+[i] = f_{i+1}`, `i ∈ [0, κ−2]` (GF(2) bits).
    pub h_plus: Vec<u8>,
    /// `h−` coefficients: `h−[i] = f_{−(i+1)}` (GF(2) bits).
    pub h_minus: Vec<u8>,
    /// The length-m extension-field inner-product reduction.
    pub fact1: Fact1Proof,
}

/// The terminal packed claims for Algorithm 4 (the packed PCS):
/// `(v1, ρA)` and `(v2, ρB)` at the shared row MLE point `y`
/// (with the Fact-1 forward challenge order — reverse before the PCS).
pub struct InnerProdTerminal {
    pub y: Vec<Gf128>,
    pub v1: Gf128,
    pub v2: Gf128,
    pub rho_a: Vec<Gf128>,
    pub rho_b: Vec<Gf128>,
}

/// Prove `c = 〈a, b〉` over GF(2) for packed matrices `A, B ∈ F^{m×κ}`
/// (Algorithm 1). Returns the proof and the prover-side terminal data.
pub fn prove_innerprod(
    a: &PlaneSet,
    b: &PlaneSet,
    c: Gf128,
    t: &mut Transcript,
) -> (InnerProdProof, InnerProdTerminal) {
    let m = a.m;
    debug_assert_eq!(b.m, m);
    debug_assert_eq!(a.kappa, KAPPA);
    debug_assert_eq!(b.kappa, KAPPA);

    // --- Steps 1–4: the correction polynomials from the cross-correlations.
    let mut h_plus = vec![0u8; KAPPA - 1];
    let mut h_minus = vec![0u8; KAPPA - 1];
    for i in 0..KAPPA - 1 {
        h_plus[i] = f_cross(a, b, (i + 1) as isize) as u8;
        h_minus[i] = f_cross(a, b, -((i + 1) as isize)) as u8;
    }
    t.absorb_bytes("ip-hplus", &h_plus);
    t.absorb_bytes("ip-hminus", &h_minus);

    // --- Step 6: x ←$ E× and the monomial/reciprocal weights.
    let x = challenge_nonzero(t, "ip-x");
    let x_inv = magnetar_field::inv(x);
    let mut rho_a = Vec::with_capacity(KAPPA);
    let mut rho_b = Vec::with_capacity(KAPPA);
    {
        let mut pa = ONE;
        let mut pb = ONE;
        for _ in 0..KAPPA {
            rho_a.push(pa);
            rho_b.push(pb);
            pa = mul(pa, x);
            pb = mul(pb, x_inv);
        }
    }

    // --- Step 8: the packed encodings ā = AρA, b̄ = BρB.
    let a_vec = encode_e_words(a, &rho_a, m);
    let b_vec = encode_e_words(b, &rho_b, m);

    // --- Step 9: c̄ and Fact 1.
    // h+(x) = Σ_i h_plus[i]·x^i (Horner), then scaled by x;
    // x^{−1}·h−(x^{−1}) = Σ_i h_minus[i]·x^{−(i+1)}.
    let mut c_bar = c;
    {
        let mut hp = ZERO;
        for &bit in h_plus.iter().rev() {
            hp = mul(hp, x) + Gf128(bit as u128);
        }
        c_bar += mul(x, hp);
        let mut hm = ZERO;
        for &bit in h_minus.iter().rev() {
            hm = mul(hm, x_inv) + Gf128(bit as u128);
        }
        c_bar += mul(x_inv, hm);
    }
    let mut av = a_vec;
    let mut bv = b_vec;
    // NOTE: no prover-side assert 〈ā, b̄〉 = c̄ — `c` is the caller's claim and
    // tamper tests deliberately feed stale claims; the verifier's Fact-1
    // round check (against c̄ recomputed from c and h±) is the soundness gate.
    let fact1 = fact1::prove(&mut av, &mut bv, c_bar, t);

    let terminal = InnerProdTerminal {
        y: fact1.challenges.clone(),
        v1: fact1.terminals.0,
        v2: fact1.terminals.1,
        rho_a,
        rho_b,
    };
    (
        InnerProdProof {
            h_plus,
            h_minus,
            fact1,
        },
        terminal,
    )
}

/// Verify (Algorithm 1): returns the terminal packed claims.
pub fn verify_innerprod(
    c: Gf128,
    proof: &InnerProdProof,
    t: &mut Transcript,
) -> Result<InnerProdTerminal, &'static str> {
    if proof.h_plus.len() != KAPPA - 1 || proof.h_minus.len() != KAPPA - 1 {
        return Err("innerprod: malformed message sizes");
    }
    if proof.h_plus.iter().any(|&b| b > 1) || proof.h_minus.iter().any(|&b| b > 1) {
        return Err("innerprod: coefficients must be GF(2)");
    }
    t.absorb_bytes("ip-hplus", &proof.h_plus);
    t.absorb_bytes("ip-hminus", &proof.h_minus);
    let x = challenge_nonzero(t, "ip-x");
    let x_inv = magnetar_field::inv(x);
    let mut rho_a = Vec::with_capacity(KAPPA);
    let mut rho_b = Vec::with_capacity(KAPPA);
    {
        let mut pa = ONE;
        let mut pb = ONE;
        for _ in 0..KAPPA {
            rho_a.push(pa);
            rho_b.push(pb);
            pa = mul(pa, x);
            pb = mul(pb, x_inv);
        }
    }
    let mut c_bar = c;
    {
        let mut hp = ZERO;
        for &bit in proof.h_plus.iter().rev() {
            hp = mul(hp, x) + Gf128(bit as u128);
        }
        c_bar += mul(x, hp);
        let mut hm = ZERO;
        for &bit in proof.h_minus.iter().rev() {
            hm = mul(hm, x_inv) + Gf128(bit as u128);
        }
        c_bar += mul(x_inv, hm);
    }
    let (y, v1, v2) = fact1::verify(c_bar, &proof.fact1, t)?;
    Ok(InnerProdTerminal {
        y,
        v1,
        v2,
        rho_a,
        rho_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::KAPPA;

    thread_local! {
        static RNG: std::cell::Cell<u128> = const { std::cell::Cell::new(0x51ce5eedbadbad0e) };
    }

    fn gf() -> Gf128 {
        RNG.with(|c| {
            let mut r = c.get();
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            c.set(r);
            Gf128(r)
        })
    }

    fn test_transcript() -> Transcript {
        Transcript::new("mikros-innerprod-test", &[])
    }

    /// The dense Laurent reference: evaluate `f(γ) = 〈gA(γ), gB(γ)〉` at a
    /// field point γ for every exponent, and confirm the identity
    /// `f(γ) = c + γ·h+(γ) + γ^{−1}·h−(γ^{−1})`.
    #[test]
    fn laurent_identity_matches_reference() {
        let m = 128usize;
        let mut a = PlaneSet::zeros(m, KAPPA);
        let mut b = PlaneSet::zeros(m, KAPPA);
        for j in 0..KAPPA {
            for i in 0..m {
                a.set_bit(j, i, gf().0 & 1 == 1);
                b.set_bit(j, i, gf().0 & 1 == 1);
            }
        }
        let mut t = test_transcript();
        let c = inner_product_f2(&a, &b);
        let (proof, _term) = prove_innerprod(&a, &b, c, &mut t);

        // dense f_τ reference: f_τ = Σ_{j−j'=τ} 〈a^(j), b^(j')〉
        let bit_dot = |j: usize, jp: usize| -> bool {
            let mut parity = 0u32;
            for i in 0..m {
                parity ^= (a.bit(j, i) & b.bit(jp, i)) as u32;
            }
            parity == 1
        };
        for i in 0..KAPPA - 1 {
            let mut want = false;
            for j in (i + 1)..KAPPA {
                want ^= bit_dot(j, j - (i + 1));
            }
            assert_eq!(proof.h_plus[i], want as u8, "h+[{i}] = f_{}", i + 1);
            let mut want_m = false;
            for j in (i + 1)..KAPPA {
                want_m ^= bit_dot(j - (i + 1), j);
            }
            assert_eq!(proof.h_minus[i], want_m as u8, "h−[{i}] = f_-{}", i + 1);
        }

        // the Laurent identity at random points: f(γ) vs c + γh+(γ) + γ⁻¹h−(γ⁻¹)
        for _ in 0..4 {
            let gamma = gf();
            if gamma == ZERO {
                continue;
            }
            let gi = magnetar_field::inv(gamma);
            // gA(γ)[r] = Σ_j a^(j)[r]·γ^{j},  gB(γ)[r] = Σ_j b^(j)[r]·γ^{−j}
            let mut f_g = ZERO;
            for r in 0..m {
                let mut ga = ZERO;
                let mut gb = ZERO;
                let mut pw = ONE;
                let mut pwi = ONE;
                for j in 0..KAPPA {
                    if a.bit(j, r) {
                        ga += pw;
                    }
                    if b.bit(j, r) {
                        gb += pwi;
                    }
                    pw = mul(pw, gamma);
                    pwi = mul(pwi, gi);
                }
                f_g += mul(ga, gb);
            }
            let mut hp = ZERO;
            for &bit in proof.h_plus.iter().rev() {
                hp = mul(hp, gamma) + Gf128(bit as u128);
            }
            let mut hm = ZERO;
            for &bit in proof.h_minus.iter().rev() {
                hm = mul(hm, gi) + Gf128(bit as u128);
            }
            assert_eq!(
                f_g,
                c + mul(gamma, hp) + mul(gi, hm),
                "f(γ) = c + γ·h+(γ) + γ⁻¹·h−(γ⁻¹)"
            );
        }
    }

    #[test]
    fn roundtrip_and_tamper() {
        let m = 256usize;
        let mut a = PlaneSet::zeros(m, KAPPA);
        let mut b = PlaneSet::zeros(m, KAPPA);
        for j in 0..KAPPA {
            for i in 0..m {
                a.set_bit(j, i, gf().0 & 1 == 1);
                b.set_bit(j, i, gf().0 & 1 == 1);
            }
        }
        let c = inner_product_f2(&a, &b);
        let mut t = test_transcript();
        let (proof, term) = prove_innerprod(&a, &b, c, &mut t);
        let mut tv = test_transcript();
        let vterm = verify_innerprod(c, &proof, &mut tv).expect("the honest proof must verify");
        assert_eq!(vterm.v1, term.v1, "v1 binding");
        assert_eq!(vterm.v2, term.v2, "v2 binding");
        assert_eq!(vterm.rho_a, term.rho_a, "ρA binding");
        assert_eq!(vterm.rho_b, term.rho_b, "ρB binding");

        // the terminal claims against direct dense evaluation:
        // v1 = 〈AρA, ts(y)〉, v2 = 〈BρB, ts(y)〉
        let ts_eval = |v: &[Gf128], pts: &[Gf128]| -> Gf128 {
            let mut cur = v.to_vec();
            for &p in pts {
                let half = cur.len() / 2;
                for k in 0..half {
                    cur[k] = cur[k] + mul(p, cur[k] + cur[k + half]);
                }
                cur.truncate(half);
            }
            cur[0]
        };
        let a_vec = encode_e_words(&a, &term.rho_a, m);
        let b_vec = encode_e_words(&b, &term.rho_b, m);
        assert_eq!(term.v1, ts_eval(&a_vec, &term.y), "v1 = 〈AρA, ts(y)〉");
        assert_eq!(term.v2, ts_eval(&b_vec, &term.y), "v2 = 〈BρB, ts(y)〉");

        // tamper h+: one flipped coefficient must break the reduction
        {
            let mut bad = InnerProdProof {
                h_plus: proof.h_plus.clone(),
                h_minus: proof.h_minus.clone(),
                fact1: clone_fact1(&proof.fact1),
            };
            bad.h_plus[7] ^= 1;
            let mut tb = test_transcript();
            assert!(verify_innerprod(c, &bad, &mut tb).is_err(), "h+ tamper");
        }
        // wrong claim: the c ≠ 〈a,b〉 route must reject
        {
            let c2 = c + Gf128(1);
            let mut tv = test_transcript();
            // re-derive: a wrong claim c2 with the honest h±/fact1 — the
            // reduced claim c̄ mismatches the Fact-1 first round sum.
            assert!(
                verify_innerprod(c2, &proof, &mut tv).is_err(),
                "wrong claim must reject"
            );
        }
    }

    /// The end-to-end knowledge-soundness shape: a witness tamper (flip a
    /// live bit where the product is 1) with the STALE claim must reject.
    #[test]
    fn witness_tamper_rejected() {
        let m = 128usize;
        let mut a = PlaneSet::zeros(m, KAPPA);
        let mut b = PlaneSet::zeros(m, KAPPA);
        for j in 0..KAPPA {
            for i in 0..m {
                a.set_bit(j, i, gf().0 & 1 == 1);
                b.set_bit(j, i, gf().0 & 1 == 1);
            }
        }
        let c = inner_product_f2(&a, &b);
        // flip a live bit
        let mut a2 = a.clone();
        'outer: for j in 0..KAPPA {
            for i in 0..m {
                if a.bit(j, i) && b.bit(j, i) {
                    a2.set_bit(j, i, false);
                    break 'outer;
                }
            }
        }
        assert_ne!(
            inner_product_f2(&a2, &b),
            c,
            "the flip must change the claim"
        );
        let mut t = test_transcript();
        let (bad_proof, _) = prove_innerprod(&a2, &b, c, &mut t);
        let mut tv = test_transcript();
        assert!(
            verify_innerprod(c, &bad_proof, &mut tv).is_err(),
            "tampered witness with a stale claim must reject"
        );
    }

    fn clone_fact1(p: &Fact1Proof) -> Fact1Proof {
        Fact1Proof {
            rounds: p.rounds.clone(),
            terminals: p.terminals,
            challenges: p.challenges.clone(),
        }
    }
}
