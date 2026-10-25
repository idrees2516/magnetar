//! The **small-value round window** (ePrint 2025/1117 §5.1, 2026/587 §5.1
//! + Appendix C.5, specialized to characteristic two): compute the first
//!
//! `v` rounds' messages in **one pass** over the un-folded bit tables,
//! instead of one full-cube pass per round.
//!
//! ## The char-2 specialization (the monomial-basis window)
//!
//! The windowable shape is the binary relation's outer sum-check:
//! `g(x) = eq(τ, x) · (D0(x)·D1(x) + D2(x))` with `D0, D1, D2` **bit-valued**
//! on the Boolean cube (the small-value setting). The window rounds bind
//! the top `v` joint variables `x_{ell-1}, …, x_{ell-v}` (the
//! `HighBitsFirst` prefix); every other position index is a *suffix*
//! `x' ∈ {0,1}^{ell-v}` and the window sub-cube is the strided set
//! `{x' + m·2^{ell-v} : m ∈ {0,1}^v}`.
//!
//! Decompose `eq(τ, x) = eq(τ_w, m) · eq(τ_s, x')` (window ∥ suffix) and
//! expand the bit tables over the window's sub-cube in the **monomial
//! basis**: in characteristic two the multilinear Möbius coefficients of
//! a bit-valued function are bits (plain XOR-folds), so
//!
//! * `D0hat[m](x')`, `D1hat[m](x')`, `D2hat[m](x')` are bits — computed
//!   with AND/XOR only (the papers' small-value arithmetic, "popcount
//!   arithmetic" in the char-2 tower setting);
//! * the product `D0·D1` has monomial coefficients
//!   `chat[e](x') ∈ {0,1}` with exponent vectors `e ∈ {0,1,2}^v` — again
//!   pure AND/XOR (the convolution of two bit masks);
//! * the **eq-weighted pack** collapses the whole suffix dimension into
//!   just `3^v + 2^v` field accumulators:
//!   `Q[e] = Σ_{x'} eq(τ_s, x')·chat[e](x')` — one 16-byte select per set
//!   monomial per suffix. The window's state is *independent of the
//!   suffix count*: `O(3^v)` memory, not `O(3^v · 2^{ell-v})`.
//!
//! ## The round messages from the state
//!
//! Round `j`'s polynomial sums the future window variables over Booleans,
//! and in char two `Σ_{b∈{0,1}} b^k = 1` for every `k ≥ 1` — the future
//! part of every monomial collapses to 1. The bound window variables
//! contribute their challenge products (folded into the accumulators
//! incrementally), so
//!
//! `m_j(X) = C · eq(τ_{w,j}, X) · (A_0 + A_1·X + A_2·X²)`,
//!
//! with `C = Π_{i<j} eq(τ_{w,i}, r_i)` and `A_e` the digit-`j` bucket sums
//! of the state — a handful of field multiplications per round. The
//! messages are bit-identical to the generic engine's (the state is the
//! same polynomial; cross-validated by tests).
//!
//! ## The fast-forward (587 §C.5.6)
//!
//! After the window, the engine's tables must reach the state they would
//! have had after `v` generic folds. The bound table is the sub-cube MLE
//! at the challenges: `bound[x'] = Σ_m eq(r_{<v}, m)·f[m·stride + x']` —
//! and since the un-folded entries are **bits**, every term is a select
//! (`mul_bit`), no CLMUL. The eq factor takes its usual slab gathers and
//! the coefficient absorbs the per-round `eq(τ, r)/eq(τ, s)` deltas.

use crate::RoundMsg;
use magnetar_field::{inv, mul, mul_bit, Gf128, ONE, ZERO};
use rayon::prelude::*;

/// The window's mutable state (owned by the engine while active).
pub(crate) struct WindowState {
    pub v: usize,
    pub ell: usize,
    /// `2^(ell - v)`: the suffix count and the sub-cube stride.
    pub stride: usize,
    /// The eq-weighted, past-folded monomial accumulators of `D0·D1`
    /// (exponent vectors in `{0,1,2}^v`, mixed-radix index `Σ e_t·3^t`).
    q_ab: Vec<Gf128>,
    /// The same for `D2` (exponent vectors in `{0,1}^v`), indexed in the
    /// SAME mixed-radix space (`Σ e_t·3^t` with `e_t ∈ {0,1}`).
    q_c: Vec<Gf128>,
    /// The multilinear-pattern → mixed-radix monomial map `[m] → Σ bit_t·3^t`.
    mono3: Vec<u16>,
    /// `Π_{i<round} eq(τ_{w,i}, r_i)` — the eq factor's bound-window
    /// constant.
    c_eq: Gf128,
    /// The eq point's window coordinates in ROUND order:
    /// `tau_w[j] = τ[ell-1-j]`.
    tau_w: Vec<Gf128>,
    /// The per-window-pattern MLE weights `eq(r_{<round}, m)` for the
    /// fast-forward contraction (maintained incrementally).
    eq_r: Vec<Gf128>,
    /// The convolution's mixed-radix index table `[m1][m2] → monomial`.
    idx_table: Vec<u16>,
    /// Window rounds completed.
    pub round: usize,
}

/// The alternating SWAR masks for the in-place Möbius butterfly over a
/// `u64` truth table (bit `t` = coordinate `t`, little-endian).
#[inline]
fn swirl_mask(j: usize) -> u64 {
    // Positions with bit j set: blocks of 2^j ones every 2^{j+1}.
    match j {
        0 => 0xAAAA_AAAA_AAAA_AAAA,
        1 => 0xCCCC_CCCC_CCCC_CCCC,
        2 => 0xF0F0_F0F0_F0F0_F0F0,
        3 => 0xFF00_FF00_FF00_FF00,
        4 => 0xFFFF_0000_FFFF_0000,
        _ => 0xFFFF_FFFF_0000_0000,
    }
}

impl WindowState {
    /// Set up the window geometry. The heavy pass runs separately
    /// ([`WindowState::run_pass`]) so construction stays cheap.
    pub(crate) fn new(ell: usize, v: usize, tau: &[Gf128]) -> Self {
        assert!((1..=6).contains(&v) && v < ell, "window geometry");
        let stride = 1usize << (ell - v);
        let n_ab = 3usize.pow(v as u32);
        let n_c = 1usize << v;
        // The convolution index: monomial exponents add digit-wise in
        // base 3 (each side contributes 0/1 per coordinate).
        let mut idx_table = vec![0u16; n_c * n_c];
        for m1 in 0..n_c {
            for m2 in 0..n_c {
                let mut idx = 0usize;
                let mut w = 1usize;
                for t in 0..v {
                    let e = ((m1 >> t) & 1) + ((m2 >> t) & 1);
                    idx += e * w;
                    w *= 3;
                }
                idx_table[m1 * n_c + m2] = idx as u16;
            }
        }
        let tau_w: Vec<Gf128> = (0..v).map(|j| tau[ell - 1 - j]).collect();
        let mono3: Vec<u16> = (0..n_c)
            .map(|m| {
                let mut idx = 0usize;
                let mut w = 1usize;
                for t in 0..v {
                    if (m >> t) & 1 == 1 {
                        idx += w;
                    }
                    w *= 3;
                }
                idx as u16
            })
            .collect();
        WindowState {
            v,
            ell,
            stride,
            q_ab: vec![ZERO; n_ab],
            q_c: vec![ZERO; n_ab],
            mono3,
            c_eq: ONE,
            tau_w,
            eq_r: vec![ONE; n_c],
            idx_table,
            round: 0,
        }
    }

    /// The single pass over the un-folded tables: per suffix, gather the
    /// three tables' window sub-cubes, Möbius-transform them into bit
    /// masks, convolve `D0 ⊗ D1`, and eq-weight-pack the set monomials
    /// into the accumulators. Parallel over suffix chunks; the per-chunk
    /// accumulators combine order-free by `+` (R1).
    pub(crate) fn run_pass(
        &mut self,
        d0: &[Gf128],
        d1: &[Gf128],
        d2: &[Gf128],
        eq_table: &[Gf128],
        tau: &[Gf128],
    ) {
        let stride = self.stride;
        let v = self.v;
        self.run_pass_inner(
            |x| mobius_mask(d0, x, stride, v),
            |x| mobius_mask(d1, x, stride, v),
            |x| mobius_mask(d2, x, stride, v),
            eq_table,
            tau,
        );
    }

    /// The **mask-major** window pass: the three bit tables arrive
    /// packed — one `u64` window truth-table word per suffix (the
    /// `apply_simd_bit_masks` layout) — so the pass reads ONE word per
    /// (table, suffix) instead of `2^v` strided 16-byte entries. The
    /// resulting state (and therefore every message) is bit-identical to
    /// [`Self::run_pass`]'s.
    pub(crate) fn run_pass_masks(
        &mut self,
        m0: &[u64],
        m1: &[u64],
        m2: &[u64],
        eq_table: &[Gf128],
        tau: &[Gf128],
    ) {
        let v = self.v;
        self.run_pass_inner(
            |x| mobius_from_packed(m0[x], v),
            |x| mobius_from_packed(m1[x], v),
            |x| mobius_from_packed(m2[x], v),
            eq_table,
            tau,
        );
    }

    /// The generic pass core: `f_i(x)` returns table `i`'s Möbius mask
    /// at suffix `x`.
    #[allow(clippy::too_many_arguments)]
    fn run_pass_inner(
        &mut self,
        f0: impl Fn(usize) -> u64 + Send + Sync,
        f1: impl Fn(usize) -> u64 + Send + Sync,
        f2: impl Fn(usize) -> u64 + Send + Sync,
        eq_table: &[Gf128],
        tau: &[Gf128],
    ) {
        let v = self.v;
        let stride = self.stride;
        let n_c = 1usize << v;
        let n_ab = 3usize.pow(v as u32);
        let idx_table = &self.idx_table;
        // The eq stripe: the window pattern m* with eq(τ_w, m*) ≠ 0 —
        // m*_t = 1 exactly when τ at that coordinate is ONE (for
        // non-bit τ both slices are nonzero; 0 is fine).
        let mut m_star = 0usize;
        let mut eq_hi = ONE;
        for t in 0..v {
            let tau_t = tau[ell_t(self.ell, v, t)];
            if tau_t == ONE {
                m_star |= 1 << t;
                // eq(1, 1) = 1
            } else {
                // slice 0: eq(τ, 0) = 1 + τ
                eq_hi = mul(eq_hi, ONE + tau_t);
            }
        }
        let eq_base = m_star * stride;
        let eq_hi_inv = inv(eq_hi);

        // The dense-mask complement form. The pair convolution is
        // GF(2)-bilinear in the two masks, so with `a`/`b` the SPARSER of
        // `(m0, ~m0)`/`(m1, ~m1)` and `u0`/`u1` the complement flags:
        // `conv(m0, m1) = conv(a, b) + u0·Σ_{i1∈b} GSUM[i1]
        //                  + u1·Σ_{i0∈a} GCOL[i0] + u0·u1·GTOTAL`
        // where `GSUM[i1]` XORs the bitmaps over ALL i0 (GCOL transposed,
        // GTOTAL over both) — precomputed once. The pair count drops from
        // `p0·p1` to `min(p0, 64−p0)·min(p1, 64−p1)`: a fully dense
        // window sub-cube (the structured tables of the real circuits)
        // costs ~a dozen word XORs instead of 4096 pair iterations.
        let words = n_ab.div_ceil(64).max(1);
        // The valid window-pattern bits (v ≤ 6, but v < 6 masks leave the
        // high bits of the u64 clear — the complements must stay in range).
        let valid = if n_c == 64 {
            u64::MAX
        } else {
            (1u64 << n_c as u32) - 1
        };
        let mut gsum = vec![[0u64; 16]; n_c];
        let mut gcol = vec![[0u64; 16]; n_c];
        for i0 in 0..n_c {
            for i1 in 0..n_c {
                let idx = idx_table[i0 * n_c + i1] as usize;
                gsum[i1][idx >> 6] ^= 1u64 << (idx & 63);
                gcol[i0][idx >> 6] ^= 1u64 << (idx & 63);
            }
        }
        let mut gtotal = [0u64; 16];
        for g in &gsum {
            for (t, &w) in g.iter().enumerate().take(words) {
                gtotal[t] ^= w;
            }
        }

        // Parallel at the right granularity: small strides still spread
        // across cores (the previous PAR_THRESHOLD floor serialized the
        // 2^11-suffix passes entirely); tiny strides stay sequential.
        let chunk = (stride / (rayon::current_num_threads().max(1) * 4))
            .max(256)
            .min(stride)
            .max(1);
        let mut acc: (Vec<Gf128>, Vec<Gf128>) = (0..stride)
            .into_par_iter()
            .chunks(chunk)
            .map(|suffixes| {
                let (f0, f1, f2) = (&f0, &f1, &f2);
                let mut q_ab = vec![ZERO; n_ab];
                let mut q_c = vec![ZERO; n_ab];
                for x in suffixes {
                    // The three window sub-cubes' Möbius masks.
                    let m0 = f0(x);
                    let m1 = f1(x);
                    let m2 = f2(x);
                    // The eq-weighted pack: eq(τ_s, x') = the stripe entry
                    // scaled by 1/eq(τ_w, m*) (applied once at the end).
                    let w = eq_table[eq_base + x];
                    if w == ZERO {
                        continue;
                    }
                    // The convolution D0 ⊗ D1 into the 3^v monomial set —
                    // the complement form (see above).
                    let pc0 = m0.count_ones();
                    let pc1 = m1.count_ones();
                    let u0 = pc0 as usize > n_c / 2;
                    let u1 = pc1 as usize > n_c / 2;
                    let a = if u0 { m0 ^ valid } else { m0 };
                    let b = if u1 { m1 ^ valid } else { m1 };
                    let mut prod = [0u64; 16];
                    let mut sa = a;
                    while sa != 0 {
                        let i0 = sa.trailing_zeros() as usize;
                        sa &= sa - 1;
                        let mut sb = b;
                        while sb != 0 {
                            let i1 = sb.trailing_zeros() as usize;
                            sb &= sb - 1;
                            let idx = idx_table[i0 * n_c + i1] as usize;
                            prod[idx >> 6] ^= 1u64 << (idx & 63);
                        }
                    }
                    let mut corr = [0u64; 16];
                    if u0 {
                        let mut sb = b;
                        while sb != 0 {
                            let g = &gsum[sb.trailing_zeros() as usize];
                            for (t, &gw) in g.iter().enumerate().take(words) {
                                corr[t] ^= gw;
                            }
                            sb &= sb - 1;
                        }
                    }
                    if u1 {
                        let mut sa2 = a;
                        while sa2 != 0 {
                            let g = &gcol[sa2.trailing_zeros() as usize];
                            for (t, &gw) in g.iter().enumerate().take(words) {
                                corr[t] ^= gw;
                            }
                            sa2 &= sa2 - 1;
                        }
                    }
                    if u0 && u1 {
                        for (t, &gw) in gtotal.iter().enumerate().take(words) {
                            corr[t] ^= gw;
                        }
                    }
                    for (t, &cw) in corr.iter().enumerate().take(words) {
                        prod[t] ^= cw;
                    }
                    for (wi, word) in prod.iter().enumerate().take(words) {
                        let mut s = *word;
                        while s != 0 {
                            let idx = (wi << 6) + s.trailing_zeros() as usize;
                            s &= s - 1;
                            q_ab[idx] += w;
                        }
                    }
                    let mut s2 = m2;
                    while s2 != 0 {
                        let m = s2.trailing_zeros() as usize;
                        s2 &= s2 - 1;
                        q_c[self.mono3[m] as usize] += w;
                    }
                }
                (q_ab, q_c)
            })
            .reduce(
                || (vec![ZERO; n_ab], vec![ZERO; n_ab]),
                |mut a, b| {
                    for (x, y) in a.0.iter_mut().zip(b.0.iter()) {
                        *x += *y;
                    }
                    for (x, y) in a.1.iter_mut().zip(b.1.iter()) {
                        *x += *y;
                    }
                    a
                },
            );
        // The single post-scale by 1/eq(τ_w, m*).
        if eq_hi_inv != ONE {
            for q in acc.0.iter_mut() {
                *q = mul(*q, eq_hi_inv);
            }
            for q in acc.1.iter_mut() {
                *q = mul(*q, eq_hi_inv);
            }
        }
        self.q_ab = acc.0;
        self.q_c = acc.1;
    }

    /// The current round's message, extracted from the monomial state.
    /// `coeffs` are the spec's term coefficients `[c_ab, c_c]`.
    pub(crate) fn message(&self, coeffs: &[Gf128]) -> RoundMsg {
        let j = self.round;
        debug_assert!(j < self.v);
        // This round binds window coordinate (v-1-j) (the rounds bind the
        // top joint variables first; coordinate t = joint variable
        // ell-v+t).
        let coord = self.v - 1 - j;
        let n_ab = self.q_ab.len();
        // The digit buckets at this coordinate, weighted by the future
        // coordinates' τ factors. The rounds bind the window coordinates
        // TOP-DOWN, so at round `j` the *future* (still-Boolean)
        // coordinates are `t < coord`. Summing the product
        // `eq(τ_t, b)·b^{e}` over `b ∈ {0,1}` gives `1` for `e = 0` and
        // `τ_t` for every `e ≥ 1` (b² = b on Booleans) — each future
        // coordinate the monomial covers contributes exactly `τ_t`. The
        // past coordinates (t > coord) are already challenge-folded into
        // the state.
        let mut a = [ZERO, ZERO, ZERO];
        let mut b = [ZERO, ZERO, ZERO];
        // τ per window coordinate (index = coordinate t).
        let tau_coord: Vec<Gf128> = (0..self.v).map(|t| self.tau_w[self.v - 1 - t]).collect();
        for idx in 0..n_ab {
            let mut rem = idx;
            let mut e_coord = 0usize;
            let mut fw = ONE;
            for t in 0..self.v {
                let e = rem % 3;
                rem /= 3;
                if t == coord {
                    e_coord = e;
                } else if t < coord && e > 0 {
                    fw = mul(fw, tau_coord[t]);
                }
            }
            a[e_coord] += mul(self.q_ab[idx], fw);
            b[e_coord] += mul(self.q_c[idx], fw);
        }
        // The eq factor's linear part at this round: eq(τ, X) = (1+τ)+X.
        let tau = self.tau_w[j];
        let t0 = ONE + tau;
        let t1 = ONE;
        let c = self.c_eq;
        // m(X) = C · (t0 + t1·X) · [c_ab·(A0 + A1 X + A2 X²)
        //                          + c_c·(B0 + B1 X)]
        let (ca, cc) = (coeffs[0], coeffs[1]);
        let s0 = mul(ca, a[0]) + mul(cc, b[0]);
        let s1 = mul(ca, a[1]) + mul(cc, b[1]);
        let s2 = mul(ca, a[2]);
        let p0 = mul(c, mul(t0, s0));
        let p1 = mul(c, mul(t0, s1) + mul(t1, s0));
        let p2 = mul(c, mul(t0, s2) + mul(t1, s1));
        let p3 = mul(c, mul(t1, s2));
        RoundMsg(vec![p0, p1, p2, p3])
    }

    /// Bind the round's challenge: fold the monomial state's coordinate,
    /// advance the eq constants.
    pub(crate) fn bind(&mut self, r: Gf128) {
        let j = self.round;
        let coord = self.v - 1 - j;
        let w3 = 3usize.pow(coord as u32);
        // Digit folds at `coord`: 1 → 0 with ×r FIRST (its source bucket
        // must not include the 2 → 1 results), then 2 → 1 with ×r².
        let n_ab = self.q_ab.len();
        let n_block = n_ab / (w3 * 3);
        for blk in 0..n_block {
            let base = blk * w3 * 3;
            // Per-monomial digit folds at `coord`: each digit-1 entry
            // folds into its own digit-0 slot (×r), each digit-2 entry
            // into its digit-1 slot (×r²). Reading both sources before
            // writing keeps the passes independent.
            for k in 0..w3 {
                let v1 = self.q_ab[base + w3 + k];
                let v2 = self.q_ab[base + 2 * w3 + k];
                let v1c = self.q_c[base + w3 + k];
                self.q_ab[base + k] += mul(v1, r);
                self.q_ab[base + w3 + k] = mul(v2, mul(r, r));
                self.q_ab[base + 2 * w3 + k] = ZERO;
                self.q_c[base + k] += mul(v1c, r);
                self.q_c[base + w3 + k] = ZERO;
            }
        }
        // C ×= eq(τ_w[j], r) and eq_r ×= eq(r, m's coord bit).
        let tau = self.tau_w[j];
        self.c_eq = mul(self.c_eq, ONE + tau + r);
        let n_c = self.eq_r.len();
        for m in 0..n_c {
            let mb = Gf128(((m >> coord) & 1) as u128);
            self.eq_r[m] = mul(self.eq_r[m], ONE + r + mb);
        }
        self.round += 1;
    }

    /// The fast-forward's MLE weights (one per window pattern).
    pub(crate) fn eq_r(&self) -> &[Gf128] {
        &self.eq_r
    }

    /// Test access to the raw monomial state.
    #[cfg(test)]
    pub(crate) fn debug_state(&self) -> (&[Gf128], &[Gf128]) {
        (&self.q_ab, &self.q_c)
    }
}

/// The joint variable of window coordinate `t`.
#[inline]
fn ell_t(ell: usize, v: usize, t: usize) -> usize {
    ell - v + t
}

/// The window sub-cube's Möbius transform of `table` at suffix `x`, as a
/// bit mask (bit `m` = the monomial-m coefficient, itself a bit).
#[inline]
fn mobius_mask(table: &[Gf128], x: usize, stride: usize, v: usize) -> u64 {
    let mut m = 0u64;
    for t in 0..(1usize << v) {
        if table[t * stride + x].0 & 1 != 0 {
            m |= 1u64 << t;
        }
    }
    for j in 0..v {
        // new[t] = old[t] ^ old[t ^ 2^j] for t with bit j set: the
        // shifted-up operand supplies old[t − 2^j] at position t.
        m ^= (m << (1 << j)) & swirl_mask(j);
    }
    m
}

/// The Möbius butterfly over an already-packed window truth table (the
/// mask-major form `apply_simd_bit_masks` produces): identical to
/// [`mobius_mask`] without the strided extraction.
#[inline]
fn mobius_from_packed(mut m: u64, v: usize) -> u64 {
    for j in 0..v {
        m ^= (m << (1 << j)) & swirl_mask(j);
    }
    m
}

/// Contract a still-bit-valued table at the window challenges: the sub-cube
/// MLE at `r_{<v}`, computed with selects (`eq_r[m]·bit`). Parallel over
/// the suffix range; used by the engine's fast-forward.
/// The mask-major contraction: identical to [`contract_bit_table`] but
/// the bit table arrives packed (one `u64` window truth-table word per
/// suffix) — a select-dot over each word's set bits instead of `2^v`
/// strided 16-byte reads.
pub(crate) fn contract_bit_masks(
    masks: &[u64],
    eq_r: &[Gf128],
    stride: usize,
    v: usize,
) -> Vec<Gf128> {
    assert_eq!(masks.len(), stride);
    assert_eq!(eq_r.len(), 1usize << v);
    let mut out = vec![ZERO; stride];
    let chunk = if stride >= 2 * magnetar_field::PAR_THRESHOLD {
        (stride / (rayon::current_num_threads().max(1) * 4))
            .clamp(magnetar_field::PAR_THRESHOLD, stride)
    } else {
        stride
    };
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, o)| {
        let base = ci * chunk;
        for (rel, dst) in o.iter_mut().enumerate() {
            let mut s = masks[base + rel];
            while s != 0 {
                let m = s.trailing_zeros() as usize;
                s &= s - 1;
                *dst += eq_r[m];
            }
        }
    });
    out
}

pub(crate) fn contract_bit_table(
    table: &[Gf128],
    eq_r: &[Gf128],
    stride: usize,
    v: usize,
) -> Vec<Gf128> {
    assert_eq!(table.len(), eq_r.len() * stride);
    let n_c = 1usize << v;
    let mut out = vec![ZERO; stride];
    let chunk = if stride >= 2 * magnetar_field::PAR_THRESHOLD {
        (stride / (rayon::current_num_threads().max(1) * 4))
            .clamp(magnetar_field::PAR_THRESHOLD, stride)
    } else {
        stride
    };
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, o)| {
        let base = ci * chunk;
        for m in 0..n_c {
            let w = eq_r[m];
            if w == ZERO {
                continue;
            }
            let src = &table[m * stride + base..m * stride + base + o.len()];
            for (dst, &s) in o.iter_mut().zip(src.iter()) {
                *dst += mul_bit(w, s);
            }
        }
    });
    out
}

/// The default window size: `min(6, high)` — the papers' single early
/// window. The window's per-suffix pack costs `Θ(3^v)` select-adds
/// against the replaced rounds' `Θ(2^v)` CLMUL passes per position, so
/// the crossover sits in the high single digits; `v = 6` is also the
/// SWAR limit (the monomial butterflies live in one `u64` truth table).
/// The `MAGNETAR_WINDOW` environment variable overrides, 0 disables.
pub(crate) fn window_size(high: usize) -> usize {
    if let Ok(w) = std::env::var("MAGNETAR_WINDOW") {
        if let Ok(w) = w.parse::<usize>() {
            return w.min(high);
        }
    }
    6.min(high)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rand::RngExt::random::<u128>(rng))
    }

    /// The Moebius mask must equal the direct XOR-fold over subsets.
    #[test]
    fn mobius_mask_matches_direct() {
        let mut rng = rand::rng();
        for v in [1usize, 3, 6] {
            let n = 1usize << v;
            let bits: Vec<bool> = (0..n)
                .map(|_| rand::RngExt::random::<bool>(&mut rng))
                .collect();
            let table: Vec<Gf128> = bits.iter().map(|&b| Gf128(b as u128)).collect();
            let m = mobius_mask(&table, 0, 1, v);
            for t in 0..n {
                // hat[t] = XOR of bits over subsets of t.
                let mut want = false;
                for s in 0..=t {
                    if t & s == s && bits[s] {
                        want = !want;
                    }
                }
                assert_eq!((m >> t) & 1, want as u64, "v={v} t={t}");
            }
        }
    }

    /// After one bind, the state's coordinate-sums must equal the true
    /// bound polynomial's coefficients: coef(d...) = Σ_{d_bound} state.
    #[test]
    fn bind_state_matches_bound_coefficients() {
        let mut rng = rand::rng();
        let ell = 7usize;
        let v = 3usize;
        let n = 1usize << ell;
        let d0t: Vec<Gf128> = (0..n)
            .map(|_| Gf128(rand::RngExt::random::<bool>(&mut rng) as u128))
            .collect();
        let d1t: Vec<Gf128> = (0..n)
            .map(|_| Gf128(rand::RngExt::random::<bool>(&mut rng) as u128))
            .collect();
        let d2t: Vec<Gf128> = (0..n)
            .map(|_| Gf128(rand::RngExt::random::<bool>(&mut rng) as u128))
            .collect();
        let tau: Vec<Gf128> = (0..ell)
            .map(|_| Gf128(rand::RngExt::random::<u128>(&mut rng)))
            .collect();
        let eqt = magnetar_field::EqBits::build(&tau).values;
        let mut w = WindowState::new(ell, v, &tau);
        w.run_pass(&d0t, &d1t, &d2t, &eqt, &tau);
        // Brute-force the pre-bind state: Qhat[(e2,e1,e0)] = Σ_{x'} eq_s(x')
        // · chat[(e2,e1,e0)](x') where chat = the monomial coefficients of
        // the product over the window sub-cube.
        let stride = 1usize << (ell - v);
        let mono_of = |bits: &[bool]| -> Vec<bool> {
            // The Moebius coefficients over the 3 window coords (v=3):
            let mut out = vec![false; 27];
            for m1 in 0..8usize {
                for m2 in 0..8usize {
                    if !(bits[m1] && bits[m2]) {
                        continue;
                    }
                    let mut idx = 0usize;
                    let mut wgt = 1usize;
                    for t in 0..v {
                        let e = ((m1 >> t) & 1) + ((m2 >> t) & 1);
                        idx += e * wgt;
                        wgt *= 3;
                    }
                    out[idx] = !out[idx];
                }
            }
            out
        };
        let mobius = |tab: &[Gf128], x: usize| -> Vec<bool> {
            let mut vals = [false; 8];
            for t in 0..8usize {
                vals[t] = tab[t * stride + x].0 & 1 == 1;
            }
            // butterflies
            for j in 0..v {
                for t in 0..8usize {
                    if t & (1 << j) != 0 {
                        vals[t] ^= vals[t ^ (1 << j)];
                    }
                }
            }
            vals.to_vec()
        };
        let mut qhat = vec![Gf128(0); 27];
        for x in 0..stride {
            let m0 = mobius(&d0t, x);
            let m1m = mobius(&d1t, x);
            // chat = the XOR-convolution mask
            let mut chat = [false; 27];
            for a in 0..8usize {
                if !m0[a] {
                    continue;
                }
                for b in 0..8usize {
                    if !m1m[b] {
                        continue;
                    }
                    let mut idx = 0usize;
                    let mut wgt = 1usize;
                    for t in 0..v {
                        let e = ((a >> t) & 1) + ((b >> t) & 1);
                        idx += e * wgt;
                        wgt *= 3;
                    }
                    chat[idx] = !chat[idx];
                }
            }
            let _ = mono_of;
            for (idx, &c) in chat.iter().enumerate() {
                if c {
                    qhat[idx] += eqt[x];
                }
            }
        }
        let (q_ab, _q_c) = w.debug_state();
        // The pass scales by 1/eq_hi: eq_hi = eq(tau_w, m*). Compute it.
        let mut eq_hi = ONE;
        for t in 0..v {
            let tv = tau[ell - v + t];
            if tv != ONE {
                eq_hi = mul(eq_hi, ONE + tv);
            }
        }
        let scale = inv(eq_hi);
        for idx in 0..27 {
            assert_eq!(q_ab[idx], mul(qhat[idx], scale), "pre-bind state idx={idx}");
        }
        // Bind coordinate 2 at r and check the folded sums.
        let r = Gf128(rand::RngExt::random::<u128>(&mut rng));
        w.bind(r);
        let (q_ab2, _q_c2) = w.debug_state();
        // The true bound coefficients: coef(d1,d0) = Σ_{d2} r^{d2}·Qhat.
        for d1 in 0..3usize {
            for d0 in 0..3usize {
                let mut want = ZERO;
                for d2 in 0..3usize {
                    want += mul(
                        mul(qhat[d2 * 9 + d1 * 3 + d0], scale),
                        if d2 == 0 {
                            ONE
                        } else if d2 == 1 {
                            r
                        } else {
                            mul(r, r)
                        },
                    );
                }
                let got = q_ab2[d1 * 3 + d0] + q_ab2[9 + d1 * 3 + d0] + q_ab2[18 + d1 * 3 + d0];
                assert_eq!(got, want, "bound coef d1={d1} d0={d0}");
            }
        }
    }

    /// The bit-table contraction must equal v rounds of the generic fold.
    #[test]
    fn contract_matches_generic_fold() {
        let mut rng = rand::rng();
        for (ell, v) in [(6usize, 4usize), (9, 6)] {
            let n = 1usize << ell;
            let table: Vec<Gf128> = (0..n)
                .map(|_| Gf128(rand::RngExt::random::<bool>(&mut rng) as u128))
                .collect();
            let rs: Vec<Gf128> = (0..v).map(|_| rg(&mut rng)).collect();
            let n_c = 1usize << v;
            let mut eq_r = vec![ONE; n_c];
            for (i, &r) in rs.iter().enumerate() {
                // Round i binds window coordinate (v-1-i) (the top-down
                // bind order).
                let coord = v - 1 - i;
                for m in 0..n_c {
                    let mb = Gf128(((m >> coord) & 1) as u128);
                    eq_r[m] = mul(eq_r[m], ONE + r + mb);
                }
            }
            let got = contract_bit_table(&table, &eq_r, n >> v, v);
            // Reference: fold v times at the top bits.
            let mut want = table.clone();
            for i in 0..v {
                let bit = ell - 1 - i;
                let r = rs[i];
                let half = 1usize << bit;
                for base in (0..want.len()).step_by(2 * half) {
                    for o in 0..half {
                        let lo = want[base + o];
                        let hi = want[base + o + half];
                        want[base + o] = lo + mul(r, lo + hi);
                    }
                }
                want.truncate(want.len() / 2);
            }
            assert_eq!(got, want, "ell={ell} v={v}");
        }
    }
}
