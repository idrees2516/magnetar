//! BinarySpartan on the Mikros stack — the production SNARK path for R1CS
//! (Setty, *BinarySpartan: Spartan over binary fields*, IACR ePrint
//! 2026/1656): Spartan instantiated on this repo's fields and the packed
//! multi-instance layout of Algorithm 6.
//!
//! ## What it is
//!
//! A transparent, plausibly post-quantum SNARK for the bit-level R1CS
//! `S_L w ∘ S_R w = S_O w` with κ = 128 packed instances, polylogarithmic
//! proofs, a **linear-time prover**, and the verifier's matrix work at the
//! real-Spartan operating point (direct sparse-MLE evaluation in O(nnz) —
//! the circuit is public; the *witness* is what is committed, through this
//! repo's fold-chain PCS in Ligerito's role).
//!
//! The protocol (Spartan's outer + inner shape, this stack's fields):
//!
//! 1. **The public-I/O bind** (Algorithm 6's steps 3–5 verbatim): the
//!    masked-eq Fact-1 reduction of `〈ξ_io, w〉` to a witness MLE claim
//!    at `y_io` — one plain PCS opening.
//! 2. **The outer sum-check** — the eq-weighted satisfaction over the
//!    joint (instance ‖ constraint-row) cube:
//!    `0 = Σ_{(j,i)} eq(τ_j, j)·eq(τ_i, i)·[L(j,i)·R(j,i) + O(j,i)]`
//!    with L, R, O the packed images `S⋆W` (Algorithm-2 layout: plane =
//!    instance, rows = constraints), `τ_j ∈ E^7` and `τ_i ∈ E^{log N}`
//!    verifier challenges. It runs in two phases:
//!    * **The instance phase** (7 rounds, degree-3 messages): the
//!      prover-side state never materializes an E-valued cube. Per free
//!      plane it forms the challenge-weighted combinations of the
//!      *original* GF(2) planes (XOR-selects, one pass) and takes the
//!      `l`-weighted dots — the row-eq weight `l = eq(τ_i, ·)` rides the
//!      dots, the bound coordinates' `eq(τ_j, ρ)` products ride the
//!      running claim, and each round's message is
//!      `[(1+τ) + X]·[u0 + u1·X + u2·X²]` in the characteristic-two
//!      MLE-bind algebra. Constant memory, work halving per round.
//!    * **The row phase** (log N rounds): after the instance point `j*`
//!      is fixed, the images fold once at `eq(j*, ·)` (bitsliced
//!      plane-combination passes) into E-valued row tables, and the
//!      engine finishes the sum-check with the eq-slab discipline — the
//!      public `eq(τ_i, ·)` factor never costs a fold multiplication
//!      (the never-folded-slab optimization the engine already carries).
//! 3. **The inner sum-check** (Spartan's second): the terminal image
//!    claims `v⋆ = ⋆̂(j*, i*)` reduce through the matrices by the identity
//!    `⋆̂(j*, i*) = Σ_y Ŝ⋆(i*, y)·Ŵ(y, j*)` — an α-batched Fact-1 over
//!    the witness columns: the matrix side `t_comb = Σ⋆ α⋆·Ŝ⋆(i*, ·)`
//!    (the verifier direct-evaluates the sparse MLEs, O(nnz)), the
//!    witness side the `eq(j*,·)`-folded packed table.
//! 4. **The openings**: the io claim through the plain fold-chain
//!    evaluation; the witness claim `Ŵ(y_in, j*)` through Algorithm 4's
//!    packed opening (the U coordinate claims + the char-2 W recursion),
//!    anchored to the same committed table.
//!
//! ## The BinarySpartan technique map on this stack
//!
//! | BinarySpartan (2026/1656) | Here |
//! |---|---|
//! | Spartan over binary fields (outer + inner) | this module |
//! | Ligerito commitment + DP24 ring switching | `magnetar-pcs` (the fold-chain backend all openings use) |
//! | Phalanx SIMD R1CS | the packed multi-instance layout (Algorithm 6's `S⋆W` planes) |
//! | eq/sum-check optimizations (Gruen 2023, Dao–Thaler 2024/1210, BDDT) | the engine's never-folded eq slabs; the 64K byte product tables (`gf8`) |
//! | hash workloads at the 200K hashes/s class | the Table-4 circuits through this path (`mikros-bench`) |
//!
//! Not carried over: SuperSpartan's *next multilinear extension* — the
//! paper text was not retrievable this session (eprint PDF access is
//! Cloudflare-gated); it is documented as not-implemented in
//! `docs/STATUS.md` rather than guessed at.

use crate::fact1::{self, Fact1Proof};
use crate::pcs_open::{self, PackedOpenProof};
use crate::planes::PlaneSet;
use crate::r1cs::{dot_gf, eq_build, transpose_ps, BinaryR1cs};
use magnetar_field::Gf128;
use magnetar_field::{mul, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_pcs::eval::{prove_eval, verify_eval, PcsParams, PcsProof, PcsStatement};
use magnetar_sumcheck::engine::{SumcheckProver, SumcheckVerifier};
use magnetar_sumcheck::{BindOrder, EqFactor, RoundMsg, SumcheckSpec, Term};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The eq tensor of a point in little-endian bit order (`bit b of the
/// index ↔ point[b]`), as a length-`2^pt.len()` vector.
fn tensor(pt: &[Gf128]) -> Vec<Gf128> {
    let mut v = vec![ONE];
    for &p in pt {
        let half = v.len();
        let mut nv = vec![ZERO; half * 2];
        for i in 0..half {
            nv[i] = mul(v[i], ONE + p);
            nv[half + i] = mul(v[i], p);
        }
        v = nv;
    }
    v
}

/// The two-entry factor list of an eq point in bit order: factor `r` ↔
/// bit `r` (`(1+pt[r], pt[r])` in char 2).
fn eq_factors(pt: &[Gf128]) -> Vec<(Gf128, Gf128)> {
    pt.iter().map(|&p| (ONE + p, p)).collect()
}

/// `Σ_j w[j]·ps.plane(j)` — the eq-weighted combination of a PlaneSet's
/// planes into an E-vector over the rows (the bitsliced fold: one XOR
/// select per set plane bit; the E-scalar never multiplies, it selects).
fn fold_planes_at(ps: &PlaneSet, w: &[Gf128]) -> Vec<Gf128> {
    let n = ps.m;
    let words = ps.words;
    let mut out = vec![ZERO; n];
    for j in 0..ps.kappa {
        let wj = w[j];
        if wj == ZERO {
            continue;
        }
        let plane = ps.plane(j);
        for wb in 0..words {
            let pw = plane[wb];
            if pw == 0 {
                continue;
            }
            let base = wb * 64;
            for r in 0..64 {
                if (pw >> r) & 1 == 1 {
                    out[base + r] += wj;
                }
            }
        }
    }
    out
}

/// The challenge-weighted combination of a PlaneSet's planes over a bound
/// prefix: `comb[b | x<<k | jp<<(k+1)] = Σ_b c[b]·ps.plane(...)` — the
/// per-round instance-phase materialization (a fresh E-vector per call;
/// `fold_planes_at` is the k=7 special case with the full plane index).
#[allow(clippy::too_many_arguments)]
fn combine_planes(ps: &PlaneSet, c: &[Gf128], k: usize, x: usize, jp: usize, out: &mut [Gf128]) {
    let words = ps.words;
    for b in 0..(1usize << k) {
        let cb = c[b];
        if cb == ZERO {
            continue;
        }
        let p = b | (x << k) | (jp << (k + 1));
        let plane = ps.plane(p);
        for wb in 0..words {
            let pw = plane[wb];
            if pw == 0 {
                continue;
            }
            let base = wb * 64;
            for r in 0..64 {
                if (pw >> r) & 1 == 1 {
                    out[base + r] += cb;
                }
            }
        }
    }
}

/// `Σ_i l[i]·a[i]` (the E dot).
fn dot2(l: &[Gf128], a: &[Gf128]) -> Gf128 {
    let mut acc = ZERO;
    for i in 0..a.len() {
        if a[i] != ZERO {
            acc += mul(l[i], a[i]);
        }
    }
    acc
}

/// `Σ_i l[i]·a[i]·b[i]` (the l-weighted product dot).
fn dot3(l: &[Gf128], a: &[Gf128], b: &[Gf128]) -> Gf128 {
    let mut acc = ZERO;
    for i in 0..a.len() {
        if a[i] != ZERO {
            acc += mul(l[i], mul(a[i], b[i]));
        }
    }
    acc
}

/// The matrix column-table at a row point: `t[y] = Σ_{i: (i,y) ∈ S}
/// eq(i*, i)` — O(nnz), the verifier-side real-Spartan cost.
fn row_point_table(s: &crate::r1cs::SparseMatrix, eq_i: &[Gf128]) -> Vec<Gf128> {
    let mut t = vec![ZERO; s.n];
    for (i, cols) in s.rows.iter().enumerate() {
        let w = eq_i[i];
        if w == ZERO {
            continue;
        }
        for &y in cols {
            t[y as usize] += w;
        }
    }
    t
}

/// The plain PCS point from a Fact-1 challenge list (Fact-1 binds MSB
/// first; the fold-chain PCS binds LSB first — the batch.rs boundary).
fn pcs_point(y: &[Gf128]) -> Vec<Gf128> {
    y.iter().rev().copied().collect()
}

// ---------------------------------------------------------------------------
// The proof
// ---------------------------------------------------------------------------

pub struct SpartanProof {
    /// The io bind's claim (steps 3–5 of Algorithm 6, verbatim).
    pub io_claim: Gf128,
    pub fact1_io: Fact1Proof,
    /// The 7 instance-round messages: each the four degree-3
    /// coefficients `[X⁰, X¹, X², X³]`.
    pub outer_inst: Vec<[Gf128; 4]>,
    /// The row-phase engine round messages (coefficient vectors).
    pub outer_row: Vec<Vec<Gf128>>,
    /// The outer terminal's image claims `v_L, v_R, v_O` (the images'
    /// MLEs at (j*, i*)) — the engine's dense-evaluation claims.
    pub terminals: [Gf128; 3],
    /// The inner's α combination (transcript-sampled, carried for the
    /// caller's inspection).
    pub alphas: [Gf128; 3],
    pub inner: Fact1Proof,
    /// The inner terminal's t-side claim `t_comb(y_in)` — checked by the
    /// verifier against its direct sparse-MLE evaluation.
    pub t_term: Gf128,
    /// The witness claim from the inner: `(y_in (MSB-first), Ŵ(y_in, j*))`.
    pub w_claim: (Vec<Gf128>, Gf128),
    /// The packed opening (Algorithm 4) of the witness claim.
    pub po: PackedOpenProof,
    /// The plain opening of the io claim.
    pub pcs_io: PcsProof,
    pub stmt: PcsStatement,
}

// ---------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn prove_spartan(
    cs: &BinaryR1cs,
    setup: &crate::batch::BatchSetup,
    params: &PcsParams,
    packed_io: &[Gf128],
    t: &mut Transcript,
) -> SpartanProof {
    let n = cs.s_l.n;
    let log_n = n.trailing_zeros() as usize;
    let w = &setup.w_table;

    // ---- 1. The public-I/O bind (Algorithm 6's steps 3–5 verbatim).
    let x_io: Vec<Gf128> = (0..log_n).map(|_| t.challenge_field("sp-io-x")).collect();
    let eq_io = eq_build(&x_io);
    let mut io_claim = ZERO;
    for (k, &pos) in cs.io_positions.iter().enumerate() {
        io_claim += mul(eq_io[pos as usize], packed_io[k]);
    }
    let io_sorted = &cs.io_positions;
    let mut masked = eq_io.clone();
    let mut ii = 0usize;
    for i in 0..n {
        while ii < io_sorted.len() && (io_sorted[ii] as usize) < i {
            ii += 1;
        }
        let in_io = ii < io_sorted.len() && io_sorted[ii] as usize == i;
        if !in_io {
            masked[i] = ZERO;
        }
    }
    debug_assert_eq!(dot_gf(&masked, w), io_claim);
    let mut a_io = w.to_vec();
    let mut m_io = masked.clone();
    let fact1_io = fact1::prove(&mut a_io, &mut m_io, io_claim, t);
    let y_io = fact1_io.challenges.clone();
    let w_io = fact1_io.terminals.0;

    // ---- 2. The outer sum-check.
    // 2a. The challenges: τ_j (7 instance coords), τ_i (log N row coords).
    let tau_j: Vec<Gf128> = (0..7).map(|_| t.challenge_field("sp-tauj")).collect();
    let tau_i: Vec<Gf128> = (0..log_n).map(|_| t.challenge_field("sp-taui")).collect();
    // The row-eq weight tensor in the row-bit order.
    let l = crate::planes::RowWeightTensor::new(eq_factors(&tau_i), true);
    debug_assert_eq!(l.m, n);

    // 2b. The instance phase: 7 rounds, degree-3 messages, constant
    // memory. State: c (the bound-prefix weights), running.
    let mut c: Vec<Gf128> = vec![ONE]; // eq() of the empty prefix
    let mut running = ZERO; // the claim
                            // The accumulated instance-point eq weight `Π_{m<k} eq(τ_j[m], ρ_m)`:
                            // every round's message carries it — the summand's bound coordinates
                            // contribute their eq factors to the remaining claim (the toy circuit
                            // validated trivially zero on both sides of this identity; the SHA
                            // differential found it: round 1's sum disagreed with the chain).
    let mut eq_accum = ONE;
    let mut outer_inst: Vec<[Gf128; 4]> = Vec::with_capacity(7);
    let mut jstar: Vec<Gf128> = Vec::with_capacity(7);
    for k in 0..7 {
        // eq over the free instance coords (bits k+1..6), little-endian
        // in the free part's own bits: factor m' ↔ plane bit k+1+m'.
        let free_pt: Vec<Gf128> = (k + 1..7).map(|m| tau_j[m]).collect();
        let eqrem = tensor(&free_pt);
        let n_free = eqrem.len();
        let mut u0 = ZERO;
        let mut u1 = ZERO;
        let mut u2 = ZERO;
        // Scratch: 6 combinations at a time (L0, L1, R0, R1, O0, O01).
        let mut la = vec![ZERO; n];
        let mut lb = vec![ZERO; n];
        let mut ra = vec![ZERO; n];
        let mut rb = vec![ZERO; n];
        let mut oa = vec![ZERO; n];
        let mut ob = vec![ZERO; n];
        for jp in 0..n_free {
            for v in la.iter_mut() {
                *v = ZERO;
            }
            for v in lb.iter_mut() {
                *v = ZERO;
            }
            for v in ra.iter_mut() {
                *v = ZERO;
            }
            for v in rb.iter_mut() {
                *v = ZERO;
            }
            for v in oa.iter_mut() {
                *v = ZERO;
            }
            for v in ob.iter_mut() {
                *v = ZERO;
            }
            combine_planes(&setup.sw[0], &c, k, 0, jp, &mut la);
            combine_planes(&setup.sw[0], &c, k, 1, jp, &mut lb);
            combine_planes(&setup.sw[1], &c, k, 0, jp, &mut ra);
            combine_planes(&setup.sw[1], &c, k, 1, jp, &mut rb);
            combine_planes(&setup.sw[2], &c, k, 0, jp, &mut oa);
            combine_planes(&setup.sw[2], &c, k, 1, jp, &mut ob);
            let d00 = dot3(&l.words, &la, &ra);
            let d01 = dot3(&l.words, &la, &rb);
            let d10 = dot3(&l.words, &lb, &ra);
            let d11 = dot3(&l.words, &lb, &rb);
            let e0 = dot2(&l.words, &oa);
            // O01 = O0 + O1 (the X¹ coefficient of the O bind).
            for i in 0..n {
                ob[i] += oa[i];
            }
            let e01 = dot2(&l.words, &ob);
            let wj = eqrem[jp];
            if wj != ZERO {
                u0 += mul(wj, d00 + e0);
                u1 += mul(wj, d01 + d10 + e01);
                u2 += mul(wj, d00 + d01 + d10 + d11);
            }
        }
        let tau = tau_j[k];
        let msg: [Gf128; 4] = [
            mul(eq_accum, mul(ONE + tau, u0)),
            mul(eq_accum, mul(ONE + tau, u1) + u0),
            mul(eq_accum, mul(ONE + tau, u2) + u1),
            mul(eq_accum, u2),
        ];
        outer_inst.push(msg);
        // eq_accum ← eq_accum · eq(τ_j[k], ρ_k)
        // Absorb, sample, fold the running claim.
        let mut buf = Vec::with_capacity(4 * 16);
        for &cf in &msg {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(cf));
        }
        t.absorb_bytes("sp-inst", &buf);
        let rk = t.challenge_field("sp-inst");
        eq_accum = mul(eq_accum, ONE + tau_j[k] + rk);
        let mut mr = ZERO;
        let mut xp = Gf128(1);
        for &cf in &msg {
            mr += mul(cf, xp);
            xp = mul(xp, rk);
        }
        running = mr;
        // c ← c ⊗ (1+rk, rk) with the NEW variable at the TOP bit of the
        // bound pattern: c[b] keeps the old bits at 0..k-1 and the round-k
        // value lands at bit k — the plane index p = b | (x<<k) | (jp<<(k+1))
        // then addresses the instances in the plane's own bit order. (The
        // first draft shifted the old bits up — a reversed bound pattern
        // that only surfaced at round 2, where the toy circuit's sparse
        // images could not see it.)
        let half = c.len();
        let mut nc = vec![ZERO; half * 2];
        for b in 0..half {
            nc[b] = mul(c[b], ONE + rk);
            nc[b + half] = mul(c[b], rk);
        }
        c = nc;
        jstar.push(rk);
    }
    // The instance point's eq value (the scalar the row phase's term
    // coefficients carry): eq(τ_j, j*) = Π_m eq(τ_j[m], ρ_m).
    let eqr_j = tensor(&jstar); // eq(j*, ·) over the planes
    let eq_jstar = eq_accum; // Π_m eq(τ_j[m], ρ_m) — the accumulated weight
    debug_assert_eq!(
        {
            let mut v = ONE;
            for (m, &rho) in jstar.iter().enumerate() {
                v = mul(v, ONE + tau_j[m] + rho);
            }
            v
        },
        eq_jstar,
        "the accumulated factor must equal the recomputation"
    );
    debug_assert_ne!(eq_jstar, ZERO, "negligible-probability degenerate point");

    // 2c. The images fold once at eq(j*, ·): E-valued row tables.
    let l_star = fold_planes_at(&setup.sw[0], &eqr_j);
    let r_star = fold_planes_at(&setup.sw[1], &eqr_j);
    let o_star = fold_planes_at(&setup.sw[2], &eqr_j);

    // 2d. The row phase through the engine (the eq-slab discipline on the
    // public eq(τ_i, ·) factor; the term coefficients carry eq(τ_j, j*)).
    let spec = SumcheckSpec {
        num_vars: log_n,
        dense: vec![l_star, r_star, o_star],
        public: vec![Box::new(EqFactor::new(tau_i.clone()))],
        terms: vec![
            Term {
                coeff: eq_jstar,
                dense: vec![0, 1],
                public: vec![0],
            },
            Term {
                coeff: eq_jstar,
                dense: vec![2],
                public: vec![0],
            },
        ],
        claim: running,
    };
    let mut row_msgs: Vec<Vec<Gf128>> = Vec::with_capacity(log_n);
    let mut row_challenges: Vec<Gf128> = Vec::with_capacity(log_n);
    let (v_l, v_o, v_r);
    {
        let mut p = SumcheckProver::new(&spec, BindOrder::Increasing);
        for _ in 0..log_n {
            let msg = p.message();
            let mut buf = Vec::with_capacity(msg.0.len() * 16);
            for &cf in &msg.0 {
                buf.extend_from_slice(&magnetar_hash::fe_to_bytes(cf));
            }
            t.absorb_bytes("sp-row", &buf);
            let r = t.challenge_field("sp-row");
            p.fold_msg(&RoundMsg(msg.0.clone()), r);
            row_msgs.push(msg.0.clone());
            row_challenges.push(r);
        }
        debug_assert_eq!(p.remaining_vars(), 0);
        let (claims, _) = p.finish();
        // The engine folds dense in the spec's order [L, R, O].
        v_l = claims[0];
        v_r = claims[1];
        v_o = claims[2];
    }

    // ---- 3. The inner sum-check (Spartan's second).
    let alphas: [Gf128; 3] = std::array::from_fn(|_| t.challenge_field("sp-alpha"));
    let v_comb = mul(alphas[0], v_l) + mul(alphas[1], v_r) + mul(alphas[2], v_o);
    // i* in row-bit order = the engine's challenges (Increasing binds
    // bit r at round r).
    let i_star: Vec<Gf128> = row_challenges.clone();
    let eq_i = eq_build(&i_star);
    let t_l = row_point_table(&cs.s_l, &eq_i);
    let t_r = row_point_table(&cs.s_r, &eq_i);
    let t_o = row_point_table(&cs.s_o, &eq_i);
    let mut t_comb = vec![ZERO; n];
    for y in 0..n {
        t_comb[y] = mul(alphas[0], t_l[y]) + mul(alphas[1], t_r[y]) + mul(alphas[2], t_o[y]);
    }
    // The witness instance-fold: the packed table's transpose (plane =
    // instance, rows = witness columns), folded at eq(j*, ·).
    let w_alayout = transpose_ps(&setup.packed_t);
    let wf = fold_planes_at(&w_alayout, &eqr_j);
    let mut tv = t_comb.clone();
    let mut wv = wf.clone();
    let v_inner = dot_gf(&t_comb, &wf);
    debug_assert_eq!(v_inner, v_comb, "the Spartan inner identity");
    let inner = fact1::prove(&mut tv, &mut wv, v_comb, t);
    let y_in = inner.challenges.clone();
    let t_term = inner.terminals.0;
    let w_term = inner.terminals.1;

    // ---- 4. The openings.
    // 4a. The io claim: the plain fold-chain opening (the batch.rs
    // boundary order: Fact-1 MSB-first → PCS LSB-first).
    let io_point = pcs_point(&y_io);
    let pcs_io = prove_eval(params, w, &io_point, w_io, t);
    // 4b. The witness claim Ŵ(y_in, j*): Algorithm 4's packed opening on
    // the witness A-layout (plane = instance, rows = witness columns),
    // ξ = eq(y_in) over the rows, ρ = eq(j*, ·) over the planes.
    // The ξ list in little-endian bit order: factor r ↔ the witness
    // column index's bit r (the A-layout's row space); the packed open's
    // W-recursion and PCS opening reverse its internal Fact-1 point to
    // match (pcs_open.rs, the spartan-module fix).
    let y_bits: Vec<Gf128> = (0..log_n).map(|r| y_in[log_n - 1 - r]).collect();
    let xi = eq_factors(&y_bits);
    let po = pcs_open::prove_packed_open(&w_alayout, w, &xi, &eqr_j, w_term, params, t);

    SpartanProof {
        io_claim,
        fact1_io,
        outer_inst,
        outer_row: row_msgs,
        terminals: [v_l, v_r, v_o],
        alphas,
        inner,
        t_term,
        w_claim: (y_in, w_term),
        po,
        pcs_io,
        stmt: setup.stmt.clone(),
    }
}

// ---------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn verify_spartan(
    cs: &BinaryR1cs,
    params: &PcsParams,
    packed_io: &[Gf128],
    proof: &SpartanProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let n = cs.s_l.n;
    let log_n = n.trailing_zeros() as usize;

    // ---- 1. The public-I/O bind.
    let x_io: Vec<Gf128> = (0..log_n).map(|_| t.challenge_field("sp-io-x")).collect();
    let eq_io = eq_build(&x_io);
    let mut io_claim = ZERO;
    for (k, &pos) in cs.io_positions.iter().enumerate() {
        io_claim += mul(eq_io[pos as usize], packed_io[k]);
    }
    if io_claim != proof.io_claim {
        return Err("spartan: io claim mismatch");
    }
    let io_sorted = &cs.io_positions;
    let mut masked = eq_io.clone();
    let mut ii = 0usize;
    for i in 0..n {
        while ii < io_sorted.len() && (io_sorted[ii] as usize) < i {
            ii += 1;
        }
        let in_io = ii < io_sorted.len() && io_sorted[ii] as usize == i;
        if !in_io {
            masked[i] = ZERO;
        }
    }
    let (y_io_pt, w_io, w_io_pub) = fact1::verify(io_claim, &proof.fact1_io, t)?;
    {
        let eq_y = eq_build(&y_io_pt.iter().rev().copied().collect::<Vec<_>>());
        let mut want = ZERO;
        for &pos in cs.io_positions.iter() {
            want += mul(eq_io[pos as usize], eq_y[pos as usize]);
        }
        if want != w_io_pub {
            return Err("spartan: io public terminal mismatch");
        }
    }

    // ---- 2. The outer sum-check.
    let tau_j: Vec<Gf128> = (0..7).map(|_| t.challenge_field("sp-tauj")).collect();
    let tau_i: Vec<Gf128> = (0..log_n).map(|_| t.challenge_field("sp-taui")).collect();
    if proof.outer_inst.len() != 7 {
        return Err("spartan: instance-round count");
    }
    // The instance rounds: Σ_{t≥1} c_t = running; running' = m(ρ).
    let mut running = ZERO;
    let mut jstar: Vec<Gf128> = Vec::with_capacity(7);
    for k in 0..7 {
        let msg = &proof.outer_inst[k];
        let s123 = msg[1] + msg[2] + msg[3];
        if s123 != running {
            eprintln!(
                "DBG: instance round {k}: sum={:x} running={:x} msg=[{:x},{:x},{:x},{:x}]",
                s123.0, running.0, msg[0].0, msg[1].0, msg[2].0, msg[3].0
            );
            return Err("spartan: instance round sum");
        }
        let mut buf = Vec::with_capacity(4 * 16);
        for &cf in msg {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(cf));
        }
        t.absorb_bytes("sp-inst", &buf);
        let rk = t.challenge_field("sp-inst");
        let mut mr = ZERO;
        let mut xp = Gf128(1);
        for &cf in msg {
            mr += mul(cf, xp);
            xp = mul(xp, rk);
        }
        running = mr;
        jstar.push(rk);
    }
    let eqr_j = tensor(&jstar);
    let mut eq_jstar = ONE;
    for (m, &rho) in jstar.iter().enumerate() {
        eq_jstar = mul(eq_jstar, ONE + tau_j[m] + rho);
    }
    if eq_jstar == ZERO {
        // A degenerate instance point unpins the outer→inner binding (a
        // negligible-probability, transcript-determined event): abort.
        return Err("spartan: degenerate instance point");
    }

    // The row phase: the engine verifier with the eq-slab public factor.
    if proof.outer_row.len() != log_n {
        return Err("spartan: row-round count");
    }
    let public: Vec<Box<dyn magnetar_sumcheck::PublicFactor + Send + Sync>> =
        vec![Box::new(EqFactor::new(tau_i.clone()))];
    let terms = vec![
        Term {
            coeff: eq_jstar,
            dense: vec![0, 1],
            public: vec![0],
        },
        Term {
            coeff: eq_jstar,
            dense: vec![2],
            public: vec![0],
        },
    ];
    let mut vfr = SumcheckVerifier::new(log_n, running, &public, terms, BindOrder::Increasing);
    for m in &proof.outer_row {
        vfr.round(&RoundMsg(m.clone()), t, "sp-row")?;
    }
    let (i_star, _) = vfr.finish(&proof.terminals)?;
    // i_star is already in original (row-bit) order.

    // ---- 3. The inner sum-check.
    let alphas: [Gf128; 3] = std::array::from_fn(|_| t.challenge_field("sp-alpha"));
    let v_comb = mul(alphas[0], proof.terminals[0])
        + mul(alphas[1], proof.terminals[1])
        + mul(alphas[2], proof.terminals[2]);
    let eq_i = eq_build(&i_star);
    let t_l = row_point_table(&cs.s_l, &eq_i);
    let t_r = row_point_table(&cs.s_r, &eq_i);
    let t_o = row_point_table(&cs.s_o, &eq_i);
    let (y_in, t_ret, w_ret) = fact1::verify(v_comb, &proof.inner, t)?;
    // The t-side terminal vs the direct sparse-MLE evaluation (the
    // real-Spartan verifier cost, O(nnz)): t_ret == Σ⋆ α⋆ · Ŝ⋆(i*, y_in).
    {
        let y_bits: Vec<Gf128> = (0..log_n).map(|r| y_in[log_n - 1 - r]).collect();
        let eq_y = eq_build(&y_bits);
        let ev = |tab: &[Gf128]| -> Gf128 {
            let mut acc = ZERO;
            for y in 0..n {
                if tab[y] != ZERO {
                    acc += mul(eq_y[y], tab[y]);
                }
            }
            acc
        };
        let direct = mul(alphas[0], ev(&t_l)) + mul(alphas[1], ev(&t_r)) + mul(alphas[2], ev(&t_o));
        if direct != t_ret {
            return Err("spartan: inner t-side direct-evaluation mismatch");
        }
    }
    if y_in != proof.w_claim.0 || w_ret != proof.w_claim.1 {
        return Err("spartan: witness claim mismatch");
    }

    // ---- 4. The openings.
    // 4a. The io claim: the plain fold-chain opening.
    let io_point = pcs_point(&y_io_pt);
    verify_eval(&proof.stmt, params, &io_point, w_io, &proof.pcs_io, t)?;
    // 4b. The witness claim: Algorithm 4's packed opening.
    let y_bits: Vec<Gf128> = (0..log_n).map(|r| y_in[log_n - 1 - r]).collect();
    let xi = eq_factors(&y_bits);
    pcs_open::verify_packed_open(&proof.stmt, params, &xi, &eqr_j, w_ret, &proof.po, t)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{self, BatchProof};
    use crate::circuits::CircuitBuilder;
    use crate::interp::KAPPA;
    use magnetar_field::ONE;

    fn emit_toy(b: &mut CircuitBuilder, v0: bool, v1: bool) {
        // x0·x1 = x3 ∧ (x0+x3)·x2 = 0, with x2 private (= 0).
        let x0 = b.io_bit(v0);
        let x1 = b.io_bit(v1);
        let x2 = b.bit(false);
        let x3 = b.io_bit(v0 & v1);
        let p = b.and(x0, x1);
        b.assert_eq(p, x3);
        let lhs = b.xor(&[x0, x3]);
        let pr = b.and(lhs, x2);
        let zero = b.bit(false);
        b.assert_eq(pr, zero);
    }

    fn toy_cs() -> (crate::r1cs::BinaryR1cs, Vec<Vec<bool>>) {
        let mut b = CircuitBuilder::new();
        emit_toy(&mut b, false, false);
        let cs = b.build();
        let witnesses: Vec<Vec<bool>> = (0..KAPPA)
            .map(|j| {
                let mut rb = CircuitBuilder::replay();
                emit_toy(&mut rb, j & 1 == 1, (j >> 1) & 1 == 1);
                rb.w
            })
            .collect();
        (cs, witnesses)
    }

    fn prove_toy(
        cs: &crate::r1cs::BinaryR1cs,
        witnesses: &[Vec<bool>],
    ) -> (batch::BatchSetup, Vec<Gf128>, SpartanProof, Transcript) {
        let params = PcsParams::default();
        let setup = batch::setup_batch(cs, witnesses, &params);
        let packed_io = batch::packed_io_words(cs, witnesses);
        let mut t = Transcript::new("mikros-spartan-test", &[]);
        let proof = prove_spartan(cs, &setup, &params, &packed_io, &mut t);
        (setup, packed_io, proof, t)
    }

    fn clone_f1(p: &Fact1Proof) -> Fact1Proof {
        Fact1Proof {
            rounds: p.rounds.clone(),
            terminals: p.terminals,
            challenges: p.challenges.clone(),
        }
    }

    #[test]
    fn spartan_roundtrip_and_tamper() {
        let (cs, witnesses) = toy_cs();
        assert!(witnesses.iter().all(|w| cs.satisfied_by(w)));
        let params = PcsParams::default();
        let (setup, packed_io, proof, _t) = prove_toy(&cs, &witnesses);
        let mut tv = Transcript::new("mikros-spartan-test", &[]);
        let vres = verify_spartan(&cs, &params, &packed_io, &proof, &mut tv);
        assert!(
            vres.is_ok(),
            "the honest spartan proof must verify (err = {:?})",
            vres.err()
        );

        // tamper: the public io words
        {
            let mut bad_io = packed_io.clone();
            bad_io[0] += ONE;
            let mut tv = Transcript::new("mikros-spartan-test", &[]);
            assert!(
                verify_spartan(&cs, &params, &bad_io, &proof, &mut tv).is_err(),
                "a flipped io word must reject"
            );
        }
        // tamper: an instance-round coefficient (the round sum check fires
        // at the NEXT round, or the terminal at the end).
        {
            let mut bad = clone_sp(&proof);
            bad.outer_inst[3][2] += ONE;
            let mut tv = Transcript::new("mikros-spartan-test", &[]);
            assert!(
                verify_spartan(&cs, &params, &packed_io, &bad, &mut tv).is_err(),
                "a tampered instance-round message must reject"
            );
        }
        // tamper: a row-phase message coefficient.
        {
            let mut bad = clone_sp(&proof);
            bad.outer_row[2][1] += ONE;
            let mut tv = Transcript::new("mikros-spartan-test", &[]);
            assert!(
                verify_spartan(&cs, &params, &packed_io, &bad, &mut tv).is_err(),
                "a tampered row-phase message must reject"
            );
        }
        // tamper: the outer terminal image claim v_L.
        {
            let mut bad = clone_sp(&proof);
            bad.terminals[0] += ONE;
            let mut tv = Transcript::new("mikros-spartan-test", &[]);
            assert!(
                verify_spartan(&cs, &params, &packed_io, &bad, &mut tv).is_err(),
                "a tampered terminal claim must reject"
            );
        }
        // tamper: the inner Fact-1's rounds.
        {
            let mut bad = clone_sp(&proof);
            bad.inner.rounds[0][0] += ONE;
            let mut tv = Transcript::new("mikros-spartan-test", &[]);
            assert!(
                verify_spartan(&cs, &params, &packed_io, &bad, &mut tv).is_err(),
                "a tampered inner round must reject"
            );
        }
        // tamper: the packed opening's coordinate claims.
        {
            let mut bad = clone_sp(&proof);
            bad.po.v_msg[0] += ONE;
            let mut tv = Transcript::new("mikros-spartan-test", &[]);
            assert!(
                verify_spartan(&cs, &params, &packed_io, &bad, &mut tv).is_err(),
                "a tampered packed opening must reject"
            );
        }
        // tamper: the witness claim value.
        {
            let mut bad = clone_sp(&proof);
            bad.w_claim.1 += ONE;
            let mut tv = Transcript::new("mikros-spartan-test", &[]);
            assert!(
                verify_spartan(&cs, &params, &packed_io, &bad, &mut tv).is_err(),
                "a tampered witness claim must reject"
            );
        }
        // tamper: an unsatisfying batch (the batch.rs discipline: the
        // R1CS residual check must reject somewhere in the chain).
        {
            let mut w2 = witnesses.clone();
            w2[1][2] = true; // breaks row 1: (x0+x3)·x2 = 1 ≠ 0
            assert!(!cs.satisfied_by(&w2[1]));
            let (setup2, packed_io2, bad, _t2) = prove_toy(&cs, &w2);
            let _ = setup2;
            let mut tv = Transcript::new("mikros-spartan-test", &[]);
            assert!(
                verify_spartan(&cs, &params, &packed_io2, &bad, &mut tv).is_err(),
                "an unsatisfying batch must reject"
            );
        }
        let _ = &setup;
    }

    fn clone_sp(p: &SpartanProof) -> SpartanProof {
        SpartanProof {
            io_claim: p.io_claim,
            fact1_io: clone_f1(&p.fact1_io),
            outer_inst: p.outer_inst.clone(),
            outer_row: p.outer_row.clone(),
            terminals: p.terminals,
            alphas: p.alphas,
            inner: clone_f1(&p.inner),
            t_term: p.t_term,
            w_claim: (p.w_claim.0.clone(), p.w_claim.1),
            po: PackedOpenProof {
                v_msg: p.po.v_msg.clone(),
                fact1: clone_f1(&p.po.fact1),
                w_pub: p.po.w_pub,
                pcs: clone_pcs(&p.po.pcs),
            },
            pcs_io: clone_pcs(&p.pcs_io),
            stmt: p.stmt.clone(),
        }
    }

    fn clone_pcs(p: &PcsProof) -> PcsProof {
        PcsProof {
            rounds: p.rounds.clone(),
            level_roots: p.level_roots.clone(),
            terminal_claim: p.terminal_claim,
            queries: p.queries.clone(),
            tail: p.tail.clone(),
        }
    }

    /// The dense E-arithmetic reference for the outer's instance phase and
    /// the terminal claims: the messages recomputed from a materialized
    /// joint-cube state (the standard folded sum-check), byte-compared
    /// against the proof's; the terminal claims compared against the direct
    /// joint-cube MLE evaluations. The Spartan inner identity is asserted
    /// brute-force: `⋆̂(j*, i*) = Σ_y Ŝ⋆(i*, y)·Ŵ(y, j*)`.
    #[test]
    fn spartan_dense_reference_and_inner_identity() {
        let (cs, witnesses) = toy_cs();
        let n = cs.s_l.n;
        let log_n = n.trailing_zeros() as usize;
        let params = PcsParams::default();
        let setup = batch::setup_batch(&cs, &witnesses, &params);
        let packed_io = batch::packed_io_words(&cs, &witnesses);
        let mut t = Transcript::new("mikros-spartan-test", &[]);
        let proof = prove_spartan(&cs, &setup, &params, &packed_io, &mut t);

        // ---- replay the verifier to recover (τ_j, τ_i, ρ_0..6, i*).
        let mut tv = Transcript::new("mikros-spartan-test", &[]);
        // skip the io bind stage (its challenges don't affect the outer's
        // reference): replay the io labels by driving the real verify path
        // far enough — simplest: recompute them here.
        // (the io Fact-1 consumes transcript state; re-derive:)
        let _ = &mut tv;
        // Run the full verifier to completion for the transcript state, and
        // separately track the outer challenges. We re-derive them by
        // mirroring the verify sampling order.
        let vres = verify_spartan(&cs, &params, &packed_io, &proof, &mut tv);
        assert!(vres.is_ok());
        // The challenges are not returned; re-derive by replaying labels in
        // order — the deterministic Fiat–Shamir chain from a fresh
        // transcript, consuming exactly the io stage first.
        let mut tr = Transcript::new("mikros-spartan-test", &[]);
        // io stage: the x_io challenges + the Fact-1 (absorbs internally).
        for _ in 0..log_n {
            tr.challenge_field("sp-io-x");
        }
        fact1::verify(proof.io_claim, &proof.fact1_io, &mut tr).unwrap();
        let tau_j: Vec<Gf128> = (0..7).map(|_| tr.challenge_field("sp-tauj")).collect();
        let tau_i: Vec<Gf128> = (0..log_n).map(|_| tr.challenge_field("sp-taui")).collect();
        let mut rho: Vec<Gf128> = Vec::new();
        for k in 0..7 {
            // absorb the proof's message exactly as the verifier does
            let mut buf = Vec::with_capacity(4 * 16);
            for &cf in &proof.outer_inst[k] {
                buf.extend_from_slice(&magnetar_hash::fe_to_bytes(cf));
            }
            tr.absorb_bytes("sp-inst", &buf);
            rho.push(tr.challenge_field("sp-inst"));
        }

        // ---- the dense reference state over the joint cube.
        // state_*[j_rem][i]: the images folded at the bound prefix.
        let to_e = |b: bool| if b { ONE } else { ZERO };
        let init = |k: usize| -> Vec<Vec<Gf128>> {
            (0..128)
                .map(|j| (0..n).map(|i| to_e(setup.sw[k].bit(j, i))).collect())
                .collect()
        };
        let mut st_l = init(0);
        let mut st_r = init(1);
        let mut st_o = init(2);
        // the images fold: after binding j_0..j_{k-1}, state[j] is the
        // eq(ρ_{<k})-weighted combination of the original planes.
        let fold_state = |st: &mut Vec<Vec<Gf128>>, k: usize, r: Gf128| {
            // st[j] covers the remaining instance cube; the j_k bit = the
            // LSB of the current index. new[j'] = (1+r)·st[2j'] + r·st[2j'+1].
            let half = st.len() / 2;
            let mut out = vec![Vec::new(); half];
            for jp in 0..half {
                let lo = &st[2 * jp];
                let hi = &st[2 * jp + 1];
                let v = lo
                    .iter()
                    .zip(hi.iter())
                    .map(|(&a, &b)| mul(ONE + r, a) + mul(r, b))
                    .collect();
                out[jp] = v;
            }
            *st = out;
            let _ = k;
        };
        let l = crate::planes::RowWeightTensor::new(eq_factors(&tau_i), true);
        let mut ref_running = ZERO;
        for k in 0..7 {
            let free_pt: Vec<Gf128> = (k + 1..7).map(|m| tau_j[m]).collect();
            let eqrem = tensor(&free_pt);
            let n_free = eqrem.len();
            let mut u0 = ZERO;
            let mut u1 = ZERO;
            let mut u2 = ZERO;
            for jp in 0..n_free {
                // the current state's halves at the j_k bit (its LSB):
                // planes 2·jp (x=0) and 2·jp+1 (x=1).
                let la = &st_l[2 * jp];
                let lb = &st_l[2 * jp + 1];
                let ra = &st_r[2 * jp];
                let rb = &st_r[2 * jp + 1];
                let oa = &st_o[2 * jp];
                let ob = &st_o[2 * jp + 1];
                let wj = eqrem[jp];
                let d00: Gf128 = (0..n).map(|i| mul(l.words[i], mul(la[i], ra[i]))).sum();
                let d01: Gf128 = (0..n).map(|i| mul(l.words[i], mul(la[i], rb[i]))).sum();
                let d10: Gf128 = (0..n).map(|i| mul(l.words[i], mul(lb[i], ra[i]))).sum();
                let d11: Gf128 = (0..n).map(|i| mul(l.words[i], mul(lb[i], rb[i]))).sum();
                let e0: Gf128 = (0..n).map(|i| mul(l.words[i], oa[i])).sum();
                let e01: Gf128 = (0..n).map(|i| mul(l.words[i], oa[i] + ob[i])).sum();
                u0 += mul(wj, d00 + e0);
                u1 += mul(wj, d01 + d10 + e01);
                u2 += mul(wj, d00 + d01 + d10 + d11);
            }
            let tau = tau_j[k];
            let want = [
                mul(ONE + tau, u0),
                mul(ONE + tau, u1) + u0,
                mul(ONE + tau, u2) + u1,
                u2,
            ];
            assert_eq!(
                proof.outer_inst[k], want,
                "instance round {k}: the lazy message must match the dense reference"
            );
            // the running-claim chain check
            let s123 = want[1] + want[2] + want[3];
            assert_eq!(s123, ref_running, "round {k} sum identity");
            let mut mr = ZERO;
            let mut xp = Gf128(1);
            for &cf in &want {
                mr += mul(cf, xp);
                xp = mul(xp, rho[k]);
            }
            ref_running = mr;
            fold_state(&mut st_l, k, rho[k]);
            fold_state(&mut st_r, k, rho[k]);
            fold_state(&mut st_o, k, rho[k]);
        }
        // sanity: the reference's running claim equals the verifier's row
        // claim source (the prover's `running` at the phase boundary is the
        // engine's claim; recovered here via the last round evaluation).
        // ---- the terminal claims vs the direct joint-cube MLE evals.
        // i*: the row-phase engine challenges in bit order — replay from
        // the verify transcript: they follow the row messages.
        // The row point: absorb each row message then sample — exactly
        // the verifier's sequence (the earlier draft sampled the
        // challenges WITHOUT the absorbs: a different transcript position
        // entirely — every downstream comparison was against the wrong
        // point while the protocol itself was correct).
        let i_star: Vec<Gf128> = {
            let mut pt = vec![ZERO; log_n];
            for (r, m) in proof.outer_row.iter().enumerate() {
                let mut buf = Vec::with_capacity(m.len() * 16);
                for &cf in m {
                    buf.extend_from_slice(&magnetar_hash::fe_to_bytes(cf));
                }
                tr.absorb_bytes("sp-row", &buf);
                pt[r] = tr.challenge_field("sp-row");
            }
            pt
        };
        let eqr_j = tensor(&rho);
        let eq_i = eq_build(&i_star);
        let mle = |st: &[Vec<Gf128>]| -> Gf128 {
            // Σ_{(j,i)} eq(j*, j)·eq(i*, i)·st[j][i] with st the folded
            // state (st has one row left = the image at j*).
            let mut acc = ZERO;
            for i in 0..n {
                if st[0][i] != ZERO {
                    acc += mul(eqr_j[0], mul(eq_i[i], st[0][i]));
                }
            }
            acc
        };
        // st_* after 7 folds: one plane left; eqr_j[0] = eq(j*,j*)?? — the
        // fold consumed all instance variables; the direct MLE needs the
        // FULL joint-cube sum, not the folded state. Use the brute joint
        // sum from the ORIGINAL images instead:
        let joint_mle = |k: usize| -> Gf128 {
            let mut acc = ZERO;
            for j in 0..128usize {
                for i in 0..n {
                    if setup.sw[k].bit(j, i) {
                        acc += mul(eqr_j[j], eq_i[i]);
                    }
                }
            }
            acc
        };
        // The fold the prover fed the engine must equal the reference's
        // folded state (the c-combination identity), and its MLE at i*
        // (the plain interleaved fold at the round challenges) must equal
        // the joint brute force — pinning the row phase's point order.
        {
            let my_l = fold_planes_at(&setup.sw[0], &eqr_j);
            assert_eq!(my_l, st_l[0], "the j*-fold equals the reference state");
            let mut tab = my_l.clone();
            for k in 0..log_n {
                let r = i_star[k];
                let half = tab.len() / 2;
                let mut nv = vec![ZERO; half];
                for j in 0..half {
                    nv[j] = mul(ONE + r, tab[2 * j]) + mul(r, tab[2 * j + 1]);
                }
                tab = nv;
            }
            assert_eq!(
                tab[0],
                joint_mle(0),
                "the manual fold at i* == the joint MLE"
            );
        }
        assert_eq!(proof.terminals[0], joint_mle(0), "v_L = L-hat(j*, i*)");
        assert_eq!(proof.terminals[1], joint_mle(1), "v_R = R-hat(j*, i*)");
        assert_eq!(proof.terminals[2], joint_mle(2), "v_O = O-hat(j*, i*)");
        // the unused folded-state helper
        let _ = mle;
        // ---- the Spartan inner identity, brute force:
        // v_L = Σ_y S-hat_L(i*, y)·W-hat(y, j*).
        let t_l = row_point_table(&cs.s_l, &eq_i);
        let t_r = row_point_table(&cs.s_r, &eq_i);
        let t_o = row_point_table(&cs.s_o, &eq_i);
        let w_alayout = transpose_ps(&setup.packed_t);
        let wf = fold_planes_at(&w_alayout, &eqr_j);
        let dot_at = |tab: &[Gf128]| -> Gf128 {
            let mut acc = ZERO;
            for y in 0..n {
                if tab[y] != ZERO && wf[y] != ZERO {
                    acc += mul(tab[y], wf[y]);
                }
            }
            acc
        };
        assert_eq!(
            proof.terminals[0],
            dot_at(&t_l),
            "v_L = <S-hat_L(i*), W-hat(j*)>"
        );
        assert_eq!(
            proof.terminals[1],
            dot_at(&t_r),
            "v_R = <S-hat_R(i*), W-hat(j*)>"
        );
        assert_eq!(
            proof.terminals[2],
            dot_at(&t_o),
            "v_O = <S-hat_O(i*), W-hat(j*)>"
        );
        // and the full witness claim: W-hat(y_in, j*) via the packed-open
        // claim form vs the direct per-column eq folds.
        // the α's from the replay transcript (the deterministic chain —
        // equal to the prover's), then the inner Fact-1.
        let alphas: [Gf128; 3] = std::array::from_fn(|_| tr.challenge_field("sp-alpha"));
        assert_eq!(alphas, proof.alphas, "the replayed α's must match");
        let v_comb = mul(alphas[0], proof.terminals[0])
            + mul(alphas[1], proof.terminals[1])
            + mul(alphas[2], proof.terminals[2]);
        let (y_in, _t_ret, w_ret) = fact1::verify(v_comb, &proof.inner, &mut tr).unwrap();
        let y_bits: Vec<Gf128> = (0..log_n).map(|r| y_in[log_n - 1 - r]).collect();
        let eq_y = eq_build(&y_bits);
        let mut want = ZERO;
        for y in 0..n {
            if wf[y] != ZERO {
                want += mul(eq_y[y], wf[y]);
            }
        }
        assert_eq!(w_ret, want, "W-hat(y_in, j*) = the packed-open claim form");
    }

    /// The cross-check against Algorithm 6: the same instances verify
    /// through both SNARKs (the shared statement oracle), and the
    /// unsatisfiable case rejects in both.
    #[test]
    fn spartan_cross_check_algorithm6() {
        use crate::gf8::Gf8;
        use crate::interp::InterpData;
        let f8 = Gf8::new();
        let mut data = InterpData::new(&f8);
        data.build_psi_monomials(&f8);
        let (cs, witnesses) = toy_cs();
        let params = PcsParams::default();
        let setup = batch::setup_batch(&cs, &witnesses, &params);
        let packed_io = batch::packed_io_words(&cs, &witnesses);

        // Algorithm 6
        let mut t6 = Transcript::new("mikros-batch-test", &[]);
        let p6 = batch::prove_batch(&f8, &data, &cs, &setup, &params, &packed_io, &mut t6);
        let mut tv6 = Transcript::new("mikros-batch-test", &[]);
        assert!(batch::verify_batch(&f8, &data, &cs, &params, &packed_io, &p6, &mut tv6).is_ok());

        // BinarySpartan on the same setup
        let mut tsp = Transcript::new("mikros-spartan-test", &[]);
        let psp = prove_spartan(&cs, &setup, &params, &packed_io, &mut tsp);
        let mut tvsp = Transcript::new("mikros-spartan-test", &[]);
        assert!(
            verify_spartan(&cs, &params, &packed_io, &psp, &mut tvsp).is_ok(),
            "both SNARKs must verify the same statement"
        );
        let _ = &proofs_marker(&p6, &psp);
    }

    fn proofs_marker(_a: &BatchProof, _b: &SpartanProof) {}

    #[test]
    fn engine_finish_vs_manual_fold() {
        use magnetar_field::Gf128;
        use magnetar_field::{mul, ONE};
        use magnetar_sumcheck::engine::SumcheckProver;
        use magnetar_sumcheck::{BindOrder, EqFactor, SumcheckSpec, Term};
        let mut seed = 0xABCDu128;
        let mut gf = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            Gf128(seed)
        };
        let log_n = 5usize;
        let n = 1 << log_n;
        let l: Vec<Gf128> = (0..n).map(|_| gf()).collect();
        let r: Vec<Gf128> = (0..n).map(|_| gf()).collect();
        let o: Vec<Gf128> = (0..n).map(|_| gf()).collect();
        let tau: Vec<Gf128> = (0..log_n).map(|_| gf()).collect();
        let eq_j = gf();
        let spec = SumcheckSpec {
            num_vars: log_n,
            dense: vec![l.clone(), r.clone(), o.clone()],
            public: vec![Box::new(EqFactor::new(tau.clone()))],
            terms: vec![
                Term {
                    coeff: eq_j,
                    dense: vec![0, 1],
                    public: vec![0],
                },
                Term {
                    coeff: eq_j,
                    dense: vec![2],
                    public: vec![0],
                },
            ],
            claim: Gf128(0),
        };
        let mut t = Transcript::new("engine-probe", &[]);
        let mut challs = Vec::new();
        let msgs = {
            let mut p = SumcheckProver::new(&spec, BindOrder::Increasing);
            let mut ms = Vec::new();
            for _ in 0..log_n {
                let msg = p.message();
                let mut buf = Vec::new();
                for &cf in &msg.0 {
                    buf.extend_from_slice(&magnetar_hash::fe_to_bytes(cf));
                }
                t.absorb_bytes("probe", &buf);
                let rr = t.challenge_field("probe");
                p.fold_msg(&magnetar_sumcheck::RoundMsg(msg.0.clone()), rr);
                challs.push(rr);
                ms.push(msg.0.clone());
            }
            ms
        };
        let _ = msgs;
        let fin = {
            let mut p = SumcheckProver::new(
                &SumcheckSpec {
                    num_vars: log_n,
                    dense: vec![l.clone(), r.clone(), o.clone()],
                    public: vec![Box::new(EqFactor::new(tau.clone()))],
                    terms: vec![
                        Term {
                            coeff: eq_j,
                            dense: vec![0, 1],
                            public: vec![0],
                        },
                        Term {
                            coeff: eq_j,
                            dense: vec![2],
                            public: vec![0],
                        },
                    ],
                    claim: Gf128(0),
                },
                BindOrder::Increasing,
            );
            for k in 0..log_n {
                let msg = p.message();
                p.fold_msg(&magnetar_sumcheck::RoundMsg(msg.0.clone()), challs[k]);
            }
            p.finish()
        };
        let (fclaims, frunning): (Vec<Gf128>, Gf128) = fin;
        // the manual interleaved folds
        let mf = |tab: &[Gf128]| -> Gf128 {
            let mut tb = tab.to_vec();
            for k in 0..log_n {
                let rr = challs[k];
                let half = tb.len() / 2;
                let mut nv = vec![Gf128(0); half];
                for j in 0..half {
                    nv[j] = mul(ONE + rr, tb[2 * j]) + mul(rr, tb[2 * j + 1]);
                }
                tb = nv;
            }
            tb[0]
        };
        println!(
            "probe: claims=({:x},{:x},{:x}) manual=({:x},{:x},{:x})",
            fclaims[0].0,
            fclaims[1].0,
            fclaims[2].0,
            mf(&l).0,
            mf(&r).0,
            mf(&o).0
        );
        assert_eq!(fclaims[0], mf(&l), "engine claim 0 vs manual fold");
        assert_eq!(fclaims[1], mf(&r), "engine claim 1 vs manual fold");
        assert_eq!(fclaims[2], mf(&o), "engine claim 2 vs manual fold");
        let _ = frunning;
    }

    /// The SHA-256-scale differential: the same dense-reference machinery as
    /// the toy test, at N = 2^17 — found the round-1 divergence the toy could
    /// not see (whatever its cause turns out to be).
    #[test]
    #[ignore] // heavy: ~1-2 min, the release container
    fn spartan_sha256_dense_differential() {
        use crate::circuits::sha256_compression;
        let mut b = crate::circuits::CircuitBuilder::new();
        let state: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let block: [u32; 16] = core::array::from_fn(|i| 0x100 + i as u32);
        sha256_compression(&mut b, state, block);
        let cs = b.build();
        let n = cs.s_l.n;
        let log_n = n.trailing_zeros() as usize;
        let witnesses: Vec<Vec<bool>> = (0..crate::interp::KAPPA)
            .map(|j| {
                let mut rb = crate::circuits::CircuitBuilder::replay();
                let blk: [u32; 16] = core::array::from_fn(|i| 0x100 + i as u32 + j as u32);
                sha256_compression(&mut rb, state, blk);
                rb.w
            })
            .collect();
        for w in &witnesses {
            assert!(cs.satisfied_by(w));
        }
        let params = PcsParams::default();
        let setup = batch::setup_batch(&cs, &witnesses, &params);
        let packed_io = batch::packed_io_words(&cs, &witnesses);
        let mut t = Transcript::new("sp-sha-diff", &[]);
        let proof = prove_spartan(&cs, &setup, &params, &packed_io, &mut t);

        // the replay
        let mut tr = Transcript::new("sp-sha-diff", &[]);
        for _ in 0..log_n {
            tr.challenge_field("sp-io-x");
        }
        fact1::verify(proof.io_claim, &proof.fact1_io, &mut tr).unwrap();
        let tau_j: Vec<Gf128> = (0..7).map(|_| tr.challenge_field("sp-tauj")).collect();
        let tau_i: Vec<Gf128> = (0..log_n).map(|_| tr.challenge_field("sp-taui")).collect();
        let mut rho: Vec<Gf128> = Vec::new();
        for k in 0..7 {
            let mut buf = Vec::with_capacity(4 * 16);
            for &cf in &proof.outer_inst[k] {
                buf.extend_from_slice(&magnetar_hash::fe_to_bytes(cf));
            }
            tr.absorb_bytes("sp-inst", &buf);
            rho.push(tr.challenge_field("sp-inst"));
        }

        // the dense reference state
        let to_e = |x: bool| if x { ONE } else { ZERO };
        let init = |k: usize| -> Vec<Vec<Gf128>> {
            (0..128)
                .map(|j| (0..n).map(|i| to_e(setup.sw[k].bit(j, i))).collect())
                .collect()
        };
        let mut st_l = init(0);
        let mut st_r = init(1);
        let mut st_o = init(2);
        let l = crate::planes::RowWeightTensor::new(eq_factors(&tau_i), true);
        let mut ref_running = ZERO;
        let mut eq_accum = ONE;
        for k in 0..7 {
            let free_pt: Vec<Gf128> = (k + 1..7).map(|m| tau_j[m]).collect();
            let eqrem = tensor(&free_pt);
            let n_free = eqrem.len();
            let mut u0 = ZERO;
            let mut u1 = ZERO;
            let mut u2 = ZERO;
            for jp in 0..n_free {
                let la = &st_l[2 * jp];
                let lb = &st_l[2 * jp + 1];
                let ra = &st_r[2 * jp];
                let rb = &st_r[2 * jp + 1];
                let oa = &st_o[2 * jp];
                let ob = &st_o[2 * jp + 1];
                let wj = eqrem[jp];
                let d00: Gf128 = (0..n).map(|i| mul(l.words[i], mul(la[i], ra[i]))).sum();
                let d01: Gf128 = (0..n).map(|i| mul(l.words[i], mul(la[i], rb[i]))).sum();
                let d10: Gf128 = (0..n).map(|i| mul(l.words[i], mul(lb[i], ra[i]))).sum();
                let d11: Gf128 = (0..n).map(|i| mul(l.words[i], mul(lb[i], rb[i]))).sum();
                let e0: Gf128 = (0..n).map(|i| mul(l.words[i], oa[i])).sum();
                let e01: Gf128 = (0..n).map(|i| mul(l.words[i], oa[i] + ob[i])).sum();
                u0 += mul(wj, d00 + e0);
                u1 += mul(wj, d01 + d10 + e01);
                u2 += mul(wj, d00 + d01 + d10 + d11);
            }
            let tau = tau_j[k];
            let want = [
                mul(eq_accum, mul(ONE + tau, u0)),
                mul(eq_accum, mul(ONE + tau, u1) + u0),
                mul(eq_accum, mul(ONE + tau, u2) + u1),
                mul(eq_accum, u2),
            ];
            assert_eq!(proof.outer_inst[k], want, "SHA round {k}");
            let s123 = want[1] + want[2] + want[3];
            assert_eq!(s123, ref_running, "SHA round {k} sum");
            let mut mr = ZERO;
            let mut xp = Gf128(1);
            for &cf in &want {
                mr += mul(cf, xp);
                xp = mul(xp, rho[k]);
            }
            ref_running = mr;
            eq_accum = mul(eq_accum, ONE + tau_j[k] + rho[k]);
            // fold
            let fold1 = |st: &mut Vec<Vec<Gf128>>, r: Gf128| {
                let half = st.len() / 2;
                let mut out = vec![Vec::new(); half];
                for jp in 0..half {
                    let lo = &st[2 * jp];
                    let hi = &st[2 * jp + 1];
                    let v: Vec<Gf128> = lo
                        .iter()
                        .zip(hi.iter())
                        .map(|(&a, &b)| mul(ONE + r, a) + mul(r, b))
                        .collect();
                    out[jp] = v;
                }
                *st = out;
            };
            fold1(&mut st_l, rho[k]);
            fold1(&mut st_r, rho[k]);
            fold1(&mut st_o, rho[k]);
            // the true remaining sum from the folded state — the chain oracle
            let nfree = st_l.len();
            let mut free_pt: Vec<Gf128> = (k + 1..7).map(|m| tau_j[m]).collect();
            let _ = &mut free_pt;
            let eqr = tensor(&free_pt);
            let mut true_rem = ZERO;
            for jp in 0..n_free {
                let w = eqr.get(jp).copied().unwrap_or(ONE);
                let mut jp_sum = ZERO;
                for i in 0..n {
                    let lv = st_l[jp][i];
                    let rv = st_r[jp][i];
                    let ov = st_o[jp][i];
                    if lv != ZERO || rv != ZERO || ov != ZERO {
                        jp_sum += mul(l.words[i], mul(lv, rv) + ov);
                    }
                }
                true_rem += mul(w, jp_sum);
            }
            // The true remaining sum (the chain oracle): the folded state's
            // eqr-weighted product sum times the accumulated eq factor must
            // equal the running claim m_k(ρ_k) — this is the invariant that
            // caught both the missing-eq-factor and the c-ordering bugs.
            {
                let tr = mul(eq_accum, true_rem);
                assert_eq!(
                    tr, ref_running,
                    "round {k}: the state's true remaining sum vs the chain"
                );
            }
        }
        let _ = ref_running;
        println!("SHA differential: all 7 instance rounds match the dense reference");
    }
}
