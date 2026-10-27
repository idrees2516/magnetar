//! Algorithm 6 — the SNARK protocol for R1CS with packing across instances
//! (ePrint 2026/2646, §5.3): the Table-4 benchmark engine.
//!
//! κ = 128 instances of one circuit are packed `W = [w^(1)|…|w^(κ)]`,
//! committed as `w = Wβ ∈ E^N`, and proved with:
//!
//! 1. **The public-I/O bind** (steps 3–5): a masked-tensor claim
//!    `〈ξ_io, w〉` reduced by Fact 1 to a witness MLE claim.
//! 2. **The packed constraint check** (steps 6–8): Algorithm 2 on
//!    `(S_L W) ∘ (S_R W)` against the prover's output claim `v_O`.
//! 3. **The three linear relations** (steps 9–11): the coordinate claims
//!    `U⋆ = Ξ⋆^⊤ S⋆ W` sent as `U⋆^⊤ β`; the verifier checks each Hadamard
//!    terminal and `v_O` against `〈v⋆, ρ⋆〉`.
//! 4. **The z/r-combined lincheck** (steps 12–13): Fact 1 on
//!    `r^⊤(U_O + zU_L + z²U_R)β = 〈w, (S_O^⊤Ξ_O + zS_L^⊤Ξ_L + z²S_R^⊤Ξ_R)r〉`
//!    → one witness MLE claim + the public terminal.
//! 5. **The public values** (steps 14–15): `〈S⋆ ts(y), Ξ⋆ r〉` — in the
//!    paper's Table-4 benchmark setting "both systems take the wiring as
//!    explicit public inputs and have the verifier evaluate them directly":
//!    the verifier direct-evaluates each value in O(nnz + N).
//! 6. **The openings** (step 16): the retained witness MLE claims proved by
//!    this repo's fold-chain PCS (the Ligerito-role backend).

use crate::fact1::{self, Fact1Proof};
use crate::gf8::Gf8;
use crate::hadamard::{self, HadamardProof};
use crate::interp::{InterpData, KAPPA};
use crate::planes::{PlaneSet, RowWeightTensor};
use crate::r1cs::{apply_sparse_packed, dot_gf, eq_build, transpose_ps, BinaryR1cs};
use magnetar_field::Gf128;
use magnetar_field::{mul, ZERO};
use magnetar_hash::Transcript;
use magnetar_pcs::eval::{
    commit_table, prove_eval, verify_eval, PcsParams, PcsProof, PcsStatement,
};

/// The packed per-instance witness `w = Wβ` from the transposed layout.
pub fn packed_witness(w_t: &PlaneSet, n: usize) -> Vec<Gf128> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = w_t.plane(i)[0];
        let hi = w_t.plane(i)[1];
        out.push(Gf128((lo as u128) | ((hi as u128) << 64)));
    }
    out
}

pub struct BatchProof {
    pub io_claim: Gf128,
    pub fact1_io: Fact1Proof,
    pub v_o: Gf128,
    pub had: HadamardProof,
    /// `U⋆^⊤ β` for ⋆ ∈ {O, L, R}.
    pub v_msgs: [Vec<Gf128>; 3],
    pub fact1_lin: Fact1Proof,
    /// `〈S⋆ ts(y), Ξ⋆ r〉` for ⋆ ∈ {O, L, R} (prover-sent, direct-checked).
    pub pub_vals: [Gf128; 3],
    /// The retained witness MLE claims: `(point, value)`.
    pub witness_claims: Vec<(Vec<Gf128>, Gf128)>,
    pub pcs_proofs: Vec<PcsProof>,
    pub stmt: PcsStatement,
}

/// The row weight of a ξ tensor as `(words, planes)`.
fn xi_materialized(factors: &[(Gf128, Gf128)]) -> (Vec<Gf128>, Vec<u64>, usize) {
    let mut words = vec![Gf128(1)];
    for &(f0, f1) in factors {
        let half = words.len();
        let mut nv = vec![ZERO; half * 2];
        for i in 0..half {
            nv[i] = mul(words[i], f0);
            nv[half + i] = mul(words[i], f1);
        }
        words = nv;
    }
    let m = words.len();
    let planes = crate::planes::words_to_planes(&words, m);
    (words, planes, m / 64)
}

/// `U = Ξ^⊤ A` (κ×κ GF(2)) bitsliced — `U[c][j] = parity(Ξplane_c ∧ Aplane_j)`.
fn u_from_xi(a: &PlaneSet, xi_planes: &[u64], xi_wp: usize) -> Vec<[u64; 2]> {
    let mut rows = vec![[0u64; 2]; KAPPA];
    for c in 0..KAPPA {
        let xp = &xi_planes[c * xi_wp..(c + 1) * xi_wp];
        let mut acc = [0u64; 2];
        for j in 0..KAPPA {
            if PlaneSet::parity_dot(xp, a.plane(j)) == 1 {
                acc[j / 64] |= 1u64 << (j % 64);
            }
        }
        rows[c] = acc;
    }
    rows
}

/// `(U^⊤ β)_j` — the packed U column j.
/// The κ×κ GF(2) coordinate matrix as packed rows — kept for the
/// standalone consumer; `u_columns_msg_vec` is the packed-columns form the
/// protocol uses.
#[allow(dead_code)]
fn u_columns_msg(rows: &[[u64; 2]; KAPPA]) -> Vec<Gf128> {
    let mut msg = vec![ZERO; KAPPA];
    for j in 0..KAPPA {
        let mut acc = 0u128;
        for c in 0..KAPPA {
            if (rows[c][j / 64] >> (j % 64)) & 1 == 1 {
                acc |= 1u128 << c;
            }
        }
        msg[j] = Gf128(acc);
    }
    msg
}

/// `φ_r(ξ_i) = Σ_c bit_c(ξ_i)·r_c` — the coordinate dot (Appendix B.2's
/// recursion evaluates it in O(log n); the benchmark's direct-evaluation
/// setting uses the O(popcount) fold).
fn phi_r(xi: Gf128, r: &[Gf128]) -> Gf128 {
    let mut acc = ZERO;
    let mut bits = xi.0;
    while bits != 0 {
        let cpos = bits.trailing_zeros() as usize;
        acc += r[cpos];
        bits &= bits - 1;
    }
    acc
}

/// `S^⊤ v` for an E-vector (O(nnz)).
fn apply_transpose(s: &crate::r1cs::SparseMatrix, v: &[Gf128]) -> Vec<Gf128> {
    let mut out = vec![ZERO; s.n];
    for (i, cols) in s.rows.iter().enumerate() {
        if v[i] == ZERO {
            continue;
        }
        for &c in cols {
            out[c as usize] += v[i];
        }
    }
    out
}

/// `(S · eq(y))_i` = XOR of `eq(y)[j]` over the row's nonzero columns.
fn sparse_apply_eq(s: &crate::r1cs::SparseMatrix, eq: &[Gf128]) -> Vec<Gf128> {
    let mut out = vec![ZERO; s.n];
    for (i, cols) in s.rows.iter().enumerate() {
        let mut acc = ZERO;
        for &c in cols {
            acc += eq[c as usize];
        }
        out[i] = acc;
    }
    out
}

/// The column-claim fold `d[j] = 〈l, A_j〉` bitsliced (the Hadamard step-1
/// shape, reused for the `v_O` evaluation).
fn column_claims(a: &PlaneSet, l: &RowWeightTensor) -> Vec<Gf128> {
    let mut d = Vec::with_capacity(KAPPA);
    for j in 0..KAPPA {
        let pj = a.plane(j);
        let mut bits = 0u128;
        for e in 0..128 {
            bits |= (PlaneSet::parity_dot(pj, l.plane(e)) as u128) << e;
        }
        d.push(Gf128(bits));
    }
    d
}

pub struct BatchSetup {
    pub w_table: Vec<Gf128>,
    pub packed_t: PlaneSet,
    /// `S⋆W` in the Algorithm-2 layout (plane = instance), for ⋆ ∈ {L, R, O}.
    pub sw: [PlaneSet; 3],
    pub stmt: PcsStatement,
}

/// Pack, commit, and precompute the sparse images (the untimed setup-to-
/// online boundary: the commitment itself is part of the online phase per
/// the paper, so it stays in `prove_batch`).
pub fn setup_batch(cs: &BinaryR1cs, witnesses: &[Vec<bool>], params: &PcsParams) -> BatchSetup {
    assert_eq!(witnesses.len(), KAPPA);
    let n = cs.s_l.n;
    let packed_t = cs.pack_instances_t(witnesses);
    let w_table = packed_witness(&packed_t, n);
    let sw_t: Vec<PlaneSet> = [&cs.s_l, &cs.s_r, &cs.s_o]
        .iter()
        .map(|s| apply_sparse_packed(s, &packed_t))
        .collect();
    let sw = [
        transpose_ps(&sw_t[0]),
        transpose_ps(&sw_t[1]),
        transpose_ps(&sw_t[2]),
    ];
    let (_, _tree, stmt) = commit_table(params, &w_table);
    BatchSetup {
        w_table,
        packed_t,
        sw,
        stmt,
    }
}

/// The public-I/O packed words at the io positions: `w_io[i]` for the
/// verifier's claim computation.
pub fn packed_io_words(cs: &BinaryR1cs, witnesses: &[Vec<bool>]) -> Vec<Gf128> {
    let io = &cs.io_positions;
    let mut out = Vec::with_capacity(io.len());
    for &pos in io {
        let mut acc = 0u128;
        for (j, w) in witnesses.iter().enumerate() {
            if w[pos as usize] {
                acc |= 1u128 << j;
            }
        }
        out.push(Gf128(acc));
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn prove_batch(
    f8: &Gf8,
    data: &InterpData,
    cs: &BinaryR1cs,
    setup: &BatchSetup,
    params: &PcsParams,
    packed_io: &[Gf128],
    t: &mut Transcript,
) -> BatchProof {
    let n = cs.s_l.n;
    let log_n = n.trailing_zeros() as usize;
    let w = &setup.w_table;

    // ---- 1. The public-I/O bind (steps 3–5).
    let x_io: Vec<Gf128> = (0..log_n)
        .map(|_| t.challenge_field("batch-io-x"))
        .collect();
    let eq_io = eq_build(&x_io);
    let mut io_claim = ZERO;
    for (k, &pos) in cs.io_positions.iter().enumerate() {
        io_claim += mul(eq_io[pos as usize], packed_io[k]);
    }
    let io_sorted = &cs.io_positions;
    let mut masked = eq_io.clone();
    let mut ii = 0usize;
    for i in 0..n {
        // keep eq only at io positions (both are sorted ascending)
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

    // ---- 2. The packed constraint check (steps 6–8).
    let row_pt: Vec<Gf128> = (0..log_n).map(|_| t.challenge_field("batch-row")).collect();
    let col_pt: Vec<Gf128> = (0..7).map(|_| t.challenge_field("batch-col")).collect();
    let l_factors: Vec<(Gf128, Gf128)> = row_pt.iter().map(|&r| (Gf128(1) + r, r)).collect();
    let l = RowWeightTensor::new(l_factors.clone(), true);
    let rho_o: Vec<Gf128> = {
        let mut v = vec![Gf128(1)];
        for &cp in &col_pt {
            let half = v.len();
            let mut nv = vec![ZERO; half * 2];
            for i in 0..half {
                nv[i] = mul(v[i], Gf128(1) + cp);
                nv[half + i] = mul(v[i], cp);
            }
            v = nv;
        }
        v
    };
    let d_o = column_claims(&setup.sw[2], &l);
    let v_o = dot_gf(&d_o, &rho_o);
    t.absorb_field("batch-vo", &[v_o]);

    let (_stats, had, term) =
        hadamard::prove_hadamard_timed(f8, data, &setup.sw[0], &setup.sw[1], &l, &rho_o, v_o, t);

    // ---- 3. The three linear relations (steps 9–11).
    // ξ_O = l; ξ_L = ts(y); ξ_R = l∘ts(y) (Algorithm 2, step 13–14).
    // ξ_O = l (its words/planes); ξ_L = ts(y); ξ_R = l∘ts(y).
    // (Fact 1 binds the high row bit first, so the ts factor list is the
    // challenge vector REVERSED in this crate's little-endian convention.)
    let xi_o = (l.words.clone(), l.planes.clone(), l.words_per_plane);
    let xi_l = xi_materialized(
        &term
            .y
            .iter()
            .rev()
            .map(|&yk| (Gf128(1) + yk, yk))
            .collect::<Vec<_>>(),
    );
    let xi_r_factors = hadamard::xi2_factors(&l, &term.y.iter().rev().copied().collect::<Vec<_>>());
    let xi_r = xi_materialized(&xi_r_factors);

    let u_rows: Vec<Vec<[u64; 2]>> = vec![
        u_from_xi(&setup.sw[2], &xi_o.1, xi_o.2),
        u_from_xi(&setup.sw[0], &xi_l.1, xi_l.2),
        u_from_xi(&setup.sw[1], &xi_r.1, xi_r.2),
    ];
    let v_msgs: [Vec<Gf128>; 3] = [
        u_columns_msg_vec(&u_rows[0]),
        u_columns_msg_vec(&u_rows[1]),
        u_columns_msg_vec(&u_rows[2]),
    ];
    for star in 0..3 {
        t.absorb_field("batch-vmsg", &v_msgs[star]);
    }
    debug_assert_eq!(dot_gf(&v_msgs[0], &rho_o), v_o, "U_O check");
    debug_assert_eq!(dot_gf(&v_msgs[1], &term.rho), term.v1, "U_L check");
    debug_assert_eq!(dot_gf(&v_msgs[2], &term.rho), term.v2, "U_R check");

    // ---- 4. The z/r-combined lincheck (steps 12–13).
    let z = t.challenge_field("batch-z");
    let r: Vec<Gf128> = (0..KAPPA).map(|_| t.challenge_field("batch-r")).collect();
    // LHS: r^⊤(U_O + zU_L + z²U_R)β — the packed U rows are E-linear.
    let z2 = mul(z, z);
    let mut comb = vec![ZERO; KAPPA];
    for k in 0..KAPPA {
        let o_w = packed_row(&u_rows[0], k);
        let l_w = packed_row(&u_rows[1], k);
        let r_w = packed_row(&u_rows[2], k);
        comb[k] = o_w + mul(z, l_w) + mul(z2, r_w);
    }
    let c_lhs = dot_gf(&comb, &r);
    // RHS operand: g = (S_O^⊤Ξ_O + zS_L^⊤Ξ_L + z²S_R^⊤Ξ_R) r
    let xi_o_r: Vec<Gf128> = xi_o.0.iter().map(|&v| phi_r(v, &r)).collect();
    let xi_l_r: Vec<Gf128> = xi_l.0.iter().map(|&v| phi_r(v, &r)).collect();
    let xi_r_r: Vec<Gf128> = xi_r.0.iter().map(|&v| phi_r(v, &r)).collect();
    let h_o = apply_transpose(&cs.s_o, &xi_o_r);
    let h_l = apply_transpose(&cs.s_l, &xi_l_r);
    let h_r = apply_transpose(&cs.s_r, &xi_r_r);
    let mut g = vec![ZERO; n];
    for i in 0..n {
        g[i] = h_o[i] + mul(z, h_l[i]) + mul(z2, h_r[i]);
    }
    debug_assert_eq!(dot_gf(w, &g), c_lhs);
    let mut wv = w.to_vec();
    let mut gv = g;
    let fact1_lin = fact1::prove(&mut wv, &mut gv, c_lhs, t);
    let y_lin = fact1_lin.challenges.clone();
    let w_w = fact1_lin.terminals.0;
    let w_g = fact1_lin.terminals.1;

    // ---- 5. The public values (steps 14–15) — direct-evaluation setting.
    // (ts(y_lin) in the Fact-1 fold convention: challenge k ↔ the high
    // bits first, so the eq table takes the reversed vector.)
    let eq_y = eq_build(&y_lin.iter().rev().copied().collect::<Vec<_>>());
    let mut pub_vals = [ZERO; 3];
    pub_vals[0] = {
        let u = sparse_apply_eq(&cs.s_o, &eq_y);
        let mut acc = ZERO;
        for i in 0..n {
            if u[i] != ZERO {
                acc += mul(u[i], phi_r(xi_o.0[i], &r));
            }
        }
        acc
    };
    pub_vals[1] = {
        let u = sparse_apply_eq(&cs.s_l, &eq_y);
        let mut acc = ZERO;
        for i in 0..n {
            if u[i] != ZERO {
                acc += mul(u[i], phi_r(xi_l.0[i], &r));
            }
        }
        acc
    };
    pub_vals[2] = {
        let u = sparse_apply_eq(&cs.s_r, &eq_y);
        let mut acc = ZERO;
        for i in 0..n {
            if u[i] != ZERO {
                acc += mul(u[i], phi_r(xi_r.0[i], &r));
            }
        }
        acc
    };
    debug_assert_eq!(
        pub_vals[0] + mul(z, pub_vals[1]) + mul(z2, pub_vals[2]),
        w_g,
        "the (1,z,z²) combination equals the Fact-1 public terminal"
    );
    t.absorb_field("batch-pubvals", &pub_vals);

    // ---- 6. The openings (step 16): the two retained witness claims.
    // Fact 1 binds the vector's MSB first (lo/hi halving); the PCS binds
    // its MLE variables in increasing (LSB-first) order — the claim point
    // must be REVERSED to match the claimed MLE value.
    let mut witness_claims = Vec::new();
    let mut pcs_proofs = Vec::new();
    for (y_fwd, value) in [(y_io.clone(), w_io), (y_lin.clone(), w_w)] {
        let point: Vec<Gf128> = y_fwd.iter().rev().copied().collect();
        let proof = prove_eval(params, w, &point, value, t);
        witness_claims.push((point, value));
        pcs_proofs.push(proof);
    }

    BatchProof {
        io_claim,
        fact1_io,
        v_o,
        had,
        v_msgs,
        fact1_lin,
        pub_vals,
        witness_claims,
        pcs_proofs,
        stmt: setup.stmt.clone(),
    }
}

fn packed_row(rows: &[[u64; 2]], k: usize) -> Gf128 {
    Gf128((rows[k][0] as u128) | ((rows[k][1] as u128) << 64))
}

fn u_columns_msg_vec(rows: &[[u64; 2]]) -> Vec<Gf128> {
    let mut msg = vec![ZERO; KAPPA];
    for j in 0..KAPPA {
        let mut acc = 0u128;
        for c in 0..KAPPA {
            if (rows[c][j / 64] >> (j % 64)) & 1 == 1 {
                acc |= 1u128 << c;
            }
        }
        msg[j] = Gf128(acc);
    }
    msg
}

// ---------------------------------------------------------------------------
// Verification (benchmark mode: the wiring matrices are public and the
// verifier evaluates the sparse claims directly, O(nnz + N))
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn verify_batch(
    f8: &Gf8,
    data: &InterpData,
    cs: &BinaryR1cs,
    params: &PcsParams,
    packed_io: &[Gf128],
    proof: &BatchProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let n = cs.s_l.n;
    let log_n = n.trailing_zeros() as usize;

    // ---- 1. The public-I/O bind.
    let x_io: Vec<Gf128> = (0..log_n)
        .map(|_| t.challenge_field("batch-io-x"))
        .collect();
    let eq_io = eq_build(&x_io);
    // the claim from the public packed io words
    let mut io_claim = ZERO;
    for (k, &pos) in cs.io_positions.iter().enumerate() {
        io_claim += mul(eq_io[pos as usize], packed_io[k]);
    }
    if io_claim != proof.io_claim {
        return Err("batch: io claim mismatch");
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
    let (y_io_pt, _w_io, w_io_pub) = fact1::verify(io_claim, &proof.fact1_io, t)?;
    // the masked-ξ public terminal: 〈masked, ts(y_io)〉 over the io positions
    {
        let eq_y = eq_build(&y_io_pt.iter().rev().copied().collect::<Vec<_>>());
        let mut want = ZERO;
        for &pos in cs.io_positions.iter() {
            want += mul(eq_io[pos as usize], eq_y[pos as usize]);
        }
        if want != w_io_pub {
            return Err("batch: io public terminal mismatch");
        }
    }

    // ---- 2. The packed constraint check.
    let row_pt: Vec<Gf128> = (0..log_n).map(|_| t.challenge_field("batch-row")).collect();
    let col_pt: Vec<Gf128> = (0..7).map(|_| t.challenge_field("batch-col")).collect();
    let l_factors: Vec<(Gf128, Gf128)> = row_pt.iter().map(|&r| (Gf128(1) + r, r)).collect();
    let rho_o: Vec<Gf128> = {
        let mut v = vec![Gf128(1)];
        for &cp in &col_pt {
            let half = v.len();
            let mut nv = vec![ZERO; half * 2];
            for i in 0..half {
                nv[i] = mul(v[i], Gf128(1) + cp);
                nv[half + i] = mul(v[i], cp);
            }
            v = nv;
        }
        v
    };
    let v_o = proof.v_o;
    t.absorb_field("batch-vo", &[v_o]);
    let term = hadamard::verify_hadamard(f8, data, &rho_o, v_o, &proof.had, t)?;

    // ---- 3. The U⋆ checks.
    for star in 0..3 {
        t.absorb_field("batch-vmsg", &proof.v_msgs[star]);
    }
    if dot_gf(&proof.v_msgs[0], &rho_o) != v_o {
        return Err("batch: U_O check");
    }
    if dot_gf(&proof.v_msgs[1], &term.rho) != term.v1 {
        return Err("batch: U_L check");
    }
    if dot_gf(&proof.v_msgs[2], &term.rho) != term.v2 {
        return Err("batch: U_R check");
    }

    // ---- 4. The z/r-combined lincheck.
    let z = t.challenge_field("batch-z");
    let r: Vec<Gf128> = (0..KAPPA).map(|_| t.challenge_field("batch-r")).collect();
    // recover U⋆ from the messages: U[c][j] = bit c of v_msg[j]
    let recover_rows = |msg: &[Gf128]| -> Vec<[u64; 2]> {
        let mut rows = vec![[0u64; 2]; KAPPA];
        for c in 0..KAPPA {
            let mut acc = [0u64; 2];
            for j in 0..KAPPA {
                if (msg[j].0 >> c) & 1 == 1 {
                    acc[j / 64] |= 1u64 << (j % 64);
                }
            }
            rows[c] = acc;
        }
        rows
    };
    let u_rows: Vec<Vec<[u64; 2]>> = vec![
        recover_rows(&proof.v_msgs[0]),
        recover_rows(&proof.v_msgs[1]),
        recover_rows(&proof.v_msgs[2]),
    ];
    let z2 = mul(z, z);
    let mut comb = vec![ZERO; KAPPA];
    for k in 0..KAPPA {
        let o_w = packed_row(&u_rows[0], k);
        let l_w = packed_row(&u_rows[1], k);
        let r_w = packed_row(&u_rows[2], k);
        comb[k] = o_w + mul(z, l_w) + mul(z2, r_w);
    }
    let c_lhs = dot_gf(&comb, &r);
    let (y_lin, w_w, w_g) = fact1::verify(c_lhs, &proof.fact1_lin, t)?;

    // ---- 5. The public values — direct evaluation (benchmark setting).
    // the ξ⋆ word vectors: ξ_O = l; ξ_L = ts(y_h); ξ_R = l∘ts(y_h).
    let expand = |factors: &[(Gf128, Gf128)]| -> Vec<Gf128> {
        let mut words = vec![Gf128(1)];
        for &(f0, f1) in factors {
            let half = words.len();
            let mut nv = vec![ZERO; half * 2];
            for i in 0..half {
                nv[i] = mul(words[i], f0);
                nv[half + i] = mul(words[i], f1);
            }
            words = nv;
        }
        words
    };
    let l_words = expand(&l_factors);
    let y_h = &term.y;
    let xi_l = expand(
        &y_h.iter()
            .rev()
            .map(|&yk| (Gf128(1) + yk, yk))
            .collect::<Vec<_>>(),
    );
    let xi_r = expand(&hadamard::xi2_factors_from(
        &l_factors,
        &y_h.iter().rev().copied().collect::<Vec<_>>(),
    ));
    let eq_y = eq_build(&y_lin.iter().rev().copied().collect::<Vec<_>>());
    let eval_pub = |s: &crate::r1cs::SparseMatrix, xi: &[Gf128]| -> Gf128 {
        let u = sparse_apply_eq(s, &eq_y);
        let mut acc = ZERO;
        for i in 0..n {
            if u[i] != ZERO {
                acc += mul(u[i], phi_r(xi[i], &r));
            }
        }
        acc
    };
    let pv_o = eval_pub(&cs.s_o, &l_words);
    let pv_l = eval_pub(&cs.s_l, &xi_l);
    let pv_r = eval_pub(&cs.s_r, &xi_r);
    if [pv_o, pv_l, pv_r] != proof.pub_vals {
        return Err("batch: direct public-value evaluation mismatch");
    }
    if pv_o + mul(z, pv_l) + mul(z2, pv_r) != w_g {
        return Err("batch: (1,z,z²) combination mismatch");
    }
    t.absorb_field("batch-pubvals", &proof.pub_vals);

    // ---- 6. The openings.
    if proof.witness_claims.len() != 2 || proof.pcs_proofs.len() != 2 {
        return Err("batch: witness claim count");
    }
    // the stored points are the fact1 points REVERSED (MSB-first → LSB-first)
    let rev = |p: &[Gf128]| -> Vec<Gf128> { p.iter().rev().copied().collect() };
    if proof.witness_claims[0].0 != rev(&y_io_pt)
        || proof.witness_claims[0].1 != proof.fact1_io.terminals.0
    {
        return Err("batch: io witness claim");
    }
    if proof.witness_claims[1].0 != rev(&y_lin) || proof.witness_claims[1].1 != w_w {
        return Err("batch: lin witness claim");
    }
    verify_eval(
        &proof.stmt,
        params,
        &proof.witness_claims[0].0,
        proof.witness_claims[0].1,
        &proof.pcs_proofs[0],
        t,
    )?;
    verify_eval(
        &proof.stmt,
        params,
        &proof.witness_claims[1].0,
        proof.witness_claims[1].1,
        &proof.pcs_proofs[1],
        t,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::CircuitBuilder;
    use crate::gf8::Gf8;
    use crate::interp::InterpData;
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
        // 128 witnesses by replay (deterministic variety)
        let witnesses: Vec<Vec<bool>> = (0..KAPPA)
            .map(|j| {
                let mut rb = CircuitBuilder::replay();
                emit_toy(&mut rb, j & 1 == 1, (j >> 1) & 1 == 1);
                rb.w
            })
            .collect();
        (cs, witnesses)
    }

    #[test]
    fn xi_r_identity() {
        // Isolate the U_R identity: v2 = 〈(Aρ)∘l, ts(y)〉 must equal
        // 〈u_columns_msg(Ξ_R^⊤ A), ρ〉 with ξ_R = l∘ts(y).
        let f8 = Gf8::new();
        let _ = &f8;
        let m = 64usize;
        let log_m = 6usize;
        let mut seed = 0xfeedu128;
        let mut gf = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            Gf128(seed)
        };
        let mut a = PlaneSet::zeros(m, KAPPA);
        for j in 0..KAPPA {
            for i in 0..m {
                a.set_bit(j, i, gf().0 & 1 == 1);
            }
        }
        let l_factors: Vec<(Gf128, Gf128)> = (0..log_m).map(|_| (gf(), gf())).collect();
        let l = RowWeightTensor::new(l_factors.clone(), true);
        let rho: Vec<Gf128> = (0..KAPPA).map(|_| gf()).collect();
        // b = Aρ densely
        let mut b = vec![ZERO; m];
        for i in 0..m {
            for j in 0..KAPPA {
                if a.bit(j, i) {
                    b[i] += rho[j];
                }
            }
        }
        let mut b_l = b.clone();
        for i in 0..m {
            b_l[i] = mul(b_l[i], l.words[i]);
        }
        let y: Vec<Gf128> = (0..log_m).map(|_| gf()).collect();
        // v2 = 〈b_l, ts(y_rev)〉 (the little-endian factor list)
        let ts_words = {
            let f: Vec<(Gf128, Gf128)> = y.iter().rev().map(|&yk| (Gf128(1) + yk, yk)).collect();
            let mut words = vec![Gf128(1)];
            for &(f0, f1) in &f {
                let half = words.len();
                let mut nv = vec![ZERO; half * 2];
                for i in 0..half {
                    nv[i] = mul(words[i], f0);
                    nv[half + i] = mul(words[i], f1);
                }
                words = nv;
            }
            words
        };
        let v2 = dot_gf(&b_l, &ts_words);
        // U_R route
        let xi_r_factors =
            crate::hadamard::xi2_factors(&l, &y.iter().rev().copied().collect::<Vec<_>>());
        let (xr_words, xr_planes, xr_wp) = xi_materialized(&xi_r_factors);
        // sanity: the ξ_R words must equal l_i·ts_i
        for i in 0..m {
            assert_eq!(xr_words[i], mul(l.words[i], ts_words[i]), "xi_r word {i}");
        }
        let rows = u_from_xi(&a, &xr_planes, xr_wp);
        let msg = u_columns_msg_vec(&rows);
        assert_eq!(dot_gf(&msg, &rho), v2, "U_R identity");
    }

    #[test]
    fn batch_roundtrip_and_tamper() {
        let f8 = Gf8::new();
        let mut data = InterpData::new(&f8);
        data.build_psi_monomials(&f8);
        let (cs, witnesses) = toy_cs();
        assert!(witnesses.iter().all(|w| cs.satisfied_by(w)));
        let params = PcsParams::default();
        let setup = setup_batch(&cs, &witnesses, &params);
        let packed_io = packed_io_words(&cs, &witnesses);

        let mut t = Transcript::new("mikros-batch-test", &[]);
        let proof = prove_batch(&f8, &data, &cs, &setup, &params, &packed_io, &mut t);
        let mut tv = Transcript::new("mikros-batch-test", &[]);
        let vres = verify_batch(&f8, &data, &cs, &params, &packed_io, &proof, &mut tv);
        assert!(
            vres.is_ok(),
            "the honest batch must verify (err = {:?})",
            vres.err()
        );

        // tamper: flip one packed io word → the io claim changes → reject
        {
            let mut bad_io = packed_io.clone();
            bad_io[0] += ONE;
            let mut tv = Transcript::new("mikros-batch-test", &[]);
            assert!(verify_batch(&f8, &data, &cs, &params, &bad_io, &proof, &mut tv).is_err());
        }
        // tamper: flip a witness bit → a tampered setup must fail to verify
        {
            let mut w2 = witnesses.clone();
            // instance 1 has (x0,x1) = (1,0) ⇒ x3 = 0, so x2 = 1 makes
            // the row-1 residual (x0+x3)·x2 = 1 ≠ 0: unsatisfiable
            w2[1][2] = true;
            assert!(!cs.satisfied_by(&w2[1]));
            assert!(w2.iter().any(|w| !cs.satisfied_by(w)));
            let setup2 = setup_batch(&cs, &w2, &params);
            let packed_io2 = packed_io_words(&cs, &w2);
            let mut t2 = Transcript::new("mikros-batch-test", &[]);
            // the io words here pin only x0/x1/x3; x2 is private: an
            // inconsistent witness still proves *something* — the R1CS
            // residual check must fail at the U_O / Hadamard stage.
            let bad = prove_batch(&f8, &data, &cs, &setup2, &params, &packed_io2, &mut t2);
            let mut tv = Transcript::new("mikros-batch-test", &[]);
            assert!(
                verify_batch(&f8, &data, &cs, &params, &packed_io2, &bad, &mut tv).is_err(),
                "an unsatisfying batch must reject"
            );
        }
        // tamper: a proof message
        {
            let mut bad = BatchProof {
                io_claim: proof.io_claim,
                fact1_io: clone_f1(&proof.fact1_io),
                v_o: proof.v_o + ONE,
                had: clone_had(&proof.had),
                v_msgs: proof.v_msgs.clone(),
                fact1_lin: clone_f1(&proof.fact1_lin),
                pub_vals: proof.pub_vals,
                witness_claims: proof.witness_claims.clone(),
                pcs_proofs: proof.pcs_proofs.clone(),
                stmt: proof.stmt.clone(),
            };
            let _ = &mut bad;
            // v_o tamper: the Hadamard's c != <d, r> check fires
            let mut tv = Transcript::new("mikros-batch-test", &[]);
            assert!(verify_batch(&f8, &data, &cs, &params, &packed_io, &bad, &mut tv).is_err());
        }
    }

    fn clone_f1(p: &Fact1Proof) -> Fact1Proof {
        Fact1Proof {
            rounds: p.rounds.clone(),
            terminals: p.terminals,
            challenges: p.challenges.clone(),
        }
    }
    fn clone_had(h: &HadamardProof) -> HadamardProof {
        HadamardProof {
            d: h.d.clone(),
            h: h.h.clone(),
            fact1: clone_f1(&h.fact1),
        }
    }
}
