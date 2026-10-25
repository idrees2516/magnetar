//! Virtual Spartan (paper §5.3, Construction 2): reduce **two** SIMD R1CS
//! relations sharing one bit-valued witness matrix `W` to one evaluation
//! claim per witness segment.
//!
//! Steps (mirroring the paper):
//! 1. Two outer sum-checks run **in lockstep**: the rounds binding the SIMD
//!    instance variables share one challenge (both end at the same
//!    `r_inst`); the row-variable rounds get separate challenges
//!    (`r_row,t`). Each outer proves
//!    `0 = Σ_x eq(τ_t, x)·(Â_t z · B̂_t z + Ĉ_t z)` over the joint
//!    (instance ‖ row) space.
//! 2. Six terminal values `v_{t,M}`, `t ∈ {bin, fld}`, `M ∈ {A, B, C}`,
//!    checked by the engine's terminal identity (paper step 3).
//! 3. Random α-combination `v* = Σ α_{t,M} v_{t,M}`.
//! 4. One inner sum-check over the **column** variables:
//!    `v* = Σ_y [Σ α_{t,M} M̂_t(r_row,t, y)]·ẑ_f(r_inst, y)`, where
//!    `ẑ_f(r_inst, ·)` is the witness matrix folded at the shared instance
//!    point — the random fold of all `q` instances (paper §4.2).
//! 5. Equation (1) at the terminal: the MSB of the column point splits
//!    `ẑ` into the witness half and the public `(1, x)` half; the witness
//!    half splits over aligned segment subcubes as
//!    `Σ_i eq(q_i, b_i)·Ŵ_i(r_inst, r_i)` (paper steps 5–8).
//!
//! Output: one `SegmentClaim` per non-padding segment — direct segments go
//! to MosaicPCS, virtual segments to their owning arguments.

use crate::r1cs::SparseMatrix;
use crate::SegmentClaim;
use magnetar_field::{eval_mle, mul, Gf128, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_sumcheck::{
    engine::{SumcheckProver, SumcheckVerifier},
    public::{EqFactor, TableFactor},
    BindOrder, SumcheckSpec, Term,
};

/// The two relations of §5.1: `bin` over `B ⊂ K`, `fld` over `K`, sharing
/// the same column layout and witness matrix.
#[derive(Clone, Debug)]
pub struct DualRelation {
    pub m: usize, // column count, power of two
    pub a_bin: SparseMatrix,
    pub b_bin: SparseMatrix,
    pub c_bin: SparseMatrix,
    pub a_fld: SparseMatrix,
    pub b_fld: SparseMatrix,
    pub c_fld: SparseMatrix,
}

/// A declared witness segment (paper's segment layout, in declared order).
#[derive(Clone, Debug)]
pub struct SegmentSpec {
    pub name: String,
    /// Segment length in columns (power of two).
    pub len: usize,
    /// Virtual segments are discharged by auxiliary arguments; direct
    /// segments are committed.
    pub direct: bool,
}

/// The aligned-segment layout over the witness half of the columns.
///
/// Segments are placed **in declared order** at offsets aligned to their
/// size (the largest-first reordering is avoided deliberately: callers
/// address columns by declared segment identity). Gaps are padding
/// segments, identically zero.
#[derive(Clone, Debug)]
pub struct SegmentLayout {
    pub witness_cols: usize,
    pub specs: Vec<SegmentSpec>,
    /// `offsets[i]` = start column of segment i.
    pub offsets: Vec<usize>,
}

impl SegmentLayout {
    pub fn build(witness_cols: usize, specs: Vec<SegmentSpec>) -> Self {
        assert!(witness_cols.is_power_of_two());
        for s in &specs {
            assert!(s.len.is_power_of_two());
            assert!(s.len <= witness_cols);
        }
        let mut offsets = Vec::with_capacity(specs.len());
        let mut cursor = 0usize;
        for s in &specs {
            let align = s.len;
            let aligned = cursor.div_ceil(align) * align;
            offsets.push(aligned);
            cursor = aligned + s.len;
        }
        assert!(
            cursor <= witness_cols,
            "segment layout overflows the witness half"
        );
        SegmentLayout {
            witness_cols,
            specs,
            offsets,
        }
    }

    /// Local variable count of segment i.
    pub fn local_vars(&self, i: usize) -> usize {
        self.specs[i].len.trailing_zeros() as usize
    }

    /// Suffix length (bits) of segment i's aligned subcube.
    pub fn suffix_bits(&self, i: usize) -> usize {
        self.witness_vars() - self.local_vars(i)
    }

    /// The fixed Boolean suffix `b_i` of segment i.
    pub fn b_i(&self, i: usize) -> usize {
        self.offsets[i] >> self.local_vars(i)
    }

    /// Total witness-half variables.
    pub fn witness_vars(&self) -> usize {
        self.witness_cols.trailing_zeros() as usize
    }

    /// Number of segments (declared).
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// The index of a segment by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.specs.iter().position(|s| s.name == name)
    }
}

/// The Virtual Spartan proof (non-interactive via Fiat–Shamir).
#[derive(Clone, Debug)]
pub struct VirtualSpartanProof {
    /// Lockstep (shared) outer rounds: both relations' messages.
    pub outer_shared: Vec<(Vec<Gf128>, Vec<Gf128>)>,
    /// Row rounds per relation (separate challenges).
    pub outer_bin: Vec<Vec<Gf128>>,
    pub outer_fld: Vec<Vec<Gf128>>,
    /// The six terminal values `v_{t,M}` (bin A, B, C, fld A, B, C).
    pub terminals: [Gf128; 6],
    /// Inner sum-check rounds.
    pub inner: Vec<Vec<Gf128>>,
    /// The inner terminal claim `v_z = ẑ_f(r_col)`.
    pub v_z: Gf128,
    /// Per-segment evaluation values (padding segments send 0).
    pub segment_values: Vec<Gf128>,
}

/// Absorb a round-message pair and sample one shared challenge.
fn shared_round(
    p1: &mut SumcheckProver,
    p2: &mut SumcheckProver,
    t: &mut Transcript,
    label: &str,
) -> (Gf128, (Vec<Gf128>, Vec<Gf128>)) {
    let m1 = p1.message().0;
    let m2 = p2.message().0;
    let mut buf = Vec::with_capacity((m1.len() + m2.len()) * 16);
    for c in m1.iter().chain(m2.iter()) {
        buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
    }
    t.absorb_bytes(label, &buf);
    let r = t.challenge_field(label);
    p1.fold_msg(&magnetar_sumcheck::RoundMsg(m1.clone()), r);
    p2.fold_msg(&magnetar_sumcheck::RoundMsg(m2.clone()), r);
    (r, (m1, m2))
}

fn outer_spec(
    _a: &SparseMatrix,
    _b: &SparseMatrix,
    _c: &SparseMatrix,
    az: Vec<Gf128>,
    bz: Vec<Gf128>,
    cz: Vec<Gf128>,
    tau: &[Gf128],
    num_vars: usize,
) -> SumcheckSpec {
    SumcheckSpec {
        num_vars,
        dense: vec![az, bz, cz],
        public: vec![Box::new(EqFactor::new(tau.to_vec()))],
        terms: vec![
            Term {
                coeff: ONE,
                dense: vec![0, 1],
                public: vec![0],
            },
            Term {
                coeff: ONE,
                dense: vec![2],
                public: vec![0],
            },
        ],
        claim: ZERO,
    }
}

/// Prove Virtual Spartan over the **packed** witness: `w_packed` is
/// the bit-packed w-half (128 witness columns per `u128`, row-major
/// per instance), `x_cols` the public x columns (shared across every
/// instance). The packing is the paper's instance-batching discipline:
/// the eq(r_inst)-fold of the bit half is a select-XOR pass over the
/// packed words, the x-fold is the identity (Σ_u eq = 1), and the
/// matrices' shared x-part is applied once.
///
/// Returns the proof and the per-segment claims.
#[allow(clippy::too_many_arguments)]
pub fn prove_virtual_spartan(
    rel: &DualRelation,
    layout: &SegmentLayout,
    w_packed: &[u128],
    x_cols: &[Gf128],
    q: usize,
    t: &mut Transcript,
) -> (VirtualSpartanProof, Vec<SegmentClaim>) {
    let m = rel.m;
    let w_cols = m / 2;
    let s = q.trailing_zeros() as usize; // instance vars
                                         // The outer sum-checks run over (instance ‖ constraint-row), each
                                         // relation over its OWN padded row count (only q and m are shared,
                                         // paper Construction 2), with separate row challenges.
    let rows_bin = rel.a_bin.rows.max(1);
    let rows_fld = rel.a_fld.rows.max(1);
    debug_assert!(rows_bin.is_power_of_two());
    debug_assert!(rows_fld.is_power_of_two());
    let row_vars_bin = rows_bin.trailing_zeros() as usize;
    let row_vars_fld = rows_fld.trailing_zeros() as usize;
    let joint_vars_bin = s + row_vars_bin;
    let joint_vars_fld = s + row_vars_fld;
    let w128 = (w_cols + 127) / 128;
    debug_assert_eq!(w_packed.len(), q * w128);
    // Sample the two τ points (per-relation joint spaces).
    let tau_bin = t.challenge_field_vec("vs-tau-bin", joint_vars_bin);
    let tau_fld = t.challenge_field_vec("vs-tau-fld", joint_vars_fld);
    // Dense tables: block-diagonal applications.
    struct _Pt(std::time::Instant);
    impl Drop for _Pt {
        fn drop(&mut self) {
            if std::env::var("MAGNETAR_TIME").is_ok() {
                eprintln!(
                    "[time] {:<24} {:>9.1} ms",
                    "vs-apply",
                    self.0.elapsed().as_secs_f64() * 1e3
                );
            }
        }
    }
    struct _PtR {
        label: &'static str,
        start: std::time::Instant,
    }
    impl _PtR {
        fn new(label: &'static str) -> Self {
            _PtR {
                label,
                start: std::time::Instant::now(),
            }
        }
    }
    impl Drop for _PtR {
        fn drop(&mut self) {
            if std::env::var("MAGNETAR_TIME").is_ok() {
                eprintln!(
                    "[time] {:<24} {:>9.1} ms",
                    self.label,
                    self.start.elapsed().as_secs_f64() * 1e3
                );
            }
        }
    }
    let _pt_apply = _Pt(std::time::Instant::now());
    // Packed application (small-value setting): every `entry · z_c` on a
    // bit column is `entry` or `0` — a select, no CLMUL — and the shared
    // x columns contribute once, not per instance. A table is bit-valued
    // exactly when ITS matrix is binary and all its referenced columns
    // are bits (checked per matrix, never assumed).
    let apply_bits =
        |mat: &crate::SparseMatrix| -> (Vec<Gf128>, bool) {
            mat.apply_simd_packed(w_packed, x_cols, q, w_cols)
        };
    let (az_bin, az_bits) = apply_bits(&rel.a_bin);
    let (bz_bin, bz_bits) = apply_bits(&rel.b_bin);
    let (cz_bin, cz_bits) = apply_bits(&rel.c_bin);
    let az_fld = apply_bits(&rel.a_fld).0;
    let bz_fld = apply_bits(&rel.b_fld).0;
    let cz_fld = apply_bits(&rel.c_fld).0;
    let spec_bin = outer_spec(
        &rel.a_bin,
        &rel.b_bin,
        &rel.c_bin,
        az_bin,
        bz_bin,
        cz_bin,
        &tau_bin,
        joint_vars_bin,
    );
    let spec_fld = outer_spec(
        &rel.a_fld,
        &rel.b_fld,
        &rel.c_fld,
        az_fld,
        bz_fld,
        cz_fld,
        &tau_fld,
        joint_vars_fld,
    );
    drop(_pt_apply);
    // Lockstep: instance rounds share challenges (HighBitsFirst(s) binds the
    // high s instance variables first), row rounds separate.
    let order = BindOrder::HighBitsFirst { high: s };
    let _pt_r = _PtR::new("vs-outer-rounds");
    // The bin outer takes the **small-value round window** (the
    // sum-check papers' round batching): its `az/bz/cz` are bit-valued and
    // its shape is `eq·(az·bz + cz)`, so the first `min(v, s)` shared
    // rounds come from ONE pass over the un-folded tables (selects, not
    // CLMUL) — messages bit-identical to the generic engine's. The fld
    // prover's tables are field-valued: generic engine.
    let mut p_bin = SumcheckProver::with_dense_bits_auto_window(
        &spec_bin,
        order,
        vec![az_bits, bz_bits, cz_bits],
    );
    let mut p_fld = SumcheckProver::with_dense_bits(&spec_fld, order, vec![false; 3]);
    let mut outer_shared = Vec::with_capacity(s);
    let mut r_inst_ch = Vec::with_capacity(s);
    for i in 0..s {
        let (r, msgs) = shared_round(&mut p_bin, &mut p_fld, t, &format!("vs-out-shared-{i}"));
        r_inst_ch.push(r);
        outer_shared.push(msgs);
    }
    let row_vars_outer = row_vars_bin.max(row_vars_fld);
    let mut outer_bin = Vec::with_capacity(row_vars_bin);
    let mut outer_fld = Vec::with_capacity(row_vars_fld);
    let mut row_ch_bin = Vec::new();
    let mut row_ch_fld = Vec::new();
    // Row rounds with separate cubes: all of the bin relation's row rounds
    // first (one message + challenge each), then all of the fld relation's.
    for i in 0..row_vars_bin {
        let label = format!("vs-out-rowbin-{i}");
        let m1 = p_bin.message();
        let mut buf = Vec::with_capacity(m1.0.len() * 16);
        for c in &m1.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(&label, &buf);
        let r1 = t.challenge_field(&label);
        p_bin.fold_msg(&m1, r1);
        row_ch_bin.push(r1);
        outer_bin.push(m1.0);
    }
    for i in 0..row_vars_fld {
        let label = format!("vs-out-rowfld-{i}");
        let m2 = p_fld.message();
        let mut buf = Vec::with_capacity(m2.0.len() * 16);
        for c in &m2.0 {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(&label, &buf);
        let r2 = t.challenge_field(&label);
        p_fld.fold_msg(&m2, r2);
        row_ch_fld.push(r2);
        outer_fld.push(m2.0);
    }
    let _ = row_vars_outer;
    drop(_pt_r);
    let (claims_bin, _) = p_bin.finish();
    let (claims_fld, _) = p_fld.finish();
    // The engine's terminal check is the paper's step-3 identity; the
    // verifier recomputes it. Collect the six terminals.
    let terminals = [
        claims_bin[0],
        claims_bin[1],
        claims_bin[2],
        claims_fld[0],
        claims_fld[1],
        claims_fld[2],
    ];
    for (i, v) in terminals.iter().enumerate() {
        t.absorb_field(&format!("vs-terminal-{i}"), &[*v]);
    }
    // α-batching (paper step 3).
    let alphas = t.challenge_field_vec("vs-alpha", 6);
    let v_star: Gf128 = alphas
        .iter()
        .zip(terminals.iter())
        .map(|(&a, &v)| mul(a, v))
        .fold(ZERO, |x, y| x + y);
    // Shared instance point: the HighBitsFirst(s) order bound the original
    // variables m-1, m-2, …, m-s with the shared challenges in order.
    // The shared instance point: rounds bind the joint variables
    // (joint_vars−1) … (joint_vars−s), which are the instance coordinates
    // (s−1) … 0 in the instance-local indexing.
    let mut r_inst = vec![ZERO; s];
    for (round, &c) in r_inst_ch.iter().enumerate() {
        r_inst[s - 1 - round] = c;
    }
    let r_row_bin: Vec<Gf128> = row_ch_bin.clone();
    let r_row_fld: Vec<Gf128> = row_ch_fld.clone();
    // Inner sum-check over the COLUMN variables (log m).
    let row_vars = m.trailing_zeros() as usize;
    let m_combo = build_m_combo(rel, &alphas, &r_row_bin, &r_row_fld);
    // The eq(r_inst)-folded z in ONE packed select pass over the bit
    // words (the paper's instance-batching discipline): the w-half folds
    // by XOR of eq[u] on set bits — no CLMUL, no per-column gather — and
    // the x-half is IDENTICAL across instances, so Σ_u eq[u]·x = x·(Σ eq)
    // = x: a copy. The `1` column folds to Σ eq = 1.
    let wf = fold_packed_w(w_packed, q, w_cols, &r_inst);
    let mut z_fold = Vec::with_capacity(m);
    z_fold.extend_from_slice(&wf);
    z_fold.push(ONE);
    z_fold.extend_from_slice(x_cols);
    debug_assert_eq!(z_fold.len(), m);
    let inner_spec = SumcheckSpec {
        num_vars: row_vars,
        dense: vec![z_fold],
        public: vec![Box::new(TableFactor::new(m_combo))],
        terms: vec![Term {
            coeff: ONE,
            dense: vec![0],
            public: vec![0],
        }],
        claim: v_star,
    };
    let mut inner_p = SumcheckProver::with_dense_bits(
        &inner_spec,
        BindOrder::Increasing,
        // `z_fold` is the eq-weighted instance fold — field-valued even
        // when the witness is bits.
        vec![false],
    );
    let mut inner = Vec::with_capacity(row_vars);
    let mut r_col_ch = Vec::with_capacity(row_vars);
    for _ in 0..row_vars {
        let msg = inner_p.message();
        inner.push(msg.0.clone());
        r_col_ch.push(inner_p.absorb_and_fold(t, "vs-inner"));
    }
    let (inner_claims, _) = inner_p.finish();
    let v_z = inner_claims[0];
    t.absorb_field("vs-vz", &[v_z]);
    // Equation (1) + segment split. Column point r_col (increasing order).
    let r_col: Vec<Gf128> = r_col_ch;
    let r_low: Vec<Gf128> = r_col[..row_vars - 1].to_vec();
    // Per-segment values: v_i = Ŵ_f evaluated at (r_local of segment i).
    // Every segment's eq(r_inst)-folded table is the packed fold's slice
    // at the segment's column range (segments tile the w-half at aligned
    // offsets) — the strided per-column dots are gone; only the small
    // MLE evaluation at r_local remains.
    let mut segment_values = Vec::with_capacity(layout.len());
    let mut segment_claims = Vec::with_capacity(layout.len());
    for i in 0..layout.len() {
        let lv = layout.local_vars(i);
        let base = layout.offsets[i];
        let len = 1usize << lv;
        // r_local: the low `lv` bits of r_low.
        let r_local: Vec<Gf128> = r_low[..lv].to_vec();
        let seg_table = &wf[base..base + len];
        let v = if lv == 0 {
            seg_table[0]
        } else {
            eval_mle(seg_table, &r_local)
        };
        segment_values.push(v);
        segment_claims.push(SegmentClaim {
            segment: i,
            r_inst: r_inst.clone(),
            r_local: r_local.clone(),
            value: v,
        });
    }
    for (i, v) in segment_values.iter().enumerate() {
        t.absorb_field(&format!("vs-seg-{i}"), &[*v]);
    }
    (
        VirtualSpartanProof {
            outer_shared,
            outer_bin,
            outer_fld,
            terminals,
            inner,
            v_z,
            segment_values,
        },
        segment_claims,
    )
}

/// The α-combination of matrix MLEs as a table over the column cube.
fn build_m_combo(
    rel: &DualRelation,
    alphas: &[Gf128],
    r_row_bin: &[Gf128],
    r_row_fld: &[Gf128],
) -> Vec<Gf128> {
    let m = rel.m;
    let mut combo = vec![ZERO; m];
    let parts = [
        (&rel.a_bin, r_row_bin, alphas[0]),
        (&rel.b_bin, r_row_bin, alphas[1]),
        (&rel.c_bin, r_row_bin, alphas[2]),
        (&rel.a_fld, r_row_fld, alphas[3]),
        (&rel.b_fld, r_row_fld, alphas[4]),
        (&rel.c_fld, r_row_fld, alphas[5]),
    ];
    for (mat, r_row, alpha) in parts {
        let table = mat.eval_mle_table(r_row);
        for (c, v) in combo.iter_mut().zip(table.iter()) {
            *c += mul(alpha, *v);
        }
    }
    combo
}

/// The eq(r_inst)-fold of the packed bit half: `out[y] = Σ_u
/// eq[u]·w(u, y)`, computed as ONE sequential pass over the packed
/// words — for each instance, `eq[u]` is XOR-selected into exactly the
/// columns whose bit is set (a 16-byte XOR per *set* bit, no CLMUL, no
/// per-column gather). XOR accumulation is order-free, so the result
/// is bit-identical to the naive per-column dot. Sparse-word skipping
/// plus trailing-zero iteration keeps the work proportional to the
/// set-bit count.
fn fold_packed_w(
    w_packed: &[u128],
    q: usize,
    w_cols: usize,
    r_inst: &[Gf128],
) -> Vec<Gf128> {
    let eq = magnetar_field::EqBits::build(r_inst).values;
    let w128 = (w_cols + 127) / 128;
    debug_assert_eq!(w_packed.len(), q * w128);
    let mut out = vec![ZERO; w_cols];
    for u in 0..q {
        let e = eq[u];
        if e == ZERO {
            continue;
        }
        let row = &w_packed[u * w128..(u + 1) * w128];
        for (k, &word) in row.iter().enumerate() {
            let col_base = k * 128;
            // Skip set bits beyond the w range (a partial last word).
            let mut hi = if col_base + 64 < w_cols {
                (word >> 64) as u64
            } else {
                0
            };
            let mut lo = if col_base + 64 <= w_cols {
                word as u64
            } else {
                // Partial low half: mask off columns ≥ w_cols.
                word as u64 & (u64::MAX >> (64 - (w_cols - col_base) % 64).max(0).min(64))
            };
            while lo != 0 {
                let b = lo.trailing_zeros() as usize;
                out[col_base + b] += e;
                lo &= lo - 1;
            }
            while hi != 0 {
                let b = hi.trailing_zeros() as usize;
                out[col_base + 64 + b] += e;
                hi &= hi - 1;
            }
        }
    }
    out
}

/// Verify a Virtual Spartan proof. Returns the accepted per-segment claims
/// (the verifier's view; values match the prover's).
#[allow(clippy::too_many_arguments)]
pub fn verify_virtual_spartan(
    rel: &DualRelation,
    layout: &SegmentLayout,
    q: usize,
    public: &[Vec<Gf128>], // per-instance public inputs
    proof: &VirtualSpartanProof,
    t: &mut Transcript,
) -> Result<Vec<SegmentClaim>, &'static str> {
    let m = rel.m;
    let s = q.trailing_zeros() as usize;
    let rows_bin = rel.a_bin.rows.max(1);
    let rows_fld = rel.a_fld.rows.max(1);
    let row_vars_bin = rows_bin.trailing_zeros() as usize;
    let row_vars_fld = rows_fld.trailing_zeros() as usize;
    let joint_vars_bin = s + row_vars_bin;
    let joint_vars_fld = s + row_vars_fld;
    // τ points (per-relation joint spaces).
    let tau_bin = t.challenge_field_vec("vs-tau-bin", joint_vars_bin);
    let tau_fld = t.challenge_field_vec("vs-tau-fld", joint_vars_fld);
    let order = BindOrder::HighBitsFirst { high: s };
    let mk_terms = || {
        vec![
            Term {
                coeff: ONE,
                dense: vec![0, 1],
                public: vec![0],
            },
            Term {
                coeff: ONE,
                dense: vec![2],
                public: vec![0],
            },
        ]
    };
    let pf_bin: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(EqFactor::new(tau_bin.clone()))];
    let pf_fld: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(EqFactor::new(tau_fld.clone()))];
    let mut v_bin = SumcheckVerifier::new(joint_vars_bin, ZERO, &pf_bin, mk_terms(), order);
    let mut v_fld = SumcheckVerifier::new(joint_vars_fld, ZERO, &pf_fld, mk_terms(), order);
    // Shared rounds: absorb both messages, one challenge, fold both.
    let mut r_inst_ch = Vec::with_capacity(s);
    for (i, (m1, m2)) in proof.outer_shared.iter().enumerate() {
        let label = format!("vs-out-shared-{i}");
        let mut buf = Vec::with_capacity((m1.len() + m2.len()) * 16);
        for c in m1.iter().chain(m2.iter()) {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(&label, &buf);
        let r = t.challenge_field(&label);
        v_bin
            .round_external(&magnetar_sumcheck::RoundMsg(m1.clone()), r)
            .map_err(|_| "vs: shared bin round check failed")?;
        v_fld
            .round_external(&magnetar_sumcheck::RoundMsg(m2.clone()), r)
            .map_err(|_| "vs: shared fld round check failed")?;
        r_inst_ch.push(r);
    }
    // Row rounds: the bin relation's rounds first, then the fld relation's
    // (separate cubes, one message + challenge each).
    let mut row_ch_bin = Vec::with_capacity(row_vars_bin);
    let mut row_ch_fld = Vec::with_capacity(row_vars_fld);
    for (i, m1) in proof.outer_bin.iter().enumerate() {
        let label = format!("vs-out-rowbin-{i}");
        let mut buf = Vec::with_capacity(m1.len() * 16);
        for c in m1.iter() {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(&label, &buf);
        let r1 = t.challenge_field(&label);
        v_bin
            .round_external(&magnetar_sumcheck::RoundMsg(m1.clone()), r1)
            .map_err(|_| "vs: row bin round failed")?;
        row_ch_bin.push(r1);
    }
    for (i, m2) in proof.outer_fld.iter().enumerate() {
        let label = format!("vs-out-rowfld-{i}");
        let mut buf = Vec::with_capacity(m2.len() * 16);
        for c in m2.iter() {
            buf.extend_from_slice(&magnetar_hash::fe_to_bytes(*c));
        }
        t.absorb_bytes(&label, &buf);
        let r2 = t.challenge_field(&label);
        v_fld
            .round_external(&magnetar_sumcheck::RoundMsg(m2.clone()), r2)
            .map_err(|_| "vs: row fld round failed")?;
        row_ch_fld.push(r2);
    }
    // Terminal checks (paper step 3): eq(τ_t, r_x_t)·(vA·vB + vC) = running.
    let [vb_a, vb_b, vb_c, vf_a, vf_b, vf_c] = proof.terminals;
    v_bin
        .finish(&[vb_a, vb_b, vb_c])
        .map_err(|_| "vs: bin terminal identity failed")?;
    v_fld
        .finish(&[vf_a, vf_b, vf_c])
        .map_err(|_| "vs: fld terminal identity failed")?;
    for (i, v) in proof.terminals.iter().enumerate() {
        t.absorb_field(&format!("vs-terminal-{i}"), &[*v]);
    }
    let alphas = t.challenge_field_vec("vs-alpha", 6);
    let v_star: Gf128 = alphas
        .iter()
        .zip(proof.terminals.iter())
        .map(|(&a, &v)| mul(a, v))
        .fold(ZERO, |x, y| x + y);
    // Shared instance point (HighBitsFirst: rounds bind m-1, m-2, …).
    // The shared instance point: rounds bind the joint variables
    // (joint_vars−1) … (joint_vars−s), which are the instance coordinates
    // (s−1) … 0 in the instance-local indexing.
    let mut r_inst = vec![ZERO; s];
    for (round, &c) in r_inst_ch.iter().enumerate() {
        r_inst[s - 1 - round] = c;
    }
    // Inner sum-check: the verifier builds M_combo itself.
    let m_combo = build_m_combo(rel, &alphas, &row_ch_bin, &row_ch_fld);
    let inner_terms = vec![Term {
        coeff: ONE,
        dense: vec![0],
        public: vec![0],
    }];
    // The inner sum-check runs over the shared column space: log2(m) vars.
    let row_vars = m.trailing_zeros() as usize;
    let pub_inner: Vec<Box<dyn magnetar_sumcheck::public::PublicFactor + Send + Sync>> =
        vec![Box::new(TableFactor::new(m_combo))];
    let mut inner_v = SumcheckVerifier::new(
        row_vars,
        v_star,
        &pub_inner,
        inner_terms,
        BindOrder::Increasing,
    );
    let mut r_col_ch = Vec::with_capacity(row_vars);
    for msg in &proof.inner {
        let ch = inner_v
            .round_capturing(&magnetar_sumcheck::RoundMsg(msg.clone()), t, "vs-inner")
            .map_err(|_| "vs: inner round failed")?;
        r_col_ch.push(ch);
    }
    inner_v
        .finish(&[proof.v_z])
        .map_err(|_| "vs: inner terminal failed")?;
    t.absorb_field("vs-vz", &[proof.v_z]);
    // Equation (1): v_z = (1+msb)·[Σ_i eq(q_i,b_i)·v_i] + msb·public_fold.
    let r_col: Vec<Gf128> = r_col_ch;
    let msb = r_col[row_vars - 1];
    let r_low: Vec<Gf128> = r_col[..row_vars - 1].to_vec();
    // Public fold of the (1, x) half.
    let half = m / 2;
    let eq_inst = magnetar_field::EqBits::build(&r_inst).values;
    let eq_low = magnetar_field::EqBits::build(&r_low).values;
    let mut public_fold = ZERO;
    for (u, x) in public.iter().enumerate() {
        let mut term = eq_low[0];
        for (j, &xv) in x.iter().enumerate() {
            term += mul(eq_low[j + 1], xv);
        }
        public_fold += mul(eq_inst[u], term);
    }
    // Segment combination: Σ_i eq(q_i, b_i)·v_i.
    let mut seg_sum = ZERO;
    for i in 0..layout.len() {
        let lv = layout.local_vars(i);
        let b = layout.b_i(i);
        // q_i: the suffix coordinates of r_low (high bits).
        let suffix_bits = layout.suffix_bits(i);
        let q_pt: Vec<Gf128> = r_low[lv..lv + suffix_bits].to_vec();
        // eq(q_i, b_i) = Π_j eq(q_j, b_j).
        let mut e = ONE;
        for (j, &qv) in q_pt.iter().enumerate() {
            e = mul(e, eq_char2_pub(qv, (b >> j) & 1 == 1));
        }
        seg_sum += mul(e, proof.segment_values[i]);
        let _ = lv;
    }
    let lhs = mul(ONE + msb, seg_sum) + mul(msb, public_fold);
    if lhs != proof.v_z {
        return Err("vs: equation (1) segment identity failed");
    }
    for (i, v) in proof.segment_values.iter().enumerate() {
        t.absorb_field(&format!("vs-seg-{i}"), &[*v]);
    }
    // Output claims.
    let claims = (0..layout.len())
        .map(|i| SegmentClaim {
            segment: i,
            r_inst: r_inst.clone(),
            r_local: r_low[..layout.local_vars(i)].to_vec(),
            value: proof.segment_values[i],
        })
        .collect();
    let _ = half;
    Ok(claims)
}

#[inline]
fn eq_char2_pub(x: Gf128, b: bool) -> Gf128 {
    if b {
        x
    } else {
        ONE + x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rg(rng: &mut impl rand::Rng) -> Gf128 {
        Gf128(rand::RngExt::random::<u128>(rng))
    }

    fn rand_mat(rng: &mut impl rand::Rng, rows: usize, cols: usize, n: usize) -> SparseMatrix {
        use rand::RngExt;
        let mut mat = SparseMatrix::empty(rows, cols);
        for _ in 0..n {
            mat.entries.push((
                rng.random_range(0..rows),
                rng.random_range(0..cols),
                rg(&mut *rng),
            ));
        }
        mat
    }

    #[test]
    fn virtual_spartan_roundtrip() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let q = 1usize;
        let m = 8usize;
        let layout = SegmentLayout::build(
            m / 2,
            vec![
                SegmentSpec {
                    name: "s0".into(),
                    len: 2,
                    direct: true,
                },
                SegmentSpec {
                    name: "s1".into(),
                    len: 2,
                    direct: false,
                },
            ],
        );
        // Random sparse A, B; witness with z[0] = 1; C derived so the R1CS
        // holds: C has entries (r, 0, (Az ∘ Bz)_r) and z[0] = 1.
        let a = rand_mat(&mut rng, m, m, 10);
        let b = rand_mat(&mut rng, m, m, 10);
        let public: Vec<Vec<Gf128>> = vec![(0..m / 2 - 1).map(|_| rg(&mut rng)).collect()];
        let half = m / 2;
        let mut joint = vec![ZERO; q * m];
        joint[0] = ONE; // z[0] = 1 (for the derived C matrix)
        joint[half] = ONE; // the R1CS layout constant at column m/2
        for (j, &xv) in public[0].iter().enumerate() {
            joint[half + 1 + j] = xv;
        }
        // Fill the witness half with random bits (they are the W we prove).
        for i in 1..half {
            joint[i] = Gf128(rng.random::<bool>() as u128);
        }
        let az = a.apply(&joint);
        let bz = b.apply(&joint);
        let mut c = SparseMatrix::empty(m, m);
        for r in 0..m {
            let want = mul(az[r], bz[r]);
            c.entries.push((r, 0, want));
        }
        let rel = DualRelation {
            m,
            a_bin: a.clone(),
            b_bin: b.clone(),
            c_bin: c.clone(),
            a_fld: a.clone(),
            b_fld: b.clone(),
            c_fld: c.clone(),
        };
        // The packed form: w-bits per instance (128 columns per u128)
        // plus the shared x vector — this test's witness matches the
        // protocol layout z = (w, 1, x) with a single shared x.
        let w_cols = m / 2;
        let w128 = (w_cols + 127) / 128;
        let mut w_packed = vec![0u128; q * w128];
        for u in 0..q {
            for k in 0..w128 {
                let mut word = 0u128;
                for c in 0..128.min(w_cols - k * 128) {
                    word |= (joint[u * m + k * 128 + c].0 & 1) << c;
                }
                w_packed[u * w128 + k] = word;
            }
        }
        let x_cols: Vec<Gf128> = (0..m / 2 - 1)
            .map(|j| joint[m / 2 + 1 + j])
            .collect();
        let mut t = Transcript::new("vs", &[b"test"]);
        let (proof, claims) =
            prove_virtual_spartan(&rel, &layout, &w_packed, &x_cols, q, &mut t);
        let mut tv = Transcript::new("vs", &[b"test"]);
        let vclaims = verify_virtual_spartan(&rel, &layout, q, &public, &proof, &mut tv)
            .expect("honest vs proof must verify");
        assert_eq!(vclaims.len(), claims.len());
        for (a, b) in claims.iter().zip(vclaims.iter()) {
            assert_eq!(a.value, b.value);
            assert_eq!(a.r_local, b.r_local);
        }
        // Tamper: segment value.
        let mut p = proof.clone();
        p.segment_values[0] += ONE;
        let mut tv = Transcript::new("vs", &[b"test"]);
        assert!(verify_virtual_spartan(&rel, &layout, q, &public, &p, &mut tv).is_err());
        // Tamper: terminal value.
        let mut p = proof.clone();
        p.terminals[0] += ONE;
        let mut tv = Transcript::new("vs", &[b"test"]);
        assert!(verify_virtual_spartan(&rel, &layout, q, &public, &p, &mut tv).is_err());
        // Tamper: inner message.
        let mut p = proof.clone();
        p.inner[0][1] += ONE;
        let mut tv = Transcript::new("vs", &[b"test"]);
        assert!(verify_virtual_spartan(&rel, &layout, q, &public, &p, &mut tv).is_err());
        // Tamper: outer row message.
        let mut p = proof.clone();
        p.outer_bin[0][1] += ONE;
        let mut tv = Transcript::new("vs", &[b"test"]);
        assert!(verify_virtual_spartan(&rel, &layout, q, &public, &p, &mut tv).is_err());
    }
}
