//! Algorithm 5 — the SNARK protocol for PLONK with a single instance
//! (ePrint 2026/2646, §5.2 and Appendix B.1): the last protocol piece of
//! the Mikros stack, composing the crate's building blocks end to end.
//!
//! **The constraint system** (Definition 1): an `N`-gate arithmetic circuit
//! over `F = GF(2)` with selectors `q_L, q_R, q_M, q_O, q_C ∈ F^N`, an
//! augmented assignment `w = (w_L ‖ w_R ‖ w_O ‖ io) ∈ F^{3N+n_io}`, the
//! gate identity
//!
//! ```text
//! q_L∘w_L + q_R∘w_R + q_M∘(w_L∘w_R) + q_O∘w_O + q_C = 0
//! ```
//!
//! and the copy constraint `σ(w) = w` over the wiring permutation
//! `σ : [3N + n_io] → [3N + n_io]`.
//!
//! **The protocol** (Algorithm 5's eighteen steps):
//!
//! 1. *Commit* (steps 2–3): the three wire matrices `W_⋆ ∈ F^{m×κ}`
//!    packed `w_⋆ = W_⋆ β` and committed as ONE vector of logical length
//!    `3m` before any challenge; the public vector — the five selector
//!    packings and the `3·d_L` wire-segment restrictions of the
//!    permuted-position digit tables — of length `m_pub = (5 + 3·d_L)·m`
//!    is preprocessed and committed (`d_L = ⌈log₂ L⌉`, `L = 3N + n_io`).
//! 2. *The gate layer* (steps 5–12): one random projection `x ←$ E^{log₂N}`
//!    of the gate residual; the five partial claims `c_L, c_R, c_M, c_O,
//!    c_C` (summing to zero) each reduced by Algorithm 2 — the selector
//!    side to public evaluations, the wire side to packed-PCS claims —
//!    with the virtual product vector `w_M = w_L∘w_R` discharged by a
//!    second, nested Algorithm 2. Five selector and five wire claims.
//! 3. *The copy layer* (steps 13–17): the joint value–position encoding
//!    over `K_pos = GF(2³²)` (Lemma 2 — the codes are the position bits
//!    plus the witness bit at coordinate 31), zero-padded to `4N`, and
//!    the packed permutation proof of [`crate::copy`] at width
//!    `κ_pos = 4`. Each terminal expands — the column/segment structure
//!    `n = jN + r` matches the wire packing's row/column split after
//!    regrouping `l₀`'s tensor factors — into six wire claims and
//!    `3·d_L` permuted-digit claims; the `ι` side and the public-I/O
//!    segment are evaluated directly by the verifier (Proposition 5,
//!    `O(κ_pos + n_io + log L)`).
//! 4. *The openings* (step 18): the `16 + 3·d_L` retained linear claims
//!    (five selector + five wire + six copy-wire + `3·d_L` digit) split
//!    into a **witness group** (the `3m` commitment; 11 claims) and a
//!    **public group** (the `m_pub` commitment) — per claim the
//!    column-projection message `U^⊤β` (Algorithm 4's step), a fresh
//!    independent `r ←$ E^κ` after all column messages are fixed, the
//!    zero-extended projections summed, and ONE Fact-1 per group reducing
//!    the sum to a single MLE opening — **two PCS openings** total, the
//!    paper's batched-terminal structure (Appendix C.6).
//!
//! The verifier never touches the `O(N)` tables: its work is the
//! `O(κ·log N)` claim algebra, the `O(n_io + d_L)` public-I/O evaluation,
//! and the two PCS verifications.

use crate::copy::{iota_eval_padded, prove_copy, verify_copy, CopyProof};
use crate::fact1::{self, Fact1Proof};
use crate::gf32::Gf32;
use crate::gf8::Gf8;
use crate::hadamard::{
    prove_hadamard_timed, verify_hadamard, HadamardProof, HadamardProverTerminal,
};
use crate::interp::{InterpData, KAPPA};
use crate::pcs_open::compute_u;
use crate::planes::{words_to_planes, PlaneSet, RowWeightTensor};
use crate::r1cs::dot_gf;
use magnetar_field::Gf128;
use magnetar_field::{mul, ONE, ZERO};
use magnetar_hash::Transcript;
use magnetar_pcs::eval::{
    commit_table, prove_eval, verify_eval, PcsParams, PcsProof, PcsStatement,
};

// ---------------------------------------------------------------------------
// The constraint system and its builder
// ---------------------------------------------------------------------------

/// A PLONK circuit over GF(2): the selectors, the wiring permutation, and
/// the public-I/O count (Definition 1). `N = m·κ` gates.
pub struct PlonkCircuit {
    pub n: usize,
    pub q_l: Vec<bool>,
    pub q_r: Vec<bool>,
    pub q_m: Vec<bool>,
    pub q_o: Vec<bool>,
    pub q_c: Vec<bool>,
    /// `sigma[i] = σ(i+1) − 1` — the 0-based wiring permutation over
    /// `[0, L)` with `L = 3N + n_io`.
    pub sigma: Vec<u32>,
    pub n_io: usize,
}

impl PlonkCircuit {
    pub fn l_len(&self) -> usize {
        3 * self.n + self.n_io
    }

    /// `d_L = ⌈log₂ L⌉` (the position-digit count; the padded shape
    /// `L ≤ 4N` gives `d_L = log₂(4N)` for the powers of two used here).
    pub fn d_l(&self) -> usize {
        self.l_len().next_power_of_two().trailing_zeros() as usize
    }

    /// Do the selectors hold on the witness (the gate identity, Eq. (2))?
    pub fn gates_hold(&self, w_l: &[bool], w_r: &[bool], w_o: &[bool]) -> bool {
        for g in 0..self.n {
            let acc = (self.q_l[g] & w_l[g])
                ^ (self.q_r[g] & w_r[g])
                ^ (self.q_m[g] & w_l[g] & w_r[g])
                ^ (self.q_o[g] & w_o[g])
                ^ self.q_c[g];
            if acc {
                return false;
            }
        }
        true
    }

    /// Does the copy constraint hold (`w_i = w_{σ₀(i)}` for all `i < L`)?
    pub fn copies_hold(&self, w: &[bool]) -> bool {
        (0..self.l_len()).all(|i| w[i] == w[self.sigma[i] as usize])
    }
}

/// A wire name (logical variable) of the circuit under construction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Wire(pub u32);

/// The PLONK circuit builder: gates emit their selector rows and their
/// wire cells; wires are logical variables whose cells (gate sides and the
/// public-I/O cells) form the copy cycles of `σ` via union-find.
pub struct PlonkBuilder {
    /// Per gate: the five selector bits.
    sels: Vec<[bool; 5]>,
    /// Per gate: the (left, right, output) wires.
    gate_wires: Vec<(Wire, Wire, Wire)>,
    /// The witness value of every wire.
    wire_val: std::collections::HashMap<Wire, bool>,
    /// The public-I/O wires in declaration order.
    io_wires: Vec<Wire>,
    next_wire: u32,
    /// The gate count target (pads with no-op gates to the power of two).
    pad_to: Option<usize>,
}

impl Default for PlonkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PlonkBuilder {
    pub fn new() -> Self {
        PlonkBuilder {
            sels: Vec::new(),
            gate_wires: Vec::new(),
            wire_val: std::collections::HashMap::new(),
            io_wires: Vec::new(),
            next_wire: 1,
            pad_to: None,
        }
    }

    /// Pad the gate count to exactly `n` (a power of two ≥ κ²) with
    /// no-op gates on fresh unused wires.
    pub fn with_pad_to(mut self, n: usize) -> Self {
        self.pad_to = Some(n);
        self
    }

    fn fresh(&mut self, v: bool) -> Wire {
        let w = Wire(self.next_wire);
        self.next_wire += 1;
        self.wire_val.insert(w, v);
        w
    }

    /// `z = a ⊕ b` — one XOR gate.
    pub fn xor(&mut self, a: Wire, b: Wire) -> Wire {
        let v = self.wire_val[&a] ^ self.wire_val[&b];
        let z = self.fresh(v);
        self.sels.push([true, true, false, true, false]);
        self.gate_wires.push((a, b, z));
        z
    }

    /// `z = a ∧ b` — one AND gate.
    pub fn and(&mut self, a: Wire, b: Wire) -> Wire {
        let v = self.wire_val[&a] & self.wire_val[&b];
        let z = self.fresh(v);
        self.sels.push([false, false, true, true, false]);
        self.gate_wires.push((a, b, z));
        z
    }

    /// A public-I/O input bit (a fresh wire whose io cell is its only
    /// copy anchor until the wire feeds gates).
    pub fn pub_io(&mut self, v: bool) -> Wire {
        let w = self.fresh(v);
        self.io_wires.push(w);
        w
    }

    /// A private witness wire (no io cell — its copy anchors are only the
    /// gate sides that consume it). The rollup circuits' state inputs.
    pub fn witness(&mut self, v: bool) -> Wire {
        self.fresh(v)
    }

    /// Bind an existing wire as a public output: its io cell joins the
    /// wire's copy cycle (the value becomes public and copy-enforced).
    pub fn mark_io(&mut self, w: Wire) {
        self.io_wires.push(w);
    }

    /// Assemble the circuit: pad the gates to the power-of-two `N`, assign
    /// the cells (gate sides [0, 3N), io cells [3N, 3N + n_io)), and build
    /// `σ` as the disjoint cycles of each wire's cells.
    pub fn build(self) -> (PlonkCircuit, PlonkWitness) {
        let n_target = self
            .pad_to
            .unwrap_or_else(|| self.sels.len().max(64).next_power_of_two());
        assert!(n_target.is_power_of_two());
        assert!(
            n_target >= KAPPA * KAPPA,
            "Theorem 1 requires N ≥ κ² = 2^14 (got {n_target})"
        );
        let n_io = self.io_wires.len();
        let mut sels = self.sels.clone();
        let mut gate_wires = self.gate_wires.clone();
        while sels.len() < n_target {
            // a no-op gate: q_O = 1 with a fresh false output
            sels.push([false, false, false, true, false]);
            gate_wires.push((Wire(0), Wire(0), Wire(0)));
        }
        assert_eq!(sels.len(), n_target);
        let n = n_target;
        let l = 3 * n + n_io;

        // the cell assignment: gate g's sides → cells g, n+g, 2n+g;
        // io wire k → cell 3n + k. Each wire collects its cells.
        let mut wire_cells: std::collections::HashMap<Wire, Vec<u32>> =
            std::collections::HashMap::new();
        for (g, &(wl, wr, wo)) in gate_wires.iter().enumerate() {
            wire_cells.entry(wl).or_default().push(g as u32);
            wire_cells.entry(wr).or_default().push((n + g) as u32);
            wire_cells.entry(wo).or_default().push((2 * n + g) as u32);
        }
        for (k, &w) in self.io_wires.iter().enumerate() {
            wire_cells.entry(w).or_default().push((3 * n + k) as u32);
        }

        // σ: each wire's cells form one cycle
        let mut sigma = vec![0u32; l];
        for cs in wire_cells.values() {
            for i in 0..cs.len() {
                sigma[cs[i] as usize] = cs[(i + 1) % cs.len()];
            }
        }
        debug_assert!(
            {
                let mut seen = vec![false; l];
                let mut ok = true;
                for &s in &sigma {
                    if seen[s as usize] {
                        ok = false;
                        break;
                    }
                    seen[s as usize] = true;
                }
                ok
            },
            "σ covers every position exactly once"
        );

        // the witness: the augmented assignment over [0, L)
        let mut w = vec![false; l];
        for (g, &(wl, wr, wo)) in gate_wires.iter().enumerate() {
            w[g] = self.wire_val.get(&wl).copied().unwrap_or(false);
            w[n + g] = self.wire_val.get(&wr).copied().unwrap_or(false);
            w[2 * n + g] = self.wire_val.get(&wo).copied().unwrap_or(false);
        }
        for (k, &wire) in self.io_wires.iter().enumerate() {
            w[3 * n + k] = self.wire_val[&wire];
        }

        let circuit = PlonkCircuit {
            n,
            q_l: sels.iter().map(|s| s[0]).collect(),
            q_r: sels.iter().map(|s| s[1]).collect(),
            q_m: sels.iter().map(|s| s[2]).collect(),
            q_o: sels.iter().map(|s| s[3]).collect(),
            q_c: sels.iter().map(|s| s[4]).collect(),
            sigma,
            n_io,
        };
        (circuit, PlonkWitness { w })
    }
}

/// The satisfying assignment: the augmented vector `w ∈ F^L`.
pub struct PlonkWitness {
    pub w: Vec<bool>,
}

// ---------------------------------------------------------------------------
// The packed layouts
// ---------------------------------------------------------------------------

/// Pack a length-`N` GF(2) vector into the `m × κ` plane layout (plane `j`
/// holds `v[j·m .. (j+1)·m]`, `m = N/κ` — the Algorithm-2 column-major
/// convention).
pub fn pack_planes(v: &[bool], n: usize) -> PlaneSet {
    let m = n / KAPPA;
    assert_eq!(v.len(), n);
    let mut ps = PlaneSet::zeros(m, KAPPA);
    for flat in 0..n {
        if v[flat] {
            let (j, i) = (flat / m, flat % m);
            ps.set_bit(j, i, true);
        }
    }
    ps
}

/// The packed commitment words `Aβ` of a plane set, via the transposed
/// layout (row-major readout of the packed E-words).
fn packed_words(ps: &PlaneSet) -> Vec<Gf128> {
    let m = ps.m;
    let tr = crate::r1cs::transpose_ps(ps);
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let lo = tr.plane(i)[0];
        let hi = tr.plane(i)[1];
        out.push(Gf128((lo as u128) | ((hi as u128) << 64)));
    }
    out
}

// ---------------------------------------------------------------------------
// The terminal-claim bookkeeping (Algorithm 4's interface)
// ---------------------------------------------------------------------------

/// One retained linear claim of the grouped openings:
/// `value = ξ^⊤ A ρ` with `ξ` a two-entry-factor row tensor over `m` rows
/// and `ρ ∈ E^κ` an arbitrary column weight, against the block `block` of
/// a committed group vector.
#[derive(Clone)]
pub struct LinearClaim {
    pub value: Gf128,
    /// The ξ row factors (little-endian, factor `k` ↔ row bit `k`).
    pub xi_factors: Vec<(Gf128, Gf128)>,
    pub rho: Vec<Gf128>,
    /// The block index in the group's committed vector.
    pub block: usize,
}

/// Expand a little-endian two-entry-factor tensor into its E-words.
pub(crate) fn factors_words(factors: &[(Gf128, Gf128)]) -> Vec<Gf128> {
    let mut v = vec![ONE];
    for &(f0, f1) in factors {
        let half = v.len();
        let mut nv = vec![ZERO; half * 2];
        for i in 0..half {
            nv[i] = mul(v[i], f0);
            nv[half + i] = mul(v[i], f1);
        }
        v = nv;
    }
    v
}

/// The little-endian factors of `ts(y)` for a Fact-1 bind-order point `y`
/// (factor `k` ↔ row bit `k`).
pub(crate) fn ts_factors_le(y: &[Gf128]) -> Vec<(Gf128, Gf128)> {
    y.iter().rev().map(|&yk| (ONE + yk, yk)).collect()
}

/// The column-claim + projection step of Algorithm 4, per claim, and the
/// grouped Fact-1 + single PCS opening (Appendix C.6's batched terminals):
///
/// * prover: per claim `U = Ξ^⊤ A` → the message `v_msg = U^⊤ β`; after
///   ALL messages are fixed, per-claim fresh `r ←$ E^κ`; the operand
///   `Ξr` zero-extended into the claim's block; ONE Fact-1 over the group
///   vector against the summed operand; ONE PCS opening of the Fact-1
///   witness terminal.
pub struct GroupOpening {
    /// Per claim: `U^⊤ β` (κ E-elements).
    pub v_msgs: Vec<Vec<Gf128>>,
    pub fact1: Fact1Proof,
    pub pcs: PcsProof,
}

/// The group's committed data (prover-side tables + the statement).
pub struct GroupCommitment {
    pub stmt: PcsStatement,
    /// The concatenated packed words (the Fact-1 vector; zero blocks pad
    /// the block count to a power of two for Fact 1's halving).
    pub table: Vec<Gf128>,
    pub blocks: usize,
    pub m: usize,
}

/// Commit a group: the per-claim matrices packed and concatenated, padded
/// with zero blocks to a power-of-two block count.
pub fn commit_group(params: &PcsParams, mats: &[PlaneSet]) -> GroupCommitment {
    let m = mats[0].m;
    let blocks = mats.len().max(1).next_power_of_two();
    let mut table = Vec::with_capacity(blocks * m);
    for ps in mats {
        table.extend_from_slice(&packed_words(ps));
    }
    let zero = PlaneSet::zeros(m, KAPPA);
    for _ in mats.len()..blocks {
        table.extend_from_slice(&packed_words(&zero));
    }
    let (_, _, stmt) = commit_table(params, &table);
    GroupCommitment {
        stmt,
        table,
        blocks,
        m,
    }
}

/// The prover side of the grouped opening (see [`GroupOpening`]).
fn prove_group_open(
    claims: &[LinearClaim],
    matrices: &[&PlaneSet],
    group: &GroupCommitment,
    params: &PcsParams,
    t: &mut Transcript,
) -> GroupOpening {
    use std::time::Instant;
    let time_on = std::env::var("MIKROS_TIME").is_ok();
    let m = group.m;
    assert_eq!(claims.len(), matrices.len());
    let mut v_msgs = Vec::with_capacity(claims.len());
    let mut us = Vec::new();
    let mut words_cache = Vec::with_capacity(claims.len());
    let t_msgs = Instant::now();
    // the distinct-ξ cache: many claims share one ξ (the 3·d_L digit
    // claims of Algorithm 5 share the copy terminal's ξ_wire) — the words
    // expansion and the plane transpose are computed once per DISTINCT
    // factor list, not once per claim.
    /// A distinct-ξ cache entry: (the factor key, the words, the planes).
    type XiCacheEntry = (Vec<(Gf128, Gf128)>, Vec<Gf128>, Vec<u64>);
    let mut xi_cache: Vec<XiCacheEntry> = Vec::new();
    for (claim, mat) in claims.iter().zip(matrices.iter()) {
        let entry = match xi_cache.iter().position(|(k, _, _)| *k == claim.xi_factors) {
            Some(i) => &xi_cache[i],
            None => {
                let words = factors_words(&claim.xi_factors);
                debug_assert_eq!(words.len(), m);
                let planes = words_to_planes(&words, m);
                xi_cache.push((claim.xi_factors.clone(), words, planes));
                xi_cache.last().expect("just pushed")
            }
        };
        let words = entry.1.clone();
        words_cache.push(words);
        let xi_planes = &entry.2;
        let rows = compute_u(mat, xi_planes, m / 64);
        let mut v_msg = vec![ZERO; KAPPA];
        for j in 0..KAPPA {
            let mut acc = 0u128;
            for c in 0..KAPPA {
                if (rows[c][j / 64] >> (j % 64)) & 1 == 1 {
                    acc |= 1u128 << c;
                }
            }
            v_msg[j] = Gf128(acc);
        }
        t.absorb_field("grp-vmsg", &v_msg);
        debug_assert_eq!(dot_gf(&v_msg, &claim.rho), claim.value);
        v_msgs.push(v_msg);
        us.push(rows);
    }
    // after ALL column messages: the independent per-claim projections
    let t_op = Instant::now();
    let mut rs: Vec<Vec<Gf128>> = Vec::with_capacity(claims.len());
    for _ in 0..claims.len() {
        let r: Vec<Gf128> = (0..KAPPA).map(|_| t.challenge_field("grp-r")).collect();
        rs.push(r);
    }
    // the summed operand: per claim Ξr zero-extended into its block — the
    // ξ-masked weight gather (one bitslice + 128 masked-XOR passes per
    // claim, against the per-row popcount walk)
    let mut operand = vec![ZERO; group.blocks * m];
    let mut c_sum = ZERO;
    for (ci, claim) in claims.iter().enumerate() {
        let r = &rs[ci];
        // c = r^⊤ U β = 〈a, Ξr〉 (per claim)
        let mut c = ZERO;
        for k in 0..KAPPA {
            if r[k] != ZERO {
                let mut acc = ZERO;
                for cj in 0..KAPPA {
                    if (us[ci][k][cj / 64] >> (cj % 64)) & 1 == 1 {
                        acc += Gf128(1u128 << cj);
                    }
                }
                c += mul(acc, r[k]);
            }
        }
        c_sum += c;
        let base = claim.block * m;
        let words = &words_cache[ci];
        for i in 0..m {
            let mut acc = ZERO;
            let mut bits = words[i].0;
            while bits != 0 {
                let cpos = bits.trailing_zeros() as usize;
                acc += r[cpos];
                bits &= bits - 1;
            }
            operand[base + i] += acc;
        }
    }
    let mut av = group.table.clone();
    let mut op = operand;
    let t_f1 = Instant::now();
    let fact1 = fact1::prove(&mut av, &mut op, c_sum, t);
    if time_on {
        eprintln!(
            "group open ({} claims, {} blocks): msgs {:.0} | operand {:.0} | fact1+pcs {:.0} ms",
            claims.len(),
            group.blocks,
            t_msgs.elapsed().as_secs_f64() * 1e3,
            t_op.elapsed().as_secs_f64() * 1e3,
            t_f1.elapsed().as_secs_f64() * 1e3,
        );
    }
    let (v1, _v2) = fact1.terminals;
    let y = fact1.challenges.clone();
    // Fact 1 binds MSB-first; the PCS binds LSB-first — reverse.
    let point: Vec<Gf128> = y.iter().rev().copied().collect();
    let pcs = prove_eval(params, &group.table, &point, v1, t);
    GroupOpening { v_msgs, fact1, pcs }
}

/// Verify a group opening: per claim `v = 〈v_msg, ρ〉` (all column messages
/// fixed first), the per-claim `r` re-derivation, the Fact-1 roundtrip,
/// the public-side terminal via the W-recursion with block-prefix weights,
/// and the PCS opening.
fn verify_group_open(
    claims: &[LinearClaim],
    group: &GroupCommitment,
    params: &PcsParams,
    proof: &GroupOpening,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    if proof.v_msgs.len() != claims.len() {
        return Err("group: claim count");
    }
    // ALL column messages are fixed first (absorbed and checked), then the
    // independent per-claim projections are sampled — the paper's "fix all
    // column messages before sampling independent projection vectors".
    for (ci, claim) in claims.iter().enumerate() {
        let v_msg = &proof.v_msgs[ci];
        if v_msg.len() != KAPPA {
            return Err("group: v_msg length");
        }
        if dot_gf(v_msg, &claim.rho) != claim.value {
            return Err("group: v != <v_msg, rho>");
        }
        t.absorb_field("grp-vmsg", v_msg);
    }
    let mut rs: Vec<Vec<Gf128>> = Vec::with_capacity(claims.len());
    for _ in 0..claims.len() {
        let r: Vec<Gf128> = (0..KAPPA).map(|_| t.challenge_field("grp-r")).collect();
        rs.push(r);
    }
    // the summed claim: per claim c = r^⊤ U β with U recovered from the
    // message (U[k][cj] = bit k of v_msg[cj])
    let mut c_sum = ZERO;
    for (ci, _claim) in claims.iter().enumerate() {
        let v_msg = &proof.v_msgs[ci];
        let r = &rs[ci];
        let mut c = ZERO;
        for k in 0..KAPPA {
            if r[k] != ZERO {
                let mut acc = ZERO;
                for cj in 0..KAPPA {
                    if (v_msg[cj].0 >> k) & 1 == 1 {
                        acc += Gf128(1u128 << cj);
                    }
                }
                c += mul(acc, r[k]);
            }
        }
        c_sum += c;
    }
    let (y, v1, v2) = fact1::verify(c_sum, &proof.fact1, t)?;
    // the public-side terminal: Σ_claims prefix_block(y)·w_recursion(...)
    let log_blocks = group.blocks.trailing_zeros() as usize;
    debug_assert_eq!(y.len(), log_blocks + group.m.trailing_zeros() as usize);
    let mut w_check = ZERO;
    for (ci, claim) in claims.iter().enumerate() {
        // the block-prefix tensor factor: y's FIRST log_blocks coords are
        // the block bits (MSB-first over the group index)
        let mut prefix = ONE;
        for b in 0..log_blocks {
            let coord = y[b];
            let bit = (claim.block >> (log_blocks - 1 - b)) & 1;
            prefix = mul(prefix, if bit == 1 { coord } else { ONE + coord });
        }
        // w_recursion pairs the little-endian ξ factors with the INNER
        // point in little-endian bit order — the Fact-1 bind order is
        // MSB-first, so the inner suffix reverses.
        let y_inner: Vec<Gf128> = y[log_blocks..].iter().rev().copied().collect();
        let w = crate::pcs_open::w_recursion(&claim.xi_factors, &y_inner, &rs[ci]);
        w_check += mul(prefix, w);
    }
    if w_check != v2 {
        return Err("group: W-recursion mismatch");
    }
    // the PCS binds its MLE variables LSB-first — the Fact-1 point reverses
    // (the same convention as batch.rs's openings).
    let point: Vec<Gf128> = y.iter().rev().copied().collect();
    verify_eval(&group.stmt, params, &point, v1, &proof.pcs, t)
}

// ---------------------------------------------------------------------------
// The protocol driver (Algorithm 5)
// ---------------------------------------------------------------------------

/// The preprocessed public data: the packed planes and the two group
/// commitments (the circuit tables are public; the witness planes stay
/// prover-side but are packed here once).
pub struct PlonkSetup {
    pub params: PcsParams,
    pub f8: Gf8,
    pub interp: InterpData,
    pub f32: Gf32,
    pub pos_interp: crate::copy::PosInterp,
    /// The wire planes `[w_L, w_R, w_O]` (m × κ) and the virtual `w_M`.
    pub w_planes: [PlaneSet; 3],
    pub w_m: PlaneSet,
    /// The selector planes `[q_L, q_R, q_M, q_O, q_C]`.
    pub q_planes: [PlaneSet; 5],
    /// The permuted-position digit tables, segment-major:
    /// `digit_planes[s·d_L + j']` = bit `j'` of `σ₀(s·N + ·)`.
    pub digit_planes: Vec<PlaneSet>,
    pub wit_group: GroupCommitment,
    pub pub_group: GroupCommitment,
}

/// Per-phase prover timings (ms).
#[derive(Default, Clone, Copy, Debug)]
pub struct PlonkStats {
    pub gate_ms: f64,
    pub copy_ms: f64,
    pub open_ms: f64,
    pub total_ms: f64,
    /// The Hadamard sub-anatomy (the four gate proofs + the nested one).
    pub hadamard_ms: f64,
}

pub struct PlonkProof {
    /// `(c_L, c_R, c_M, c_O, c_C)` — the gate layer's partial claims.
    pub c_claims: [Gf128; 5],
    /// The four selector–wire Algorithm-2 proofs, order `[L, R, O, M]`.
    pub gate_proofs: [HadamardProof; 4],
    /// The nested Algorithm-2 discharging the virtual `w_M` claim.
    pub nested: HadamardProof,
    /// The copy-constraint permutation proof.
    pub copy: CopyProof,
    /// The six wire values (2 operands × 3 segments) — the recombination
    /// terms of the two copy terminals.
    pub wire_values: [[Gf128; 3]; 2],
    /// The `3·d_L` permuted-digit values (`d_L` entries × 3 segments).
    pub digit_values: Vec<[Gf128; 3]>,
    /// The witness group's batched opening (11 claims).
    pub wit_open: GroupOpening,
    /// The public group's batched opening (5 + 3·d_L claims).
    pub pub_open: GroupOpening,
}

/// Pack the circuit + witness into the proving layout and commit the two
/// groups (the setup-to-online boundary; the Merkle trees build here).
pub fn setup_plonk(
    circuit: &PlonkCircuit,
    witness: &PlonkWitness,
    params: PcsParams,
) -> PlonkSetup {
    let n = circuit.n;
    let m = n / KAPPA;
    assert!(m >= KAPPA, "Theorem 1: m ≥ κ (N ≥ 2^14)");
    assert!(n.is_power_of_two());
    assert_eq!(witness.w.len(), circuit.l_len());

    let w_l = &witness.w[0..n];
    let w_r = &witness.w[n..2 * n];
    let w_o = &witness.w[2 * n..3 * n];
    let w_planes = [
        pack_planes(w_l, n),
        pack_planes(w_r, n),
        pack_planes(w_o, n),
    ];
    let mut w_m = PlaneSet::zeros(m, KAPPA);
    for j in 0..KAPPA {
        w_m.and_into(j, &w_planes[0], &w_planes[1]);
    }
    let q_planes = [
        pack_planes(&circuit.q_l, n),
        pack_planes(&circuit.q_r, n),
        pack_planes(&circuit.q_m, n),
        pack_planes(&circuit.q_o, n),
        pack_planes(&circuit.q_c, n),
    ];
    let d_l = circuit.d_l();
    let mut digit_planes = Vec::with_capacity(3 * d_l);
    for s in 0..3 {
        for jp in 0..d_l {
            let table: Vec<bool> = (0..n)
                .map(|r| (circuit.sigma[s * n + r] >> jp) & 1 == 1)
                .collect();
            digit_planes.push(pack_planes(&table, n));
        }
    }

    let f8 = Gf8::new();
    let mut interp = InterpData::new(&f8);
    interp.build_psi_monomials(&f8);
    let f32 = Gf32::new();
    let pos_interp = crate::copy::PosInterp::new(&f32);

    let wit_group = commit_group(&params, &w_planes);
    let mut pub_mats: Vec<PlaneSet> = q_planes.to_vec();
    pub_mats.extend(digit_planes.iter().cloned());
    let pub_group = commit_group(&params, &pub_mats);

    PlonkSetup {
        params,
        f8,
        interp,
        f32,
        pos_interp,
        w_planes,
        w_m,
        q_planes,
        digit_planes,
        wit_group,
        pub_group,
    }
}

/// The bitsliced column-claim fold `d[j] = 〈l, A_j〉` (the Hadamard
/// step-1 shape) and the full claim `c = 〈d, r〉`.
fn gate_claim(a: &PlaneSet, l: &RowWeightTensor, r: &[Gf128]) -> (Vec<Gf128>, Gf128) {
    let mut d = Vec::with_capacity(KAPPA);
    for j in 0..KAPPA {
        let pj = a.plane(j);
        let mut bits = 0u128;
        for e in 0..128 {
            bits |= (PlaneSet::parity_dot(pj, l.plane(e)) as u128) << e;
        }
        d.push(Gf128(bits));
    }
    let c = dot_gf(&d, r);
    (d, c)
}

/// The truncated tensor expansion: `l[r]` for `r < keep` (the public-I/O
/// prefix of a terminal row weight).
fn l_prefix(factors: &[(Gf128, Gf128)], keep: usize) -> Vec<Gf128> {
    let mut l_vals = vec![ONE];
    for &(f0, f1) in factors {
        let half = l_vals.len();
        let k = keep.min(half * 2);
        let mut next = vec![ZERO; k];
        for i in 0..k.min(half) {
            next[i] = mul(l_vals[i], f0);
        }
        for i in half..k {
            next[i] = mul(l_vals[i - half], f1);
        }
        l_vals = next;
    }
    l_vals
}

/// An empty Hadamard proof (the array-assembly mover's placeholder).
fn hadamard_dummy() -> HadamardProof {
    HadamardProof {
        d: Vec::new(),
        h: Vec::new(),
        fact1: Fact1Proof {
            rounds: Vec::new(),
            terminals: (ZERO, ZERO),
            challenges: Vec::new(),
        },
    }
}

/// The packed E-word of row `i` of a plane set (its β-combination) — the
/// per-row readout of the transposed layout (the expansion phase now
/// hoists the whole-plane transpose instead).
#[allow(dead_code)]
fn packed_row_word(ps: &PlaneSet, i: usize) -> Gf128 {
    let mut acc = 0u128;
    for j in 0..KAPPA {
        if ps.bit(j, i) {
            acc |= 1u128 << j;
        }
    }
    Gf128(acc)
}

/// Prove (Algorithm 5). The caller creates the transcript; the io values
/// are absorbed first (the public-input binding).
pub fn prove_plonk(
    circuit: &PlonkCircuit,
    witness: &PlonkWitness,
    setup: &PlonkSetup,
    t: &mut Transcript,
) -> (PlonkProof, PlonkStats) {
    use std::time::Instant;
    let t_all = Instant::now();
    let mut stats = PlonkStats::default();
    let n = circuit.n;
    let m = n / KAPPA;
    let log_m = m.trailing_zeros() as usize;
    let log_n = n.trailing_zeros() as usize;
    let d_l = circuit.d_l();
    let l_len = circuit.l_len();
    let n_io = circuit.n_io;
    assert!(l_len <= 4 * n, "L = 3N + n_io ≤ 4N");

    // ---- the public-input binding.
    let io_words: Vec<Gf128> = witness.w[3 * n..3 * n + n_io]
        .iter()
        .map(|&b| if b { ONE } else { ZERO })
        .collect();
    t.absorb_field("plonk-io", &io_words);

    // ---- step 5: x ←$ E^{log₂ N}; ξ = ts(x) factored into (l, r).
    let x: Vec<Gf128> = (0..log_n).map(|_| t.challenge_field("plonk-x")).collect();
    // flat gate index g = c·m + i: the top 7 coords are the column bits
    // (MSB-first), the rest the row bits.
    let col_coords = &x[..7];
    let row_coords = &x[7..];
    let l_factors: Vec<(Gf128, Gf128)> = (0..log_m)
        .map(|k| {
            let yk = row_coords[log_m - 1 - k];
            (ONE + yk, yk)
        })
        .collect();
    let l = RowWeightTensor::new(l_factors.clone(), true);
    let col_factors: Vec<(Gf128, Gf128)> = (0..7)
        .map(|a| (ONE + col_coords[6 - a], col_coords[6 - a]))
        .collect();
    let r_vec = factors_words(&col_factors);

    // ---- steps 6–7: the five partial claims (Eq. 12–13). The first four
    // are Hadamard claims `c⋆ = 〈ξ, q⋆ ∘ w⋆〉` (the product planes folded
    // bitsliced against l, then dotted with r); c_C is the constant
    // selector's plain evaluation.
    let mut prod = PlaneSet::zeros(m, KAPPA);
    let mut claim_of = |qa: &PlaneSet, wa: &PlaneSet| -> Gf128 {
        let words = prod.words;
        for j in 0..KAPPA {
            // prod_j = qa_j ∧ wa_j (fresh, not accumulated)
            let pj = prod.plane_mut(j);
            let sq = qa.plane(j);
            let sw = wa.plane(j);
            for k in 0..words {
                pj[k] = sq[k] & sw[k];
            }
        }
        let (_, c) = gate_claim(&prod, &l, &r_vec);
        c
    };
    let c_l = claim_of(&setup.q_planes[0], &setup.w_planes[0]);
    let c_r = claim_of(&setup.q_planes[1], &setup.w_planes[1]);
    let c_o = claim_of(&setup.q_planes[3], &setup.w_planes[2]);
    let c_m = claim_of(&setup.q_planes[2], &setup.w_m);
    let (_, c_c) = gate_claim(&setup.q_planes[4], &l, &r_vec);
    debug_assert_eq!(
        c_l + c_r + c_m + c_o + c_c,
        ZERO,
        "the gate residual sums to zero"
    );
    let c_claims = [c_l, c_r, c_m, c_o, c_c];
    t.absorb_field("plonk-c", &c_claims);

    // ---- steps 8–12: the four Algorithm-2 reductions + the nested w_M.
    let t_gate = Instant::now();
    // the operand order is (L, R, O, M) with the matching claims
    // (c_l, c_r, c_o, c_m) — c_claims stores (c_l, c_r, c_m, c_o, c_c).
    let operands = [
        (&setup.q_planes[0], &setup.w_planes[0], c_claims[0]), // L
        (&setup.q_planes[1], &setup.w_planes[1], c_claims[1]), // R
        (&setup.q_planes[3], &setup.w_planes[2], c_claims[3]), // O
        (&setup.q_planes[2], &setup.w_m, c_claims[2]),         // M
    ];
    let mut gate_proofs: Vec<HadamardProof> = Vec::with_capacity(4);
    let mut gate_terms: Vec<HadamardProverTerminal> = Vec::with_capacity(4);
    let mut hadamard_ms = 0.0f64;
    for (q, w, claim) in operands.iter() {
        let (st, proof, term) =
            prove_hadamard_timed(&setup.f8, &setup.interp, q, w, &l, &r_vec, *claim, t);
        hadamard_ms += st.total_ms;
        gate_proofs.push(proof);
        gate_terms.push(term);
    }
    // the nested proof: the w_M terminal's product-vector claim
    // v = l₂^⊤ (W_L ∘ W_R) ρ with l₂ = l∘ts(y) (the ξ2 factors)
    let w_m_term = &gate_terms[3];
    let y_m = w_m_term.y.clone();
    let y_m_rev: Vec<Gf128> = y_m.iter().rev().copied().collect();
    let nested_l_factors = crate::hadamard::xi2_factors(&l, &y_m_rev);
    let nested_l = RowWeightTensor::new(nested_l_factors.clone(), true);
    let (st, nested, nested_term) = prove_hadamard_timed(
        &setup.f8,
        &setup.interp,
        &setup.w_planes[0],
        &setup.w_planes[1],
        &nested_l,
        &w_m_term.rho,
        w_m_term.v2,
        t,
    );
    hadamard_ms += st.total_ms;
    stats.hadamard_ms = hadamard_ms;
    stats.gate_ms = t_gate.elapsed().as_secs_f64() * 1e3;

    // ---- steps 13–14: the copy layer.
    let t_copy = Instant::now();
    let theta32 = setup.f32.embed(1u32 << 31);
    let t_code = Instant::now();
    let mut code_a = vec![0u32; 4 * n];
    let mut code_b = vec![0u32; 4 * n];
    for flat in 0..l_len {
        let w_bit = witness.w[flat] as u32;
        code_a[flat] = circuit.sigma[flat] | (w_bit << 31);
        code_b[flat] = (flat as u32) | (w_bit << 31);
    }
    if std::env::var("MIKROS_TIME").is_ok() {
        eprintln!(
            "  code build: {:.0} ms (L = {l_len})",
            t_code.elapsed().as_secs_f64() * 1e3
        );
    }
    let (copy, copy_terminal) = prove_copy(
        &setup.f32,
        &setup.pos_interp,
        &code_a,
        &code_b,
        n, // m_pos = 4N/κ_pos = N
        t,
    );
    stats.copy_ms = t_copy.elapsed().as_secs_f64() * 1e3;
    let t_expand = Instant::now();

    // ---- steps 15–16: expand the two copy terminals.
    let mut wire_values = [[ZERO; 3]; 2];
    let mut digit_values: Vec<[Gf128; 3]> = vec![[ZERO; 3]; d_l];
    let mut wit_claims: Vec<LinearClaim> = Vec::new();
    let mut wit_mats: Vec<&PlaneSet> = Vec::new();
    let mut pub_claims: Vec<LinearClaim> = Vec::new();
    let mut pub_mats: Vec<&PlaneSet> = Vec::new();

    // the gate layer's three wire claims (the L/R/O proofs' second
    // terminals; their ξ2 factors are l∘ts(y))
    for (star, block) in [(0usize, 0usize), (1, 1), (2, 2)] {
        let term = &gate_terms[star];
        let y_rev: Vec<Gf128> = term.y.iter().rev().copied().collect();
        let xi2 = crate::hadamard::xi2_factors(&l, &y_rev);
        wit_claims.push(LinearClaim {
            value: term.v2,
            xi_factors: xi2,
            rho: term.rho.clone(),
            block,
        });
        wit_mats.push(&setup.w_planes[block]);
    }
    // the nested proof's two wire claims (w_L and w_R)
    {
        let y_rev: Vec<Gf128> = nested_term.y.iter().rev().copied().collect();
        wit_claims.push(LinearClaim {
            value: nested_term.v1,
            xi_factors: ts_factors_le(&nested_term.y),
            rho: nested_term.rho.clone(),
            block: 0,
        });
        let xi2n = crate::hadamard::xi2_factors(&nested_l, &y_rev);
        wit_claims.push(LinearClaim {
            value: nested_term.v2,
            xi_factors: xi2n,
            rho: nested_term.rho.clone(),
            block: 1,
        });
        wit_mats.push(&setup.w_planes[0]);
        wit_mats.push(&setup.w_planes[1]);
    }
    // the selector claims: the proofs' order (L, R, O, M) → selector
    // blocks (0, 1, 3, 2) (the commit order is [q_L, q_R, q_M, q_O, q_C])
    for (star, block) in [(0usize, 0usize), (1, 1), (2, 3), (3, 2)] {
        let term = &gate_terms[star];
        pub_claims.push(LinearClaim {
            value: term.v1,
            xi_factors: ts_factors_le(&term.y),
            rho: term.rho.clone(),
            block,
        });
        pub_mats.push(&setup.q_planes[block]);
    }
    pub_claims.push(LinearClaim {
        value: c_c,
        xi_factors: l_factors.clone(),
        rho: r_vec.clone(),
        block: 4,
    });
    pub_mats.push(&setup.q_planes[4]);

    // the β-packed row words of every plane — ONE transpose per plane,
    // hoisted out of the op loop (the packed_row_word walk was the
    // expansion phase's cost)
    let t_tr = std::time::Instant::now();
    let w_rows: Vec<Vec<Gf128>> = (0..3).map(|s| packed_words(&setup.w_planes[s])).collect();
    let digit_rows: Vec<Vec<Gf128>> = (0..3 * d_l)
        .map(|k| packed_words(&setup.digit_planes[k]))
        .collect();
    let ms_tr = t_tr.elapsed().as_secs_f64() * 1e3;
    let mut rw = vec![ZERO; m];
    for op in 0..2 {
        let (claim_v, l0_factors, rho4) = &copy_terminal.claims[op];
        let _ = claim_v;
        // l₀'s factor split: the low log_m factors are the wire rows, the
        // high 7 the wire columns
        let xi_wire: Vec<(Gf128, Gf128)> = l0_factors[..log_m].to_vec();
        let col_wire: Vec<(Gf128, Gf128)> = l0_factors[log_m..].to_vec();
        let rho_tilde = factors_words(&col_wire);
        // the ξ row weights: identical across the s/jp loops — expanded once
        let xi_words = factors_words(&xi_wire);
        for s in 0..3 {
            // wire_val = θ·ρ_s·(ξ^⊤ W_s ρ̃) — as a claim with the scaled
            // column weight θ·ρ_s·ρ̃: rw = W_s rows ⋅ col_weight (one masked
            // gather), then the m-length dot with the ξ row weights.
            let col_weight: Vec<Gf128> = rho_tilde
                .iter()
                .map(|&rv| mul(mul(theta32, rho4[s]), rv))
                .collect();
            for (i, v) in rw.iter_mut().enumerate() {
                *v = ZERO;
                let mut bits = w_rows[s][i].0;
                while bits != 0 {
                    let cpos = bits.trailing_zeros() as usize;
                    *v += col_weight[cpos];
                    bits &= bits - 1;
                }
            }
            let val = magnetar_field::avx::dot_vv(&xi_words, &rw, m);
            wire_values[op][s] = val;
            wit_claims.push(LinearClaim {
                value: val,
                xi_factors: xi_wire.clone(),
                rho: col_weight,
                block: s,
            });
            wit_mats.push(&setup.w_planes[s]);
        }
        // the digit claims (the ς side — operand 0)
        if op == 0 {
            for jp in 0..d_l {
                for s in 0..3 {
                    let col_weight: Vec<Gf128> =
                        rho_tilde.iter().map(|&rv| mul(rho4[s], rv)).collect();
                    for (i, v) in rw.iter_mut().enumerate() {
                        *v = ZERO;
                        let mut bits = digit_rows[s * d_l + jp][i].0;
                        while bits != 0 {
                            let cpos = bits.trailing_zeros() as usize;
                            *v += col_weight[cpos];
                            bits &= bits - 1;
                        }
                    }
                    let val = magnetar_field::avx::dot_vv(&xi_words, &rw, m);
                    digit_values[jp][s] = val;
                    pub_claims.push(LinearClaim {
                        value: val,
                        xi_factors: xi_wire.clone(),
                        rho: col_weight,
                        block: 5 + s * d_l + jp,
                    });
                    pub_mats.push(&setup.digit_planes[s * d_l + jp]);
                }
            }
        }
    }
    drop((w_rows, digit_rows, rw));
    if std::env::var("MIKROS_TIME").is_ok() {
        eprintln!("  expand detail: transposes {ms_tr:.0} ms");
    }
    t.absorb_field("plonk-wv", &{
        let mut v = Vec::new();
        for op in 0..2 {
            v.extend_from_slice(&wire_values[op]);
        }
        v
    });
    let digit_flat: Vec<Gf128> = digit_values
        .iter()
        .flat_map(|s| s.iter().copied())
        .collect();
    t.absorb_field("plonk-dv", &digit_flat);
    if std::env::var("MIKROS_TIME").is_ok() {
        eprintln!(
            "expand phases: {:.0} ms",
            t_expand.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- step 18: the two grouped openings.
    let t_open = Instant::now();
    let wit_open = prove_group_open(&wit_claims, &wit_mats, &setup.wit_group, &setup.params, t);
    let pub_open = prove_group_open(&pub_claims, &pub_mats, &setup.pub_group, &setup.params, t);
    stats.open_ms = t_open.elapsed().as_secs_f64() * 1e3;

    stats.total_ms = t_all.elapsed().as_secs_f64() * 1e3;
    let gate_proofs_arr: [HadamardProof; 4] = match gate_proofs.len() {
        4 => [
            std::mem::replace(&mut gate_proofs[0], hadamard_dummy()),
            std::mem::replace(&mut gate_proofs[1], hadamard_dummy()),
            std::mem::replace(&mut gate_proofs[2], hadamard_dummy()),
            std::mem::replace(&mut gate_proofs[3], hadamard_dummy()),
        ],
        _ => unreachable!("four gate proofs"),
    };
    (
        PlonkProof {
            c_claims,
            gate_proofs: gate_proofs_arr,
            nested,
            copy,
            wire_values,
            digit_values,
            wit_open,
            pub_open,
        },
        stats,
    )
}

/// Verify (Algorithm 5). `io_values` are the public input/output bits.
pub fn verify_plonk(
    circuit: &PlonkCircuit,
    setup: &PlonkSetup,
    io_values: &[bool],
    proof: &PlonkProof,
    t: &mut Transcript,
) -> Result<(), &'static str> {
    let n = circuit.n;
    let m = n / KAPPA;
    let log_m = m.trailing_zeros() as usize;
    let log_n = n.trailing_zeros() as usize;
    let d_l = circuit.d_l();
    let n_io = circuit.n_io;
    assert_eq!(io_values.len(), n_io, "the public-I/O arity");

    // ---- the public-input binding.
    let io_words: Vec<Gf128> = io_values
        .iter()
        .map(|&b| if b { ONE } else { ZERO })
        .collect();
    t.absorb_field("plonk-io", &io_words);

    // ---- the gate challenges.
    let x: Vec<Gf128> = (0..log_n).map(|_| t.challenge_field("plonk-x")).collect();
    let col_coords = &x[..7];
    let row_coords = &x[7..];
    let l_factors: Vec<(Gf128, Gf128)> = (0..log_m)
        .map(|k| {
            let yk = row_coords[log_m - 1 - k];
            (ONE + yk, yk)
        })
        .collect();
    let col_factors: Vec<(Gf128, Gf128)> = (0..7)
        .map(|a| (ONE + col_coords[6 - a], col_coords[6 - a]))
        .collect();
    let r_vec = factors_words(&col_factors);

    // ---- the gate claims and their sum.
    let c_claims = &proof.c_claims;
    let sum = c_claims.iter().fold(ZERO, |acc, &v| acc + v);
    if sum != ZERO {
        return Err("plonk: c_L + c_R + c_M + c_O + c_C ≠ 0");
    }
    t.absorb_field("plonk-c", c_claims);

    // ---- the four Algorithm-2 verifications (selector + wire terminals).
    // gate_proofs order: [L, R, O, M] with claims [c_l, c_r, c_o, c_m].
    let claim_order = [c_claims[0], c_claims[1], c_claims[3], c_claims[2]];
    let mut terms = Vec::with_capacity(4);
    for (star, &claim) in claim_order.iter().enumerate() {
        let term = verify_hadamard(
            &setup.f8,
            &setup.interp,
            &r_vec,
            claim,
            &proof.gate_proofs[star],
            t,
        )?;
        terms.push(term);
    }
    // the nested w_M proof: its claim is the M-proof's second terminal
    // (the row weight l∘ts(y) is reconstructed from the parent's y).
    let w_m_term = &terms[3];
    let y_m = w_m_term.y.clone();
    let y_m_rev: Vec<Gf128> = y_m.iter().rev().copied().collect();
    let nested_l_factors = crate::hadamard::xi2_factors_from(&l_factors, &y_m_rev);
    let nested_term = verify_hadamard(
        &setup.f8,
        &setup.interp,
        &w_m_term.rho,
        w_m_term.v2,
        &proof.nested,
        t,
    )?;
    let _ = &nested_l_factors;

    // ---- the copy layer.
    let copy_terminal = verify_copy(&setup.f32, &setup.pos_interp, n, &proof.copy, t)?;

    // ---- the recombinations (step 17): both copy terminals.
    t.absorb_field("plonk-wv", &{
        let mut v = Vec::new();
        for op in 0..2 {
            v.extend_from_slice(&proof.wire_values[op]);
        }
        v
    });
    let digit_flat: Vec<Gf128> = proof
        .digit_values
        .iter()
        .flat_map(|s| s.iter().copied())
        .collect();
    t.absorb_field("plonk-dv", &digit_flat);
    if proof.digit_values.len() != d_l {
        return Err("plonk: digit value count");
    }

    let theta32 = setup.f32.embed(1u32 << 31);
    let theta = |jp: usize| -> Gf128 { setup.f32.embed(1u32 << jp) };

    for op in 0..2 {
        let (claim_v, l0_factors, rho4) = &copy_terminal.claims[op];
        // the io parts (verifier-direct): the public-I/O prefix of l₀
        // against column 3's ρ
        let l_pre = l_prefix(l0_factors, n_io);
        let mut io_wire = ZERO;
        let mut io_digit = vec![ZERO; d_l];
        for (r, &lv) in l_pre.iter().enumerate() {
            io_wire += mul(lv, if io_values[r] { ONE } else { ZERO });
            for jp in 0..d_l {
                let bit = (circuit.sigma[3 * n + r] >> jp) & 1 == 1;
                if bit {
                    io_digit[jp] += lv;
                }
            }
        }
        io_wire = mul(mul(theta32, rho4[3]), io_wire);
        for jp in 0..d_l {
            io_digit[jp] = mul(rho4[3], io_digit[jp]);
        }
        // the wire part (same structure both operands; the values differ)
        let wire_part = proof.wire_values[op].iter().fold(ZERO, |acc, &v| acc + v) + io_wire;
        // operand a adds the ς digit part; operand b the ι evaluation
        if op == 0 {
            let mut vars_part = ZERO;
            for jp in 0..d_l {
                let mut acc = io_digit[jp];
                for s in 0..3 {
                    acc += proof.digit_values[jp][s];
                }
                vars_part += mul(theta(jp), acc);
            }
            if *claim_v != vars_part + wire_part {
                return Err("plonk: the ς-side recombination failed");
            }
        } else {
            let iota = iota_eval_padded(&setup.f32, l0_factors, rho4, n_io);
            if *claim_v != iota + wire_part {
                return Err("plonk: the ι-side recombination failed");
            }
        }
    }

    // ---- step 18: the two grouped openings.
    // the witness group's 11 claims
    let mut wit_claims: Vec<LinearClaim> = Vec::new();
    for (star, block) in [(0usize, 0usize), (1, 1), (2, 2)] {
        let term = &terms[star];
        let y_rev: Vec<Gf128> = term.y.iter().rev().copied().collect();
        let xi2 = crate::hadamard::xi2_factors_from(&l_factors, &y_rev);
        wit_claims.push(LinearClaim {
            value: term.v2,
            xi_factors: xi2,
            rho: term.rho.clone(),
            block,
        });
    }
    {
        let y_rev: Vec<Gf128> = nested_term.y.iter().rev().copied().collect();
        wit_claims.push(LinearClaim {
            value: nested_term.v1,
            xi_factors: ts_factors_le(&nested_term.y),
            rho: nested_term.rho.clone(),
            block: 0,
        });
        let xi2n = crate::hadamard::xi2_factors_from(&nested_l_factors, &y_rev);
        wit_claims.push(LinearClaim {
            value: nested_term.v2,
            xi_factors: xi2n,
            rho: nested_term.rho.clone(),
            block: 1,
        });
    }
    // the six copy wire claims (values from the proof's wire_values)
    for op in 0..2 {
        let (_, l0_factors, rho4) = &copy_terminal.claims[op];
        let xi_wire: Vec<(Gf128, Gf128)> = l0_factors[..log_m].to_vec();
        let col_wire: Vec<(Gf128, Gf128)> = l0_factors[log_m..].to_vec();
        let rho_tilde = factors_words(&col_wire);
        for s in 0..3 {
            let col_weight: Vec<Gf128> = rho_tilde
                .iter()
                .map(|&rv| mul(mul(theta32, rho4[s]), rv))
                .collect();
            wit_claims.push(LinearClaim {
                value: proof.wire_values[op][s],
                xi_factors: xi_wire.clone(),
                rho: col_weight,
                block: s,
            });
        }
    }
    verify_group_open(
        &wit_claims,
        &setup.wit_group,
        &setup.params,
        &proof.wit_open,
        t,
    )?;

    // the public group: 5 selector claims + 3·d_L digit claims
    let mut pub_claims: Vec<LinearClaim> = Vec::new();
    // the proofs' order (L, R, O, M) → selector blocks (0, 1, 3, 2)
    for (star, block) in [(0usize, 0usize), (1, 1), (2, 3), (3, 2)] {
        let term = &terms[star];
        pub_claims.push(LinearClaim {
            value: term.v1,
            xi_factors: ts_factors_le(&term.y),
            rho: term.rho.clone(),
            block,
        });
    }
    pub_claims.push(LinearClaim {
        value: c_claims[4],
        xi_factors: l_factors.clone(),
        rho: r_vec.clone(),
        block: 4,
    });
    {
        let (_, l0_factors, rho4) = &copy_terminal.claims[0];
        let xi_wire: Vec<(Gf128, Gf128)> = l0_factors[..log_m].to_vec();
        let col_wire: Vec<(Gf128, Gf128)> = l0_factors[log_m..].to_vec();
        let rho_tilde = factors_words(&col_wire);
        for jp in 0..d_l {
            for s in 0..3 {
                let col_weight: Vec<Gf128> = rho_tilde.iter().map(|&rv| mul(rho4[s], rv)).collect();
                pub_claims.push(LinearClaim {
                    value: proof.digit_values[jp][s],
                    xi_factors: xi_wire.clone(),
                    rho: col_weight,
                    block: 5 + s * d_l + jp,
                });
            }
        }
    }
    verify_group_open(
        &pub_claims,
        &setup.pub_group,
        &setup.params,
        &proof.pub_open,
        t,
    )
}

// ---------------------------------------------------------------------------
// The Table-5 circuit family (a simulated binary circuit) and the tests
// ---------------------------------------------------------------------------

/// A deterministic simulated binary circuit (the paper's Table-5 workload):
/// `n_io` public input bits, then random XOR / AND gates over prior wires,
/// with the final gate's value bound as the public output. Every wire used
/// more than once forms a copy cycle (the wiring permutation σ).
pub fn simulated_circuit(log_n: usize, seed: u64) -> (PlonkCircuit, PlonkWitness) {
    let n = 1usize << log_n;
    assert!(n >= KAPPA * KAPPA);
    let mut rng = seed;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut b = PlonkBuilder::new().with_pad_to(n);
    let n_io = 64usize.min(n / 4);
    let mut io_wires = Vec::with_capacity(n_io + 1);
    for _ in 0..n_io {
        io_wires.push(b.pub_io(next() & 1 == 1));
    }
    let mut pool: Vec<Wire> = io_wires.clone();
    // the gate budget: reserve one gate for the output binding
    let gates = n - 1;
    let mut created = 0usize;
    while created < gates {
        let take = pool.len();
        let a = pool[(next() as usize) % take];
        let be = pool[(next() as usize) % take];
        let z = if next() & 1 == 1 {
            b.xor(a, be)
        } else {
            b.and(a, be)
        };
        pool.push(z);
        created += 1;
        // keep the pool bounded (the wiring fan-out stays O(1) on average)
        if pool.len() > 4 * n_io + 64 {
            let cut = (next() as usize) % (pool.len() - 1);
            pool.remove(cut);
        }
    }
    // the public output: the last gate's wire is marked public (its io
    // cell joins the wire's copy cycle)
    let last = *pool.last().expect("gates created");
    b.mark_io(last);
    let (circuit, witness) = b.build();
    (circuit, witness)
}

// ---------------------------------------------------------------------------
// The ERC-20 transfer circuit (the state-transition core)
// ---------------------------------------------------------------------------

/// The per-transfer gate shape of [`erc20_circuit`]: the two 256-bit
/// carry-save adders (the debit `b_s' = b_s − amount` as `b_s + ¬amount + 1`,
/// the credit `b_r' = b_r + amount` with uint256 wrap-around semantics) and
/// the `¬amount` materialization.
pub const ERC20_ROWS_PER_TRANSFER: usize = 256 * (2 * 5 + 1);

/// One 256-bit ripple-carry adder over the builder's bit gates:
/// `sum = x + y + cin` with the carry chain `c_i = x_i·y_i ⊕ (x_i⊕y_i)·c_{i-1}`
/// (5 rows per bit; the `x⊕y` row is shared by the carry and sum terms).
/// The 256-bit little-endian wire materialization of a four-lane value.
fn wires256(b: &mut PlonkBuilder, v: &[u64; 4]) -> Vec<Wire> {
    let mut out = Vec::with_capacity(256);
    for w in 0..4 {
        for i in 0..64 {
            out.push(b.witness((v[w] >> i) & 1 == 1));
        }
    }
    out
}

fn adder256(b: &mut PlonkBuilder, x: &[Wire], y: &[Wire], cin: Wire) -> (Vec<Wire>, Wire) {
    assert_eq!(x.len(), 256);
    assert_eq!(y.len(), 256);
    let mut sum = Vec::with_capacity(256);
    let mut carry = cin;
    for i in 0..256 {
        let xy = b.xor(x[i], y[i]);
        let t = b.and(x[i], y[i]);
        let uc = b.and(xy, carry);
        let c1 = b.xor(t, uc);
        let s = b.xor(xy, carry);
        sum.push(s);
        carry = c1;
    }
    (sum, carry)
}

/// An ERC-20 transfer batch state-transition circuit: `transfers`
/// independent `transfer(sender, recipient, amount)` applications of the
/// standard token semantics over uint256:
///
/// * **debit** `b_s' = b_s − amount` — computed as `b_s + ¬amount + 1`
///   (two's complement); the adder's carry-out is 1 exactly when
///   `b_s ≥ amount`, and is bound as a public io bit (the no-underflow
///   assertion — the witness satisfies it by construction);
/// * **credit** `b_r' = b_r + amount` — plain uint256 addition (the
///   overflow wraps: Solidity's `balance[to] += amount` does not check);
/// * the `amount` bits are public io (the rollup's public calldata); the
///   balances stay private witness (the state-transition core — the
///   storage-root binding is the documented follow-up).
///
/// Returns the circuit, the witness, and the number of transfers actually
/// placed (the caller pads to the power-of-two gate count).
pub fn erc20_circuit(
    transfers: usize,
    pad_to: usize,
    seed: u64,
) -> (PlonkCircuit, PlonkWitness, usize) {
    assert!(transfers >= 1, "at least one transfer");
    assert!(pad_to.is_power_of_two() && pad_to >= KAPPA * KAPPA);
    assert!(
        pad_to >= transfers * ERC20_ROWS_PER_TRANSFER,
        "the pad target must hold every transfer's rows"
    );
    let mut rng = seed;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut b = PlonkBuilder::new().with_pad_to(pad_to);
    let one = b.witness(true); // the constant-1 wire (the ⊕1 of ¬amount)
    let zero = b.witness(false); // the add-side carry-in
    let mut placed = 0usize;
    for _ in 0..transfers {
        // the private state: b_s, b_r, amount ∈ {0..2^256}, amount ≤ b_s
        let bs: [u64; 4] = [next(), next(), next(), next()];
        let br: [u64; 4] = [next(), next(), next(), next()];
        // amount ≤ b_s: the 256-bit AND `amount = b_s & mask` (a bitwise
        // AND is ≤ its operand, so the debit never underflows)
        let mask: [u64; 4] = [next(), next(), next(), next()];
        let amt: [u64; 4] = [
            bs[0] & mask[0],
            bs[1] & mask[1],
            bs[2] & mask[2],
            bs[3] & mask[3],
        ];
        let bs_w = wires256(&mut b, &bs);
        let br_w = wires256(&mut b, &br);
        let amt_w = wires256(&mut b, &amt);
        // the public calldata: the amount bits (io)
        for w in &amt_w {
            b.mark_io(*w);
        }
        // ¬amount (the two's complement operand)
        let not_amt: Vec<Wire> = amt_w.iter().map(|w| b.xor(*w, one)).collect();
        // debit: b_s + ¬amount + 1 — cin = the constant-1 wire
        let (_bs_out, cout_debit) = adder256(&mut b, &bs_w, &not_amt, one);
        // the no-underflow bit is public (carry-out = b_s ≥ amount)
        b.mark_io(cout_debit);
        // credit: b_r + amount (uint256 wrap)
        let (br_out, _cout_credit) = adder256(&mut b, &br_w, &amt_w, zero);
        let _ = br_out;
        placed += 1;
    }
    let (circuit, witness) = b.build();
    (circuit, witness, placed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnetar_hash::Transcript;

    /// A small simulated circuit at the minimum Theorem-1 size (N = 2^14).
    fn build_test_circuit() -> (PlonkCircuit, PlonkWitness, Vec<bool>) {
        let (circuit, witness) = simulated_circuit(14, 0x5eed_1234);
        let io: Vec<bool> = witness.w[3 * circuit.n..3 * circuit.n + circuit.n_io].to_vec();
        (circuit, witness, io)
    }

    #[test]
    fn erc20_roundtrip_and_tamper() {
        use crate::plonk::erc20_circuit;
        // 4 transfers at the Theorem-1 minimum pad 2^14: 4×2816 = 11264 rows
        let (circuit, witness, placed) = erc20_circuit(4, 1 << 14, 7);
        assert_eq!(placed, 4);
        let n = circuit.n;
        assert_eq!(n, 1 << 14);
        // honest: gates + copy + the no-underflow io bits are all true
        assert!(circuit.gates_hold(
            &witness.w[0..n],
            &witness.w[n..2 * n],
            &witness.w[2 * n..3 * n]
        ));
        assert!(circuit.copies_hold(&witness.w));
        // every transfer's carry-out io bit is 1 (amount ≤ balance by
        // construction) — the io cells sit at [3n, 3n + n_io)
        let io: Vec<bool> = witness.w[3 * n..3 * n + circuit.n_io].to_vec();
        assert_eq!(io.len(), 4 * 257);
        assert!(io.iter().enumerate().all(|(k, &b)| (k % 257) != 256 || b));

        let params = PcsParams::default();
        let setup = setup_plonk(&circuit, &witness, params);
        let base = || Transcript::new("mikros-erc20-test", &[]);
        let mut t = base();
        let (proof, _stats) = prove_plonk(&circuit, &witness, &setup, &mut t);
        let mut tv = base();
        assert!(verify_plonk(&circuit, &setup, &io, &proof, &mut tv).is_ok());
        // tamper: flip one carry-out io bit (an underflowed transfer) — the
        // io vector is part of the verifier's statement, so the tampered
        // run must be rejected at the io binding or downstream.
        let mut bad_io = io.clone();
        bad_io[256] ^= true;
        let mut tv2 = base();
        let r = verify_plonk(&circuit, &setup, &bad_io, &proof, &mut tv2);
        assert!(r.is_err(), "tampered io must be rejected");
    }

    #[test]
    fn plonk_roundtrip_and_structure() {
        let (circuit, witness, io) = build_test_circuit();
        // the circuit is honestly satisfied
        let n = circuit.n;
        assert!(circuit.gates_hold(
            &witness.w[0..n],
            &witness.w[n..2 * n],
            &witness.w[2 * n..3 * n]
        ));
        assert!(circuit.copies_hold(&witness.w));
        assert!(circuit
            .sigma
            .iter()
            .all(|&s| (s as usize) < circuit.l_len()));

        let params = PcsParams::default();
        let setup = setup_plonk(&circuit, &witness, params);
        let mut t = Transcript::new("mikros-plonk-test", &[]);
        let (proof, stats) = prove_plonk(&circuit, &witness, &setup, &mut t);
        eprintln!(
            "plonk N=2^14: gate {:.1} ms (hadamard {:.1}), copy {:.1} ms, open {:.1} ms, total {:.1} ms",
            stats.gate_ms, stats.hadamard_ms, stats.copy_ms, stats.open_ms, stats.total_ms
        );
        let mut tv = Transcript::new("mikros-plonk-test", &[]);
        match verify_plonk(&circuit, &setup, &io, &proof, &mut tv) {
            Ok(()) => {}
            Err(e) => panic!("the honest proof must verify: {e}"),
        }
    }

    #[test]
    fn plonk_tamper_classes() {
        let (circuit, witness, io) = build_test_circuit();
        let params = PcsParams::default();
        let setup = setup_plonk(&circuit, &witness, params);
        let mut t = Transcript::new("mikros-plonk-test", &[]);
        let (proof, _) = prove_plonk(&circuit, &witness, &setup, &mut t);

        // 1. corrupt a gate claim (the sum check)
        {
            let mut bad = clone_proof(&proof);
            bad.c_claims[2] += ONE;
            let mut tv = Transcript::new("mikros-plonk-test", &[]);
            assert!(verify_plonk(&circuit, &setup, &io, &bad, &mut tv).is_err());
        }
        // 2. corrupt a wire value (the recombination)
        {
            let mut bad = clone_proof(&proof);
            bad.wire_values[0][1] += ONE;
            let mut tv = Transcript::new("mikros-plonk-test", &[]);
            assert!(verify_plonk(&circuit, &setup, &io, &bad, &mut tv).is_err());
        }
        // 3. corrupt a digit value (the ς recombination)
        {
            let mut bad = clone_proof(&proof);
            bad.digit_values[0][0] += ONE;
            let mut tv = Transcript::new("mikros-plonk-test", &[]);
            assert!(verify_plonk(&circuit, &setup, &io, &bad, &mut tv).is_err());
        }
        // 4. corrupt a witness-group v_msg (the opening)
        {
            let mut bad = clone_proof(&proof);
            bad.wit_open.v_msgs[0][3] += ONE;
            let mut tv = Transcript::new("mikros-plonk-test", &[]);
            assert!(verify_plonk(&circuit, &setup, &io, &bad, &mut tv).is_err());
        }
        // 5. corrupt the copy proof (a cubic round)
        {
            let mut bad = clone_proof(&proof);
            let mut ct = bad.copy.unfolds[0].e_tree.clone();
            for level in ct.iter_mut() {
                if !level.rounds.is_empty() {
                    level.rounds[0][2] += ONE;
                    break;
                }
            }
            bad.copy.unfolds[0].e_tree = ct;
            let mut tv = Transcript::new("mikros-plonk-test", &[]);
            assert!(verify_plonk(&circuit, &setup, &io, &bad, &mut tv).is_err());
        }
        // 6. an UNSATISFIED circuit: flip one witness wire bit (a private
        // gate input) — the gate claims or the copy relation must fail
        {
            let mut bad_w = witness.w.clone();
            // flip a wire that feeds a gate (position < N, not io)
            bad_w[7] ^= true;
            let bad_witness = PlonkWitness { w: bad_w };
            assert!(
                !circuit.gates_hold(
                    &bad_witness.w[0..circuit.n],
                    &bad_witness.w[circuit.n..2 * circuit.n],
                    &bad_witness.w[2 * circuit.n..3 * circuit.n]
                ) || !circuit.copies_hold(&bad_witness.w),
                "the flip must break the circuit"
            );
            let bad_setup = setup_plonk(&circuit, &bad_witness, PcsParams::default());
            let mut t2 = Transcript::new("mikros-plonk-test", &[]);
            let (bad_proof, _) = prove_plonk(&circuit, &bad_witness, &bad_setup, &mut t2);
            let mut tv = Transcript::new("mikros-plonk-test", &[]);
            assert!(
                verify_plonk(&circuit, &bad_setup, &io, &bad_proof, &mut tv).is_err(),
                "the tampered witness must reject (its commitments differ)"
            );
        }
    }

    fn clone_fact1(p: &Fact1Proof) -> Fact1Proof {
        Fact1Proof {
            rounds: p.rounds.clone(),
            terminals: p.terminals,
            challenges: p.challenges.clone(),
        }
    }

    fn clone_proof(p: &PlonkProof) -> PlonkProof {
        fn clone_had(h: &HadamardProof) -> HadamardProof {
            HadamardProof {
                d: h.d.clone(),
                h: h.h.clone(),
                fact1: clone_fact1(&h.fact1),
            }
        }
        fn clone_group(g: &GroupOpening) -> GroupOpening {
            GroupOpening {
                v_msgs: g.v_msgs.clone(),
                fact1: clone_fact1(&g.fact1),
                pcs: g.pcs.clone(),
            }
        }
        PlonkProof {
            c_claims: p.c_claims,
            gate_proofs: [
                clone_had(&p.gate_proofs[0]),
                clone_had(&p.gate_proofs[1]),
                clone_had(&p.gate_proofs[2]),
                clone_had(&p.gate_proofs[3]),
            ],
            nested: clone_had(&p.nested),
            copy: crate::copy::clone_copy_proof(&p.copy),
            wire_values: p.wire_values,
            digit_values: p.digit_values.clone(),
            wit_open: clone_group(&p.wit_open),
            pub_open: clone_group(&p.pub_open),
        }
    }
}
