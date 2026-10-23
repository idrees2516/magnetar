//! Proof-size instrumentation (the cost model of the research roadmap's
//! build order: "measure the committed-bit inventory and the
//! query/path/digest split of the proof directly, so every subsequent
//! lever lands on measured constants").
//!
//! [`proof_stats`] walks a finished proof and reports (1) the committed-bit
//! inventory per Mosaic component (and the virtual bits the protocol never
//! commits), (2) the PCS geometry (codeword length, Merkle depth, query
//! count, digest size), and (3) a byte breakdown per protocol component.

use crate::protocol::{KeyGen, Proof};

/// The measured cost model of one proof.
#[derive(Clone, Debug)]
pub struct ProofStats {
    pub cycles: usize,
    /// Total committed bits (the Mosaic `G`).
    pub committed_bits: usize,
    /// Bits of the virtual segments (never committed — the paper's
    /// Virtual-Spartan saving).
    pub virtual_bits: usize,
    /// Per-component committed bits.
    pub components: Vec<(String, usize)>,
    /// The packed table's variables (before the codeword extension).
    pub pcs_n: usize,
    /// The codeword rate parameter.
    pub pcs_k: usize,
    /// Codeword length `2^(n+k)`.
    pub codeword_len: usize,
    /// Merkle depth per query.
    pub merkle_depth: usize,
    /// Correlated path queries in the PCS evaluation argument.
    pub queries: usize,
    /// Digest size in bytes.
    pub digest_bytes: usize,
    /// Total proof size in bytes (field elements at 16 bytes, digests at
    /// 32).
    pub bytes_total: usize,
    /// Per-component byte breakdown.
    pub bytes_breakdown: Vec<(String, usize)>,
}

impl ProofStats {
    /// A compact human-readable report.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("cycles: {}\n", self.cycles));
        s.push_str(&format!(
            "committed bits: {} ({:.1} KiB) across {} components\n",
            self.committed_bits,
            self.committed_bits as f64 / 8192.0,
            self.components.len()
        ));
        s.push_str(&format!(
            "virtual (uncommitted) bits: {} ({:.1}% of the witness)\n",
            self.virtual_bits,
            100.0 * self.virtual_bits as f64 / (self.virtual_bits + self.committed_bits) as f64
        ));
        let mut comps: Vec<_> = self.components.clone();
        comps.sort_by_key(|c| std::cmp::Reverse(c.1));
        for (name, bits) in comps.iter().take(10) {
            s.push_str(&format!("  {name}: {bits} bits\n"));
        }
        s.push_str(&format!(
            "PCS: n = {} vars, rate 2^-{} (codeword {} entries), Merkle depth {}, {} queries, {}-byte digests\n",
            self.pcs_n, self.pcs_k, self.codeword_len, self.merkle_depth, self.queries, self.digest_bytes
        ));
        s.push_str(&format!(
            "proof: {} bytes ({:.1} KiB)\n",
            self.bytes_total,
            self.bytes_total as f64 / 1024.0
        ));
        let mut br: Vec<_> = self.bytes_breakdown.clone();
        br.sort_by_key(|c| std::cmp::Reverse(c.1));
        for (name, b) in br {
            s.push_str(&format!(
                "  {name}: {b} bytes ({:.0}%)\n",
                100.0 * b as f64 / self.bytes_total as f64
            ));
        }
        s
    }
}

/// Count field elements in a nested structure via a light recursive walk
/// of the proof's known shapes.
fn count_fields_proof(proof: &Proof) -> (usize, Vec<(String, usize)>) {
    let p = &proof.inner;
    let _fe = |v: &[magnetar_field::Gf128]| v.len();
    let mut total = 0usize;
    let mut br = Vec::new();
    let mut add = |name: &str, n: usize| {
        total += n * 16;
        br.push((name.to_string(), n * 16));
    };
    // Commitment statement.
    add("commitment+stmt", 32 / 16 + 3);
    // Virtual Spartan: the outer rounds, terminals, inner, segment values.
    let mut vs_n = 0usize;
    for (m1, m2) in &p.vs.outer_shared {
        vs_n += m1.len() + m2.len();
    }
    for m in p.vs.outer_bin.iter().chain(p.vs.outer_fld.iter()) {
        vs_n += m.len();
    }
    vs_n += 6; // terminals
    vs_n += p.vs.inner.iter().map(|m| m.len()).sum::<usize>();
    vs_n += 1 + p.vs.segment_values.len();
    add("virtual-spartan", vs_n);
    // Register Twist.
    let twist_n = |t: &magnetar_memory::TwistProof| -> usize {
        let mut n = 0usize;
        for r in &t.read_runs {
            n += r.rounds.iter().map(|m| m.len()).sum::<usize>() + r.r_joint.len() + 3;
        }
        n += t.write_run.rounds.iter().map(|m| m.len()).sum::<usize>()
            + t.write_run.r_joint.len()
            + 3;
        for r in t.read_recons.iter().chain(std::iter::once(&t.write_recon)) {
            n += r.rounds.iter().map(|m| m.len()).sum::<usize>()
                + 2
                + r.wfa_component_claims.len()
                + 1;
        }
        for per in &t.st1 {
            for rounds in per {
                n += rounds.iter().map(|m| m.len()).sum::<usize>() + 1;
            }
        }
        for per in &t.onehot {
            for oh in per {
                n += oh.rounds.iter().map(|m| m.len()).sum::<usize>() + oh.r.len() * 2 + 3;
            }
        }
        n += t
            .read_component_claims
            .iter()
            .map(|c| c.len())
            .sum::<usize>()
            + t.write_component_claims.len();
        n
    };
    add("reg-twist", twist_n(&p.reg_twist) + 3);
    add("ram-twist", twist_n(&p.ram_twist) + 2 + 1);
    // Psi proofs.
    let psi_n = |ps: &[magnetar_memory::PsiProof; 2]| -> usize {
        ps.iter()
            .map(|x| {
                x.rounds.iter().map(|m| m.len()).sum::<usize>() + x.component_claims.len() * 8 + 1
            })
            .sum()
    };
    add(
        "reg+ram psi",
        p.reg_psi
            .iter()
            .map(|x| {
                x.rounds.iter().map(|m| m.len()).sum::<usize>() + x.component_claims.len() * 8 + 1
            })
            .sum::<usize>()
            + psi_n(&p.ram_psi),
    );
    // Routers.
    let router_n = |r: &magnetar_router::RouterProof| -> usize {
        r.rounds.iter().map(|m| m.len()).sum::<usize>()
            + r.terminal.len()
            + 1
            + r.q_values.len()
            + r.offset_values.len()
    };
    add(
        "routers",
        router_n(&p.shift_router) + router_n(&p.store_router) + 2,
    );
    // Bytecode Shout.
    let bus_n = p
        .bus_shout
        .phi_rounds
        .iter()
        .map(|m| m.len())
        .sum::<usize>()
        + p.bus_shout.phi_component_claims.len() * 10
        + p.bus_shout
            .pc_psi
            .rounds
            .iter()
            .map(|m| m.len())
            .sum::<usize>()
        + p.bus_shout.pc_psi.component_claims.len() * 8;
    let mut oh_n = 0usize;
    for oh in &p.bus_shout.onehot {
        oh_n += oh.rounds.iter().map(|m| m.len()).sum::<usize>() + oh.r.len() * 2 + 3;
    }
    add("bytecode-shout", bus_n + oh_n);
    // Selector checks.
    let sel_n = |s: &magnetar_memory::SelectorShoutProof| -> usize {
        s.per_factor
            .iter()
            .map(|r| {
                r.rounds.iter().map(|m| m.len()).sum::<usize>() + r.component_claims.len() * 10
            })
            .sum()
    };
    add("selectors", sel_n(&p.sel_shift) + sel_n(&p.sel_store));
    // The cross-step argument.
    let cs_n = p
        .cross_step
        .s1_rounds
        .iter()
        .map(|m| m.len())
        .sum::<usize>()
        + p.cross_step
            .s2_rounds
            .iter()
            .map(|m| m.len())
            .sum::<usize>()
        + 5;
    add("cross-step", cs_n);
    // The final-state bindings (segments).
    let fs_n = |fs: &Option<magnetar_memory::FinalStateProof>| -> usize {
        fs.as_ref()
            .map(|f| {
                f.zero_rounds.iter().map(|m| m.len()).sum::<usize>()
                    + f.evo_rounds.iter().map(|m| m.len()).sum::<usize>()
                    + f.component_claims.len()
                    + 2
            })
            .unwrap_or(0)
    };
    add("final-state", fs_n(&p.final_reg) + fs_n(&p.final_ram));
    // Mosaic.
    let mosaic_n = proof
        .mosaic
        .batch_rounds
        .iter()
        .map(|m| m.len())
        .sum::<usize>()
        + 1
        + ring_fields(&proof.mosaic.ring);
    add("mosaic+pcs-eval", mosaic_n);
    (total, br)
}

fn ring_fields(r: &magnetar_pcs::ring::RingProof) -> usize {
    // Level roots, the terminal claim, the query levels, the tail with
    // paths, and the anchors.
    let mut n = r.level_roots.len() * (32 / 16) + 1;
    for q in &r.queries {
        for lv in &q.levels {
            n += 2 + (lv.a_path.len() + lv.b_path.len()) * (32 / 16);
        }
    }
    for (_v, path) in &r.tail {
        n += 1 + path.len() * (32 / 16);
    }
    for a in &r.anchors {
        n += 1 + (a.main_path.len() + a.c_path.len()) * (32 / 16);
    }
    n
}

/// Measure a proof's cost model.
pub fn proof_stats(pk: &KeyGen, proof: &Proof) -> ProofStats {
    let (bytes_total, bytes_breakdown) = count_fields_proof(proof);
    let mut components = Vec::new();
    let mut committed = 0usize;
    for (i, c) in pk.mosaic.components.iter().enumerate() {
        let bits = 1usize << c.vars;
        committed += bits;
        components.push((c.name.clone(), bits));
        let _ = i;
    }
    // The virtual segments' bits.
    let n = pk.trace_len.trailing_zeros() as usize;
    let mut virtual_bits = 0usize;
    for (name, len, direct) in crate::protocol::segments() {
        if !*direct {
            virtual_bits += len << n;
            let _ = name;
        }
    }
    ProofStats {
        cycles: pk.trace_len,
        committed_bits: committed,
        virtual_bits,
        components,
        pcs_n: proof.pcs_n,
        pcs_k: proof.pcs_k,
        codeword_len: 1usize << (proof.pcs_n + proof.pcs_k),
        merkle_depth: proof.pcs_n + proof.pcs_k,
        queries: pk.pcs.queries,
        digest_bytes: 32,
        bytes_total,
        bytes_breakdown,
    }
}
