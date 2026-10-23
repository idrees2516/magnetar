//! The wire-size model: the exact serialized byte count of a proof, with
//! one 16-byte word per committed field element (`GF(2^128)`), 32-byte
//! digests, and 8-byte length prefixes. This is the **uncompressed**
//! wire format; production transports would compress the field elements
//! (round coefficients carry far less entropy than the challenges), so
//! this is the conservative upper bound.
//!
//! The `stats` module's cost model is analytic; this walks the **actual
//! proof object** so the benchmark harness reports honest, measured
//! sizes: `Proof` (the aggregated multi-segment artifact) and
//! `AggregateProof` (the tree-IVC chunked artifact).

use crate::ivc::AggregateProof;
use crate::protocol::{Proof, SegmentProof};
use magnetar_field::Gf128;
use magnetar_memory::twist::{PortRun, ReconRun};
use magnetar_memory::{
    BusShoutProof, OneHotProof, PsiProof, SelectorShoutProof, StageSelectProof, TwistProof,
};
use magnetar_pcs::{PathQuery, RingProof};
use magnetar_router::RouterProof;
use magnetar_spartan::VirtualSpartanProof;

const GF: usize = 16;
const DIGEST: usize = 32;
const LEN: usize = 8;

fn v_gf(v: &[Gf128]) -> usize {
    LEN + v.len() * GF
}
fn v_digest(v: &[[u8; 32]]) -> usize {
    LEN + v.len() * DIGEST
}
fn level_open(l: &magnetar_pcs::LevelOpen) -> usize {
    3 * GF + v_digest(&l.a_path) + v_digest(&l.b_path) + v_digest(&l.c_path)
}
fn path_query(q: &PathQuery) -> usize {
    LEN + q.levels.iter().map(level_open).sum::<usize>()
}
fn ring_proof(p: &RingProof) -> usize {
    let rounds: usize = p.rounds.iter().map(|r| v_gf(r)).sum();
    let queries: usize = p.queries.iter().map(path_query).sum();
    let tail: usize = p.tail.iter().map(|(_v, path)| GF + v_digest(path)).sum();
    let anchors: usize = p
        .anchors
        .iter()
        .map(|a| GF + GF + v_digest(&a.main_path) + v_digest(&a.c_path))
        .sum();
    rounds + v_digest(&p.level_roots) + GF + queries + tail + anchors + LEN
}
fn mosaic_proof(p: &crate::mosaic::MosaicProof) -> usize {
    let rounds: usize = p.batch_rounds.iter().map(|r| v_gf(r)).sum();
    rounds + GF + ring_proof(&p.ring)
}
fn port_run(p: &PortRun) -> usize {
    let rounds: usize = p.rounds.iter().map(|r| v_gf(r)).sum();
    rounds + v_gf(&p.r_joint) + 2 * GF
}
fn recon_run(p: &ReconRun) -> usize {
    let rounds: usize = p.rounds.iter().map(|r| v_gf(r)).sum();
    rounds + 2 * GF + v_gf(&p.wfa_component_claims) + v_gf(&p.r_cycle)
}
fn onehot(oh: &OneHotProof) -> usize {
    let rounds: usize = oh.rounds.iter().map(|r| v_gf(r)).sum();
    rounds + 3 * GF + v_gf(&oh.r) + v_gf(&oh.r_prime)
}
fn twist_proof(p: &TwistProof) -> usize {
    let reads: usize = p.read_runs.iter().map(port_run).sum();
    let recons: usize = p.read_recons.iter().map(recon_run).sum();
    let st1: usize = p
        .st1
        .iter()
        .map(|per_port| {
            LEN + per_port
                .iter()
                .map(|per_comp| LEN + per_comp.iter().map(|r| v_gf(r)).sum::<usize>())
                .sum::<usize>()
        })
        .sum();
    let st1_claims: usize = p.st1_claims.iter().map(|per| v_gf(per)).sum();
    let onehots: usize = p
        .onehot
        .iter()
        .map(|per| LEN + per.iter().map(onehot).sum::<usize>())
        .sum();
    reads
        + port_run(&p.write_run)
        + recons
        + recon_run(&p.write_recon)
        + p.read_component_claims
            .iter()
            .map(|per| v_gf(per))
            .sum::<usize>()
        + v_gf(&p.write_component_claims)
        + st1
        + st1_claims
        + onehots
}
fn psi_proof(p: &PsiProof) -> usize {
    let rounds: usize = p.rounds.iter().map(|r| v_gf(r)).sum();
    rounds + p.component_claims.len() * (GF + GF + LEN)
}
fn bus_proof(p: &BusShoutProof) -> usize {
    let phi: usize = p.phi_rounds.iter().map(|r| v_gf(r)).sum();
    phi + p.phi_component_claims.len() * (GF + GF + LEN)
        + psi_proof(&p.pc_psi)
        + p.onehot.iter().map(onehot).sum::<usize>()
}
fn selector_proof(p: &SelectorShoutProof) -> usize {
    p.per_factor
        .iter()
        .map(|f| {
            let rounds: usize = f.rounds.iter().map(|r| v_gf(r)).sum();
            rounds + f.component_claims.len() * (GF + GF + LEN)
        })
        .sum()
}
fn stage_proof(p: &StageSelectProof) -> usize {
    let rounds: usize = p.rounds.iter().map(|r| v_gf(r)).sum();
    rounds + 4 * GF
}
fn router_proof(p: &RouterProof) -> usize {
    let rounds: usize = p.rounds.iter().map(|r| v_gf(r)).sum();
    rounds + GF + v_gf(&p.q_values) + v_gf(&p.offset_values) + v_gf(&p.terminal)
}
fn vs_proof(p: &VirtualSpartanProof) -> usize {
    let shared: usize = p.outer_shared.iter().map(|(a, b)| v_gf(a) + v_gf(b)).sum();
    let ob: usize = p.outer_bin.iter().map(|r| v_gf(r)).sum();
    let of: usize = p.outer_fld.iter().map(|r| v_gf(r)).sum();
    let inner: usize = p.inner.iter().map(|r| v_gf(r)).sum();
    shared + ob + of + 6 * GF + inner + GF + v_gf(&p.segment_values)
}
fn state_identity(p: &crate::ivc::StateIdentityProof) -> usize {
    let rounds: usize = p.rounds.iter().map(|r| v_gf(r)).sum();
    rounds + 2 * GF
}

fn segment_proof(p: &SegmentProof) -> usize {
    vs_proof(&p.vs)
        + twist_proof(&p.reg_twist)
        + p.reg_psi.iter().map(psi_proof).sum::<usize>()
        + p.reg_index_claims.len() * GF
        + twist_proof(&p.ram_twist)
        + p.ram_psi.iter().map(psi_proof).sum::<usize>()
        + p.ram_addr_claims.len() * GF
        + GF
        + v_gf(&p.inc_point)
        + GF
        + router_proof(&p.shift_router)
        + router_proof(&p.store_router)
        + p.store_x_claims.len() * GF
        + bus_proof(&p.bus_shout)
        + stage_proof(&p.stage_select)
        + selector_proof(&p.sel_stage)
        + selector_proof(&p.sel_shift)
        + selector_proof(&p.sel_store)
}

impl Proof {
    /// The uncompressed wire size in bytes.
    pub fn size_bytes(&self) -> usize {
        DIGEST
            + 8 * 3
            + self.pcs_domain.len()
            + LEN
            + 8
            + self.segments.iter().map(segment_proof).sum::<usize>()
            + mosaic_proof(&self.mosaic)
    }
}

impl AggregateProof {
    /// The uncompressed wire size in bytes.
    pub fn size_bytes(&self) -> usize {
        DIGEST
            + 8 * 3
            + self.pcs_domain.len()
            + LEN
            + 8
            + self
                .chunks
                .iter()
                .map(|c| {
                    segment_proof(&c.core)
                        + state_identity(&c.ram_state)
                        + state_identity(&c.reg_state)
                })
                .sum::<usize>()
            + self.ram_state_values.len() * 2 * GF
            + self.cont_values.len() * 2 * GF
            + mosaic_proof(&self.mosaic)
    }
}

/// The peak resident set (KiB) from /proc/self/status.
pub fn peak_rss_kib() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}
