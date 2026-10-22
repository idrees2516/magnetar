//! Shared types for the claims the memory arguments leave for the PCS.

use magnetar_field::Gf128;

/// One evaluation claim on a committed component: `p̂(point) = value`,
/// with the point in the component's local coordinates (address part low,
/// cycle part high, little-endian).
#[derive(Clone, Debug, PartialEq)]
pub struct PcsPoint {
    pub point: Vec<Gf128>,
    pub value: Gf128,
}

impl PcsPoint {
    pub fn new(point: Vec<Gf128>, value: Gf128) -> Self {
        PcsPoint { point, value }
    }
}

/// All PCS claims a BinaryTwist proof leaves open, in a fixed order the
/// zkvm layer maps to Mosaic components.
#[derive(Clone, Debug)]
pub struct TwistPcsClaims {
    /// Address-component claims from the read runs: `[port][component]`.
    pub read_addr: Vec<Vec<PcsPoint>>,
    /// Address-component claims from the write run: `[component]`.
    pub write_addr: Vec<PcsPoint>,
    /// Address-component claims from the reconstruction sum-checks
    /// (the write port's components at each recon's point):
    /// `[run][component]` with `run` in `0..=p` (`0..p` = read recons,
    /// `p` = the write recon).
    pub recon_wfa: Vec<Vec<PcsPoint>>,
    /// Increment claims from the reconstruction sum-checks: `[run]`.
    pub inc: Vec<PcsPoint>,
    /// Transmitted initial-memory claims from the reconstruction
    /// sum-checks: `[run]` at the run's `(r_k ‖ r_v)` point — non-empty
    /// only when the instance's memory is not public (the tree-IVC
    /// committed-entry-image mode; the zkvm layer discharges them against
    /// the `ram-init` Mosaic component).
    pub m0: Vec<PcsPoint>,
    /// Sum-to-one terminals: `[port][component]`.
    pub st1: Vec<Vec<PcsPoint>>,
    /// Δ\* terminals: `[port][component]` — the claim at `r` and at `r'`.
    pub onehot: Vec<Vec<[PcsPoint; 2]>>,
}

/// All PCS claims a bytecode BinaryShout proof leaves open.
#[derive(Clone, Debug)]
pub struct ShoutPcsClaims {
    /// The joint Φ sum-check's fetch-component terminal claims
    /// (one per component, at the Φ point).
    pub phi: Vec<PcsPoint>,
    /// The Δ\* terminals per component (at `r` and at `r'`).
    pub onehot: Vec<[PcsPoint; 2]>,
    /// The Ψ (address-claim) sum-check terminals per component.
    pub psi: Vec<PcsPoint>,
}

/// PCS claims of a selector check: one per fetch component.
#[derive(Clone, Debug)]
pub struct SelectorPcsClaims {
    pub components: Vec<PcsPoint>,
}
