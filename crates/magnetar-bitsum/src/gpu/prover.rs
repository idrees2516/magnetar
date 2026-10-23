//! Host-side GPU prover: round orchestration over the cuTile kernels.
//!
//! Per round (the exact protocol of `crate::sumcheck::prove_planes`,
//! with the tree bind and reduction running on device):
//!
//! ```text
//! S0      <- reduce_partial hierarchy over the low half + host parity fold
//! (c0, c1) = (S0, running claim)              // absorb, draw alpha
//! masks   <- alpha coordinate masks            // H2D, 512 bytes
//! bound   <- bind_round::<TG>                  // out = lo ^ tree(lo ^ hi, alpha)
//! ```
//!
//! The tail (`< GPU_MIN_GROUPS` groups) downloads and finishes on the CPU
//! via the shared `prove_from_tail`, so the GPU and CPU provers produce
//! bit-identical proofs on the same transcript (the `gpu_parity` test).
//!
//! Memory: `16n` bytes for the input planes + `8n` for the round output +
//! `16n` transient for the packed upload = peak `<= 40n` bytes
//! (640 MiB at `2^24` entries).

use crate::sumcheck::{self, ProverState};
use crate::tree;
use crate::{BitPlanes, ProveError, ProveOutput, SumcheckProof, Transcript, F128};
use cutile::api;
use cutile::prelude::*;
use cutile::tensor::Tensor;
use std::sync::Arc;

use crate::gpu::kernels::sumcheck_kernels;

/// Groups per bind program (tile `[32, 128]`, 16 KiB of plane data).
pub const BIND_TG: usize = 32;
/// Groups per reduce block (tile `[64, 128]`, 32 KiB).
pub const REDUCE_TGR: usize = 64;
/// Groups per pack program (intermediate tiles `[16, 32, 32]`).
pub const PACK_TG: usize = 16;
/// The GPU handles rounds while at least this many groups remain
/// (2048 elements); below that launch overhead dominates and the tail
/// finishes on the host.
pub const GPU_MIN_GROUPS: usize = 64;
/// Cap on stage-1 reduction programs.
const MAX_REDUCE_PROGRAMS: usize = 1024;

/// Errors from the GPU prover.
#[derive(Debug)]
pub enum Error {
    /// Underlying cutile / CUDA failure (JIT, launch, transfer).
    Cuda(cutile::Error),
    /// Protocol-level failure from the shared core logic.
    Protocol(ProveError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Cuda(e) => write!(f, "cuda error: {e}"),
            Error::Protocol(e) => write!(f, "protocol error: {e}"),
        }
    }
}

impl From<cutile::Error> for Error {
    fn from(e: cutile::Error) -> Self {
        Error::Cuda(e)
    }
}

impl From<ProveError> for Error {
    fn from(e: ProveError) -> Self {
        Error::Protocol(e)
    }
}

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Selects the reduce tile size for a row count: the largest power of two
/// in `[1, REDUCE_TGR]` that divides `rows` (host guarantees no remainder).
fn pick_tgr(rows: usize) -> usize {
    let mut tgr = REDUCE_TGR.min(rows);
    while rows % tgr != 0 {
        tgr >>= 1;
    }
    tgr
}

/// Launches one `reduce_partial` stage at a fixed tile size.
fn launch_reduce<const TGR: i32>(
    input: &Tensor<u32>,
    nblocks: i32,
    nprog: i32,
) -> Result<Tensor<u32>> {
    let partials = api::zeros::<u32>(&[nprog as usize, 128])
        .sync()?
        .partition([1, 128]);
    // Launchers return ALL runtime arguments in parameter order
    // (out, planes, nblocks, nprog).
    let (done, _input, _nblocks, _nprog) =
        sumcheck_kernels::reduce_partial::<TGR>(partials, input, nblocks, nprog).sync()?;
    Ok(done.unpartition())
}

/// XOR-reduces the first `rows` rows of a `[rows_total, 128]` plane tensor
/// into a single field element (hierarchical `reduce_partial` + host
/// parity fold). Mirrors `bitsum_core::tree::reduce_planes`.
///
/// Stage tiling: the host picks `TGR in {64, 32, 16, 8, 4, 2, 1}` per
/// stage so `rows % TGR == 0` always holds (row counts are powers of two
/// by construction); programs cap at [`MAX_REDUCE_PROGRAMS`].
fn reduce_to_element(planes: &Tensor<u32>, rows: usize) -> Result<F128> {
    let mut owned: Option<Tensor<u32>> = None;
    let mut cur_rows = rows;

    while cur_rows > 1 {
        let cur: &Tensor<u32> = owned.as_ref().unwrap_or(planes);
        let tgr = pick_tgr(cur_rows);
        let nblocks = cur_rows / tgr;
        let nprog = nblocks.min(MAX_REDUCE_PROGRAMS);
        owned = Some(match tgr {
            64 => launch_reduce::<64>(cur, nblocks as i32, nprog as i32)?,
            32 => launch_reduce::<32>(cur, nblocks as i32, nprog as i32)?,
            16 => launch_reduce::<16>(cur, nblocks as i32, nprog as i32)?,
            8 => launch_reduce::<8>(cur, nblocks as i32, nprog as i32)?,
            4 => launch_reduce::<4>(cur, nblocks as i32, nprog as i32)?,
            2 => launch_reduce::<2>(cur, nblocks as i32, nprog as i32)?,
            _ => launch_reduce::<1>(cur, nblocks as i32, nprog as i32)?,
        });
        cur_rows = nprog;
    }

    let words: Vec<u32> = match &owned {
        Some(t) => download_words(t)?,
        None => download_words(planes)?, // rows == 1 degenerate case
    };
    let mut acc = 0u128;
    for (j, w) in words.iter().take(128).enumerate() {
        if crate::bitslice::parity32(*w) == 1 {
            acc |= 1u128 << j;
        }
    }
    Ok(F128(acc))
}

/// Uploads raw u32 words (element or plane tensors) to the device.
///
/// Exposed so the `gpu_parity` acceptance test can stage inputs without
/// depending on `cutile` directly.
pub fn upload_words(words: Vec<u32>) -> Result<Tensor<u32>> {
    Ok(api::copy_host_vec_to_device(&Arc::new(words)).sync()?)
}

/// Downloads a tensor's words to the host.
pub fn download_words(t: &Tensor<u32>) -> Result<Vec<u32>> {
    Ok(t.to_host_vec().sync()?)
}

/// One GPU bind round: `out = lo ^ tree(lo ^ hi, alpha)` on device.
///
/// Exposed for the parity test (single-round comparison against
/// `tree::bind_planes`).
pub fn bind_round_on_device(
    planes: &Tensor<u32>,
    half_groups: usize,
    alpha: F128,
) -> Result<Tensor<u32>> {
    let masks: Vec<u32> = tree::alpha_masks(alpha).to_vec();
    let alpha_masks = api::copy_host_vec_to_device(&Arc::new(masks)).sync()?;
    let out = api::zeros::<u32>(&[half_groups, 128])
        .sync()?
        .partition([BIND_TG as i32, 128]);
    let half_tiles = (half_groups / BIND_TG) as i32;
    // Returns (out, planes, alpha_masks, half_tiles).
    let (done, _planes, _masks, _half_tiles) =
        sumcheck_kernels::bind_round::<32>(out, planes, &alpha_masks, half_tiles).sync()?;
    Ok(done.unpartition())
}

/// Packs elements on device: `[groups, 32, 4]` u32 -> `[groups, 128]`
/// planes. Exposed for the parity test.
pub fn pack_on_device(elems: &Tensor<u32>, groups: usize) -> Result<Tensor<u32>> {
    let elems_view = elems.view(&[groups, 32, 4])?;
    let out = api::zeros::<u32>(&[groups, 128])
        .sync()?
        .partition([PACK_TG as i32, 128]);
    let (done, _elems) = sumcheck_kernels::pack_planes::<16>(out, &elems_view).sync()?;
    Ok(done.unpartition())
}

/// Uploads a witness table as the `[groups, 32, 4]` little-endian u32
/// tensor.
fn upload_elems(evals: &[F128]) -> Result<Tensor<u32>> {
    let mut elem_words: Vec<u32> = Vec::with_capacity(evals.len() * 4);
    for e in evals {
        let bytes = e.to_le_bytes();
        for w in bytes.chunks_exact(4) {
            elem_words.push(u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
        }
    }
    Ok(api::copy_host_vec_to_device(&Arc::new(elem_words)).sync()?)
}

/// GPU prover entry point.
///
/// Falls back to the CPU reference prover below `GPU_MIN_GROUPS * 32`
/// elements, so the function is total over all power-of-two sizes.
///
/// # Errors
///
/// [`Error::Cuda`] for JIT/launch/transfer failures,
/// [`Error::Protocol`] for the (bug-indicating) prover self-check.
pub fn prove(evals: &[F128], t: &mut Transcript) -> Result<ProveOutput> {
    let n = evals.len();
    let m = sumcheck::num_vars(n).ok_or(ProveError::NotPowerOfTwo(n))?;

    if n < GPU_MIN_GROUPS * 32 {
        // host path for small instances (launch overhead dominates)
        return Ok(sumcheck::prove(evals, t)?);
    }

    t.absorb_u64("sc-num-vars", m as u64);
    let groups = n / 32;

    // 1. upload packed elements, pack to planes on device
    let elems = upload_elems(evals)?;
    let planes = pack_on_device(&elems, groups)?;

    // 2. claimed sum over the full table
    let claimed_sum = reduce_to_element(&planes, groups)?;

    // 3. rounds on device
    let mut state = ProverState::new(claimed_sum);
    let mut cur = planes;
    let mut cur_groups = groups;
    while cur_groups >= GPU_MIN_GROUPS {
        let half_groups = cur_groups / 2;
        let s0 = reduce_to_element(&cur, half_groups)?;
        let alpha = state.push_round(s0, t);
        cur = bind_round_on_device(&cur, half_groups, alpha)?;
        cur_groups = half_groups;
    }

    // 4. tail on host, shared protocol logic
    let tail_words: Vec<u32> = download_words(&cur)?;
    let tail_planes = BitPlanes::from_group_major(tail_words);
    let tail = tail_planes.unpack();
    let final_eval = sumcheck::prove_from_tail(&tail, t, &mut state)?;

    Ok(ProveOutput {
        proof: SumcheckProof {
            num_vars: m,
            messages: state.messages,
        },
        claimed_sum,
        eval_point: state.challenges,
        final_eval,
    })
}
