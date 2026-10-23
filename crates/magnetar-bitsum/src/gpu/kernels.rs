//! cuTile GPU kernels for the bitsliced GF(2^128) multilinear sumcheck.
//!
//! This is the Phase-1 kernel spec of the GPU proving program:
//! *NVlabs cuTile (cutile-rs) bitsliced sumcheck*, exit criterion
//! ">= 2x over the CPU baseline on 1x H100".
//!
//! # What each entry computes (and its tested host oracle)
//!
//! | Entry | Computes | Host oracle (`magnetar-bitsum`) |
//! |---|---|---|
//! | [`pack_planes`] | packed `[groups, 32, 4]` words -> bitsliced `[groups, 128]` | `BitPlanes::pack` |
//! | [`bind_round`] | `out[g] = lo[g] ^ tree(lo[g] ^ hi[g], alpha)` | `tree::bind_planes` |
//! | [`reduce_partial`] | XOR-reduction of plane rows -> partials `[P, 128]` | `tree::reduce_planes` |
//!
//! The tower multiplication tree is *literally* the recursion verified by
//! `crate::tree` (bit-exact tests there): seven device functions
//! `tower_w1 .. tower_w128`, one per coordinate width, splitting tiles
//! along the plane axis with `extract`/`cat` and multiplying the low-half
//! product by the tower constant `NUS[k-1]` (constant masks, single
//! coordinate `2^k - 1`) — the compiler folds the constant subtrees.
//!
//! # Layout and tiling
//!
//! Plane tensors are `[groups, 128]` u32 (group-major rows), identical to
//! `bitsum_core::BitPlanes`. Tiles are `[TG, 128]` with `TG` groups per
//! program — the "tiling" dimension of the Phase-1 spec: each program
//! streams `TG x 512 B` of contiguous rows, so every load/store is a
//! coalesced row burst.
//!
//! # Bring-up checklist (first run on CUDA hardware)
//!
//! 1. `CUDA_TOOLKIT_PATH` must point at CUDA 13.2+ (13.3 recommended);
//!    GPU must be sm_80+ (H100 = sm_90).
//! 2. Run `cargo test -p magnetar-bitsum --features cutile --test
//!    gpu_parity` — it compares all three kernels plus the full prover
//!    against the host oracle and is the acceptance gate for this module.
//! 3. Inspect `#[cutile::entry(print_ir = true)]` output on `bind_round`:
//!    verify the constant-mask subtrees of the tower folded away
//!    (`AND` with `0x00000000` eliminated). If they did not fold, the kernel
//!    is still correct but runs the unpruned tree (~4x more tile ops);
//!    file follow-up: offline-generated circuits (Phase-2, README).
//! 4. Confirm `scan`-based row reduction semantics (custom XOR scan, last
//!    row extracted) with `print_ir`; this is the one DSL primitive whose
//!    return shape the reference docs leave ambiguous.

use cutile::prelude::*;

#[cutile::module]
mod sumcheck_kernels {
    use cutile::core::*;

    // =====================================================================
    // Tower multiplication tree (mirrors crate::tree 1:1).
    //
    // Coordinate-width ladder: 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1.
    // tower_w{W} multiplies a [TG, W] value tile by a constant whose
    // coordinate masks are the [W] mask tile `m`.
    //
    //   r0  = tower(a_lo, m_lo)           r1 = tower(a_hi, m_hi)
    //   r2  = tower(a_lo ^ a_hi, m_lo ^ m_hi)
    //   nu  = tower(r1, NUS[k-1]-masks)   // constant masks below
    //   out = (r0 ^ nu) ++ (r2 ^ r0)
    //
    // Tower constants (verified in crate::field / the basis-trace search):
    //   NUS[k] = single coordinate 2^k - 1, i.e. the full generator product
    //   g_{k-1} * ... * g_0. For tower_w{W} (W = 2^k), the nu-multiplication
    //   runs on width W/2 with mask = all zeros except coordinate W/2 - 1.
    // =====================================================================

    /// Leaf: width 1. `a & m` broadcast over the group axis.
    fn tower_w1<const TG: i32>(
        a: Tile<u32, { [TG, 1] }>,
        m: Tile<u32, { [1] }>,
    ) -> Tile<u32, { [TG, 1] }> {
        let m_b: Tile<u32, { [TG, 1] }> = m.broadcast(shape![TG, 1]);
        a & m_b
    }

    /// Width 2 (F_4 level). nu-masks for width 1: NUS[0] = coordinate 0.
    fn tower_w2<const TG: i32>(
        a: Tile<u32, { [TG, 2] }>,
        m: Tile<u32, { [2] }>,
    ) -> Tile<u32, { [TG, 2] }> {
        let a_lo: Tile<u32, { [TG, 1] }> = extract(a, [0, 0]);
        let a_hi: Tile<u32, { [TG, 1] }> = extract(a, [0, 1]);
        let m_lo: Tile<u32, { [1] }> = extract(m, [0]);
        let m_hi: Tile<u32, { [1] }> = extract(m, [1]);
        let r0 = tower_w1::<TG>(a_lo, m_lo);
        let r1 = tower_w1::<TG>(a_hi, m_hi);
        let r2 = tower_w1::<TG>(a_lo ^ a_hi, m_lo ^ m_hi);
        // NUS[0] = 1: single coordinate 0 within width 1 -> mask = all ones
        let nu: Tile<u32, { [1] }> = constant(0xFFFF_FFFFu32, shape![1]);
        let nu_r1 = tower_w1::<TG>(r1, nu);
        let out_lo = r0 ^ nu_r1;
        let out_hi = r2 ^ r0;
        cat(out_lo, out_hi, 1i32)
    }

    /// Width 4 (F_16 level). nu-masks for width 2: NUS[1] = coordinate 1.
    fn tower_w4<const TG: i32>(
        a: Tile<u32, { [TG, 4] }>,
        m: Tile<u32, { [4] }>,
    ) -> Tile<u32, { [TG, 4] }> {
        let a_lo: Tile<u32, { [TG, 2] }> = extract(a, [0, 0]);
        let a_hi: Tile<u32, { [TG, 2] }> = extract(a, [0, 2]);
        let m_lo: Tile<u32, { [2] }> = extract(m, [0]);
        let m_hi: Tile<u32, { [2] }> = extract(m, [2]);
        let r0 = tower_w2::<TG>(a_lo, m_lo);
        let r1 = tower_w2::<TG>(a_hi, m_hi);
        let r2 = tower_w2::<TG>(a_lo ^ a_hi, m_lo ^ m_hi);
        // NUS[1] = coordinate 1 within width 2: zeros[1] ++ ones[1]
        let nu: Tile<u32, { [2] }> = cat(
            constant(0u32, shape![1]),
            constant(0xFFFF_FFFFu32, shape![1]),
            0i32,
        );
        let nu_r1 = tower_w2::<TG>(r1, nu);
        let out_lo = r0 ^ nu_r1;
        let out_hi = r2 ^ r0;
        cat(out_lo, out_hi, 1i32)
    }

    /// Width 8 (F_256 level). nu-masks for width 4: NUS[2] = coordinate 3.
    fn tower_w8<const TG: i32>(
        a: Tile<u32, { [TG, 8] }>,
        m: Tile<u32, { [8] }>,
    ) -> Tile<u32, { [TG, 8] }> {
        let a_lo: Tile<u32, { [TG, 4] }> = extract(a, [0, 0]);
        let a_hi: Tile<u32, { [TG, 4] }> = extract(a, [0, 4]);
        let m_lo: Tile<u32, { [4] }> = extract(m, [0]);
        let m_hi: Tile<u32, { [4] }> = extract(m, [4]);
        let r0 = tower_w4::<TG>(a_lo, m_lo);
        let r1 = tower_w4::<TG>(a_hi, m_hi);
        let r2 = tower_w4::<TG>(a_lo ^ a_hi, m_lo ^ m_hi);
        // NUS[2] = coordinate 3 within width 4: zeros[3] ++ ones[1]
        let nu: Tile<u32, { [4] }> = cat(
            constant(0u32, shape![3]),
            constant(0xFFFF_FFFFu32, shape![1]),
            0i32,
        );
        let nu_r1 = tower_w4::<TG>(r1, nu);
        let out_lo = r0 ^ nu_r1;
        let out_hi = r2 ^ r0;
        cat(out_lo, out_hi, 1i32)
    }

    /// Width 16 (F_65536 level). nu-masks for width 8: NUS[3] = coordinate 7.
    fn tower_w16<const TG: i32>(
        a: Tile<u32, { [TG, 16] }>,
        m: Tile<u32, { [16] }>,
    ) -> Tile<u32, { [TG, 16] }> {
        let a_lo: Tile<u32, { [TG, 8] }> = extract(a, [0, 0]);
        let a_hi: Tile<u32, { [TG, 8] }> = extract(a, [0, 8]);
        let m_lo: Tile<u32, { [8] }> = extract(m, [0]);
        let m_hi: Tile<u32, { [8] }> = extract(m, [8]);
        let r0 = tower_w8::<TG>(a_lo, m_lo);
        let r1 = tower_w8::<TG>(a_hi, m_hi);
        let r2 = tower_w8::<TG>(a_lo ^ a_hi, m_lo ^ m_hi);
        // NUS[3] = coordinate 7 within width 8: zeros[7] ++ ones[1]
        let nu: Tile<u32, { [8] }> = cat(
            constant(0u32, shape![7]),
            constant(0xFFFF_FFFFu32, shape![1]),
            0i32,
        );
        let nu_r1 = tower_w8::<TG>(r1, nu);
        let out_lo = r0 ^ nu_r1;
        let out_hi = r2 ^ r0;
        cat(out_lo, out_hi, 1i32)
    }

    /// Width 32 (F_{2^32} level). nu-masks for width 16: NUS[4] = coordinate 15.
    fn tower_w32<const TG: i32>(
        a: Tile<u32, { [TG, 32] }>,
        m: Tile<u32, { [32] }>,
    ) -> Tile<u32, { [TG, 32] }> {
        let a_lo: Tile<u32, { [TG, 16] }> = extract(a, [0, 0]);
        let a_hi: Tile<u32, { [TG, 16] }> = extract(a, [0, 16]);
        let m_lo: Tile<u32, { [16] }> = extract(m, [0]);
        let m_hi: Tile<u32, { [16] }> = extract(m, [16]);
        let r0 = tower_w16::<TG>(a_lo, m_lo);
        let r1 = tower_w16::<TG>(a_hi, m_hi);
        let r2 = tower_w16::<TG>(a_lo ^ a_hi, m_lo ^ m_hi);
        // NUS[4] = coordinate 15 within width 16: zeros[15] ++ ones[1]
        let nu: Tile<u32, { [16] }> = cat(
            constant(0u32, shape![15]),
            constant(0xFFFF_FFFFu32, shape![1]),
            0i32,
        );
        let nu_r1 = tower_w16::<TG>(r1, nu);
        let out_lo = r0 ^ nu_r1;
        let out_hi = r2 ^ r0;
        cat(out_lo, out_hi, 1i32)
    }

    /// Width 64 (F_{2^64} level). nu-masks for width 32: NUS[5] = coordinate 31.
    fn tower_w64<const TG: i32>(
        a: Tile<u32, { [TG, 64] }>,
        m: Tile<u32, { [64] }>,
    ) -> Tile<u32, { [TG, 64] }> {
        let a_lo: Tile<u32, { [TG, 32] }> = extract(a, [0, 0]);
        let a_hi: Tile<u32, { [TG, 32] }> = extract(a, [0, 32]);
        let m_lo: Tile<u32, { [32] }> = extract(m, [0]);
        let m_hi: Tile<u32, { [32] }> = extract(m, [32]);
        let r0 = tower_w32::<TG>(a_lo, m_lo);
        let r1 = tower_w32::<TG>(a_hi, m_hi);
        let r2 = tower_w32::<TG>(a_lo ^ a_hi, m_lo ^ m_hi);
        // NUS[5] = coordinate 31 within width 32: zeros[31] ++ ones[1]
        let nu: Tile<u32, { [32] }> = cat(
            constant(0u32, shape![31]),
            constant(0xFFFF_FFFFu32, shape![1]),
            0i32,
        );
        let nu_r1 = tower_w32::<TG>(r1, nu);
        let out_lo = r0 ^ nu_r1;
        let out_hi = r2 ^ r0;
        cat(out_lo, out_hi, 1i32)
    }

    /// Width 128 (full GF(2^128)). nu-masks for width 64: NUS[6] = coordinate 63.
    ///
    /// This is the multiplier the bind kernel calls; `masks` are the
    /// runtime coordinate masks of the round challenge alpha.
    fn tower_w128<const TG: i32>(
        a: Tile<u32, { [TG, 128] }>,
        m: Tile<u32, { [128] }>,
    ) -> Tile<u32, { [TG, 128] }> {
        let a_lo: Tile<u32, { [TG, 64] }> = extract(a, [0, 0]);
        let a_hi: Tile<u32, { [TG, 64] }> = extract(a, [0, 64]);
        let m_lo: Tile<u32, { [64] }> = extract(m, [0]);
        let m_hi: Tile<u32, { [64] }> = extract(m, [64]);
        let r0 = tower_w64::<TG>(a_lo, m_lo);
        let r1 = tower_w64::<TG>(a_hi, m_hi);
        let r2 = tower_w64::<TG>(a_lo ^ a_hi, m_lo ^ m_hi);
        // NUS[6] = coordinate 63 within width 64: zeros[63] ++ ones[1]
        let nu: Tile<u32, { [64] }> = cat(
            constant(0u32, shape![63]),
            constant(0xFFFF_FFFFu32, shape![1]),
            0i32,
        );
        let nu_r1 = tower_w64::<TG>(r1, nu);
        let out_lo = r0 ^ nu_r1;
        let out_hi = r2 ^ r0;
        cat(out_lo, out_hi, 1i32)
    }

    // =====================================================================
    // Entry points
    // =====================================================================

    /// Binds one sumcheck variable:
    /// `out[g] = planes[g] ^ tree(planes[g] ^ planes[g + half], alpha)`.
    ///
    /// `out` is `[half_groups, 128]` partitioned `[TG, 128]` (grid =
    /// `half_groups / TG` programs); `planes` is the full `[groups, 128]`
    /// tensor; `alpha_masks` is the `[128]` coordinate-mask tensor of the
    /// round challenge (0x00000000 / 0xFFFFFFFF words); `half_tiles` is
    /// `half_groups / TG` (the tile offset of the high half).
    #[cutile::entry()]
    fn bind_round<const TG: i32>(
        out: &mut Tensor<u32, { [-1, -1] }>,
        planes: &Tensor<u32, { [-1, -1] }>,
        alpha_masks: &Tensor<u32, { [-1] }>,
        half_tiles: i32,
    ) {
        let pid = program_id(0);
        let lo: Tile<u32, { [TG, 128] }> = planes.load_tile(shape![TG, 128], [pid, 0]);
        let hi: Tile<u32, { [TG, 128] }> = planes.load_tile(shape![TG, 128], [pid + half_tiles, 0]);
        let masks: Tile<u32, { [128] }> = alpha_masks.load_tile(shape![128], [0]);
        let d = lo ^ hi;
        let e = tower_w128::<TG>(d, masks);
        out.store(lo ^ e);
    }

    /// XOR-reduces the first `nblocks * TGR` rows of `planes` into per-program
    /// partial rows.
    ///
    /// `partials` is `[nprog, 128]` partitioned `[1, 128]` (grid = `nprog`);
    /// each program scans its strided share of `[TGR, 128]` row blocks,
    /// keeping a running `[1, 128]` XOR, and stores it to `partials[pid]`.
    /// Apply hierarchically until one row remains; the host then folds the
    /// 128 parity bits into the `S0` field element.
    ///
    /// The host guarantees `rows % TGR == 0` (it picks `TGR` per stage from
    /// `{64, 32, 16, 8, 4, 2, 1}` so every launch divides the row count).
    #[cutile::entry()]
    fn reduce_partial<const TGR: i32>(
        partials: &mut Tensor<u32, { [-1, -1] }>,
        planes: &Tensor<u32, { [-1, -1] }>,
        nblocks: i32,
        nprog: i32,
    ) {
        let pid = program_id(0);
        let mut acc: Tile<u32, { [1, 128] }> = constant(0u32, shape![1, 128]);
        let mut b = pid;
        while b < nblocks {
            let t: Tile<u32, { [TGR, 128] }> = planes.load_tile(shape![TGR, 128], [b, 0]);
            // XOR-scan down the rows: the last row holds the block's XOR.
            let scanned: Tile<u32, { [TGR, 128] }> =
                scan(t, 0i32, reverse::Forward, 0u32, |x, y| x ^ y);
            let last = TGR - 1;
            let row: Tile<u32, { [1, 128] }> = extract(scanned, [last, 0]);
            acc = acc ^ row;
            b += nprog;
        }
        partials.store(acc);
    }

    /// Packs packed element words into bitsliced planes.
    ///
    /// `out` is `[groups, 128]` partitioned `[TG, 128]`; `elems` is the
    /// `[groups, 32, 4]` u32 tensor of element coordinate words (little
    /// endian). Bit `j` of element `e` of group `g` (word `j / 32`, bit
    /// `j % 32`) lands at plane `j`, lane `e`.
    ///
    /// Implementation: four unrolled 32-plane chunks (chunk `c` covers
    /// `j in [32c, 32c+32)`, so its word index is the *compile-time*
    /// constant `c`); per chunk, build the bit tensor `[TG, 32, 32]`,
    /// multiply by lane weights `2^e`, and `reduce_sum` over the element
    /// axis — an exact integer identity (sums are bounded by `2^32 - 1`,
    /// no carry between lanes). The four `[TG, 32]` chunk columns are
    /// concatenated into the output tile.
    #[cutile::entry()]
    fn pack_planes<const TG: i32>(
        out: &mut Tensor<u32, { [-1, -1] }>,
        elems: &Tensor<u32, { [-1, -1, -1] }>,
    ) {
        let pid = program_id(0);
        let w: Tile<u32, { [TG, 32, 4] }> = elems.load_tile(shape![TG, 32, 4], [pid, 0, 0]);

        // lane weights 2^e over the element axis
        let ee: Tile<u32, { [1, 32, 1] }> = iota(shape![32]).reshape(shape![1, 32, 1]);
        let weights: Tile<u32, { [1, 32, 1] }> = broadcast_scalar(1u32, shape![1, 32, 1]) << ee;

        // intra-word shift amounts s in [0, 32), broadcast over [TG, 32, 32]
        let ss: Tile<u32, { [1, 1, 32] }> = iota(shape![32]).reshape(shape![1, 1, 32]);
        let ssb: Tile<u32, { [TG, 32, 32] }> = ss.broadcast(shape![TG, 32, 32]);
        let oneb: Tile<u32, { [TG, 32, 32] }> =
            broadcast_scalar(1u32, shape![1, 1, 32]).broadcast(shape![TG, 32, 32]);
        let wgt: Tile<u32, { [TG, 32, 32] }> = weights.broadcast(shape![TG, 32, 32]);

        // chunk 0: word 0, planes j in [0, 32)
        let wc0: Tile<u32, { [TG, 32, 1] }> = extract(w, [0, 0, 0]);
        let bits0: Tile<u32, { [TG, 32, 32] }> = (wc0.broadcast(shape![TG, 32, 32]) >> ssb) & oneb;
        let col0: Tile<u32, { [TG, 32] }> = reduce_sum(bits0 * wgt, 1i32);

        // chunk 1: word 1, planes j in [32, 64)
        let wc1: Tile<u32, { [TG, 32, 1] }> = extract(w, [0, 0, 1]);
        let bits1: Tile<u32, { [TG, 32, 32] }> = (wc1.broadcast(shape![TG, 32, 32]) >> ssb) & oneb;
        let col1: Tile<u32, { [TG, 32] }> = reduce_sum(bits1 * wgt, 1i32);

        // chunk 2: word 2, planes j in [64, 96)
        let wc2: Tile<u32, { [TG, 32, 1] }> = extract(w, [0, 0, 2]);
        let bits2: Tile<u32, { [TG, 32, 32] }> = (wc2.broadcast(shape![TG, 32, 32]) >> ssb) & oneb;
        let col2: Tile<u32, { [TG, 32] }> = reduce_sum(bits2 * wgt, 1i32);

        // chunk 3: word 3, planes j in [96, 128)
        let wc3: Tile<u32, { [TG, 32, 1] }> = extract(w, [0, 0, 3]);
        let bits3: Tile<u32, { [TG, 32, 32] }> = (wc3.broadcast(shape![TG, 32, 32]) >> ssb) & oneb;
        let col3: Tile<u32, { [TG, 32] }> = reduce_sum(bits3 * wgt, 1i32);

        let half0: Tile<u32, { [TG, 64] }> = cat(col0, col1, 1i32);
        let half1: Tile<u32, { [TG, 64] }> = cat(col2, col3, 1i32);
        let planes_tile: Tile<u32, { [TG, 128] }> = cat(half0, half1, 1i32);
        out.store(planes_tile);
    }
}
