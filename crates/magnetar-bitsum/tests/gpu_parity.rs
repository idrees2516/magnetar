//! GPU parity tests: the acceptance gate for the `cutile` feature on CUDA
//! hardware. Every kernel plus the full prover is compared bit-exactly
//! against the tested host oracle.
//!
//! Skipped unless `BITSUM_GPU_TESTS=1` (set on CUDA runners):
//! ```sh
//! BITSUM_GPU_TESTS=1 cargo test -p magnetar-bitsum --features cutile --test gpu_parity --release
//! ```

use magnetar_bitsum::sumcheck;
use magnetar_bitsum::{BitPlanes, Transcript, F128};

struct SplitMix(u64);
impl magnetar_bitsum::Rng for SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x09E3_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn gpu_enabled() -> bool {
    match std::env::var("BITSUM_GPU_TESTS") {
        Ok(v) => v == "1" || v == "true",
        Err(_) => false,
    }
}

fn skip(name: &str) {
    eprintln!("SKIP {name}: set BITSUM_GPU_TESTS=1 on a CUDA 13.x + sm_80+ host to run");
}

fn random_evals(m: usize, seed: u64) -> Vec<F128> {
    let mut r = SplitMix(seed);
    (0..(1usize << m)).map(|_| F128::random(&mut r)).collect()
}

/// GPU pack kernel vs `BitPlanes::pack`.
#[test]
fn pack_parity() {
    if !gpu_enabled() {
        skip("pack_parity");
        return;
    }
    for groups in [64usize, 128, 256] {
        let evals = random_evals(
            (groups * 32).trailing_zeros() as usize,
            0x100 + groups as u64,
        );
        let host_planes = BitPlanes::pack(&evals).expect("pack");

        let mut words: Vec<u32> = Vec::with_capacity(evals.len() * 4);
        for e in &evals {
            let b = e.to_le_bytes();
            for w in b.chunks_exact(4) {
                words.push(u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            }
        }
        let elems = magnetar_bitsum::gpu::upload_words(words).expect("h2d");
        let planes = magnetar_bitsum::gpu::pack_on_device(&elems, groups).expect("gpu pack");
        let got: Vec<u32> = magnetar_bitsum::gpu::download_words(&planes).expect("d2h");
        assert_eq!(got, host_planes.words, "pack mismatch at {groups} groups");
    }
}

/// One GPU bind round vs `tree::bind_planes`.
#[test]
fn bind_round_parity() {
    if !gpu_enabled() {
        skip("bind_round_parity");
        return;
    }
    let evals = random_evals(12, 0x200); // 4096 elements = 128 groups
    let planes = BitPlanes::pack(&evals).expect("pack");
    let mut r = SplitMix(0xb1d);
    let alpha = F128::random(&mut r);

    let host_bound = magnetar_bitsum::tree::bind_planes(&planes, alpha);

    let dev_planes = magnetar_bitsum::gpu::upload_words(planes.words.clone()).expect("h2d");
    let bound = magnetar_bitsum::gpu::bind_round_on_device(&dev_planes, planes.groups / 2, alpha)
        .expect("gpu bind");
    let got: Vec<u32> = magnetar_bitsum::gpu::download_words(&bound).expect("d2h");
    assert_eq!(got, host_bound.words, "bind mismatch");
}

/// GPU reduction vs `tree::reduce_planes`.
#[test]
fn reduce_parity() {
    if !gpu_enabled() {
        skip("reduce_parity");
        return;
    }
    let evals = random_evals(12, 0x300);
    let planes = BitPlanes::pack(&evals).expect("pack");

    let dev_planes = magnetar_bitsum::gpu::upload_words(planes.words.clone()).expect("h2d");
    for count in [32usize, 64, planes.groups] {
        let host_s = magnetar_bitsum::tree::reduce_planes(&planes, count);
        let gpu_s =
            magnetar_bitsum::gpu::reduce_to_element(&dev_planes, count).expect("gpu reduce");
        assert_eq!(host_s, gpu_s, "reduce mismatch at count {count}");
    }
}

/// Full GPU prover vs CPU reference: bit-identical proofs, same transcript.
#[test]
fn full_prover_parity() {
    if !gpu_enabled() {
        skip("full_prover_parity");
        return;
    }
    for m in [11usize, 12, 14] {
        let evals = random_evals(m, 0x400 + m as u64);
        let mut t1 = Transcript::new("gpu-parity");
        let mut t2 = Transcript::new("gpu-parity");
        let cpu_out = sumcheck::prove(&evals, &mut t1).expect("cpu prove");
        let gpu_out = magnetar_bitsum::gpu::prove(&evals, &mut t2).expect("gpu prove");

        assert_eq!(cpu_out.proof, gpu_out.proof, "proof mismatch m={m}");
        assert_eq!(cpu_out.claimed_sum, gpu_out.claimed_sum);
        assert_eq!(cpu_out.eval_point, gpu_out.eval_point);
        assert_eq!(cpu_out.final_eval, gpu_out.final_eval);

        // and the GPU proof verifies with the CPU verifier + outer check
        let mut tv = Transcript::new("gpu-parity");
        let oc = sumcheck::verify_with_final_eval(
            &gpu_out.proof,
            gpu_out.claimed_sum,
            gpu_out.final_eval,
            &mut tv,
        )
        .expect("verify");
        assert_eq!(sumcheck::eval_mle(&evals, &oc.eval_point), oc.final_claim);
    }
}

/// Small instances fall back to the CPU path and still prove/verify.
#[test]
fn small_instance_fallback() {
    if !gpu_enabled() {
        skip("small_instance_fallback");
        return;
    }
    let evals = random_evals(6, 0x500);
    let mut t = Transcript::new("small");
    let out = magnetar_bitsum::gpu::prove(&evals, &mut t).expect("prove");
    let mut tv = Transcript::new("small");
    let oc = sumcheck::verify_with_final_eval(&out.proof, out.claimed_sum, out.final_eval, &mut tv)
        .expect("verify");
    assert_eq!(sumcheck::eval_mle(&evals, &oc.eval_point), oc.final_claim);
}
