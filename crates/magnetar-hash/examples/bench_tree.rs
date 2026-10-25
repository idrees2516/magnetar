//! Merkle build microbenchmark: the prefix-path leaf/node hashing.
use magnetar_field::Gf128;
use magnetar_hash::MerkleTree;
use std::time::Instant;

fn main() {
    for log2 in [16usize, 18, 20] {
        let n = 1usize << log2;
        let leaves: Vec<Gf128> = (0..n).map(|i| Gf128(i as u128 * 0x9E3779B97F4A7C15)).collect();
        // Warm the rayon pool.
        let _ = MerkleTree::build("bench/dom/level/0", &leaves[..1024]);
        let t0 = Instant::now();
        let tree = MerkleTree::build("bench/dom/level/0", &leaves);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "2^{log2} leaves: {dt:7.1} ms  ({:6.0}k leaf-hashes/s)",
            n as f64 / 1e3 / dt
        );
    }
}
