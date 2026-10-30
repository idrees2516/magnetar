//! Binary hash circuits as GF(2) R1CS (the paper's Table-4 workloads).
//!
//! Each builder emits **constraints and the witness simultaneously** — gate
//! values are known as the circuit is built, so the packed-instance prover
//! gets ready witnesses. Rotation/permutation wirings cost zero rows (wire
//! aliases); k-input XOR gates are single R1CS rows `(t1⊕…⊕tk)·1 = 0`
//! against the constant-one column; AND gates are `a·b = c` rows — the
//! paper's arithmetization.
//!
//! Public I/O: every builder marks its input and output words in
//! `CircuitBuilder::io` so Algorithm 6's public-IO binding (steps 3–5) can
//! weight them; the rest of the witness is private.

use crate::r1cs::{BinaryR1cs, SparseMatrix};

/// A circuit under construction: witness columns + constraint rows.
pub struct CircuitBuilder {
    pub n_cols: usize,
    pub w: Vec<bool>,
    /// (S_L row, S_R row, S_O row) column lists.
    pub rows: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>,
    /// Public-I/O columns (inputs and outputs).
    pub io: Vec<u32>,
    /// When false, gate values are computed but rows are not emitted — the
    /// per-instance witness-replay mode (the circuit is built once with
    /// rows and replayed without for the other κ−1 instances).
    pub emit_rows: bool,
}

pub const C1: u32 = 0; // the constant-one column

impl CircuitBuilder {
    pub fn new() -> Self {
        CircuitBuilder {
            n_cols: 1,
            w: vec![true],
            rows: Vec::new(),
            io: Vec::new(),
            emit_rows: true,
        }
    }

    /// The witness-replay builder (no row emission).
    pub fn replay() -> Self {
        CircuitBuilder {
            emit_rows: false,
            ..CircuitBuilder::new()
        }
    }

    pub fn bit(&mut self, v: bool) -> u32 {
        let idx = self.n_cols as u32;
        self.n_cols += 1;
        self.w.push(v);
        idx
    }

    pub fn io_bit(&mut self, v: bool) -> u32 {
        let idx = self.bit(v);
        if self.emit_rows {
            self.io.push(idx);
        }
        idx
    }

    pub fn word(&mut self, val: u32) -> [u32; 32] {
        let mut out = [0u32; 32];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.bit((val >> i) & 1 == 1);
        }
        out
    }

    pub fn io_word(&mut self, val: u32) -> [u32; 32] {
        let mut out = [0u32; 32];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.io_bit((val >> i) & 1 == 1);
        }
        out
    }

    pub fn word64(&mut self, val: u64) -> [u32; 64] {
        let mut out = [0u32; 64];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.bit((val >> i) & 1 == 1);
        }
        out
    }

    pub fn io_word64(&mut self, val: u64) -> [u32; 64] {
        let mut out = [0u32; 64];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.io_bit((val >> i) & 1 == 1);
        }
        out
    }

    /// `z = t1 ⊕ … ⊕ tk` — one row when k ≥ 2.
    pub fn xor(&mut self, terms: &[u32]) -> u32 {
        assert!(terms.len() >= 2);
        let mut v = false;
        for &t in terms {
            v ^= self.w[t as usize];
        }
        let z = self.bit(v);
        if self.emit_rows {
            let mut l: Vec<u32> = terms.to_vec();
            l.push(z);
            self.rows.push((l, vec![C1], Vec::new()));
        }
        z
    }

    /// `z = a ∧ b`.
    pub fn and(&mut self, a: u32, b: u32) -> u32 {
        let z = self.bit(self.w[a as usize] & self.w[b as usize]);
        if self.emit_rows {
            self.rows.push((vec![a], vec![b], vec![z]));
        }
        z
    }

    /// Bind two bits equal: `(a ⊕ b)·1 = 0`.
    pub fn assert_eq(&mut self, a: u32, b: u32) {
        if self.emit_rows {
            self.rows.push((vec![a, b], vec![C1], Vec::new()));
        }
    }

    /// A full adder over bits: returns `(sum, carry)`.
    pub fn full_adder(&mut self, a: u32, b: u32, c: u32) -> (u32, u32) {
        let axb = self.xor(&[a, b]);
        let sum = self.xor(&[axb, c]);
        let ab = self.and(a, b);
        let axbc = self.and(axb, c);
        let carry = self.xor(&[ab, axbc]);
        (sum, carry)
    }

    /// n-bit adder, LSB-first words: `a + b = s` (mod 2^n).
    pub fn adder(&mut self, a: &[u32], b: &[u32]) -> Vec<u32> {
        assert_eq!(a.len(), b.len());
        let mut s = Vec::with_capacity(a.len());
        let mut carry = self.bit(false);
        for i in 0..a.len() {
            let (sum, c) = self.full_adder(a[i], b[i], carry);
            s.push(sum);
            carry = c;
        }
        s
    }

    /// Assemble the final `BinaryR1cs`, padding rows/cols to a power of two.
    pub fn build(self) -> BinaryR1cs {
        let n_rows = self.rows.len();
        let n = self.n_cols.max(n_rows).max(64).next_power_of_two();
        // the io positions are a SET: sort (+ dedup) so downstream
        // consumers can scan them monotonically — the batch io-bind's
        // masked-eq scan requires it (the keccak output lanes are pushed
        // in (y,x) order, not column-creation order).
        let mut io = self.io;
        io.sort_unstable();
        io.dedup();
        let mut s_l = SparseMatrix::new(n);
        let mut s_r = SparseMatrix::new(n);
        let mut s_o = SparseMatrix::new(n);
        for (i, (l, r, o)) in self.rows.iter().enumerate() {
            for &c in l {
                s_l.add(i, c);
            }
            for &c in r {
                s_r.add(i, c);
            }
            for &c in o {
                s_o.add(i, c);
            }
        }
        BinaryR1cs {
            n_constraints: n_rows,
            n_witness: self.n_cols,
            s_l,
            s_r,
            s_o,
            io_positions: io,
        }
    }
}

impl Default for CircuitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SHA-256 compression
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn rotr32(w: &[u32], k: usize) -> Vec<u32> {
    let n = w.len();
    let k = k % n;
    (0..n).map(|i| w[(i + k) % n]).collect()
}

fn xor_words(b: &mut CircuitBuilder, ws: &[Vec<u32>]) -> Vec<u32> {
    let n = ws[0].len();
    (0..n)
        .map(|i| b.xor(&ws.iter().map(|w| w[i]).collect::<Vec<_>>()))
        .collect()
}

/// Build the SHA-256 compression circuit for one block. The 8 input state
/// words and 16 block words (and the 8 output words) are public I/O.
/// Returns the output state words.
pub fn sha256_compression(
    b: &mut CircuitBuilder,
    state: [u32; 8],
    block: [u32; 16],
) -> [Vec<u32>; 8] {
    // W schedule (block words are public I/O)
    let mut w: Vec<Vec<u32>> = Vec::with_capacity(64);
    for i in 0..16 {
        w.push(b.io_word(block[i]).to_vec());
    }
    for t in 16..64 {
        // σ0(x) = ROT7 ⊕ ROT18 ⊕ SHR3 ; σ1(x) = ROT17 ⊕ ROT19 ⊕ SHR10
        let sig0 = |bb: &mut CircuitBuilder, wm: &[u32]| -> Vec<u32> {
            let mut sh = vec![0u32; 32];
            for i in 0..32 {
                sh[i] = if i + 3 < 32 { wm[i + 3] } else { bb.bit(false) };
            }
            xor_words(bb, &[rotr32(wm, 7), rotr32(wm, 18), sh])
        };
        let sig1 = |bb: &mut CircuitBuilder, wm: &[u32]| -> Vec<u32> {
            let mut sh = vec![0u32; 32];
            for i in 0..32 {
                sh[i] = if i + 10 < 32 {
                    wm[i + 10]
                } else {
                    bb.bit(false)
                };
            }
            xor_words(bb, &[rotr32(wm, 17), rotr32(wm, 19), sh])
        };
        let s0 = sig0(b, &w[t - 15]);
        let s1 = sig1(b, &w[t - 2]);
        w.push(xor_words(b, &[s0, w[t - 7].clone(), s1, w[t - 16].clone()]));
    }
    // working vars — the initial state words are the public I/O bits
    let mut a = b.io_word(state[0]).to_vec();
    let mut bb = b.io_word(state[1]).to_vec();
    let mut c = b.io_word(state[2]).to_vec();
    let mut d = b.io_word(state[3]).to_vec();
    let mut e = b.io_word(state[4]).to_vec();
    let mut f = b.io_word(state[5]).to_vec();
    let mut g = b.io_word(state[6]).to_vec();
    let mut h = b.io_word(state[7]).to_vec();
    let state_words = [
        a.clone(),
        bb.clone(),
        c.clone(),
        d.clone(),
        e.clone(),
        f.clone(),
        g.clone(),
        h.clone(),
    ];
    for t in 0..64 {
        let big_sigma1 = xor_words(b, &[rotr32(&e, 6), rotr32(&e, 11), rotr32(&e, 25)]);
        let ch: Vec<u32> = (0..32)
            .map(|i| {
                // Ch = (e∧f) ⊕ (¬e∧g) = ef ⊕ (e∧g) ⊕ g
                let ef = b.and(e[i], f[i]);
                let eg = b.and(e[i], g[i]);
                b.xor(&[ef, eg, g[i]])
            })
            .collect();
        let big_sigma0 = xor_words(b, &[rotr32(&a, 2), rotr32(&a, 13), rotr32(&a, 22)]);
        let maj: Vec<u32> = (0..32)
            .map(|i| {
                let ab = b.and(a[i], bb[i]);
                let ac = b.and(a[i], c[i]);
                let bc = b.and(bb[i], c[i]);
                b.xor(&[ab, ac, bc])
            })
            .collect();
        let kw = b.word(SHA256_K[t]).to_vec();
        // T1 = h + Σ1 + Ch + K + W
        let mut acc = h.clone();
        for term in [big_sigma1, ch, kw, w[t].clone()] {
            acc = b.adder(&acc, &term);
        }
        // T2 = Σ0 + Maj
        let t2 = b.adder(&big_sigma0, &maj);
        let na = b.adder(&acc, &t2);
        let old_d = d.clone();
        d = c.clone();
        c = bb.clone();
        bb = a.clone();
        a = na;
        h = g.clone();
        g = f.clone();
        f = e.clone();
        e = b.adder(&old_d, &acc);
    }
    // outputs = the initial state words + the round words (public I/O)
    let mut out: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
    let finals: [&Vec<u32>; 8] = [&a, &bb, &c, &d, &e, &f, &g, &h];
    for i in 0..8 {
        let sum = b.adder(&state_words[i], finals[i]);
        for &bitcol in &sum {
            b.io.push(bitcol);
        }
        out[i] = sum;
    }
    out
}

// ---------------------------------------------------------------------------
// Keccak-f[1600]
// ---------------------------------------------------------------------------

const KECCAK_RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000000008009,
    0x000000000000000a,
    0x000000000000008a,
    0x0000000000000088,
    0x8000000000008009,
    0x800000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
    0x0000000000008080,
    0x8000000000008009,
    0x000000000000008a,
];

const KECCAK_ROT: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// Build the Keccak-f[1600] permutation circuit. `state[y][x]` bits are
/// 64-bit lanes (x = the column index), LSB-first. Input and output lanes
/// are public I/O. Returns the output state.
pub fn keccak_f1600(b: &mut CircuitBuilder, state: &[[[u32; 64]; 5]; 5]) -> [[[u32; 64]; 5]; 5] {
    let mut a: Vec<Vec<[u32; 64]>> = state.iter().map(|row| row.to_vec()).collect();
    for round in 0..24 {
        // θ
        let mut c: Vec<[u32; 64]> = Vec::with_capacity(5);
        for x in 0..5 {
            let mut cc = [0u32; 64];
            for i in 0..64 {
                let terms: Vec<u32> = (0..5).map(|y| a[y][x][i]).collect();
                cc[i] = b.xor(&terms);
            }
            c.push(cc);
        }
        let mut d: Vec<[u32; 64]> = Vec::with_capacity(5);
        for x in 0..5 {
            let mut dd = [0u32; 64];
            for i in 0..64 {
                let rot1 = c[(x + 1) % 5][(i + 63) % 64];
                dd[i] = b.xor(&[c[(x + 4) % 5][i], rot1]);
            }
            d.push(dd);
        }
        for y in 0..5 {
            for x in 0..5 {
                let mut na = [0u32; 64];
                for i in 0..64 {
                    na[i] = b.xor(&[a[y][x][i], d[x][i]]);
                }
                a[y][x] = na;
            }
        }
        // ρ and π: pure index remaps
        let mut bb_arr: Vec<Vec<[u32; 64]>> = vec![vec![[0u32; 64]; 5]; 5];
        for x in 0..5 {
            for y in 0..5 {
                let k = KECCAK_ROT[x + 5 * y] as usize;
                let src = a[y][x];
                let mut rotated = [0u32; 64];
                for (i, r) in rotated.iter_mut().enumerate() {
                    *r = src[(i + 64 - (k % 64)) % 64];
                }
                let (bx, by) = ((2 * x + 3 * y) % 5, y);
                bb_arr[by][bx] = rotated;
            }
        }
        a = bb_arr;
        // χ: A[x] = B[x] ⊕ ((¬B[x+1]) ∧ B[x+2])  =  B0 ⊕ (B1∧B2) ⊕ B2
        let mut out: Vec<Vec<[u32; 64]>> = vec![vec![[0u32; 64]; 5]; 5];
        for y in 0..5 {
            for x in 0..5 {
                let b0 = &a[y][x];
                let b1 = &a[y][(x + 1) % 5];
                let b2 = &a[y][(x + 2) % 5];
                let mut lane = [0u32; 64];
                for i in 0..64 {
                    let and = b.and(b1[i], b2[i]);
                    lane[i] = b.xor(&[b0[i], and, b2[i]]);
                }
                out[y][x] = lane;
            }
        }
        a = out;
        // ι
        let rc = b.word64(KECCAK_RC[round]);
        let mut lane = [0u32; 64];
        for i in 0..64 {
            lane[i] = b.xor(&[a[0][0][i], rc[i]]);
        }
        a[0][0] = lane;
    }
    let mut final_state = [[[0u32; 64]; 5]; 5];
    for y in 0..5 {
        for x in 0..5 {
            final_state[y][x] = a[y][x];
            for &bitcol in &a[y][x] {
                b.io.push(bitcol);
            }
        }
    }
    final_state
}

// ---------------------------------------------------------------------------
// BLAKE3 compression
// ---------------------------------------------------------------------------

pub const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

fn blake3_g(
    b: &mut CircuitBuilder,
    state: &mut [[u32; 32]; 16],
    a: usize,
    bb: usize,
    c: usize,
    d: usize,
    mx: [u32; 32],
    my: [u32; 32],
) {
    // a = a + b + mx ; d = (d ^ a) >>> 16 ; c = c + d ; b = (b ^ c) >>> 12
    let a1 = b.adder(&state[a], &state[bb]);
    let a2 = b.adder(&a1, &mx);
    let d_xor_a: [u32; 32] = std::array::from_fn(|i| b.xor(&[state[d][i], a2[i]]));
    let d_rot16: [u32; 32] = std::array::from_fn(|i| d_xor_a[(i + 16) % 32]);
    let c1 = b.adder(&state[c], &d_rot16);
    let b_xor_c: [u32; 32] = std::array::from_fn(|i| b.xor(&[state[bb][i], c1[i]]));
    let b_rot12: [u32; 32] = std::array::from_fn(|i| b_xor_c[(i + 12) % 32]);
    // a = a + b + my ; d = (d ^ a) >>> 8 ; c = c + d ; b = (b ^ c) >>> 7
    let a3 = b.adder(&a2, &b_rot12);
    let a4 = b.adder(&a3, &my);
    let d_xor_a2: [u32; 32] = std::array::from_fn(|i| b.xor(&[d_rot16[i], a4[i]]));
    let d_rot8: [u32; 32] = std::array::from_fn(|i| d_xor_a2[(i + 8) % 32]);
    let c2 = b.adder(&c1, &d_rot8);
    let b_xor_c2: [u32; 32] = std::array::from_fn(|i| b.xor(&[b_rot12[i], c2[i]]));
    let b_rot7: [u32; 32] = std::array::from_fn(|i| b_xor_c2[(i + 7) % 32]);
    state[a] = a4.try_into().unwrap();
    state[bb] = b_rot7;
    state[c] = c2.try_into().unwrap();
    state[d] = d_rot8;
}

/// Build the BLAKE3 compression circuit for one 64-byte block. `cv`, the
/// 16 message words, counter, block length, and flags are public I/O, as
/// are the 16 output words (the chaining-value form takes the first 8).
pub fn blake3_compress(
    b: &mut CircuitBuilder,
    cv: [u32; 8],
    block: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [Vec<u32>; 16] {
    let mut state: [[u32; 32]; 16] = [[0u32; 32]; 16];
    for i in 0..8 {
        state[i] = b.io_word(cv[i]);
    }
    // v[8..12] = IV[0..4] — circuit constants, not per-instance I/O
    for i in 0..4 {
        state[8 + i] = b.word(BLAKE3_IV[i]);
    }
    state[12] = b.io_word(block_len);
    state[13] = b.io_word(counter as u32);
    state[14] = b.io_word((counter >> 32) as u32);
    state[15] = b.io_word(flags);
    let mut msg: Vec<[u32; 32]> = block.iter().map(|m| b.io_word(*m)).collect();
    for _round in 0..7 {
        for i in 0..4 {
            blake3_g(
                b,
                &mut state,
                i,
                4 + i,
                8 + i,
                12 + i,
                msg[2 * i],
                msg[2 * i + 1],
            );
        }
        let mut perm: Vec<[u32; 32]> = vec![[0u32; 32]; 16];
        for (i, &p) in MSG_PERMUTATION.iter().enumerate() {
            perm[i] = msg[p];
        }
        msg = perm;
        for i in 0..4 {
            blake3_g(
                b,
                &mut state,
                i,
                (i + 1) % 4,
                (i + 2) % 4,
                (i + 3) % 4,
                msg[2 * i],
                msg[2 * i + 1],
            );
        }
        let mut perm: Vec<[u32; 32]> = vec![[0u32; 32]; 16];
        for (i, &p) in MSG_PERMUTATION.iter().enumerate() {
            perm[i] = msg[p];
        }
        msg = perm;
    }
    let mut out: [Vec<u32>; 16] = std::array::from_fn(|_| Vec::new());
    for i in 0..16 {
        out[i] = (0..32)
            .map(|bit| b.xor(&[state[i][bit], state[(i + 8) % 16][bit]]))
            .collect();
        for &bitcol in &out[i] {
            b.io.push(bitcol);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference SHA-256 compression (u32).
    fn ref_sha256_compress(state: [u32; 8], block: [u32; 16]) -> [u32; 8] {
        let rotr = |x: u32, k: u32| x.rotate_right(k);
        let mut w = block.to_vec();
        for t in 16..64 {
            let s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
            let s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w.push(w[t - 16] ^ s0 ^ w[t - 7] ^ s1);
        }
        let [mut a, mut bb, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for t in 0..64 {
            let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[t])
                .wrapping_add(w[t]);
            let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            let maj = (a & bb) ^ (a & c) ^ (bb & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = bb;
            bb = a;
            a = t1.wrapping_add(t2);
        }
        [
            state[0].wrapping_add(a),
            state[1].wrapping_add(bb),
            state[2].wrapping_add(c),
            state[3].wrapping_add(d),
            state[4].wrapping_add(e),
            state[5].wrapping_add(f),
            state[6].wrapping_add(g),
            state[7].wrapping_add(h),
        ]
    }

    fn word_val(b: &CircuitBuilder, w: &[u32]) -> u32 {
        let mut v = 0u32;
        for (i, &bit) in w.iter().enumerate() {
            if b.w[bit as usize] {
                v |= 1 << i;
            }
        }
        v
    }

    fn word64_val(b: &CircuitBuilder, w: &[u32]) -> u64 {
        let mut v = 0u64;
        for (i, &bit) in w.iter().enumerate() {
            if b.w[bit as usize] {
                v |= 1 << i;
            }
        }
        v
    }

    #[test]
    fn sha256_circuit_matches_reference() {
        let state = [
            0x6a09e667u32,
            0xbb67ae85,
            0x3c6ef372,
            0xa54ff53a,
            0x510e527f,
            0x9b05688c,
            0x1f83d9ab,
            0x5be0cd19,
        ];
        let block: [u32; 16] = std::array::from_fn(|i| (i as u32).wrapping_mul(0x9E3779B1));
        let expect = ref_sha256_compress(state, block);
        let mut b = CircuitBuilder::new();
        let out = sha256_compression(&mut b, state, block);
        for i in 0..8 {
            assert_eq!(word_val(&b, &out[i]), expect[i], "output word {i}");
        }
        let n_witness = b.w.len();
        let witness = b.w.clone();
        let cs = b.build();
        assert!(
            cs.satisfied_by(&witness),
            "the witness must satisfy the circuit"
        );
        println!(
            "sha256 compression: rows {} witness {}",
            cs.n_constraints, n_witness
        );
    }

    #[test]
    fn keccak_circuit_matches_reference() {
        // The reference Keccak-f (canonical pseudocode over A[x][y], stored
        // here as a[y][x] — the same convention as the circuit).
        let ref_keccak = |mut a: [[u64; 5]; 5]| -> [[u64; 5]; 5] {
            for round in 0..24 {
                let mut c = [0u64; 5];
                for x in 0..5 {
                    let mut acc = 0;
                    for y in 0..5 {
                        acc ^= a[y][x];
                    }
                    c[x] = acc;
                }
                let mut d = [0u64; 5];
                for x in 0..5 {
                    d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
                }
                for y in 0..5 {
                    for x in 0..5 {
                        a[y][x] ^= d[x];
                    }
                }
                let mut bb = [[0u64; 5]; 5];
                for x in 0..5 {
                    for y in 0..5 {
                        bb[y][(2 * x + 3 * y) % 5] = a[y][x].rotate_left(KECCAK_ROT[x + 5 * y]);
                    }
                }
                for y in 0..5 {
                    for x in 0..5 {
                        a[y][x] = bb[y][x] ^ ((!bb[y][(x + 1) % 5]) & bb[y][(x + 2) % 5]);
                    }
                }
                a[0][0] ^= KECCAK_RC[round];
            }
            a
        };
        let mut state = [[0u64; 5]; 5];
        let mut seed = 0x0123_4567_89ab_cdefu64;
        for y in 0..5 {
            for x in 0..5 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                state[y][x] = seed;
            }
        }
        let expect = ref_keccak(state);
        let mut b = CircuitBuilder::new();
        let mut lanes = [[[0u32; 64]; 5]; 5];
        for y in 0..5 {
            for x in 0..5 {
                lanes[y][x] = b.io_word64(state[y][x]);
            }
        }
        // the inputs are io; strip them so only outputs+inputs stay marked
        let out = keccak_f1600(&mut b, &lanes);
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(word64_val(&b, &out[y][x]), expect[y][x], "lane ({x},{y})");
            }
        }
        let n_witness = b.w.len();
        let witness = b.w.clone();
        let cs = b.build();
        assert!(cs.satisfied_by(&witness));
        println!("keccak: rows {} witness {}", cs.n_constraints, n_witness);
    }

    #[test]
    fn blake3_circuit_matches_reference() {
        fn ref_g(s: &mut [u32; 16], a: usize, bb: usize, c: usize, d: usize, mx: u32, my: u32) {
            s[a] = s[a].wrapping_add(s[bb]).wrapping_add(mx);
            s[d] = (s[d] ^ s[a]).rotate_right(16);
            s[c] = s[c].wrapping_add(s[d]);
            s[bb] = (s[bb] ^ s[c]).rotate_right(12);
            s[a] = s[a].wrapping_add(s[bb]).wrapping_add(my);
            s[d] = (s[d] ^ s[a]).rotate_right(8);
            s[c] = s[c].wrapping_add(s[d]);
            s[bb] = (s[bb] ^ s[c]).rotate_right(7);
        }
        let cv = [
            0x01234567u32,
            0x89abcdef,
            0x0fedcba9,
            0x11112222,
            0x33334444,
            0x55556666,
            0x77778888,
            0x9999aaaa,
        ];
        let block: [u32; 16] = std::array::from_fn(|i| (i as u32).wrapping_mul(0x85ebca6b));
        let counter = 0xdeadbeefcafeu64;
        let block_len = 64u32;
        let flags = 1u32;
        let mut st: [u32; 16] = {
            let mut v: Vec<u32> = Vec::new();
            v.extend_from_slice(&cv);
            v.extend_from_slice(&BLAKE3_IV[..4]);
            v.push(block_len);
            v.push(counter as u32);
            v.push((counter >> 32) as u32);
            v.push(flags);
            v.try_into().unwrap()
        };
        let mut m: [u32; 16] = block;
        for _ in 0..7 {
            for i in 0..4 {
                ref_g(&mut st, i, 4 + i, 8 + i, 12 + i, m[2 * i], m[2 * i + 1]);
            }
            let mut pm = [0u32; 16];
            for (i, &p) in MSG_PERMUTATION.iter().enumerate() {
                pm[i] = m[p];
            }
            m = pm;
            for i in 0..4 {
                ref_g(
                    &mut st,
                    i,
                    (i + 1) % 4,
                    (i + 2) % 4,
                    (i + 3) % 4,
                    m[2 * i],
                    m[2 * i + 1],
                );
            }
            let mut pm = [0u32; 16];
            for (i, &p) in MSG_PERMUTATION.iter().enumerate() {
                pm[i] = m[p];
            }
            m = pm;
        }
        let expect: Vec<u32> = (0..16).map(|i| st[i] ^ st[(i + 8) % 16]).collect();

        let mut b = CircuitBuilder::new();
        let out = blake3_compress(&mut b, cv, block, counter, block_len, flags);
        for i in 0..16 {
            assert_eq!(word_val(&b, &out[i]), expect[i], "word {i}");
        }
        let n_witness = b.w.len();
        let witness = b.w.clone();
        let cs = b.build();
        assert!(cs.satisfied_by(&witness));
        println!("blake3: rows {} witness {}", cs.n_constraints, n_witness);
    }
}
