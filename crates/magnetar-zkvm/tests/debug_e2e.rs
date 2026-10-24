//! Debug: print the verify error for the failing e2e tests.
use magnetar_zkvm::{keygen, prove, verify, PublicInput};

fn enc_i(imm: u32, rs1: u32, f3: u32, rd: u32, op: u32) -> u32 {
    ((imm & 0x1f) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
}

fn enc_s(f3: u32, rs2: u32, rs1: u32, imm: u32, op: u32) -> u32 {
    ((imm & 0x1f) << 20) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd_0()) | op
}
fn rd_0() -> u32 {
    0
}

#[test]
fn debug_hot_cell_error() {
    let program = vec![
        (3 << 12) | (3 << 7) | 0x03, // ld x3, 0(x0)
        enc_i(7, 0, 0, 5, 0x13),
        (5u32 << 20) | (3 << 15) | (3 << 7) | 0x33, // add x3, x3, x5
        enc_s(3, 3, 0, 0, 0x23),                    // sd x3, 0(x0)
        (3 << 12) | (10 << 7) | 0x03,               // ld x10, 0(x0)
        0x0000_0073,
    ];
    let (pk, vk) = keygen(program, 8);
    let pub_in = PublicInput {
        input: vec![7],
        output: 14,
    };
    let proof = prove(&pk, &pub_in);
    let res = verify(&vk, &pub_in, &proof);
    println!("VERIFY RESULT: {res:?}");
    assert!(res.is_ok());
}
