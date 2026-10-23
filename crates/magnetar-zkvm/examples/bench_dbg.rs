use magnetar_zkvm::protocol::{keygen, prove, verify, PublicInput};
fn main() {
    let enc_i = |imm: u32, rs1: u32, funct3: u32, rd: u32| {
        ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x13
    };
    let bne = |off: i32| -> u32 {
        let imm = off as u16;
        let i12 = ((imm >> 12) & 1) as u32;
        let i11 = ((imm >> 11) & 1) as u32;
        let i10_5 = ((imm >> 5) & 0x3f) as u32;
        let i4_1 = ((imm >> 1) & 0xf) as u32;
        (i12 << 31) | (i10_5 << 25) | (3 << 15) | (1 << 12) | (i4_1 << 8) | (i11 << 7) | 0x63
    };
    let program = vec![
        enc_i(2, 0, 0, 3),
        enc_i(1, 0, 0, 1),
        enc_i(1, 0, 0, 2),
        enc_i(0xfff, 3, 0, 3),
        (2 << 20) | (1 << 15) | (4 << 7) | 0x33,
        enc_i(0, 2, 0, 1),
        enc_i(0, 4, 0, 2),
        bne(-24),
        enc_i(0, 2, 0, 10),
        0x0000_0073,
    ];
    let trace = magnetar_cpu::interp::execute(&program, &[], 32);
    println!(
        "halted={} records={}",
        trace.final_state.halted,
        trace.records.len()
    );
    let (pk, vk) = keygen(program, 32);
    let pub_in = PublicInput {
        input: vec![],
        output: trace.final_state.output,
    };
    let proof = prove(&pk, &pub_in);
    match verify(&vk, &pub_in, &proof) {
        Ok(()) => println!("verified OK"),
        Err(e) => println!("verify error: {e}"),
    }
}
