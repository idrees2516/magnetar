//! Tree-IVC end-to-end tests: honest multi-segment proofs, the boundary
//! chain, and the tamper classes.

use magnetar_ivc::{prove_ivc, verify_ivc};
use magnetar_pcs::PcsParams;

fn enc_i(imm: u32, rs1: u32, funct3: u32, rd: u32, op: u32) -> u32 {
    ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | op
}

/// A program whose execution spans several segments: load the input,
/// accumulate it eight times, emit, halt.
fn sum_program() -> Vec<u32> {
    let mut p = vec![(3 << 12) | (1 << 7) | 0x03]; // ld x1, 0(x0)
    for _ in 0..7 {
        p.push((2u32 << 20) | (1 << 15) | (2 << 7) | 0x33); // add x2, x1, x2
    }
    p.push(enc_i(0, 2, 0, 10, 0x13)); // addi x10, x2, 0
    p.push(0x0000_0073); // ecall
    p
}

#[test]
fn ivc_four_segments_roundtrip() {
    let program = sum_program();
    let input = vec![7u64];
    let output = 7u64 * 7;
    let proof = prove_ivc(&program, &input, output, 32, 4, &PcsParams::default());
    assert_eq!(proof.statements.len(), 4);
    assert_eq!(proof.inners.len(), 4);
    for w in proof.statements.windows(2) {
        assert_eq!(w[0].final_, w[1].init);
    }
    assert!(verify_ivc(&program, &input, output, 32, &proof, &PcsParams::default()).is_ok());
}

#[test]
fn ivc_single_segment_matches_shape() {
    let program = sum_program();
    let input = vec![3u64];
    let output = 21u64;
    let proof = prove_ivc(&program, &input, output, 16, 1, &PcsParams::default());
    assert!(verify_ivc(&program, &input, output, 16, &proof, &PcsParams::default()).is_ok());
}

#[test]
fn ivc_eight_segments() {
    let program = sum_program();
    let input = vec![5u64];
    let output = 35u64;
    let proof = prove_ivc(&program, &input, output, 32, 8, &PcsParams::default());
    assert!(verify_ivc(&program, &input, output, 32, &proof, &PcsParams::default()).is_ok());
}

#[test]
fn ivc_tamper_rejected() {
    let program = sum_program();
    let input = vec![7u64];
    let output = 49u64;
    let proof = prove_ivc(&program, &input, output, 32, 4, &PcsParams::default());
    let ok = |p: &magnetar_ivc::IvcProof| {
        verify_ivc(&program, &input, output, 32, p, &PcsParams::default()).is_ok()
    };
    assert!(ok(&proof));

    // 1. A wrong output claim.
    assert!(verify_ivc(
        &program,
        &input,
        output + 1,
        32,
        &proof,
        &PcsParams::default()
    )
    .is_err());

    // 2. A broken boundary chain.
    {
        let mut p = clone_of(&proof);
        p.statements[1].init.regs[2] += 1;
        assert!(!ok(&p), "broken chain must fail");
    }
    // 3. A wrong final RAM on the last segment (the final-state binding).
    {
        let mut p = clone_of(&proof);
        p.statements[3].final_.ram[0] ^= 1;
        assert!(!ok(&p), "wrong final RAM must fail");
    }
    // 4. A wrong final register file.
    {
        let mut p = clone_of(&proof);
        p.statements[3].final_.regs[2] ^= 1;
        assert!(!ok(&p), "wrong final regs must fail");
    }
    // 5. The wrong initial halt latch.
    {
        let mut p = clone_of(&proof);
        p.statements[0].init.halted = true;
        assert!(!ok(&p), "wrong initial halt must fail");
    }
    // 6. The wrong final halt latch.
    {
        let mut p = clone_of(&proof);
        p.statements[3].final_.halted = false;
        assert!(!ok(&p), "wrong final halt must fail");
    }
    // 7. A tampered final-state sum-check round.
    {
        let mut p = clone_of(&proof);
        if let Some(fs) = p.inners[3].final_ram.as_mut() {
            fs.zero_rounds[0][1] += magnetar_field::Gf128(1);
        }
        assert!(!ok(&p), "tampered fs round must fail");
    }
    // 8. A tampered segment digest.
    {
        let mut p = clone_of(&proof);
        p.segment_digests[1][0] ^= 0xFF;
        assert!(!ok(&p), "tampered digest must fail");
    }
    // 9. A tampered Mosaic round.
    {
        let mut p = clone_of(&proof);
        p.mosaic.batch_rounds[0][1] += magnetar_field::Gf128(1);
        assert!(!ok(&p), "tampered mosaic must fail");
    }
    // 10. A tampered commitment root.
    {
        let mut p = clone_of(&proof);
        p.commitment_root[0] ^= 0xFF;
        assert!(!ok(&p), "tampered commitment must fail");
    }
    // 11. A tampered inner proof (a twist round).
    {
        let mut p = clone_of(&proof);
        p.inners[0].reg_twist.read_runs[0].rounds[0][1] += magnetar_field::Gf128(1);
        assert!(!ok(&p), "tampered twist must fail");
    }
}

fn clone_of(p: &magnetar_ivc::IvcProof) -> magnetar_ivc::IvcProof {
    magnetar_ivc::IvcProof {
        statements: p.statements.clone(),
        inners: p.inners.clone(),
        segment_digests: p.segment_digests.clone(),
        commitment_root: p.commitment_root,
        pcs_n: p.pcs_n,
        pcs_k: p.pcs_k,
        pcs_domain: p.pcs_domain.clone(),
        ell: p.ell,
        mosaic: p.mosaic.clone(),
    }
}
