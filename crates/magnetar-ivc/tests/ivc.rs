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
    assert!(verify_ivc(
        &program,
        &input,
        output,
        32,
        &proof,
        &PcsParams::default(),
        magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
    )
    .is_ok());
}

#[test]
fn ivc_single_segment_matches_shape() {
    let program = sum_program();
    let input = vec![3u64];
    let output = 21u64;
    let proof = prove_ivc(&program, &input, output, 16, 1, &PcsParams::default());
    assert!(verify_ivc(
        &program,
        &input,
        output,
        16,
        &proof,
        &PcsParams::default(),
        magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
    )
    .is_ok());
}

#[test]
fn ivc_eight_segments() {
    let program = sum_program();
    let input = vec![5u64];
    let output = 35u64;
    let proof = prove_ivc(&program, &input, output, 32, 8, &PcsParams::default());
    assert!(verify_ivc(
        &program,
        &input,
        output,
        32,
        &proof,
        &PcsParams::default(),
        magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
    )
    .is_ok());
}

#[test]
fn ivc_tamper_rejected() {
    let program = sum_program();
    let input = vec![7u64];
    let output = 49u64;
    let proof = prove_ivc(&program, &input, output, 32, 4, &PcsParams::default());
    let ok = |p: &magnetar_ivc::IvcProof| {
        verify_ivc(
            &program,
            &input,
            output,
            32,
            p,
            &PcsParams::default(),
            magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI,
        )
        .is_ok()
    };
    assert!(ok(&proof));

    // 1. A wrong output claim.
    assert!(verify_ivc(
        &program,
        &input,
        output + 1,
        32,
        &proof,
        &PcsParams::default(),
        magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
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
        p.mosaic.local_rounds[0][0][1] += magnetar_field::Gf128(1);
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

/// The streaming prover must produce **byte-identical** proofs to the
/// sequential prover (same statements, same per-segment transcripts,
/// same commitment) on 1, 2 and 4 segments — the checkpoint/replay
/// pipeline is a pure memory-path change (docs/PERFORMANCE.md R7).
#[test]
fn streaming_matches_sequential() {
    let program = sum_program();
    let input = vec![7u64];
    let output = 49u64;
    for segments in [1usize, 2, 4] {
        let total = 32;
        let seq = prove_ivc(
            &program,
            &input,
            output,
            total,
            segments,
            &PcsParams::default(),
        );
        let (stream, stats) = magnetar_ivc::prove_ivc_streaming(
            &program,
            &input,
            output,
            total,
            segments,
            &PcsParams::default(),
            magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI,
        );
        // Identical commitment and transcript digests.
        assert_eq!(seq.commitment_root, stream.commitment_root);
        assert_eq!(seq.segment_digests, stream.segment_digests);
        assert_eq!(seq.pcs_n, stream.pcs_n);
        assert_eq!(seq.ell, stream.ell);
        // Identical statements (the boundary chain).
        assert_eq!(
            format!("{:?}", seq.statements),
            format!("{:?}", stream.statements)
        );
        // Identical inner proofs and discharge.
        assert_eq!(format!("{:?}", seq.inners), format!("{:?}", stream.inners));
        assert_eq!(format!("{:?}", seq.mosaic), format!("{:?}", stream.mosaic));
        // And the streaming proof verifies.
        assert!(verify_ivc(
            &program,
            &input,
            output,
            total,
            &stream,
            &PcsParams::default(),
            magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
        )
        .is_ok());
        // The stats are populated (R10).
        assert!(stats.wall_total >= stats.wall_checkpoints + stats.wall_root);
        assert!(stats.packed_bytes > 0);
        // The single-segment root materializes the combined codeword;
        // the multi-segment (split-ring) root never does — its
        // per-segment commitment trees live inside the discharge.
        if segments == 1 {
            assert!(stats.codeword_bytes >= stats.packed_bytes);
        } else {
            assert_eq!(stats.codeword_bytes, 0);
        }
        assert!(stats.segment_table_bytes > 0);
        assert!(stats.checkpoint_bytes > 0);
        assert!(stats.threads >= 1);
    }
}

/// The streaming prover's segments verify individually and the whole
/// proof rejects the standard tamper classes (spot-check two).
#[test]
fn streaming_roundtrip_and_tamper() {
    let program = sum_program();
    let input = vec![5u64];
    let output = 35u64;
    let (proof, _stats) = magnetar_ivc::prove_ivc_streaming(
        &program,
        &input,
        output,
        32,
        4,
        &PcsParams::default(),
        magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI,
    );
    assert!(verify_ivc(
        &program,
        &input,
        output,
        32,
        &proof,
        &PcsParams::default(),
        magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
    )
    .is_ok());
    // Tamper: a broken boundary chain.
    let mut p = clone_of(&proof);
    p.statements[2].init.regs[2] += 1;
    assert!(verify_ivc(
        &program,
        &input,
        output,
        32,
        &p,
        &PcsParams::default(),
        magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
    )
    .is_err());
    // Tamper: a wrong output.
    assert!(verify_ivc(
        &program,
        &input,
        output + 1,
        32,
        &proof,
        &PcsParams::default(),
        magnetar_cpu::WITNESS_LAYOUT::DEFAULT_CPI
    )
    .is_err());
}

/// The checkpointed replay regenerates the exact full padded trace
/// (the deepest equivalence: record-level, before any proving).
#[test]
fn stream_cursor_matches_full_trace() {
    use magnetar_cpu::expand_program;
    use magnetar_cpu::interp::{execute, pad_records, StreamCursor};

    let program = sum_program();
    let input = vec![9u64];
    let total = 64usize;
    let seg_len = 16usize;
    // The sequential path: execute + pad to total.
    let mut full = execute(&program, &input, total);
    pad_records(
        &mut full.records,
        &mut full.final_state,
        &full.program,
        total,
    );
    assert_eq!(full.records.len(), total);
    // The streaming path: checkpoint + per-segment replay.
    let prog = expand_program(&program);
    let mut cursor = StreamCursor::start(&input);
    let mut checkpoints = vec![cursor.clone()];
    for _ in 0..total / seg_len {
        let _ = cursor.run_segment(&prog, seg_len);
        checkpoints.push(cursor.clone());
    }
    let mut replayed = Vec::new();
    for cp in &checkpoints[..total / seg_len] {
        let mut c = cp.clone();
        replayed.extend(c.run_segment(&prog, seg_len));
    }
    assert_eq!(full.records, replayed);
    // And the final state agrees.
    assert_eq!(cursor.st.regs, full.final_state.regs);
    assert_eq!(cursor.st.ram, full.final_state.ram);
    assert_eq!(cursor.st.halted, full.final_state.halted);
    assert_eq!(cursor.st.output, full.final_state.output);
}

/// The instance-batched streaming ladder (paper §5.1): several cycles per
/// SIMD instance through the full tree-IVC pipeline — the boundary chain,
/// the per-segment claim plumbing (`[δ-vars ‖ instance-vars]` cycle
/// points), and the shared Mosaic discharge all carry the batched shape.
#[test]
fn ivc_instance_batched_cpi() {
    let program = sum_program();
    let input = vec![7u64];
    let output = 7u64 * 7;
    for cpi in [2usize, 4] {
        let (proof, stats) = magnetar_ivc::prove_ivc_streaming(
            &program,
            &input,
            output,
            32,
            2,
            &PcsParams::default(),
            cpi,
        );
        assert!(stats.packed_bytes > 0);
        assert!(
            magnetar_ivc::verify_ivc(
                &program,
                &input,
                output,
                32,
                &proof,
                &PcsParams::default(),
                cpi
            )
            .is_ok(),
            "cpi={cpi}: the batched streaming proof must verify"
        );
        // The shape: 2 segments of 16 cycles each → 16/cpi instances.
        assert_eq!(proof.statements.len(), 2);
    }
}
