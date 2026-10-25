//! Debug helper: print the circuit's real constraint counts and nnz.
#![allow(dead_code)]

#[test]
fn print_circuit_sizes() {
    let spec = magnetar_cpu::build_circuit();
    println!("rows (padded total): {}", spec.a_bin.rows);
    println!("cols m: {}", spec.m);
    println!(
        "bin nnz: A={} B={} C={}",
        spec.a_bin.entries.len(),
        spec.b_bin.entries.len(),
        spec.c_bin.entries.len()
    );
    println!(
        "fld nnz: A={} B={} C={}",
        spec.a_fld.entries.len(),
        spec.b_fld.entries.len(),
        spec.c_fld.entries.len()
    );
    let mut maxcol = 0usize;
    for m in [
        &spec.a_bin,
        &spec.b_bin,
        &spec.c_bin,
        &spec.a_fld,
        &spec.b_fld,
        &spec.c_fld,
    ] {
        for e in &m.entries {
            maxcol = maxcol.max(e.1);
        }
    }
    println!("max z column referenced: {}", maxcol);
    // Real (unpadded) row counts: track the max row index + 1 per relation.
    let mut bin_max = 0usize;
    let mut fld_max = 0usize;
    for m in [&spec.a_bin, &spec.b_bin, &spec.c_bin] {
        for e in &m.entries {
            bin_max = bin_max.max(e.0 + 1);
        }
    }
    for m in [&spec.a_fld, &spec.b_fld, &spec.c_fld] {
        for e in &m.entries {
            fld_max = fld_max.max(e.0 + 1);
        }
    }
    println!("real bin rows: {}, real fld rows: {}", bin_max, fld_max);
}
