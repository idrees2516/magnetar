fn main() {
    let kind = magnetar_field::avx::detect();
    println!("detected: {kind:?}");
    let a = [magnetar_field::Gf128(0x1234_5678_9abc_def0_1234_5678_9abc_def0u128); 4];
    let b = [magnetar_field::Gf128(0x0f0f_0f0f_0f0f_0f0f_f0f0_f0f0_f0f0_f0f0u128); 4];
    let r = magnetar_field::avx::mul4(a, b);
    println!("mul4 result: {:x}", r[0].0);
}
