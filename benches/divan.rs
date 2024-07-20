use staged_sg_filter::{sav_gol, sav_gol_f32};
//use staged_sg_filter::utils::*;

use divan::black_box as bb;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

/*
// Define a `fibonacci` function and register it for benchmarking.
#[divan::bench]
fn divan_runtime_loop() -> i32 {
    runtime_loop(bb(10_000_00), bb(20_000_000))
}
#[divan::bench]
fn divan_constant_loop() -> i32 {
    const_trip_count_loop(bb(1))
}
#[divan::bench]
fn divan_constant_generic_loop() -> i32 {
    const_generic_trip_count_loop::<10_000_000>(bb(1))
}
#[divan::bench]
fn rolling_average_loop_vec_bench() -> Vec<f64> {
    const N: i32 = 4;
    const M: i32 = 10_000;
    let a = vec![1.0; M as usize];
    let mut b = vec![0.0; M as usize];
    rolling_average_raw_loop::<{ N }, { M }>(bb(&a), bb(&mut b));
    b.into()
}

#[divan::bench]
fn rolling_average_iter_array_bench() -> Vec<f64> {
    const N: i32 = 4;
    // Codegen options!
    // good 152
    // bad 151
    const M: usize = 10_000;
    let a = [1.0; M];
    let mut b = [0.0; M];
    rolling_average_iter_array::<{ N }, { M }>(bb(a), bb(&mut b));
    b.into()
}
#[divan::bench]
fn rolling_average_iter_vec_bench() -> Vec<f64> {
    const N: i32 = 4;
    // Codegen options: M.len == 151
    // good 152
    // bad 151
    const M: usize = 10_000;
    let a = vec![1.0; M];
    let mut b = vec![0.0; M];
    rolling_average_iter_vec::<{ N }, { M }>(bb(a), bb(&mut b));
    b.into()
}
#[divan::bench]
fn simd_average_array() -> Vec<f64> {
    const N: usize = 4;
    const M: usize = 10_000;
    const CHUNKS: i32 = 4;
    let a = [1.0; M];
    let mut b = [0.0; M];

    rolling_average_simd_array::<{ N }, { M }>(bb(a), bb(&mut b));
    b.into()
}
#[divan::bench]
fn div_rt_loop() -> f32 {
    div_runtime_loop(bb(100.0), bb(2.0))
}
#[divan::bench]
fn div_ct_loop() -> f32 {
    div_consttime_loop::<2>(bb(100.0))
}

*/
#[divan::bench(sample_size = 3, sample_count = 3)]
fn savgol_f64() -> f64 {
    let n =  100_000_000;
    let v = vec![10.0; n];
    let mut buf = vec![0.0; n];
    sav_gol::<2, 2>(bb(&mut buf), bb(&v));
    buf[0]
}
#[divan::bench(sample_size = 3, sample_count = 3)]
fn savgol_f32() -> f32 {
    let n = 100_000_000;
    let v = vec![10.0f32; n];
    let mut buf = vec![0.0f32; n];
    sav_gol_f32::<2, 2>(bb(&mut buf), bb(&v));
    buf[0]
}
