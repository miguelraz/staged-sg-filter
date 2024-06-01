//#![feature(array_chunks)]
//#![feature(portable_simd)]

pub mod coeffs2;

use iter_comprehensions::map;
use nalgebra::{DMatrix, DVector};
//use std::f64::consts;
//use std::simd::f64x4;
//use std::simd::f64x8;
//use std::simd::StdFloat;

//         .globl  staged_sg_filter::runtime_loop
//         .p2align        4, 0x90
// staged_sg_filter::runtime_loop:

//         lea r8d, [rdx + rdx]
//         xor eax, eax

//         test edx, edx

//         cmovg eax, r8d
//         add eax, ecx

//         ret
#[inline(never)]
pub fn runtime_loop(x: i32, n: i32) -> i32 {
    let mut res = x;
    for _ in 0..n {
        res += 2;
    }
    return res;
}

//         .globl  staged_sg_filter::constant_loop
//         .p2align        4, 0x90
// staged_sg_filter::constant_loop:

//         lea eax, [rcx + 20000000]
//         ret
#[inline(never)]
pub fn const_trip_count_loop(x: i32) -> i32 {
    let mut res = x;
    for _ in 0..10_000_000 {
        res += 2;
    }
    return res;
}

// staged_sg_filter::const_generic_trip_count_loop:
//         lea eax, [rcx + 20000000]
//         ret
// .... or!
//         movl $2000001, (%rsp)
#[inline(never)]
pub fn const_generic_trip_count_loop<const N: i32>(x: i32) -> i32 {
    let mut res = x;
    for _ in 0..N {
        res += 2;
    }
    res
}

//  movss xmm2, dword ptr [rip + .LCPI2_0] ; .LCPI2_0 is 0.5f
//  divss xmm2, xmm1
//  ... (repeats 99 more times)
//
#[inline(always)]
pub fn div_runtime_loop(x: f32, n: f32) -> f32 {
    let mut res = x;
    for _ in 0..100 {
        res += 1.0 / n;
    }
    res
}

#[inline(never)]
pub fn div_consttime_loop<const N: i32>(x: f32) -> f32 {
    let mut res = x;
    for _ in 0..100 {
        res += 1.0 / (N as f32);
    }
    res
}

macro_rules! const_div {
    ($x: expr) => {
        1.0 / ($x as f64)
    };
}

macro_rules! const_chunker {
    ($x: expr, $y: expr) => {
        $x / $y
    };
}

#[inline(never)]
pub fn rolling_average_raw_loop<const N: i32, const M: i32>(invec: &[f64], outvec: &mut [f64]) {
    let inv = const_div!(N);
    for i in 0..(M as usize) {
        outvec[i] = invec[i] * inv;
    }
}
#[inline(never)]
pub fn rolling_average_iter_array<const N: i32, const M: usize>(
    invec: [f64; M],
    outvec: &mut [f64; M],
) {
    let inv = const_div!(N);
    invec
        .iter()
        .zip(outvec.iter_mut())
        .for_each(|(a, b)| *b = *a * inv);
}
#[inline(never)]
pub fn rolling_average_iter_vec<const N: i32, const M: usize>(
    invec: Vec<f64>,
    outvec: &mut Vec<f64>,
) {
    let inv = const_div!(N);
    invec
        .iter()
        .zip(outvec.iter_mut())
        .for_each(|(a, b)| *b = *a * inv);
}

/*
#[inline(never)]
pub fn rolling_average_simd_array<const N: usize, const M: usize>(
    invec: [f64; M],
    outvec: &mut [f64; M],
) {
    let inv = const_div!(N);
    let vinv = f64x8::splat(inv);
    let fmul = f64x8::splat(42.0);

    outvec
        .array_chunks_mut::<8>()
        .zip(invec.array_chunks::<8>().map(|&b| f64x8::from_array(b)))
        .for_each(|(a, b)| {
            let vb = f64x8::from(b);
            let res = vb.mul_add(vinv, fmul).to_array();
            //let res = (vb * vinv).to_array();
            *a = res;
        });
}
*/

#[test]
fn rolling_avg_const_size_window_simdf64x8() {
    const N: usize = 8;
    const M: usize = 8;
    const CHUNKS: i32 = 8;
    let a = [1.0; M];
    let mut b = [0.0; M];

    //    rolling_average_simd_array::<{ N }, { M }>(a, &mut b);

    let rhs = [42.125; M];

    assert_eq!(b, rhs);
}

// J = T[(i - M - 1 )^(j - 1) for i = 1:2M + 1, j = 1:N + 1]
// e₁ = [one(T); zeros(T,N)]
// C = J' \ e₁
#[macro_export]
macro_rules!  get_rolling_coeffs {
    ($WINDOW_SIZE: expr, $M: expr) => {
        {
        let vec = map!(
            ((i - $M - 1) as i32).pow((j as u32) - 1) as f64;
                i in 1..=(2*$M + 1),
                j in 1..=($WINDOW_SIZE+1))
            .collect::<Vec<f64>>();
        let J = DMatrix::from_vec($WINDOW_SIZE + 1, 2*$M + 1, vec);
        let mut temp = vec![1.0];
        let mut zeros = vec![0.0; $M];
        temp.append(&mut zeros);
        let e_1 = DVector::from_vec(temp);
        println!("e_1 len is {}", e_1.len());
        println!("J len is {}", J.len());
        let jtick = J.adjoint();
        let decomp = jtick.lu();

        let C = decomp.solve(&e_1).expect("Linear resolution failed");
        C
        }
    };
}

fn rolling_avg_const_linear_solve<const WINDOW_SIZE: usize, const M: usize>(// invec: [f64; M],
    // outvec: [f64; M],
) -> Vec<f64> {
    // J = T[(i - M - 1 )^(j - 1) for i = 1:2M + 1, j = 1:N + 1]
    // e₁ = [one(T); zeros(T,N)]
    // C = J' \ e₁
    let C = get_rolling_coeffs!(WINDOW_SIZE, M);
    dbg!(C);
    vec![1.0]
}

#[test]
fn test_rolling_avg_const_linear_solve() {
    //let C = rolling_avg_const_linear_solve::<8, 8>(a, b);
    let n = 2;
    let m = 2;
    let c = get_rolling_coeffs!(1, 1);
    //let d = c.as_vec();
    //assert_eq!(nalgebra::dmatrix!(0.25), d);
}

mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[test]
    fn runtime_loop_test() {
        assert_eq!(runtime_loop(1, 10), 21);
    }
    #[test]
    fn constant_loop_test() {
        assert_eq!(const_trip_count_loop(1), 20_000_001);
    }
    #[test]
    fn constant_generic_loop_test() {
        assert_eq!(const_generic_trip_count_loop::<10_000_000>(1), 20_000_001);
    }
    #[test]
    fn div_rt_loop() {
        assert_eq!(div_runtime_loop(100.0, 2.0), 150.0);
    }
    #[test]
    fn div_ct_loop() {
        assert_eq!(div_consttime_loop::<2>(100.0), 150.0);
    }
    #[test]
    fn rolling_avg_const_size_window_raw_loop() {
        const N: i32 = 4;
        const M: i32 = 4;
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let mut b = vec![0.0; 4];

        rolling_average_raw_loop::<{ N }, { M }>(&a, &mut b);
        let rhs = vec![0.25, 0.25, 0.25, 0.25];

        assert_eq!(b, rhs);
    }
    #[test]
    fn rolling_avg_const_size_window_iter_array() {
        const N: i32 = 4;
        const M: usize = 4;
        let a = [1.0, 1.0, 1.0, 1.0];
        let mut b = [0.0; 4];

        rolling_average_iter_array::<{ N }, { M }>(a, &mut b);
        let rhs = [0.25, 0.25, 0.25, 0.25];

        assert_eq!(b, rhs);
    }
    #[test]
    fn rolling_avg_const_size_window_iter_vec() {
        const N: i32 = 4;
        const M: usize = 4;
        let a = vec![1.0; M];
        let mut b = vec![0.0; M];

        rolling_average_iter_vec::<{ N }, { M }>(a, &mut b);
        let rhs = vec![0.25; M];

        assert_eq!(b, rhs);
    }
}
