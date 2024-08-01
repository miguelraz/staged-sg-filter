//#![feature(array_chunks)]
//#![feature(portable_simd)]

use coeffs::get_coeffs;
use coeffs_f32::get_coeffs_f32;

pub mod coeffs;
pub mod coeffs_f32;

/// Small utility function to clean up the `sav_gol` filter
#[inline]
pub fn dot_prod_update(buf: &mut f64, data: &[f64], coeffs: &[f64]) {
    if !cfg!(feature = "std") {
    *buf = data
        .iter()
        .zip(coeffs.iter())
        .fold(0.0f64, |acc, (a, b)| a.mul_add(*b, acc));
    } else {
    *buf = data
        .iter()
        .zip(coeffs.iter())
        .fold(0.0f64, |acc, (a, b)| a * (*b) + acc);
    }
}

#[inline]
pub fn dot_prod_update_f32(buf: &mut f32, data: &[f32], coeffs: &[f32]) {
    if !cfg!(feature = "std") {
    *buf = data
        .iter()
        .zip(coeffs.iter())
        .fold(0.0f32, |acc, (a, b)| a.mul_add(*b, acc));
    } else {
    *buf = data
        .iter()
        .zip(coeffs.iter())
        .fold(0.0f32, |acc, (a, b)| a * (*b) + acc);
    }
}

#[test]
fn test_dot_prod_update() {
    let mut buf = 0.0;
    let data = vec![1.0; 4];
    let coeffs = vec![0.25; 4];
    dot_prod_update(&mut buf, &data, &coeffs);
    assert_eq!(buf, 1.0);
}
#[test]
fn test_dot_prod_update_f32() {
    let mut buf = 0.0f32;
    let data = vec![1.0f32; 4];
    let coeffs = vec![0.25f32; 4];
    dot_prod_update_f32(&mut buf, &data, &coeffs);
    assert_eq!(buf, 1.0f32);
}

/// Savitzky-Golay smoothing filter
///
/// Iterator element type is `f32`/`f64`.
///
/// If `WINDOW` is `1` then `window_size` is `3` (1 on either side plus the element itself.)
/// If `M` is `3` then `3` statistical momenta are conserved.
///
/// This filter ignores elements on the fringes (starting and ending `window_size`) elements of the array.
/// There also exists a parallel version of this filter as `par_sav_gol`, behind a feature flag `par_sav_gol`.
/// ```
///     use staged_sg_filter::sav_gol;
///     let mut v = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
///     let mut buf = vec![0.0; 7];
///     sav_gol::<1, 1>(&mut buf, &v);
///     let res = vec![0.0, 0.3333333333333333, 0.6666666666666666, 0.3333333333333333, 0.6666666666666666, 0.3333333333333333, 0.0];
///     assert_eq!(res, buf);
///```
pub fn sav_gol<const WINDOW: usize, const M: usize>(buf: &mut [f64], data: &[f64]) {
    let coeffs = get_coeffs::<WINDOW, M>();
    let window_size = 2 * WINDOW + 1;
    let body_size = data.len() - (window_size - 1);
    buf.iter_mut()
        // Start the iteration without the `windows` reaching before `buf` starts
        .skip(window_size / 2)
        .zip(data.windows(window_size))
        // Advance `body_size` iterations so that `windows` doesn't go over the end of `buf`
        .take(body_size)
        .for_each(|(buf, data)| {
            dot_prod_update(buf, data, coeffs);
        });
}

pub fn sav_gol_f32<const WINDOW: usize, const M: usize>(buf: &mut [f32], data: &[f32]) {
    let coeffs = get_coeffs_f32::<WINDOW, M>();
    let window_size = 2 * WINDOW + 1;
    let body_size = data.len() - (window_size - 1);
    buf.iter_mut()
        // Start the iteration without the `windows` reaching before `buf` starts
        .skip(window_size / 2)
        .zip(data.windows(window_size))
        // Advance `body_size` iterations so that `windows` doesn't go over the end of `buf`
        .take(body_size)
        .for_each(|(buf, data)| {
            dot_prod_update_f32(buf, data, coeffs);
        });
}

#[test]
fn test_sav_gol() {
    let v = vec![0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0];
    let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    sav_gol::<1, 1>(&mut buf, &v);
    let res = vec![
        1.0,
        3.333333333333333,
        6.666666666666666,
        3.333333333333333,
        6.666666666666666,
        3.333333333333333,
        7.0,
    ];
    assert_eq!(res, buf);
}

#[test]
fn test_sav_golf32() {
    let v = vec![0.0f32, 10.0f32, 0.0f32, 10.0f32, 0.0f32, 10.0f32, 0.0f32];
    let mut buf = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32];
    sav_gol_f32::<1, 1>(&mut buf, &v);
    let res = vec![
        1.0f32,
        3.3333335f32,
        6.666667f32,
        3.3333335f32,
        6.666667f32,
        3.3333335f32,
        7.0f32,
    ];
    assert_eq!(res, buf);
}
// dynamic data (must accept args)
// mark as #[inline(never)]
// cargo asm --lib
#[inline(never)]
pub fn asm_dump_f64(buf: &mut [f64], data: &mut [f64]) {
    sav_gol::<2, 2>(buf, &data);
}

#[inline(never)]
pub fn asm_dump_f32(buf: &mut [f32], data: &mut [f32]) {
    sav_gol_f32::<2, 2>(buf, &data);
}

#[cfg(feature = "rayon")]
pub fn par_sav_gol<const WINDOW: usize, const M: usize>(buf: &mut [f64], data: &[f64]) {
    use rayon::prelude::*;
    let coeffs = get_coeffs::<WINDOW, M>();
    let window_size = 2 * WINDOW + 1;
    let body_size = data.len() - (window_size - 1);
    buf.par_iter_mut()
        .skip(window_size / 2)
        .zip(data.par_windows(window_size))
        .take(body_size)
        .for_each(|(buf, data)| {
            dot_prod_update(buf, &data, &coeffs);
        });
}

#[cfg(feature = "rayon")]
pub fn par_sav_gol_f32<const WINDOW: usize, const M: usize>(buf: &mut [f32], data: &[f32]) {
    use rayon::prelude::*;
    let coeffs = get_coeffs_f32::<WINDOW, M>();
    let window_size = 2 * WINDOW + 1;
    let body_size = data.len() - (window_size - 1);
    buf.par_iter_mut()
        .skip(window_size / 2)
        .zip(data.par_windows(window_size))
        .take(body_size)
        .for_each(|(buf, data)| {
            dot_prod_update(buf, &data, &coeffs);
        });
}
