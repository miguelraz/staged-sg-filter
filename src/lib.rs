//#![feature(array_chunks)]
//#![feature(portable_simd)]

use coeffs::get_coeffs;

pub mod coeffs;

/// Small utility function to clean up the `sav_gol` filter
pub fn dot_prod_update(buf: &mut f64, data: &[f64], coeffs: &[f64]) {
    assert!(data.len() == coeffs.len());
    *buf = data
        .iter()
        .zip(coeffs.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
}

#[test]
fn test_dot_prod_update() {
    let mut buf = 0.0;
    let data = vec![1.0; 4];
    let coeffs = vec![0.25; 4];
    dot_prod_update(&mut buf, &data, &coeffs);
    assert_eq!(buf, 1.0);
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
///     let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
///     let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
///     sav_gol::<1, 1>(&mut buf, &v);
///     let res = vec[1.0, 3.333333333333333, 6.666666666666666, 3.333333333333333, 6.666666666666666, 3.333333333333333, 7.0, ];
///     assert_eq!(res, buf);
///```
pub fn sav_gol<const WINDOW: usize, const M: usize>(buf: &mut [f64], data: &[f64]) {
    let coeffs = get_coeffs::<WINDOW, M>();
    assert!(buf.len() == data.len());
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

#[cfg(feature = "rayon")]
pub fn par_sav_gol<const WINDOW: usize, const M: usize>(buf: &mut [f64], data: &[f64]) {
    use rayon::prelude::*;
    let coeffs = get_coeffs::<WINDOW, M>();
    assert!(buf.len() == data.len());
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
