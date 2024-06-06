# Staged Filters

* A Savitzky-Golar filter that is fast, baby.
* All (N,M) parameters are precomputed and pulled in at compile time.
* `rayon` support is available via a `rayon` feature flag
* Still some SIMD perf left on the table - newer versions will focus on perf

* Remember to compile this with `RUSTFLAGS="-C target-cpu=native"`.

This code is based on another code I adapted in Julia with much help from others, see [StagedFilters.jl](https://github.com/miguelraz/StagedFilters.jl).

## Example

## Benchmarks

The other `savgol-rs` implementation offers this speed:

```rust
// took 52s
use savgol_rs::*;
fn main() {
    let input = SavGolInput {
        data: &vec![10.0; 500_000],
        window_length: 3,
        poly_order: 1,
        derivative: 0,
    };
    let result = savgol_filter(&input);
    let data = result.unwrap();
    println!("{:?}", &data[0..10]);
}
```

whereas this crate

```rust
use staged_sg_filter::sav_gol;

fn main() {
    let v = vec![10.0; 500_000];
    let mut buf = vec![0.0; 500_000];
    sav_gol::<1, 1>(&mut buf, &v);

    println!("{:?}", &buf[0..10]);
}
```

runs in about 100ms.

Using `divan` on a vector of `10_000_000` `f64`'s gives::

```
     Running benches\divan.rs (target\release\deps\divan-fb5954a85863758b.exe)
Timer precision: 100 ns
divan      fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ savgol  205.8 ms      │ 297.4 ms      │ 218.5 ms      │ 219.6 ms      │ 100     │ 100
```

which means it's churning through about `10_000_000/0.2 ≈ 5e7` elements per second or `5e7 * 10e-9 ≈ 0.5` elements per nanosecond. Not bad!

## Notes

It's called "staged" because the computation is done in "stages", which allows the compiler to optimize the code a lot more - namely, the use of const generics in Rust provide more opportunities for profitable loop unrolling and proper SIMD lane-width usage.

You are expected to have FMA and AVX2 compatible hardware (at least). Compile with `RUSTFLAGS="-C target-cpu=native" cargo run --release` for best performance.

Decent efforts have been made to ensure

* minimal dependencies and fast builds
* auto-vectorization fires off with the help of `cargo-remark`
* as much computation is pushed to compile time with the use of precomputed coefficients and `const` generics
* the hot path is allocation and panic-free

## Algorithm

1. Calculate the coefficients of interest in Julia, copy/paste them into `coeffs/_f32.rs` appropriately and declare them as `const`.
2. Do a fixed-size rolling window dot_product with half the elements of the dot product as the `coeffs` obtained previously.
3. Update each element of a `buf`fer
4. Parallelize with Rayon

## TODO

- [X] rayon support
- [X] Just calculate NxM up to 12x12 and cache that
- [X] fma support
- [X] f32/f64 float support
- [ ] SIMD support
- [ ] `no_std` support see([Effective Rust link](https://www.lurklurk.org/effective-rust/no-std.html))
- [ ] support derivatives (stretch goal - sponsor me???)
