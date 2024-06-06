# Staged Filters

* A Savitzky-Golar filter that is fast, baby.
* All (N,M) parameters are precomputed and pulled in at compile time.
* `rayon` support is available via a `rayon` feature flag
* Still some SIMD perf left on the table - newer versions will focus on perf

* Remember to compile this with `RUSTFLAGS="-C target-cpu=native"`.

This code is based on another code I adapted in Julia with much help from others, see [StagedFilters.jl](https://github.com/miguelraz/StagedFilters.jl).

## Example

## Benchmarks

## Notes

It's called "staged" because the computation is done in "stages", which allows the compiler to optimize the code a lot more - namely, the use of const generics in Rust provide more opportunities for profitable loop unrolling and proper SIMD lane-width usage.

Decent efforts have been made to ensure

* you are expected to have FMA and AVX2 compatible hardware (at least). Compile with `RUSTFLAGS="-C target-cpu=native" cargo run --release` for best performance.
* auto-vectorization fires off with the help of `cargo-remark`
* as much computation is pushed to compile time
* the hot path is allocation and panic-free

## TODO

- [X] rayon support
- [X] Just calculate NxM up to 12x12 and cache that
- [ ] fma support
- [ ] SIMD support
- [ ] f32 support
- [ ] consider `no_std` [Effective Rust link](https://www.lurklurk.org/effective-rust/no-std.html)
- [ ] support derivatives (stretch goal - sponsor me???)
