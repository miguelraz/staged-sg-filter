# Staged Filters

* A Savitzky-Golar filter that is fast, baby.
* All (N,M) parameters are precomputed and pulled in at compile time.
* `rayon` support is available via a `rayon` feature flag
* Still some SIMD perf left on the table - newer versions will focus on perf

* Remeber to compile this with `RUSTFLAGS="-C target-cpu=native"`.

## TODO

- [X] rayon support
- [X] Just calculate NxM up to 12x12 and cache that
- [ ] fma support
- [ ] SIMD support
- [ ] f32 support
- consider `no_std` [Effective Rust link](https://www.lurklurk.org/effective-rust/no-std.html)
- [ ] support derivatives (stretch goal - sponsor me???)
