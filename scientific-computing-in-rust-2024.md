# Building a compile-time SIMD optimized smoothing filter

By Miguel Raz Guzmán Macedo 
Scientific Computing in Rust 2024

-----

# The problem

* You want to smooth your time series finance/particle physics/econometrics data
* You want do it fast, like, really **fast**
* You want to make sure data is still holding up some of its statistical properties 

-----

# The solution

* What you want is a `Savitzky-Golay` filter
* Or a weighted moving average, rolling window average, convolution...
* It's basically a fancy dot product with one vector being precomputed.
* registered on crates.io already on `staged-sg-filter`

-----

# Why it's juicy

* A dot product has perf opportunities at many levels
* Wide set of applications
* The coefficent precomputation lends itself to compile-time shenanigans and speedups

-----

# Dot products galore

It's just a map-reduce with a binary_op and a plus reduction!

* In Julia, `mapreduce(*, +, a, b)`

But it's also useful for
* The Hamming Distance for characters if you map `!=`
* The [Exact Binary Vector search powering RAGs in LLMs](https://domluna.com/blog/tiny-binary-rag) if `x` and `y` are bitvectors

----

# Benchmarks

Scipy

```python
   ...: N = 100_000_000 
#          👆
   ...: dataf32 = np.arange(1, N, dtype=np.float32)
   ...: dataf64 = np.arange(1, N, dtype=np.float64)
   ...: %timeit f(dataf32, 5, 2, mode = "wrap")
   ...: %timeit f(dataf64, 5, 2, mode = "wrap")
f32 savgol (5,2) for 100000000 elements
1.38 s ± 31.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 👆
f64 savgol (5,2) for 100000000 elements
1.69 s ± 121 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 👆
```

1. Time isn't scaling with bitwidth -> not memory bound!
2. All benchmarking issues are mine, please let me know if I've goofed

----

# Julia

```julia
julia> using BenchmarkTools
julia> using StagedFilters
julia> N = 100_000_000;
julia> data = rand(Float32, N); smoothed = similar(data);
julia> @btime smooth!($SavitzkyGolayFilter{2,2}, $data, $smoothed);
  61.839 ms (0 allocations: 0 bytes)
  # 👆 ~22x faster than Scipy
julia> dataf64 = rand(Float64, N); smoothedf64 = similar(dataf64);
julia> @btime smooth!($SavitzkyGolayFilter{2,2}, $dataf64, $smoothedf64);
  125.020 ms (0 allocations: 0 bytes)
  # 👆 ~13x faster than Scipy
```
----

# Rust

```rust
#[divan::bench(sample_size = 3, sample_count = 3)]
fn savgol_f32() -> f32 {
    let n = 100_000_000;
    let v = vec![10.0f32; n];
    let mut buf = vec![0.0f32; n];
    sav_gol_f32::<2, 2>(bb(&mut buf), bb(&v));
    buf[0]
}
```

Whelp...

```bash
mrg:~/staged-sg-filter$ RUSTFLAGS="-C target-feature=+fma,+avx2 -C target-cpu=native" cargo bench
  #                                                   👆 ~2x faster than without
     Running benches/divan.rs (target/release/deps/divan-a75fca219433bc49)
Timer precision: 20 ns
divan          fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ savgol_f32  521.4 ms      │ 530.7 ms      │ 526.2 ms      │ 526.1 ms      │ 3       │ 9
  #             👆 ~2.5x faster than Scipy
╰─ savgol_f64  1.046 s       │ 1.059 s       │ 1.048 s       │ 1.051 s       │ 3       │ 9
  #             👆 ~30% faster than Scipy
```

----

# Counting cycles

Processing 100M elements for a computer with `1e11` peakFLOP/s:

| Language| f32 time[s] | Elements / ns | ns / Element |
|---|---|---|---|
| Python | 1.38 |  .072 | 13.8 |
| Julia  | 0.06 | 1.617 | 0.618 |
| Rust   | .52 |  .19 | 5.21 | 

* Surprising result - don't know where this difference between Julia and Rust is coming from!

----

# The tight loop

```text
vfnmadd132ss  xmm0, xmm8, dword ptr [rcx + 4*r10 - 12] # xmm0 = -(xmm0 * mem) + xmm8
vfmadd231ss   xmm0, xmm3, dword ptr [rcx + 4*r10 - 8]  # xmm0 = (xmm3 * mem) + xmm0
vfmadd231ss   xmm0, xmm4, dword ptr [rcx + 4*r10 - 4]  # xmm0 = (xmm4 * mem) + xmm0
vfmadd231ss   xmm0, xmm5, dword ptr [rcx]              # xmm0 = (xmm5 * mem) + xmm0
vfnmadd231ss  xmm0, xmm13, dword ptr [rcx + 4*rdi + 8] # xmm0 = -(xmm13 * mem) + xmm0
```

Tight assembly! Entering 👩‍🍳 💋 territory

* Dot product is the most intensive part of the code -> expect series of `vfmadd`
* Exercise left to reader - exploit AVX512 for even more FLOP/s!

----

# Ingredients for an enviable tight loop 🍅🥦🥕

1. Move all uncertainty to compile time: `const` generics, `const` coefficients, `const` `fn`'s
2. Use iterator idioms with mechanical sympathy (avoid mutation)
3. use fused multiply adds and don't forget to set `RUSTFLAGS="-C target-cpu=native target-feature=+avx2,+fma" cargo run --release`
4. Use tools to take it one small step at a time 
   1.  battery of unit tests and `cargo asm` invocations

----

# Rusty implementation

```rust
pub fn sav_gol_f32<const WINDOW: usize, const M: usize>(buf: &mut [f32], data: &[f32]) {
  //                👆 1. Const generics
    let coeffs = get_coeffs_f32::<WINDOW, M>();
  //                               👆 1. Const generics
    let window_size = 2 * WINDOW + 1;
    let body_size = data.len() - (window_size - 1);
    buf.iter_mut()
        .skip(window_size / 2)
        .zip(data.windows(window_size))
                  //👆 2. Use iterator idioms (avoid mutation)
        .take(body_size)
        .for_each(|(buf, data)| {
            dot_prod_update_f32(buf, data, coeffs);
        });
}
```

----

# Const all the things

```rust
    let coeffs = get_coeffs_f32::<WINDOW, M>();
```

* Unfortunately floating point ops are not yet `const`, but will be Soon ™ (?)️

### Experiments:

* How much can you stretch compile time shenanigans?
* `nalgebra` and friends were not helpful
* Easiest to just precompute many cases in Julia and copy/paste them with vim into an unholy

```rust
pub const COEFFS_F32: [[&[f32]; 25]; 10] = ...;
```

----

# Pass the `const` ball ⚽ to your iterators

prefer

```rust
    buf.iter_mut()
        .skip(window_size / 2)
        .zip(data.windows(window_size))
        .take(body_size)
        .for_each(|(buf, data)| {
            dot_prod_update_f32(buf, data, coeffs);
        });
```

to an indexing approach - tracking the mutability of the inner index can wreck the compiler's analysis and bail out early.

* Try using `cargo-remark`
* Try figuring out all the places where Rust can take advantage of known trip counts, or even multiplying by a known constant
----

# Small bites at a time

Separating out

```rust
#[inline]
pub fn dot_prod_update(buf: &mut f64, data: &[f64], coeffs: &[f64]) {
    *buf = data
        .iter()
        .zip(coeffs.iter())
        .fold(0.0, |acc, (a, b)| a.mul_add(*b, acc));
}
```

meant I could iterate faster with unit tests and swap out `fold` for `sum`, `a.mul_add(*b, acc)` for other idioms, etc.

* Homework: Try and use an exact-sized iterator, deal with the remainder separately

----

# Low hanging fruit

```rust

pub fn div_runtime_loop(x: f32) -> f32 {
    let mut res = x;
    for _ in 0..148 {
        res += 1.0 / 2.0;
    }
    res
}
```

----

# Gives...

```asm
.LCPI0_0:
        .long   0x3f000000
example::div_runtime_loop::haf9d2398cf2f6940:
        mov     eax, 144
        movss   xmm1, dword ptr [rip + .LCPI0_0]
.LBB0_1:
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        test    eax, eax
        je      .LBB0_3
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        add     eax, -8
        jmp     .LBB0_1
.LBB0_3:
        ret
```

----

# But if the trip count is 149...

```asm
.LCPI0_0:
        .long   0x3f000000
example::div_runtime_loop::haf9d2398cf2f6940:
        movss   xmm1, dword ptr [rip + .LCPI0_0]
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        addss   xmm0, xmm1
        ; ... 149 times, very, very bad!
```

* My takeaway: be like a 5 year old and "lick"/test everything!

----

# Compilers are hard

Poke at the [link yourself](https://godbolt.org/z/5K96aoc5d):

![steve cannon](https://media.hachyderm.io/media_attachments/files/111/426/828/315/602/530/original/e81d6bc1bae5a940.png)

----


Thanks to Jubilee