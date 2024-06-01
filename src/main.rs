#![allow(dead_code)]

use iter_comprehensions::map;
use staged_sg_filter::coeffs2::make_map;

fn main() {
    // J = T[(i - M - 1 )^(j - 1) for i = 1:2M + 1, j = 1:N + 1]
    // e₁ = [one(T); zeros(T,N)]
    // C = J' \ e₁

    // N = 3
    const N: i32 = 3;
    let hmap = make_map();
    let vec = hmap.get(&(N, 1)).unwrap().to_vec();
    let coeffs: [f64; 2 * (N as usize) + 1] = vec.try_into().expect("you done goofed");

    println!("{:?}", coeffs);
}
