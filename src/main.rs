#![allow(dead_code)]
#![allow(unused_imports)]

use iter_comprehensions::map;
use staged_sg_filter::coeffs::COEFFS;

fn main() {
    // N = 3
    const N: usize = 3;
    const WINDOW_SIZE: usize = 2 * N + 1;
    //let vec = hmap.get(&(N, 1)).unwrap().to_vec();
    let coeffs = COEFFS[0][0];

    println!("{:?}", coeffs);
}
