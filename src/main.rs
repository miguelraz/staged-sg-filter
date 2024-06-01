#![allow(dead_code)]

use iter_comprehensions::map;
use nalgebra::{DMatrix, DVector};
use ndarray::*;

fn main() {
    // J = T[(i - M - 1 )^(j - 1) for i = 1:2M + 1, j = 1:N + 1]
    // e₁ = [one(T); zeros(T,N)]
    // C = J' \ e₁
    let m = 2;
    let n = 2;
    println!("asdf");
    let vec = map!(
            ((i - m - 1) as i32).pow((j as u32) - 1) as f64;
                i in 1..=(2*m+1),
                j in 1..=(n+1))
    .collect::<Vec<f64>>();
    println!("{vec:?}");
    // I was here
    let mut J = Array::from(vec);
    J.to_shape((m, n));
    let mut temp = vec![1.0];
    let mut zeros = vec![0.0; m];
    temp.append(&mut zeros);
    let e_1 = Array::from_vec(temp);
    println!("e_1 len is {}", e_1.len());
    let Jtick = J.adjoint();
    println!("J len is {}", J.len());
    println!("{:?}", Jtick.clone());
    let decomp = Jtick.svd();
    let sol = decomp.solve(&e_1).expect("couldn't solve :(");
    println!("{:?}", sol);
}
