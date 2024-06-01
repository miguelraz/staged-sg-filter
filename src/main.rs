#![allow(dead_code)]
#![allow(unused_imports)]

use iter_comprehensions::map;
use staged_sg_filter::sav_gol;

fn main() {
    let v = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let mut buf = vec![0.0; 7];
    //pub fn sav_gol<const WINDOW: usize, const M: usize>(buf: &mut Vec<f64>, data: &Vec<f64>) {
    sav_gol::<1, 1>(&mut buf, &v);

    println!("{:?}", buf);
}
