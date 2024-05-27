use iter_comprehensions::map;
fn main() {
    let m = 4;
    let n = 4;
    let vec = map!(((i - m - 1) as i32).pow((j as u32) - 1) as f64; i in 1..=(2*m), j in 1..(n+1))
        .collect::<Vec<f64>>();
    println!("{vec:?}");
}
