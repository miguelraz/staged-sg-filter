//use staged_sg_filter::par_sav_gol;
use staged_sg_filter::{sav_gol, sav_gol_f32};

fn main() {
    let n = 100_000_000;
    // f32
    let v = vec![10.0f32; n];
    let mut buf = vec![0.0f32; n];
    let start = std::time::Instant::now();
    sav_gol_f32::<1, 1>(&mut buf, &v);
    let duration = start.elapsed();

    println!("f32: {:?}", duration);
    println!("{:?}", &buf[0..3]);

    // f64
    let v = vec![10.0; n];
    let mut buf = vec![0.0; n];
    let start = std::time::Instant::now();
    sav_gol::<1, 1>(&mut buf, &v);
    let duration = start.elapsed();

    println!("f64: {:?}", duration);
    println!("{:?}", &buf[0..3]);

}
