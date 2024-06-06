//use staged_sg_filter::par_sav_gol;
use staged_sg_filter::sav_gol;

fn main() {
    let v = vec![10.0; 500_000];
    let mut buf = vec![0.0; 500_000];
    sav_gol::<1, 1>(&mut buf, &v);

    println!("{:?}", &buf[0..10]);
}
