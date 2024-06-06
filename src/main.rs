//use staged_sg_filter::par_sav_gol;
use staged_sg_filter::sav_gol;

fn main() {
    let v = vec![0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0];
    let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    sav_gol::<1, 1>(&mut buf, &v);

    println!("{:?}", buf);
}
