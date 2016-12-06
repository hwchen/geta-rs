extern crate gnuplot;

use gnuplot::{AxesCommon, Figure};
use std::fs::File;
use std::io::Read;

fn main() {
    // get mnist data and convert to vecs of u8 representing a digit
    let mut f = File::open("./mnist/mnist_test_10.csv").unwrap();
    let mut test_csv = String::new();
    f.read_to_string(&mut test_csv).unwrap();
    let test_data: Vec<Vec<_>> = test_csv.lines().map(|line| {
        line.split(',').filter_map(|s| s.parse::<u8>().ok()).collect()
    }).collect();

    let ref digit = test_data[0];
    graph_digit(digit);
}

fn graph_digit(digit: &[u8]) {
    let mut fg = Figure::new();

    fg.axes2d()
        .set_title(&format!("{}", digit[0]), &[])
        .set_y_reverse()
        //.set_x_reverse()
        .image(
            digit.iter().skip(1),
            28,
            28,
            None,
            &[]
        );
    fg.show();
}
