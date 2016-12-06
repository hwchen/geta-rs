// Remove allocations?

extern crate geta;

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

    let mut f = File::open("./mnist/mnist_train_100.csv").unwrap();
    let mut train_csv = String::new();
    f.read_to_string(&mut train_csv).unwrap();
    let train_data: Vec<Vec<_>> = train_csv.lines().map(|line| {
        line.split(',').filter_map(|s| s.parse::<u8>().ok()).collect()
    }).collect();

    // Initialize Neural Network
    let input_nodes = 784;
    let hidden_nodes = 100;
    let output_nodes = 10;
    let learning_rate = 0.3;

    let mut nn = geta::NeuralNetwork::new(
        input_nodes,
        hidden_nodes,
        output_nodes,
        learning_rate,
    );

    // Train
    for digit_data in train_data {
        let input = scale_input(&digit_data[1..]);
        let target = process_target(digit_data[0]);

        nn.train(input, target);
    }

    // Query
    let ref test_target = test_data[0];
    println!("digit: {}", test_target[0]);

    let ref test_bytes = test_target[1..];
    graph_digit(test_bytes);

    let test_output = nn.query(scale_input(test_bytes));
    println!("output for test target: {:?}", test_output);
}

pub fn scale_input(digit_data: &[u8]) -> Vec<f64> {
    digit_data.iter().map(|&x| (x as f64 / 255.0 * 0.99) + 0.01).collect()
}

pub fn process_target(target_digit: u8) -> Vec<f64> {
    let mut res = vec![0f64;10];
    res[target_digit as usize] = 0.99;
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_target() {
        assert_eq!(process_target(5), vec![0f64,0.,0.,0.,0.,0.99,0.,0.,0.,0.]);
        assert_eq!(process_target(6), vec![0f64,0.,0.,0.,0., 0.,0.99,0.,0.,0.]);
    }
}

