// Neural Network

extern crate ndarray;
extern crate rand;

use ndarray::Array2;
use rand::distributions::{IndependentSample, Normal};

use std::f64::consts;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NeuralNetwork {
    input_nodes: usize,
    hidden_nodes: usize,
    output_nodes: usize,
    learning_rate: f64,

    weights_input_to_hidden: Array2<f64>,
    weights_hidden_to_output: Array2<f64>,
}

impl NeuralNetwork {

    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        output_nodes: usize,
        learning_rate: f64,
    ) -> Self {

        // weights to hidden layer is normal distribution
        // with standard deviation derived as
        // number of hidden nodes ^ -0.5
        let i_h_weights_distr = Normal::new(0f64, (hidden_nodes as f64).powf(-0.5));
        let h_o_weights_distr = Normal::new(0f64, (output_nodes as f64).powf(-0.5));

        // Will I need this again after init?
        let mut rng = rand::thread_rng();

        NeuralNetwork {
            input_nodes: input_nodes,
            hidden_nodes: hidden_nodes,
            output_nodes: output_nodes,
            learning_rate: learning_rate,

            weights_input_to_hidden:
                Array2::<f64>::from_shape_fn(
                    (hidden_nodes, input_nodes),
                    |_| {
                        i_h_weights_distr.ind_sample(&mut rng)
                    }
                ),

            weights_hidden_to_output:
                Array2::<f64>::from_shape_fn(
                    (output_nodes, hidden_nodes),
                    |_| {
                        h_o_weights_distr.ind_sample(&mut rng)
                    }
                ),
        }
    }

    pub fn train(&mut self, input: Vec<f64>, target: Vec<f64>) {
        // TODO check that input is right shape
        let input = Array2::<f64>::from_shape_vec(
            (self.input_nodes, 1),
            input
        ).unwrap();

        let target = Array2::<f64>::from_shape_vec(
            (self.output_nodes, 1),
            target
        ).unwrap();

        // This section is to run forward

        // into hidden
        let hidden_input = self.weights_input_to_hidden.dot(&input);
        // out of hidden
        let hidden_result = hidden_input.mapv(|x| {
            sigmoid(x)
        });

        // into output
        let input_final = self.weights_hidden_to_output.dot(&hidden_result);
        // out of output
        let final_result = input_final.mapv(|x| {
            sigmoid(x)
        });

        // Now we have the output on a training pass.
        // Here, we'll find the error (difference from target)
        // and backpropogate that error to the hidden layer.
        // TODO remove allocation?
        let result_error = target - &final_result;

        let hidden_error = self.weights_hidden_to_output.t().dot(&result_error);

        // Update
        self.weights_hidden_to_output +=
            &(&result_error * &final_result * &(final_result.mapv(|x| { 1.0 - x })))
            .dot(&hidden_result.t())
            .mapv(|x| {
                x * self.learning_rate
            });

        self.weights_input_to_hidden +=
            &(&hidden_error * &hidden_result * &(hidden_result.mapv(|x| { 1.0 - x })))
            .dot(&input.t())
            .mapv(|x| {
                x * self.learning_rate
            });
    }

    pub fn query(&self, input: Vec<f64>) -> Array2<f64> {
        // TODO check that input is right shape
        let input = Array2::<f64>::from_shape_vec(
            (self.input_nodes, 1),
            input
        ).unwrap();

        // into hidden
        let hidden_input = self.weights_input_to_hidden.dot(&input);
        // out of hidden
        let hidden_result = hidden_input.mapv(|x| {
            sigmoid(x)
        });

        // into output
        let input_final = self.weights_hidden_to_output.dot(&hidden_result);
        // out of output
        let final_result = input_final.mapv(|x| {
            sigmoid(x)
        });

        final_result
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + consts::E.powf(-1.0 * x))
}

#[cfg(test)]
mod test {

    use ndarray::Array2;
    use super::*;

    #[test]
    fn default_and_init() {
        let nn: NeuralNetwork = Default::default();

        let test = NeuralNetwork {
            input_nodes: 0,
            hidden_nodes: 0,
            output_nodes: 0,
            learning_rate: 0.0,
            weights_input_to_hidden: Array2::<f64>::zeros((0,0)),
            weights_hidden_to_output: Array2::<f64>::zeros((0,0)),
        };

        assert_eq!(nn, test);

        let nn = NeuralNetwork::new(10, 30, 20, 0.5);

        let test = NeuralNetwork {
            input_nodes: 10,
            hidden_nodes: 30,
            output_nodes: 20,
            learning_rate: 0.5,
            weights_input_to_hidden: Array2::<f64>::zeros((0,0)),
            weights_hidden_to_output: Array2::<f64>::zeros((0,0)),
        };

        assert_eq!(nn.input_nodes, test.input_nodes);
        assert_eq!(nn.hidden_nodes, test.hidden_nodes);
        assert_eq!(nn.output_nodes, test.output_nodes);
        assert_eq!(nn.learning_rate, test.learning_rate);

        assert_eq!(nn.weights_input_to_hidden.len(), 300);
        assert_eq!(nn.weights_input_to_hidden.ndim(), 2);
        assert_eq!(nn.weights_input_to_hidden.dim(), (30, 10));

        assert_eq!(nn.weights_hidden_to_output.len(), 600);
        assert_eq!(nn.weights_hidden_to_output.ndim(), 2);
        assert_eq!(nn.weights_hidden_to_output.dim(), (20, 30));

        // TODO? test weights distribution?
    }

    #[test]
    fn train_test() {
        let mut nn = NeuralNetwork::new(10, 20, 10, 0.5);

        // doesn't panic
        nn.train(
            vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,0.],
            vec![2.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
        );

        assert!(false);
    }

    #[test]
    fn query_test() {
        let nn = NeuralNetwork::new(10, 30, 20, 0.5);

        // doesn't panic
        nn.query(vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,0.]);
    }
}
