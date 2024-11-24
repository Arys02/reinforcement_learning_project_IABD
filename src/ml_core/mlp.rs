use crate::ml_core::ml_traits::Forward;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct MyQMLP<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    linear3: nn::Linear<B>,
    output_layer: nn::Linear<B>,
}

impl<B: burn::prelude::Backend> MyQMLP<B> {
    pub fn new(device: &B::Device, input_state_features: usize, output_actions: usize) -> Self {
        let linear1 = nn::LinearConfig::new(input_state_features, 128)
            .with_bias(true)
            .init(device);
        let linear2 = nn::LinearConfig::new(128, 64).with_bias(true).init(device);
        let linear3 = nn::LinearConfig::new(64, 32).with_bias(true).init(device);
        let output_layer = nn::LinearConfig::new(32, output_actions)
            .with_bias(true)
            .init(device);
        MyQMLP {
            linear1,
            linear2,
            linear3,
            output_layer,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear1.forward(x);
        let x = x.tanh();
        let x = self.linear2.forward(x);
        let x = x.tanh();
        let x = self.linear3.forward(x);
        let x = x.tanh();
        self.output_layer.forward(x)
    }
}

impl<B: Backend> Forward for MyQMLP<B> {
    type B = B;

    fn forward<const DIM: usize>(&self, input: Tensor<Self::B, DIM>) -> Tensor<Self::B, DIM> {
        self.forward(input)
    }
}
