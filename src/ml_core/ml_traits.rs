use burn::prelude::*;

pub trait Forward {
    type B: Backend;
    fn forward<const DIM: usize>(&self, input: Tensor<Self::B, DIM>) -> Tensor<Self::B, DIM>;
}
