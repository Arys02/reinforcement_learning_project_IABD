use burn::prelude::{Backend, Tensor};
use rand::prelude::IteratorRandom;
use rand::Rng;

pub fn epsilon_greedy_action<
    B: Backend<FloatElem=f32, IntElem=i64>,
    const NUM_STATES_FEATURES: usize,
    const NUM_ACTIONS: usize,
>(
    q_s: &Tensor<B, 1>,
    mask_tensor: &Tensor<B, 1>,
    available_actions: impl Iterator<Item=usize>,
    epsilon: f32,
    rng: &mut impl Rng,
) -> usize {
    if rng.gen_range(0f32..=1f32) < epsilon {
        available_actions.choose(rng).unwrap()
    } else {
        let inverted_mask = mask_tensor
            .clone()
            .mul([-1f32; NUM_ACTIONS].into())
            .add([1f32; NUM_ACTIONS].into());
        let masked_q_s = (q_s.clone() * mask_tensor.clone())
            .add(inverted_mask.mul([f32::MIN; NUM_ACTIONS].into()));
        masked_q_s.clone().argmax(0).into_scalar() as usize
    }
}
/*
pub fn soft_max_with_mask_action<
    B: Backend<FloatElem=f32, IntElem=i64>,
    const NUM_STATES_FEATURES: usize,
    const NUM_ACTIONS: usize,
>(
    q_s: &Tensor<B, 1>,
    mask_tensor: &Tensor<B, 1>,
    available_actions: impl Iterator<Item=usize>,
) -> usize {
    let inverted_mask = mask_tensor
        .clone()
        .mul([-1f32; NUM_ACTIONS].into())
        .add([1f32; NUM_ACTIONS].into());

    let masked_q_s = (q_s.clone() * mask_tensor.clone())
        .add(inverted_mask.mul([f32::MIN; NUM_ACTIONS].into()));

    let positiv_x = q_s - q_s.clone().min();
    let masked_pos_x = positiv_x * mask_tensor.clone();

    let negativ_x = masked_pos_x - mask_tensor.clone();
    let exp_x = negativ_x.exp();
}

 */

//source : https://burn.dev/blog/burn-rusty-approach-to-tensor-handling/

fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    log_softmax(tensor, dim).exp()
}

fn log_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    tensor.clone() - tensor.exp().sum_dim(dim).log()
}

