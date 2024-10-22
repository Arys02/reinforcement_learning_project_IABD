use burn::prelude::{Backend, Tensor};
use rand::prelude::IteratorRandom;
use rand::Rng;

pub fn epsilon_greedy_action<
    B: Backend<FloatElem = f32, IntElem = i64>,
    const NUM_STATES_FEATURES: usize,
    const NUM_ACTIONS: usize,
>(
    q_s: &Tensor<B, 1>,
    mask_tensor: &Tensor<B, 1>,
    available_actions: impl Iterator<Item = usize>,
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
