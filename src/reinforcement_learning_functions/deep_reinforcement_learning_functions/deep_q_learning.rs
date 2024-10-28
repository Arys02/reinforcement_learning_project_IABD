use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use std::fmt::{Debug, Display};

use crate::ml_core::ml_traits::Forward;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::epsilon_greedy_action;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn deep_q_learning<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B=B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem=f32, IntElem=i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Debug + Display,
>(
    mut model: M,
    num_episodes: usize,
    gamma: f32,
    replay_capacity: usize,
    batch_size: usize,
    alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B=B::InnerBackend>,
{
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();

    //initialize replay memory D to capacity N
    let mut replay_memory: Vec<(f32, f32, f32, bool)> =
        Vec::with_capacity
            (replay_capacity);

    if batch_size > replay_capacity {
        panic!("batch_size should't be bigger than replay capacity")
    }

    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;
    let mut i_replay: usize = 0;

    let mut env = Env::default();

    for ep_id in tqdm!(0..num_episodes) {
        env.reset();

        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        if ep_id % 1000 == 0 {
            println!("Mean Score: {}", total_score / 1000.0);
            total_score = 0.0;
        }

        while !env.is_terminal() {
            let s = env.state_description();
            let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
            let mut q_s = model.forward(s_tensor);

            let mut a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s,
                &mask_tensor,
                env.available_actions_ids(),
                decayed_epsilon,
                &mut rng,
            );

            let prev_score = env.score();
            //execute action a_t in emulator and observe reward r_t
            env.step(a);
            let r = env.score() - prev_score;

            //observe state s+1
            let s_p = env.state_description();
            //x+1
            let s_p_tensor: Tensor<B, 1> = Tensor::from_floats(s_p.as_slice(), device);


            let mask_p = env.action_mask();
            let mask_p_tensor: Tensor<B, 1> = Tensor::from(mask_p).to_device(device);


            //phi  Φ t+1
            let q_s_p = Tensor::from_inner(model.valid().forward(s_p_tensor.clone().inner()));

            //we want the best action for s_p
            let mut a_p_max = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s_p,
                &mask_p_tensor,
                env.available_actions_ids(),
                -1.,
                &mut rng,
            );

            //Store transition ( greedy_ε_Φt(s, a), rt, arg_max_aΦt+1(s', a'), is_terminal)
            replay_memory.insert(
                i_replay, (
                    q_s.clone().slice([a..(a + 1)]).into_scalar(),
                    r,
                    q_s_p.clone().slice([a_p_max..(a_p_max + 1)]).into_scalar(),
                    env.is_terminal()
                ));
            i_replay = (i_replay + 1) % replay_capacity;

            if replay_memory.len() < batch_size {
                continue;
            }

            let batch: Vec<(f32, f32, f32, bool)> = replay_memory
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect();

            let y: Tensor<B, 1> = Tensor::from_floats({
                                                          let x: Vec<f32> = batch.iter().map(|(q_s, r, q_s_p, is_terminal)|
                                                              {
                                                                  if *is_terminal {
                                                                      r - q_s
                                                                  } else {
                                                                      (q_s_p * gamma + r) - q_s
                                                                  }
                                                              }
                                                          ).collect();
                                                          x.clone().as_slice()
                                                      }, device);

            let y = y.detach();

            let loss = y.powf_scalar(2f32);
            let grad_loss = loss.backward();
            let grads = GradientsParams::from_grads(grad_loss, &model);

            model = optimizer.step(alpha.into(), model, grads);
        }
        total_score += env.score();
    }

    model
}
