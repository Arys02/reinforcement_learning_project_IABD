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
    let mut replay_memory: Vec<(Tensor<B, 1>, usize, f32, Tensor<B, 1>, usize, bool)> =
        Vec::with_capacity
            (replay_capacity);

    if batch_size > replay_capacity {
        panic!("batch_size should't be bigger than replay capacity")
    }

    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;
    let mut i_replay: usize = 0;

    let mut env = Env::default();


    //let mut bar = tqdm!();

    for ep_id in tqdm!(0..num_episodes) {
        env.reset();
        //bar.update(1).unwrap();

        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        if ep_id % 1000 == 0 {
            println!("Mean Score: {}", total_score / 1000.0);
            //println!("it  {}", bar.fmt_rate());
            total_score = 0.0;
        }

        while !env.is_terminal() {
            let s = env.state_description();
            let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
            let mut q_s = model.forward(s_tensor.clone());

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
                    s_tensor.clone(),
                    a,
                    //q_s.clone().slice([a..(a + 1)]).into_scalar(),
                    r,
                    s_p_tensor.clone(),
                    a_p_max,
                    //q_s_p.clone().slice([a_p_max..(a_p_max + 1)]).into_scalar(),
                    env.is_terminal()
                ));
            i_replay = (i_replay + 1) % replay_capacity;

            if replay_memory.len() < batch_size {
                continue;
            }

            //                Φ_s          a       r   Φ_s_p         a_max  is_terminal
            let batch: Vec<(Tensor<B, 1>, usize, f32, Tensor<B, 1>, usize, bool)> = replay_memory
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect();


            let y: Tensor<B, 1> = Tensor::from_floats({
                                                          let x: Vec<f32> = batch.iter().map(
                                                              |(s_tensor, a, r, s_p_tensor,
                                                                   a_p_max, is_terminal)|
                                                              {
                                                                  let q_s_a: f32 = model.forward
                                                                  (s_tensor.clone()).slice([*a..
                                                                      (*a + 1)])
                                                                      .into_scalar();

                                                                  if *is_terminal {
                                                                      q_s_a - r
                                                                  } else {
                                                                      let q_s_p_a_p = model.forward
                                                                      (s_p_tensor.clone())
                                                                          .slice([*a_p_max..
                                                                              (*a_p_max + 1)])
                                                                          .into_scalar();

                                                                      q_s_p_a_p * gamma * r - q_s_a
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
