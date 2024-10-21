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
    let mut replay_memory: Vec<(Tensor<B, 1>, usize, f32, Tensor<B, 1>, bool)> = Vec::with_capacity
        (replay_capacity);

    if batch_size > replay_capacity {
        panic!("batch_size should't be bigger than replay capacity")
    }

    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;
    let mut i_replay: usize = 0;


    for ep_id in tqdm!(0..num_episodes) {
        let mut env = Env::default();

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
            //println!("{:?}", q_s);

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

            //Store transition (Φt, at, rt, Φt+1)
            replay_memory.insert(i_replay, (q_s.clone(), a, r, q_s_p.clone(), env.is_terminal()));
            i_replay = (i_replay + 1) % replay_capacity;

            if replay_memory.len() < batch_size {
                continue;
            }

            let batch: Vec<(Tensor<B, 1>, usize, f32, Tensor<B, 1>, bool)> = replay_memory.clone()
                .choose_multiple(&mut rng,
                                 batch_size).cloned().collect();


            let y: Tensor<B, 1> = if env.is_terminal() {
                Tensor::from_floats(
                    {
                        let x: Vec<f32> = batch
                            .clone()
                            .into_iter()
                            .map(|(s, a, r, q_s_p, end)|
                                { r - s.slice([a..(a + 1)]).into_scalar() }
                            )
                            .collect();
                        x.clone().as_slice()
                    }
                    , device)
            } else {
                Tensor::from_floats(
                    {
                        let x: Vec<f32> = batch
                            .clone()
                            .into_iter()
                            .map(|(s, a, r,
                                      q_s_p, end)| {
                                let a_p = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                                    &q_s_p,
                                    &mask_p_tensor,
                                    env.available_actions_ids(),
                                    decayed_epsilon,
                                    &mut rng,
                                );

                                #[allow(clippy::single_range_in_vec_init)]
                                let q_s_p_a_p = q_s_p.clone().slice([a_p..(a_p + 1)]).into_scalar();

                                (q_s_p_a_p * gamma + r) - s.slice([a..(a + 1)]).into_scalar()
                            })
                            .collect();
                        x.clone().as_slice()
                    }, device)
            };

            let y = y.detach();

            /*
            let q_s_a = Tensor::from_floats(
                {
                    let x: Vec<f32> = batch
                        .clone()
                        .into_iter()
                        .map(|(q_s, a, r, q_s_p, b)| {
                            q_s.slice([a..(a + 1)]).into_scalar()
                        }
                        ).collect();
                    x.clone().as_slice()
                }, device,
            );

             */


            let loss = y.powf_scalar(2f32);
            let grad_loss = loss.backward();
            let grads = GradientsParams::from_grads(grad_loss, &model);

            model = optimizer.step(alpha.into(), model, grads);

            //q_s = model.forward(s_p_tensor);

        }
        //println!("{}", env);
        total_score += env.score();
    }

    model
}
