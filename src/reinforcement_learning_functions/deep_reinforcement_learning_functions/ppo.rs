use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use std::fmt::{Debug, Display};

use crate::ml_core::ml_traits::Forward;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::epsilon_greedy_action;

use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::ppo_trajectory::trajectory::Trajectory;

use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{GradientsParams, SgdConfig};
use burn::optim::Optimizer;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::ml_core::mlp::MyQMLP;

pub fn ppo<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B=B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem=f32, IntElem=i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Debug + Display,
>(
    mut model: M,
    num_episodes: usize,
    alpha: f32,
    gamma: f32,
    epsilon: f32,
    memory_size: usize,
    start_epsilon: f32,
    final_epsilon: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B=B::InnerBackend>,
{
    /*
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();
     */

    let value_fct = MyQMLP::<B>::new(&device,
                                                 NUM_STATE_FEATURES,
                                                 1);


    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;

    let mut env = Env::default();


    for ep_id in tqdm!(0..num_episodes) {
        env.reset();

        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        if ep_id % 1000 == 0 {
            println!("Mean Score: {}", total_score / 1000.0);
            total_score = 0.0;
        }

        let mut trajectory = Trajectory::<B>::new(memory_size);


        for i in 0..memory_size{
            env.reset();
            while !env.is_terminal() {
                let s = env.state_description();
                let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

                let mask = env.action_mask();
                let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
                let pi_s = model.forward(s_tensor.clone());



                //TODO change to get a softmax random action
                let a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                    &pi_s,
                    &mask_tensor,
                    env.available_actions_ids(),
                    decayed_epsilon,
                    &mut rng,
                );

                let prev_score = env.score();
                //execute action a_t in emulator and observe reward r_t
                env.step(a);

                let r = env.score() - prev_score;

                let v_t = value_fct.forward(s_tensor.clone());



                trajectory.push(s_tensor.clone(), a, r, pi_s.log().clone(), v_t.clone());

                if env.is_terminal() {
                    env.reset();
                    break;
                }
            }
        }




        //generate an episode
        while !env.is_terminal() {
            let s = env.state_description();
            let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
            let mut pi_s = model.forward(s_tensor);


            //TODO change to get a softmax random action
            let a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &pi_s,
                &mask_tensor,
                env.available_actions_ids(),
                decayed_epsilon,
                &mut rng,
            );

            let prev_score = env.score();
            //execute action a_t in emulator and observe reward r_t
            env.step(a);
            let r = env.score() - prev_score;


            let prob = pi_s.clone().slice([a..(a+1)]).log().detach();


        }

        total_score += env.score();

        /*

        let mut g = 0.;
        //   t          pi(s | a, Φ)
        for (t, (pi_s_a,_)) in trajectory.iter().enumerate() {
            for i in (t + 1)..trajectory.len() {
                //g = g + γ^(k - t - 1) * Rk
                g = g + gamma.powf((i - t - 1) as f32) * trajectory[i].1;
            }

            let loss = pi_s_a.clone().mul_scalar(g);

            let grad_loss = loss.backward();
            let grads = GradientsParams::from_grads(grad_loss, &model);
            model = optimizer.step((alpha * gamma.powf(t as f32)).into(), model, grads);
        }

         */

    }

    model
}

