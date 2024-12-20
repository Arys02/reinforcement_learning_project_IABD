use crate::training_observer::TrainingObserver;
use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use std::fmt::{Debug, Display};
use std::time::{Duration, Instant};
use crate::ml_core::ml_traits::Forward;

use rand::distributions::Distribution;


use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, SgdConfig};
use burn::optim::Optimizer;
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use ndarray_rand::rand_distr::num_traits::Pow;
use rand::distributions::WeightedIndex;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::logger::Logger;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::utils::{epsilon_greedy_action, soft_max_with_mask_action};
use crate::training_observer::{Hyperparameters, TrainingEvent};

pub fn reinforce<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B=B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem=f32, IntElem=i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Debug,
>(
    mut model: M,
    num_episodes: usize,
    alpha: f32,
    gamma: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B=B::InnerBackend>,
{
    let mut env = Env::default();

    #[cfg(feature = "logging")]
    let hyperparameters = Hyperparameters {
        num_episodes,
        gamma,
        alpha,
        start_epsilon,
        final_epsilon,
        log_interval: 500,
        replay_capacity: 0,
        batch_size: 0,
    };

    #[cfg(feature = "logging")]
    let log_interval = hyperparameters.log_interval;

    #[cfg(feature = "logging")]
    let model_name = format!("reinforce2_{}_model2", env.get_name());

    #[cfg(feature = "logging")]
    let mut observer = Logger::new(&model_name, &format!("{}_{}_{}", num_episodes, gamma, alpha));

    #[cfg(feature = "logging")]
    observer.on_event(&TrainingEvent::HyperparametersLogged {
        hyperparameters: hyperparameters.clone(),
    });

    #[cfg(feature = "logging")]
    let mut log_total_score: f32 = 0.0;

    #[cfg(feature = "logging")]
    let mut log_total_steps: usize = 0;

    #[cfg(feature = "logging")]
    let mut win_count: usize = 0;

    #[cfg(feature = "logging")]
    let mut best_score: f32 = f32::MIN;

    #[cfg(feature = "logging")]
    let mut log_total_time: Duration = Duration::new(0, 0);

    #[cfg(feature = "logging")]
    let mut log_total_loss: f32 = 0.0;


    /*
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();

     */
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();


    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;
    let mut mean_step = 0.0;


    for ep_id in tqdm!(0..num_episodes) {
        env.reset();

        //LOGGER ?
        let mut episode_reward = 0.0;
        let mut episode_steps = 0;
        let episode_start_time = Instant::now();

        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;


        if ep_id % 100 == 0 {
            println!("Mean Score: {}, Mean nb_steps : {}", total_score / 100.0, mean_step / 100.);
            total_score = 0.0;
            mean_step = 0.0;
        }


        let mut trajectory = Vec::new();

        let mut nb_max_step = 0.0;
        //generate an episode
        while !env.is_terminal() && nb_max_step < 100.0 {
            mean_step += 1.;
            nb_max_step += 1.;
            let s = env.state_description();
            let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
            let mut pi_s = model.forward(s_tensor.clone());


            let a : usize =  soft_max_with_mask_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &pi_s, &mask_tensor
            );

            let prev_score = env.score();
            //execute action a_t in emulator and observe reward r_t
            env.step(a);
            let r = env.score() - prev_score;

            trajectory.push(
                (
                    s_tensor.clone(),
                    a,
                    r,
                ));
        }

        total_score += env.score();



        //   t          pi(s | a, Φ)


        let mut g = 0.;
        for (t, (s, a , r)) in trajectory.iter().enumerate().rev() {
            g =  r + g * gamma;

            //println!("{}", g * alpha * gamma.powf(t as f32));

            let pi_s = log_softmax(model.forward(s.clone()), 0);
            let pi_s_a = pi_s.slice([*a..(*a+1)]);
            let loss = pi_s_a.clone();

            let grad_loss = loss.backward();
            let grads = GradientsParams::from_grads(grad_loss, &model);
            model = optimizer.step((-alpha * gamma.powf(t as f32) * g).into(), model, grads);

            #[cfg(feature = "logging")] {
                log_total_loss += loss.mean().into_scalar();
            }
        }



        episode_reward += env.score();
        episode_steps += 1;
        #[cfg(feature = "logging")]
        let episode_duration = episode_start_time.elapsed();

        //LOGGER

        #[cfg(feature = "logging")] {
            log_total_score += episode_reward;
            log_total_steps += episode_steps;
            log_total_time += episode_duration;
        }
        //LOGGER
        #[cfg(feature = "logging")] {
            let win = episode_reward > 0.0;

            if win {
                win_count += 1;
            }

            //LOGGER

            if episode_reward > best_score {
                best_score = episode_reward;
            }

            //LOGGER
            if (ep_id + 1) % log_interval == 0 && ep_id != 0 {
                let average_score_per_episode = log_total_score / log_interval as f32;
                let average_steps_per_episode = log_total_steps as f32 / log_interval as f32;
                let average_time = log_interval as f32 / log_total_time.as_secs_f32();
                let average_loss = log_total_loss / log_interval as f32;
                let current_epsilon = decayed_epsilon;

                //LOGGER
                observer.on_event(&TrainingEvent::LoggingSummary {
                    episodes: ep_id + 1,
                    total_score: log_total_score,
                    average_score_per_episode,
                    average_steps_per_episode,
                    average_loss,
                    epsilon: current_epsilon,
                    win_count,
                    best_score,
                    average_time,
                    epoch: ep_id + 1,
                });

                //LOGGER
                println!(
                    "Summary after {} episodes:
                - Total Score: {:.2}
                - Avg Score per Episode: {:.2}
                - Avg Steps per Episode: {:.2}
                - Avg Time per Episode: {:.2}s
                - Avg Loss: {:.4}
                - Epsilon: {:.4}
                - Wins: {}
                - Best Score: {:.2}",
                    ep_id + 1,
                    log_total_score,
                    average_score_per_episode,
                    average_steps_per_episode,
                    average_time,
                    average_loss,
                    current_epsilon,
                    win_count,
                    best_score
                );

                //LOGGER
                log_total_score = 0.0;
                log_total_steps = 0;
                log_total_time = Duration::new(0, 0);
                win_count = 0;
                log_total_loss = 0.0;
            }

        }



    }

    model
}
