use std::collections::HashMap;
use crate::training_observer::TrainingObserver;
use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use std::fmt::{Debug, Display};
use std::time::{Duration, Instant};
use crate::ml_core::ml_traits::Forward;

use rand::distributions::Distribution;


use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{GradientsParams, SgdConfig};
use burn::optim::Optimizer;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::distributions::WeightedIndex;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use tensorboard_rs::summary_writer::SummaryWriter;
use crate::logger::Logger;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::utils::{epsilon_greedy_action, soft_max_with_mask_action};
use crate::training_observer::{Hyperparameters, TrainingEvent};

pub fn reinforce<
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
    start_epsilon: f32,
    final_epsilon: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B=B::InnerBackend>,
{
    let mut env = Env::default();

    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    let mut map = HashMap::new();

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


    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();


    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;
    let mut mean_nb_step = 0.0;

    let mut episode_duration = 0.0;


    for ep_id in tqdm!(0..num_episodes) {
        env.reset();

        //LOGGER ?
        let mut episode_reward = 0.0;
        let mut episode_steps = 0;
        let episode_start_time = Instant::now();

        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;



        let mut trajectory = Vec::new();

        //generate an episode
        while !env.is_terminal() {
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



        let mut g = 0.;
        //   t          pi(s | a, Φ)
        for (t, (s, a , _)) in trajectory.iter().enumerate() {
            for i in (t + 1)..trajectory.len() {
                //g = g + γ^(k - t - 1) * Rk
                g = g + gamma.powf((i - t - 1) as f32) * trajectory[i].2;
            }


            let pi_s = model.forward(s.clone());
            let pi_s_a = pi_s.slice([*a..(*a+1)]).log();
            let loss = pi_s_a.clone().mul_scalar(g);

            let grad_loss = loss.backward();
            let grads = GradientsParams::from_grads(grad_loss, &model);
            model = optimizer.step((alpha * gamma.powf(t as f32)).into(), model, grads);

            #[cfg(feature = "logging")] {
                log_total_loss += loss.mean().into_scalar();
            }
        }





        total_score += env.score();
        mean_nb_step += trajectory.len() as f32;

        episode_reward += env.score();
        episode_steps += 1;
        //#[cfg(feature = "logging")]
        //let mut episode_duration = episode_start_time.elapsed();

        #[cfg(feature = "logging2")] {
            episode_duration += episode_start_time.elapsed().as_secs_f32();

            if ep_id % 1000 == 0 {
                map.insert("Mean Score".to_string(), total_score / 1000.0);
                map.insert("Mean nb steps".to_string(), mean_nb_step / 1000.0);
                map.insert("average time per step".to_string(), episode_duration / 1000.0 );
                writer.add_scalars(&format!("reinforce/{}/{}_{}_{}", env.get_name(), num_episodes, alpha, gamma), &map, ep_id);
                println!("Mean Score: {}", total_score / 1000.0);
                total_score = 0.0;
                mean_nb_step = 0.0;
                episode_duration = 0.0;
            }
        }


        //LOGGER

        /*
        #[cfg(feature = "logging")] {
            log_total_score += episode_reward;
            log_total_steps += episode_steps;
            log_total_time += episode_duration;
        }

         */
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



    writer.flush();
    model
}
