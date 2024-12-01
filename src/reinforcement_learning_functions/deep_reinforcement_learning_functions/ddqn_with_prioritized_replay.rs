use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use crate::training_observer::{Hyperparameters, TrainingEvent, TrainingObserver};
use std::fmt::{Debug, Display};

use crate::ml_core::ml_traits::Forward;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::dqn_trajectory::prioritized_trajectory::PrioritizedTrajectory;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::epsilon_greedy_action;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig};

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::{Duration, Instant};
use crate::logger::Logger;

pub fn deep_double_q_learning_per<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Debug + Display,
>(
    mut online_model: M,
    mut target_model: M,
    num_episodes: usize,
    replay_capacity: usize,
    gamma: f32,
    alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    batch_size: usize,
    target_update_frequency: usize,
    per_alpha: f32,
    per_beta: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .init();

    // let mut optimizer = SgdConfig::new()
    //     .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
    //     .init();

    // Use PrioritizedTrajectory instead of regular Trajectory
    let mut replay_memory: PrioritizedTrajectory<B> = PrioritizedTrajectory::new(
        replay_capacity,
        per_alpha,
        per_beta
    );

    if batch_size > replay_capacity {
        panic!("batch_size shouldn't be bigger than replay capacity")
    }

    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut env = Env::default();

    #[cfg(feature = "logging")]
    let hyperparameters = Hyperparameters {
        num_episodes,
        replay_capacity,
        gamma,
        alpha,
        start_epsilon,
        final_epsilon,
        batch_size,
        log_interval: 100,
    };

    #[cfg(feature = "logging")]
    let log_interval = hyperparameters.log_interval;

    #[cfg(feature = "logging")]
    let model_name = format!("ddqn_per_farkle{}_model", env.get_name());

    #[cfg(feature = "logging")]
    let mut observer = Logger::new(&model_name);

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

    for ep_id in tqdm!(0..num_episodes) {
        env.reset();
        let mut episode_reward = 0.0;
        let mut episode_steps = 0;

        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        let episode_start_time = Instant::now();

        while !env.is_terminal() {
            let s = env.state_description();
            let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
            let q_s = online_model.forward(s_tensor.clone());

            let a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s,
                &mask_tensor,
                env.available_actions_ids(),
                decayed_epsilon,
                &mut rng,
            );

            let prev_score = env.score();
            env.step(a);
            let r = env.score() - prev_score;

            let s_p = env.state_description();
            let s_p_tensor: Tensor<B, 1> = Tensor::from_floats(s_p.as_slice(), device);

            let mask_p = env.action_mask();
            let mask_p_tensor: Tensor<B, 1> = Tensor::from(mask_p).to_device(device);

            let q_s_p_online = online_model.forward(s_p_tensor.clone());
            let a_p_max = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s_p_online,
                &mask_p_tensor,
                env.available_actions_ids(),
                -1.,
                &mut rng,
            );

            // Initial priority is set to max priority when adding new experiences
            replay_memory.push(
                s_tensor,
                a as f32,
                r,
                s_p_tensor,
                a_p_max as f32,
                if env.is_terminal() { 1.0 } else { 0. },
            );

            if replay_memory.len < batch_size {
                continue;
            }

            // Get batch with importance sampling weights
            let (mut batch, weights, indices) = replay_memory.get_prioritized_batch(batch_size);
            let (t_a, s_tensor, a_p_tensor, s_p_tensor, r_tensor, is_terminal_tensor) =
                batch.get_as_tensor(device);

            let weights_tensor: Tensor<B, 1> = Tensor::from_floats(weights.as_slice(), device);

            let a: Tensor<B, 2, Int> = t_a.unsqueeze_dim(1);

            let q_s_a = online_model.forward(s_tensor.clone());
            let b_q_s_a_scalar: Tensor<B, 1> = q_s_a.clone().gather(1, a).squeeze(1);

            let t_a_p: Tensor<B, 2, Int> = a_p_tensor.unsqueeze_dim(1);
            let q_s_p_target = target_model.forward(s_p_tensor.clone());
            let b_q_s_p_a_p_scalar: Tensor<B, 1> = q_s_p_target.clone().gather(1, t_a_p).squeeze(1);

            let y_is_t = b_q_s_a_scalar
                .clone()
                .sub(r_tensor.clone())
                .mul(is_terminal_tensor.clone())
                .detach();

            let y_no_t = b_q_s_p_a_p_scalar
                .mul_scalar(gamma)
                .add(r_tensor)
                .sub(b_q_s_a_scalar)
                .mul(is_terminal_tensor.sub_scalar(1.))
                .detach();

            let y = y_is_t.add(y_no_t).detach();

            // Calculate TD errors for priority updates
            let td_errors = y.clone().abs().into_data();

            // Update priorities in replay memory
            replay_memory.update_priorities(indices, td_errors);

            // Apply importance sampling weights to the loss
            let loss = y.powf_scalar(2f32).mul(weights_tensor);

            let grad_loss = loss.backward();
            let grads = GradientsParams::from_grads(grad_loss, &online_model);
            online_model = optimizer.step(alpha.into(), online_model, grads);

            episode_reward += r;
            episode_steps += 1;

            #[cfg(feature = "logging")] {
                log_total_loss += loss.mean().into_scalar();
            }
        }

        if (ep_id + 1) % target_update_frequency == 0 {
            target_model = online_model.clone();
        }

        #[cfg(feature = "logging")] {
            let episode_duration = episode_start_time.elapsed();
            log_total_score += episode_reward;
            log_total_steps += episode_steps;
            log_total_time += episode_duration;

            let win = episode_reward > 0.0;
            if win {
                win_count += 1;
            }

            if episode_reward > best_score {
                best_score = episode_reward;
            }

            if (ep_id + 1) % log_interval == 0 && ep_id != 0 {
                let average_score_per_episode = log_total_score / log_interval as f32;
                let average_steps_per_episode = log_total_steps as f32 / log_interval as f32;
                let average_time = log_total_time.as_secs_f32() / log_interval as f32;
                let average_loss = log_total_loss / log_interval as f32;
                let current_epsilon = decayed_epsilon;

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

                log_total_score = 0.0;
                log_total_steps = 0;
                log_total_time = Duration::new(0, 0);
                win_count = 0;
                log_total_loss = 0.0;
            }
        }
    }

    #[cfg(feature = "logging")]
    observer.close();

    online_model
}