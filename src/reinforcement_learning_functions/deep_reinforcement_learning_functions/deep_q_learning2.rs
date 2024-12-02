use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use crate::training_observer::{Hyperparameters, TrainingEvent, TrainingObserver};
use std::fmt::{Debug, Display};

use crate::ml_core::ml_traits::Forward;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::{Duration, Instant};
use burn::tensor::activation::relu;
use crate::logger::Logger;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::dqn_trajectory::trajectory::Trajectory;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::utils::epsilon_greedy_action;

pub fn deep_q_learning2<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B = B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Debug,
>(
    mut model: M,
    num_episodes: usize,
    replay_capacity: usize,
    epoch: usize,
    gamma: f32,
    alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    batch_size: usize,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{


    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();

    let mut replay_memory: Trajectory<B> = Trajectory::new(replay_capacity);

    if batch_size > replay_capacity {
        panic!("batch_size should't be bigger than replay capacity")
    }

    let mut rng = Xoshiro256PlusPlus::from_entropy();


    let mut env = Env::default();

    //LOGGER
    #[cfg(feature = "logging")]
    let hyperparameters = Hyperparameters {
        num_episodes,
        replay_capacity,
        gamma,
        alpha,
        start_epsilon,
        final_epsilon,
        batch_size,
        log_interval: 500,
    };



    #[cfg(feature = "logging")]
    let log_interval = hyperparameters.log_interval;

    #[cfg(feature = "logging")]
    let model_name = format!("dqnv4_{}_{}_model2_2", env.get_name(), epoch);

    #[cfg(feature = "logging")]
    let mut observer = Logger::new(&model_name, &format!("{}_{}_{}_{}_{}", num_episodes, replay_capacity, batch_size, gamma, alpha));

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

    let mut total_score = 0.0;
    let mut nb_steps = 0.0;

    for ep_id in tqdm!(0..num_episodes) {
        env.reset();

        //LOGGER ?
        let mut episode_reward = 0.0;
        let mut episode_steps = 0;

        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        let episode_start_time = Instant::now();

        if ep_id % 1000 == 0 {
            println!("Mean Score: {}, Mean nb_steps : {}", total_score / 1000.0, nb_steps / 1000.);
            total_score = 0.0;
            nb_steps = 0.0;
        }

        let mut count_max_step = 0;
        while !env.is_terminal() && count_max_step <= 100 {
            nb_steps += 1.;
            count_max_step += 1;
            let s = env.state_description();
            let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
            let q_s = relu(model.forward(s_tensor.clone()));

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
            let q_s_p = Tensor::from_inner(relu(model.valid().forward(s_p_tensor.clone().inner())));

            //we want the best action for s_p
            let mut a_p_max = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s_p,
                &mask_p_tensor,
                env.available_actions_ids(),
                -1.,
                &mut rng,
            );

            //t1 += now.elapsed().as_secs_f32();
            //let now = Instant::now();
            //Store transition ( greedy_ε_Φt(s, a), rt, arg_max_aΦt+1(s', a'), is_terminal)
            replay_memory.push(
                s_tensor,
                a as f32,
                r,
                s_p_tensor,
                a_p_max as f32,
                if env.is_terminal() { 1.0 } else { 0. },
            );
            //t2_1 += now.elapsed().as_secs_f32();
            //let now = Instant::now();

            if replay_memory.len < batch_size{
                continue;
            }


            total_score += env.score();

            for _ in 0..epoch {
                let mut batch = replay_memory.get_batch(batch_size);
                //t2_2 += now.elapsed().as_secs_f32();
                //let now = Instant::now();

                let batch_ =
                    batch.get_as_vector();
                //t2_3 += now.elapsed().as_secs_f32();
                //let now = Instant::now();
                let batch_s_tensor: Tensor<B, 2> = Tensor::from(Tensor::stack(batch.s_t, 0));
                let batch_s_p_tensor: Tensor<B, 2> = Tensor::from(Tensor::stack(batch.s_p_t, 0));


                ///////////////////
                let b_q_s_a = relu(model.forward(batch_s_tensor.clone())).detach();
                let b_q_s_p_a_p = relu(model.forward(batch_s_p_tensor.clone())).detach();


                let y: Tensor<B, 1> = Tensor::from_floats({
                                                              let x: Vec<f32> = batch_.into_iter()
                                                                  .enumerate().map(
                                                                  |(i, (_, a, r, _,
                                                                      a_p_max, is_terminal))|
                                                                      {
                                                                          let a = a as usize;
                                                                          let q_s_a = b_q_s_a.clone()
                                                                              .slice([i..(i + 1), a..(a + 1)])
                                                                              .into_scalar();

                                                                          if is_terminal == 1. {
                                                                              q_s_a - r
                                                                          } else {
                                                                              let a_p_max = a_p_max as usize;
                                                                              let q_s_p_a_p =
                                                                                  b_q_s_p_a_p.clone()
                                                                                      .slice([i..(i + 1), a_p_max..
                                                                                          (a_p_max + 1)])
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



                //LOGGER
                episode_reward += r;
                episode_steps += 1;

                #[cfg(feature = "logging")] {
                    log_total_loss += loss.mean().into_scalar();
                }
            }
        }
        //LOGGER

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
                let average_score_per_episode = log_total_score / (log_interval * epoch) as f32;
                let average_steps_per_episode = log_total_steps as f32 / (log_interval * epoch) as f32;
                let average_time = log_total_time.as_secs_f32() / log_interval as f32;
                let average_loss = log_total_loss / (log_interval * epoch) as f32;
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


    //LOGGER
    #[cfg(feature = "logging")]
    observer.close();

    model
}
