use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use crate::training_observer::{Hyperparameters, TrainingEvent, TrainingObserver};
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
use std::time::{Duration, Instant};

pub fn deep_q_learning<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B=B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem=f32, IntElem=i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Debug + Display,
    Obs: TrainingObserver,

>(
    mut model: M,
    hyperparameters: &Hyperparameters,
    device: &B::Device,
    observer: &mut Obs,
) -> M
where
    M::InnerModule: Forward<B=B::InnerBackend>,
{
    let num_episodes = hyperparameters.num_episodes;
    let replay_capacity = hyperparameters.replay_capacity;
    let gamma = hyperparameters.gamma;
    let alpha = hyperparameters.alpha;
    let batch_size = hyperparameters.batch_size;
    let start_epsilon = hyperparameters.start_epsilon;
    let final_epsilon = hyperparameters.final_epsilon;
    let log_interval = hyperparameters.log_interval;


    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();

    //initialize replay memory D to capacity N
    let mut replay_memory: Vec<(Tensor<B, 1>, usize, f32, Tensor<B, 1>, usize, bool)> =
        Vec::with_capacity
            (replay_capacity);

    let mut replay_memory_q: Vec<Tensor<B, 1>> = Vec::with_capacity(replay_capacity);

    if batch_size > replay_capacity {
        panic!("batch_size should't be bigger than replay capacity")
    }

    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut log_total_score: f32 = 0.0;
    let mut log_total_steps: usize = 0;
    let mut win_count: usize = 0;
    let mut best_score: f32 = f32::MIN;
    let mut log_total_time: Duration = Duration::new(0, 0);
    let mut log_total_loss: f32 = 0.0;
    let mut log_total_reward: f32 = 0.0;

    let mut i_replay: usize = 0;

    let mut env = Env::default();

    observer.on_event(&TrainingEvent::HyperparametersLogged {
        hyperparameters: hyperparameters.clone(),
    });

    //let mut bar = tqdm!();

    for ep_id in tqdm!(0..num_episodes) {
        env.reset();
        //bar.update(1).unwrap();

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
            replay_memory_q.insert(
                i_replay, s_tensor.clone(),
            );

            i_replay = (i_replay + 1) % replay_capacity;

            if replay_memory.len() < batch_size {
                continue;
            }

            //                Φ_s          a       r   Φ_s_p         a_max  is_terminal
            let batch: Vec<(Tensor<B, 1>, usize, f32, Tensor<B, 1>, usize, bool)> = replay_memory
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect();

            let (batch_q, batch_q_p) = batch.iter().map(|(x1, x2, x3, x4, x5, x6), | {
                (x1
                     .clone(),
                 x4.clone())
            })
                .collect();

            let batch_q_tensor: Tensor<B, 2> = Tensor::from(Tensor::stack(batch_q, 0));
            let batch_q_p_tensor: Tensor<B, 2> = Tensor::from(Tensor::stack(batch_q_p, 0));

            let b_q_s_a = model.forward(batch_q_tensor.clone());
            let b_q_s_p_a_p = model.forward(batch_q_p_tensor.clone());

            //output[i, j, k] = input[indices[i, j, k], j, k]; // dim = 0 output[i, j, k] =
            // input[i, indices[i, j, k], k]; // dim = 1 output[i, j, k] = input[i, j, indices[i, j, k]]; // dim = 2
            //let b = q_s_a.squeeze_dims(&[]).into_scalar();

            let y: Tensor<B, 1> = Tensor::from_floats({
                                                          let x: Vec<f32> = batch.into_iter()
                                                              .enumerate().map(
                                                              |(i, (s_tensor, a, r, s_p_tensor,
                                                                  a_p_max, is_terminal))|
                                                                  {
                                                                      let q_s_a = b_q_s_a.clone()
                                                                          .slice([i..(i + 1), a..(a + 1)])
                                                                          .into_scalar();

                                                                      if is_terminal {
                                                                          q_s_a - r
                                                                      } else {
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

            episode_reward += r;
            episode_steps += 1;

            log_total_loss += loss.mean().into_scalar();
        }
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

    model
}
