use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use std::fmt::{Debug, Display};
use std::time::Duration;
use crate::ml_core::ml_traits::Forward;


use burn::module::AutodiffModule;

use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::logger::Logger;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::ppo_trajectory::trajectory::Trajectory;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::utils::{epsilon_greedy_action, soft_max_with_mask_action};
use crate::training_observer::{Hyperparameters, TrainingEvent, TrainingObserver};

pub fn ppo<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B = B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Debug,
>(
    mut model: M,
    mut value_fct: M,
    num_episodes: usize,
    alpha: f32,
    gamma: f32,
    gae_lmd: f32,
    horizon: usize,
    n_actors: usize,
    batch_size: usize,
    num_epochs: usize,
    start_epsilon: f32,
    final_epsilon: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init::<B, M>();




    let mut total_score = 0.0;
    let mut nb_score = 0.0;

    let mut env = Env::default();

    let mut old_pi: M = model.clone();

    #[cfg(feature = "logging")]
    let hyperparameters = Hyperparameters {
        num_episodes,
        gamma,
        alpha,
        start_epsilon,
        final_epsilon,
        log_interval: 10,
        replay_capacity: 0,
        batch_size: 0,
    };
    #[cfg(feature = "logging")]
    let log_interval = hyperparameters.log_interval;

    #[cfg(feature = "logging")]
    let model_name = format!("ppo_{}_model2", env.get_name());

    #[cfg(feature = "logging")]
    let mut observer = Logger::new(&model_name, &format!("{}_{}_{}_{}_{}_{}_{}_{}", num_episodes, gamma, alpha, gae_lmd, horizon, n_actors,batch_size, num_epochs));

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
    let mut mean_step = 0.0;

    let mut nb_steps = 0.0;
    let mut episode_reward = 0.0;
    //for iteration = 1, 2, ... do
    for ep_id in tqdm!(0..num_episodes) {
        env.reset();




        let mut trajectory_vec = Vec::new();

        for _ in 0..n_actors {
            let mut trajectory = Trajectory::<B>::new(horizon);
            let mut count_trajectory = 0;
            let mut old_count_trajectory = 0;
            while count_trajectory < horizon {
                //println!("{}, {}", old_count_trajectory, count_trajectory);
                let s = env.state_description();
                let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

                let mask = env.action_mask();
                let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
                let pi_s =  old_pi.forward(s_tensor.clone());
                let v_s = value_fct.forward(s_tensor.clone());

                let a = soft_max_with_mask_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                    &pi_s, &mask_tensor
                );

                let prev_score = env.score();
                //execute action a_t in emulator and observe reward r_t

                env.step(a);

                let r = env.score() - prev_score;


                trajectory.push(s_tensor.clone(), a, r, v_s.clone());

                count_trajectory += 1;


                //println!("{}, {}", old_count_trajectory, count_trajectory);
                //println!("env_term : {}, horizon : {}", env.is_terminal(), horizon);
                if env.is_terminal() || count_trajectory == horizon {
                    //println!("end {}, {}", old_count_trajectory, count_trajectory);
                    //println!("{:?}", env);
                    total_score += r;
                    nb_score += 1.0;
                    env.reset();
                    let mut vec_rets = Vec::new();
                    let mut vec_vs = Vec::new();

                    let mut gae: Tensor<B, 1> =
                        Tensor::from_floats(vec![0.; NUM_ACTIONS].as_slice(), device);
                    for i in (old_count_trajectory..count_trajectory).rev() {
                        let v_s = trajectory.v_s[i].clone();
                        let v_s_p = trajectory.v_s[count_trajectory - i - 1].clone();
                        vec_vs.push(v_s.clone());

                        // delta = (V[t + 1] * gamma) + r_t - V[t]
                        let delta = v_s_p
                            .clone()
                            .mul_scalar(gamma)
                            .add_scalar(trajectory.r_t[i])
                            .sub(v_s.clone());
                        let tmp = gae.clone().mul_scalar(gae_lmd * gamma);
                        //println!("delta: {:?}, tmp : {:?}", delta.shape(), tmp.shape());
                        gae = delta.add(tmp);
                        //trajectory.returns.push(v_s.clone().add(gae.clone()))
                        vec_rets.push(v_s.clone().add(gae.clone()));
                    }
                    //reverse the advantage list
                    let revers_rets: Vec<Tensor<B, 1>> =
                        vec_rets.clone().into_iter().rev().collect();
                    //trajectory.returns = revers_adv.clone();

                    //println!("trajectory len {:?}", trajectory.s_t.len());
                    let revers_rets_tensor: Tensor<B, 2> =
                        Tensor::from(Tensor::stack(revers_rets.clone(), 0)).detach();
                    //let last_adv = revers_rets_tensor.clone().slice([(count_trajectory - 1)..count_trajectory]);
                    let last_adv = Tensor::from(Tensor::stack(vec_vs.clone(), 0)).detach();

                    let advantage = revers_rets_tensor.sub(last_adv).detach();

                    for (i, tensor) in advantage.clone().iter_dim(0).enumerate() {
                        trajectory.returns.push(revers_rets[i].clone());
                        trajectory.advantage.push(tensor.squeeze(0));
                    }
                    old_count_trajectory = count_trajectory;
                }
            }
            trajectory_vec.push(trajectory);
        }

        for _ in 0..num_epochs {
            let mut batch_advantages = Vec::new();
            let mut batch_s_t = Vec::new();
            let mut batch_a_t = Vec::new();
            let mut batch_rewards = Vec::new();

            for i in 0..n_actors {
                //println!(" i : {}", i);
                //println!("batchsize {} / nb actor {} : {}", batch_size, n_actors, batch_size/ n_actors);
                let batch = trajectory_vec[i].get_batch(batch_size / n_actors);
                batch_rewards.append(&mut batch.returns.clone());
                batch_a_t.append(&mut batch.a_t.clone());
                batch_s_t.append(&mut batch.s_t.clone());
                batch_advantages.append(&mut batch.advantage.clone());
            }

            //build the
            let advantage: Tensor<B, 2> =
                Tensor::from(Tensor::stack(batch_advantages.clone(), 0)).detach();

            let (variance, mean) = advantage.clone().var_mean_bias(0);
            let std = variance.sqrt();
            let normalized_advantage = advantage
                .clone()
                .sub(mean.clone())
                .div(std.add_scalar(1e-8));

            //trajectory.advantage = normalized_advantage;

            let s_tensor: Tensor<B, 2> = Tensor::from(Tensor::stack(batch_s_t.clone(), 0)).detach();

            let returns: Tensor<B, 2> =
                Tensor::from(Tensor::stack(batch_rewards.clone(), 0)).detach();

            let pi_s = log_softmax(model.forward(s_tensor.clone()).detach(), 1);
            let v_s = value_fct.forward(s_tensor.clone()).detach();

            let a_tensor: Tensor<B, 1, Int> = Tensor::from_ints(batch_a_t.as_slice(), device);
            let a: Tensor<B, 2, Int> = a_tensor.unsqueeze_dim(1);
            let pi_s_a: Tensor<B, 2> = pi_s.clone().gather(1, a.clone()).detach();

            /*
            let r_tensor: Tensor<B, 1> = Tensor::from_floats(trajectory.r_t.as_slice(), device);
            let r_tensor = r_tensor.clone().reshape([r_tensor.shape().dims[0], 1]);

            let g_tensor: Tensor<B, 1> = Tensor::from_floats(trajectory.r_cul.as_slice(), device);
            let g_tensor = r_tensor.clone().reshape([g_tensor.shape().dims[0], 1]);

             */

            let old_pi_s = log_softmax(old_pi.clone().forward(s_tensor.clone()), 1);
            let old_pi_s_a: Tensor<B, 2> = old_pi_s.clone().gather(1, a);
            let ratio = log_softmax(pi_s_a.sub(old_pi_s_a), 1);

            let clip_param = 0.2;
            let surr1 = ratio.clone().mul(normalized_advantage.clone()).clone();
            let surr2 = ratio
                .clone()
                .clamp(1. - clip_param, 1. + clip_param)
                .mul(normalized_advantage.clone());

            let diff = surr1.clone().sub(surr2.clone()).abs();
            let surr_min = (surr1.add(surr2.clone()).sub(diff.clone()).mul_scalar(0.5));

            //println!("surr_min : {:?}\n", surr_min);
            //println!("surr_min_mean : {:?}\n", surr_min.clone().mean());
            let policy_loss = -surr_min.mean();
            let critic_loss = returns.sub(v_s.clone()).powf_scalar(2f32).mean();

            let policy_grad_loss = policy_loss.backward();
            let policy_grads = GradientsParams::from_grads(policy_grad_loss, &model);
            //println!("mean loss : {:?}\n", policy_loss);

            model = optimizer.step(alpha.into(), model, policy_grads);

            let value_grad_loss = critic_loss.backward();
            let value_grads = GradientsParams::from_grads(value_grad_loss, &model);
            value_fct = optimizer.step(alpha.into(), value_fct, value_grads);
        }
        old_pi = model.clone();


        #[cfg(feature = "logging")]
        if ep_id % 10 == 0 {
            println!("Mean Score: {}", total_score / nb_score);
            observer.on_event(&TrainingEvent::LoggingSummary {
                episodes: ep_id + 1,
                total_score: log_total_score,
                average_score_per_episode: total_score / (nb_score * 10.),
                average_steps_per_episode: 0.,
                average_loss: 0.,
                epsilon: 0.,
                win_count,
                best_score,
                average_time: 0.,
                epoch: ep_id + 1,
            });
            total_score = 0.0;
            nb_score = 0.0;
        }

    }

    model
}
