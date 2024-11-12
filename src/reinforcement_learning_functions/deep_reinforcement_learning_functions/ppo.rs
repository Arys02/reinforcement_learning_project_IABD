use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use std::fmt::{Debug, Display};

use crate::ml_core::ml_traits::Forward;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::epsilon_greedy_action;

use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::ppo_trajectory::trajectory::Trajectory;

use burn::module::AutodiffModule;

use crate::ml_core::mlp::MyQMLP;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use itertools::Itertools;
use kdam::tqdm;
use ndarray_rand::rand_distr::num_traits::clamp;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn ppo<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B = B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Debug + Display,
>(
    mut model: M,
    mut value_fct: M,
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
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init::<B, M>();


    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;

    let mut env = Env::default();



    let mut old_pi : M = model.clone();



    for ep_id in tqdm!(0..num_episodes) {
        let mut trajectory = Trajectory::<B>::new(memory_size);
        env.reset();

        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        if ep_id % 1000 == 0 {
            println!("Mean Score: {}", total_score / 1000.0);
            total_score = 0.0;
        }


        let mut g = 0.;

        //build the trajectory
        while !env.is_terminal() {
            let s = env.state_description();
            let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
            let pi_s = model.forward(s_tensor.clone());

            println!("{:?}", pi_s.clone());

            //TODO change to get a softmax random action

            let a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &pi_s,
                &mask_tensor,
                env.available_actions_ids(),
                decayed_epsilon,
                &mut rng,
            );
            //println!("action : a: {}", a);

            let prev_score = env.score();
            //execute action a_t in emulator and observe reward r_t

            env.step(a);
            let s_t = env.state_description();
            let s_t_tensor: Tensor<B, 1> = Tensor::from_floats(s_t.as_slice(), device);


            let r = env.score() - prev_score;

            g = r + gamma * g;

            trajectory.push(
                s_tensor.clone(),
                a,
                r,
                g,
                pi_s.log().clone(),
                s_t_tensor.clone());

            if env.is_terminal() {
                env.reset();
                break;
            }
            //   }
        }
        total_score += env.score();

        let lmda = 0.95;
        let len = trajectory.s_t.len();

        let mut gae: Tensor<B, 1>= Tensor::from_floats(vec![0.;NUM_ACTIONS].as_slice(), device); ;
        for i in (0..len).rev() {
            let s = trajectory.s_t[i].clone();
            let s_p = trajectory.s_t[len - i - 1].clone();
            let v_s = value_fct.forward(s.clone());
            let v_s_p = value_fct.forward(s_p.clone());
            //println!("i : {}, len : {}", i, len);
            let delta = v_s_p.clone().mul_scalar(gamma).add_scalar(trajectory.r_t[i]).sub(v_s.clone());
            let tmp = gae.clone().mul_scalar(lmda * gamma);
            //println!("delta: {:?}, tmp : {:?}", delta.shape(), tmp.shape());
            gae = delta.add(tmp);
            trajectory.advantage.push(v_s.clone().add(gae.clone()))
        }
        let revers_adv : Vec<Tensor<B, 1>> = trajectory.advantage.clone().into_iter().rev().collect();

        //println!("trajectory len {:?}", trajectory.s_t.len());
        let revers_adv_tensor: Tensor::<B, 2> = Tensor::from(Tensor::stack(revers_adv.clone(), 0)).detach();
        let last_adv = revers_adv_tensor.clone().slice([(len-1)..len]);

        let advantage = revers_adv_tensor.sub(last_adv);

        let s_tensor: Tensor<B, 2> = Tensor::from(Tensor::stack(trajectory.s_t.clone(), 0)).detach();

        let pi_s = model.forward(s_tensor.clone()).detach();
        let v_s = value_fct.forward(s_tensor.clone()).detach();

        for i in 0..trajectory.s_t.len() {
            let pi_s_t: Tensor<B, 1> = pi_s.clone().slice([i..(i+1)]).squeeze(0);
        }

        let a: Tensor<B, 2, Int> = Tensor::from_ints(trajectory.a_t.as_slice(), device).unsqueeze_dim(1);
        let pi_s_a: Tensor<B, 2> = pi_s.clone().gather(1, a.clone());


        
        let r_tensor: Tensor<B, 1> = Tensor::from_floats(trajectory.r_t.as_slice(), device);
        let r_tensor= r_tensor.clone().reshape([r_tensor.shape().dims[0], 1]);

        let g_tensor: Tensor<B, 1> = Tensor::from_floats(trajectory.r_cul.as_slice(), device);
        let g_tensor= r_tensor.clone().reshape([g_tensor.shape().dims[0], 1]);
        

        total_score += env.score();

        let old_pi_s = old_pi.clone().forward(s_tensor.clone());
        let old_pi_s_a: Tensor<B, 2> = old_pi_s.clone().gather(1, a);
        let ratio = pi_s_a.sub(old_pi_s_a).log();
        let clip_param = 0.2;
        let surr1 = ratio.clone().mul(advantage.clone()).clone();
        let surr2 = ratio.clone().clamp(1. - clip_param, 1. + clip_param).mul(advantage.clone());

        let diff = surr1.clone().sub(surr2.clone()).abs();
        let surr_min = (surr1.add(diff.clone()).sub_scalar(0.5));

        let policy_loss = -surr_min.mean();
        let critic_loss = g_tensor.sub(v_s.clone()).powf_scalar(2f32).mean();

        let policy_grad_loss = policy_loss.backward();
        let policy_grads = GradientsParams::from_grads(policy_grad_loss, &model);

        model = optimizer.step(alpha.into(), model, policy_grads);
        old_pi = model.clone();


        let value_grad_loss = critic_loss.backward();
        let value_grads= GradientsParams::from_grads(value_grad_loss, &model);
        value_fct = optimizer.step(alpha.into(), value_fct, value_grads);

    }

    model
}
