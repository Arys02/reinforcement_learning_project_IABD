use crate::environement::environment_traits::DeepDiscreteActionsEnv;
use std::fmt::{Debug, Display};

use crate::ml_core::ml_traits::Forward;
use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::epsilon_greedy_action;

use crate::reinforcement_learning_functions::deep_reinforcement_learning_functions::ppo_trajectory::trajectory::Trajectory;

use burn::module::AutodiffModule;

use crate::ml_core::mlp::MyQMLP;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::SgdConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
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

    let clip = 0.2;

    let value_fct = MyQMLP::<B>::new(&device, NUM_STATE_FEATURES, 1);

    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;

    let mut env = Env::default();
    
    let mut old_pi = None;

    for ep_id in tqdm!(0..num_episodes) {
        env.reset();

        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        if ep_id % 1000 == 0 {
            println!("Mean Score: {}", total_score / 1000.0);
            total_score = 0.0;
        }

        let mut trajectory = Trajectory::<B>::new(memory_size);

        //generate episodes
        //for i in 0..memory_size{
        env.reset();

        let mut g = 0.;
        

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
            let s_t = env.state_description();
            let s_t_tensor: Tensor<B, 1> = Tensor::from_floats(s_t.as_slice(), device);


            let r = env.score() - prev_score;

            g = r + gamma * g;

            trajectory.push(s_tensor.clone(), a, r, g, pi_s.log().clone(), s_t_tensor.clone());

            if env.is_terminal() {
                env.reset();
                break;
            }
            //   }
        }
        
        let s_tensor: Tensor<B, 2> = Tensor::from(Tensor::stack(trajectory.s_t.clone(), 0)).detach();
        let pi_s = model.forward(s_tensor.clone()).detach();
        let v_s = value_fct.forward(s_tensor).detach();
        
        let s_p_tensor: Tensor<B, 2> = Tensor::from(Tensor::stack(trajectory.s_p_t.clone(), 0)).detach();
        let pi_s_p = model.forward(s_p_tensor.clone()).detach();
        let v_s_p = value_fct.forward(s_p_tensor).detach();
        
        
        let r_tensor: Tensor<B, 1> = Tensor::from_floats(trajectory.r_t.as_slice(), device);
        let r_tensor= r_tensor.clone().reshape([r_tensor.shape().dims[0], 1]);

        let g_tensor: Tensor<B, 1> = Tensor::from_floats(trajectory.r_cul.as_slice(), device);
        let g_tensor= r_tensor.clone().reshape([g_tensor.shape().dims[0], 1]);
        
        let advantage = (v_s_p.clone().sub(v_s).mul_scalar(gamma)).add(r_tensor);

        total_score += env.score();
        
        let loss = v_s_p.sub(g_tensor).powf_scalar(2f32).mean();
        

    }

    model
}
