extern crate IABD4_reinforcement_learning;

use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn_tch::LibTorchDevice;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use IABD4_reinforcement_learning::environement::environment_traits::ActionEnv;
use IABD4_reinforcement_learning::environement::environment_traits::BaseEnv;
use IABD4_reinforcement_learning::environement::environment_traits::DeepDiscreteActionsEnv;
/*
use IABD4_reinforcement_learning::environement::farkle::farkle::Farkle;
use IABD4_reinforcement_learning::environement::farkle::farkle::{NUM_ACTIONS,
                                                                 NUM_STATE_FEATURES};

 */
use IABD4_reinforcement_learning::environement::tic_tac_toe::tic_tac_toe::{TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES};
use IABD4_reinforcement_learning::ml_core::mlp::MyQMLP;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::deep_q_learning::deep_q_learning;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::epsilon_greedy_action;

type GameEnv = TicTacToeVersusRandom;

type MyBackend = burn_tch::LibTorch;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {

    //let device = &LibTorchDevice::Cuda(0);
    let device = &Default::default();


    // Create the model
    let model = MyQMLP::<MyAutodiffBackend>::new(&device,
                                                 NUM_STATE_FEATURES,
                                                 NUM_ACTIONS);

    // Train the model
    /* SARSA
    let model =
        episodic_semi_gradient_sarsa::<
            NUM_STATE_FEATURES,
            NUM_ACTIONS,
            _,
            MyAutodiffBackend,
            GameEnv,
        >(
            model,
            50_000,
            0.999f32,
            3e-3,
            1.0f32,
            1e-5f32,
            device,
        );

     */
    let model =
        deep_q_learning::<
            NUM_STATE_FEATURES,
            NUM_ACTIONS,
            _,
            MyAutodiffBackend,
            GameEnv,
        >(
            model,
            20_000,
            0.999f32,
            1000,
            100,
            3e-3,
            1.0f32,
            1e-5f32,
            &device,
        );

    // Let's play some games (press enter to show the next game)
    let device = &Default::default();
    let mut env = GameEnv::default();
    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut win = 0;
    let mut lose = 0;
    loop {
        env.reset();
        while !env.is_terminal() {
            let s = env.state_description();
            let s_tensor: Tensor<MyBackend, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<MyBackend, 1> = Tensor::from(mask).to_device(device);
            let q_s = model.valid().forward(s_tensor);


            let a = epsilon_greedy_action::<MyBackend, NUM_STATE_FEATURES, NUM_ACTIONS>(&q_s, &mask_tensor, env.available_actions_ids(), 1e-5f32, &mut rng);
            env.step(a);
        }
        println!("Score: {}", env.score);
        if env.score > 0. {
            win += 1;
        } else {
            lose += 1;
        }
        println!("Win: {}, Lose: {}", win, lose);
        println!("winrate : {}", win / (win + lose));
        let mut s = String::new();
        std::io::stdin().read_line(&mut s).unwrap();
    }
}