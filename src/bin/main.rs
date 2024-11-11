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

//use IABD4_reinforcement_learning::environement::farkle::farkle::{Farkle, NUM_ACTIONS, NUM_STATE_FEATURES};

//use IABD4_reinforcement_learning::environement::farkle_2::farkle_2::{Farkle2, NUM_ACTIONS, NUM_STATE_FEATURES};


use IABD4_reinforcement_learning::environement::tic_tac_toe::tic_tac_toe::{TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES};

use IABD4_reinforcement_learning::ml_core::mlp::MyQMLP;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::deep_q_learning::deep_q_learning;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::reinforce::reinforce;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::utils::epsilon_greedy_action;
use IABD4_reinforcement_learning::logger::Logger;
use IABD4_reinforcement_learning::training_observer::{Hyperparameters, TrainingObserver};

//type GameEnv = TicTacToeVersusRandom;
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

    let model_name = "dqn_log_tester_reduced_2";
    let mut logger = Logger::new(model_name);


    let hyperparameters = Hyperparameters {
        num_episodes: 10_000,
        replay_capacity: 10_0,
        gamma: 0.999f32,
        alpha: 3e-3,
        start_epsilon: 1.0f32,
        final_epsilon: 1e-5f32,
        batch_size: 40,
        log_interval: 200,
    };


    let trained_model = deep_q_learning::<
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
        _,
        MyAutodiffBackend,
        GameEnv,
        Logger,
    >(
        model,
        &hyperparameters,
        &device,
        &mut logger,
    );

    logger.close();

    /*
    let model = reinforce::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
        model,
        1_000_000,
        3e-3,
        0.999f32,
        1.0f32,
        1e-5f32,
        &device,
    );
     */

    // Let's play some games (press enter to show the next game)
    let mut env = GameEnv::default();
    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut win = 0.;
    let mut lose = 0.;
    for _ in 0..1000{
        env.reset();
        while !env.is_terminal() {
            let s = env.state_description();
            let s_tensor: Tensor<MyBackend, 1> = Tensor::from_floats(s.as_slice(), device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<MyBackend, 1> = Tensor::from(mask).to_device(device);
            let q_s = trained_model.valid().forward(s_tensor);


            let a = epsilon_greedy_action::<MyBackend, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s,
                &mask_tensor,
                env.available_actions_ids(),
                -1.,
                &mut rng);

            env.step(a);
        }
        if env.score > 0. {
            win += 1.;
        } else {
            lose += 1.;
        }

    }

    println!("Win: {}, Lose: {}", win, lose);
    println!("winrate : {}", win / (win + lose));
}