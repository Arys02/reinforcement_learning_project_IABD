extern crate IABD4_reinforcement_learning;

use burn::backend::Autodiff;
use kdam::tqdm;

//use IABD4_reinforcement_learning::environement::farkle::farkle::{Farkle, NUM_ACTIONS, NUM_STATE_FEATURES};

use IABD4_reinforcement_learning::environement::farkle_2::farkle_2::{Farkle2, NUM_ACTIONS, NUM_STATE_FEATURES};

//use IABD4_reinforcement_learning::environement::tic_tac_toe::tic_tac_toe::{TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES, };


use IABD4_reinforcement_learning::ml_core::mlp::MyQMLP;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::deep_q_learning::deep_q_learning;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::reinforce::reinforce;

//type GameEnv = TicTacToeVersusRandom;
type GameEnv = Farkle2;

type MyBackend = burn_tch::LibTorch;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    //let device = &LibTorchDevice::Cuda(0);
    let device = &Default::default();

    let mut wr = 0.;

    for _ in tqdm!(0..1) {


        let num_episodes = vec![1_000, 10_000, 100_000, 1_000_000];

        for num_episode in num_episodes {
            let mut model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
            model = reinforce::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
                model,
                num_episode,
                3e-3,
                0.999f32,
                1.0f32,
                1e-5f32,
                &device,
            );

        }


    }
    println!("total winrate : {}", wr / 100.);
}
