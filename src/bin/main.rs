extern crate IABD4_reinforcement_learning;

use burn::backend::Autodiff;
use kdam::tqdm;

//use IABD4_reinforcement_learning::environement::farkle::farkle::{Farkle, NUM_ACTIONS, NUM_STATE_FEATURES};

use IABD4_reinforcement_learning::environement::farkle_2::farkle_2::{Farkle2, NUM_ACTIONS, NUM_STATE_FEATURES};

//use IABD4_reinforcement_learning::environement::tic_tac_toe::tic_tac_toe::{ TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES, };
/*
use IABD4_reinforcement_learning::environement::tic_tac_toe::tic_tac_toe::{
    TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES,
};
>>>>>>> 11bfc8a (impl reinforce with mean_baseline)

 */


use IABD4_reinforcement_learning::ml_core::mlp::MyQMLP;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::ddqn_with_prioritized_replay::deep_double_q_learning_per;

//type GameEnv = TicTacToeVersusRandom;
type GameEnv = Farkle2;

type MyBackend = burn_tch::LibTorch;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    //let device = &LibTorchDevice::Cuda(0);
    let device = &Default::default();

    // Create the model

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

    /*

    let trained_model = deep_q_learning::<
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
        _,
        MyAutodiffBackend,
        GameEnv,
    >(
        model,
        10_000,
        100,
        0.999f32,
        3e-3f32,
        1.0f32,
        1e-5f32,
        40,
        &device,
    );

     */
    let mut wr = 0.;

    for _ in tqdm!(0..1) {
        let online_model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
        let target_model = online_model.clone();
        //let mut value_model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, 1);

        /*
        let model = ppo::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
            model.clone(), value_model.clone(), 100, 3e-5, 0.999f32,0.95, 2048, 2,64, 10, 1.0f32, 1e-5f32, &device
        );

         */
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
        /*

        */

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

        //let replay_capacity = vec![10, 100, 1000];
        //let batch_size = vec![5, 10, 20, 50, 100];
        //for rep in replay_capacity.clone() {
            //for batch in batch_size.clone() {
            //     if batch >= rep {
            //         continue;
            //     }
        deep_double_q_learning_per::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
                    online_model.clone(),
                    target_model.clone(),
                    10_000,
                    5_000,
                    0.99,
                    0.0001f32,
                    1.0,
                    0.01,
                    64,
                    200,
                    0.6,
                    0.4,
                    &device,
                );
            }
        }

        /*
        // Let's play some games (press enter to show the next game)
        let device = &Default::default();
        let mut env = GameEnv::default();
        let mut rng = Xoshiro256PlusPlus::from_entropy();

        let mut win = 0.;
        let mut lose = 0.;
        for _ in 0..1000 {
            env.reset();
            while !env.is_terminal() {
                let s = env.state_description();
                let s_tensor: Tensor<MyBackend, 1> = Tensor::from_floats(s.as_slice(), device);

                let mask = env.action_mask();
                let mask_tensor: Tensor<MyBackend, 1> = Tensor::from(mask).to_device(device);
                let q_s = model.valid().forward(s_tensor);

                let a = epsilon_greedy_action::<MyBackend, NUM_STATE_FEATURES, NUM_ACTIONS>(
                    &q_s,
                    &mask_tensor,
                    env.available_actions_ids(),
                    -1.,
                    &mut rng,
                );
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
        wr += win / (win + lose);
         */
    //}
    //println!("total winrate : {}", wr / 100.);
//}
