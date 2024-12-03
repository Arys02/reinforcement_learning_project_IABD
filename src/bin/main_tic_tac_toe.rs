extern crate IABD4_reinforcement_learning;

use burn::backend::Autodiff;

use IABD4_reinforcement_learning::environement::tic_tac_toe::tic_tac_toe::TicTacToeVersusRandom;
use IABD4_reinforcement_learning::environement::tic_tac_toe::tic_tac_toe::NUM_ACTIONS;
use IABD4_reinforcement_learning::environement::tic_tac_toe::tic_tac_toe::NUM_STATE_FEATURES;

use IABD4_reinforcement_learning::ml_core::mlp::MyQMLP;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::ddqn_with_prioritized_replay::deep_double_q_learning_per;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::deep_q_learning2::deep_q_learning2;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::deep_q_learning::deep_q_learning;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::double_deep_q_no_replay::deep_double_q_learning;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::ppo::ppo;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::reinforce::reinforce;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::reinforce_with_mean_baseline::reinforce_with_mean_baseline;

type GameEnv = TicTacToeVersusRandom;

type MyBackend = burn_tch::LibTorch;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = &Default::default();

    /** DQN without epochs */
    let mut model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    model = deep_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
        model.clone(),
        100_000,
        10000,
        0.999f32,
        3e-3f32,
        1.0f32,
        1e-5f32,
        5,
        &device,
    );

    /** DDQN no replay */
    let online_model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let target_model = online_model.clone();
    deep_double_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
        online_model.clone(),
        target_model.clone(),
        500_000,
        0.99,
        0.001f32,
        1e-5f32,
        1.0f32,
        20,
        &device,
    );

    /** DDQN Learning EXP Replay */
    let online_model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let target_model = online_model.clone();

    deep_double_q_learning_per::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
        online_model.clone(),
        target_model.clone(),
        5_000,
        5_000,
        0.99,
        0.001f32,
        1.0,
        0.01,
        64,
        200,
        0.6,
        0.4,
        &device,
    );


    /** Reinforce **/
    let num_episodes = vec![100_000];
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


    /** Reinforce with baseline **/
    //let num_episodes = vec![1_000, 10_000, 100_000, 1_000_000];
    let num_episodes = vec![100_000];
    for num_episode in num_episodes {
        let mut model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
        let mut value_model = MyQMLP::<MyAutodiffBackend>::new(
            &device,
            NUM_STATE_FEATURES,
            1,
        );
        model = reinforce_with_mean_baseline::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
            model,
            value_model,
            num_episode,
            1e-3,
            0.999f32,
            1.0f32,
            1e-5f32,
            &device,
        );
    }

    /** PPO **/
    let mut value_model = MyQMLP::<MyAutodiffBackend>::new(
        &device,
        IABD4_reinforcement_learning::environement::farkle_2::farkle_2::NUM_STATE_FEATURES,
        1,
    );
    let model = MyQMLP::<MyAutodiffBackend>::new(
        &device,
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
    );

    let model =
        ppo::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
            model.clone(),
            value_model.clone(),
            5000,
            3e-5,
            0.999f32,
            0.95,
            2048,
            10,
            10,
            10,
            1.0f32,
            1e-5f32,
            &device,
        );

    /*
    //let num_episodes = vec![1_000, 10_000, 100_000, 1_000_000];
    let num_episodes = vec![100_000];
    for num_episode in num_episodes {
        let mut model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
        model = reinforce::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
            model,
            num_episode,
            1e-3,
            0.999f32,
            1.0f32,
            1e-5f32,
            &device,
        );
    }
    return;

     */

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
    let num_episodes = vec![100, 200, 500, 1000];
    let num_horizon = vec![10, 100, 1000, 5000];
    let n_actors = vec![1, 2, 5, 10];
    let num_batch_size = vec![4, 8, 16, 32, 64, 128];
    let num_epoch = vec![2, 4, 8, 16, 32];

    for n_episode in &num_episodes {
        for horizon in &num_horizon {
            for actors in &n_actors {
                for batch_size in &num_batch_size {
                    if (*batch_size * *actors) >= *horizon {
                        continue;
                    }
                    for epoch in &num_epoch {
                        let mut value_model = MyQMLP::<MyAutodiffBackend>::new(
                                &device,
                                IABD4_reinforcement_learning::environement::farkle_2::farkle_2::NUM_STATE_FEATURES,
                                1,
                            );
                        let model = MyQMLP::<MyAutodiffBackend>::new(
                            &device,
                            NUM_STATE_FEATURES,
                            NUM_ACTIONS,
                        );

                        let model =
                            ppo::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
                                model.clone(),
                                value_model.clone(),
                                *n_episode,
                                3e-5,
                                0.999f32,
                                0.95,
                                *horizon,
                                *actors,
                                *batch_size,
                                *epoch,
                                1.0f32,
                                1e-5f32,
                                &device,
                            );
                    }
                }
            }
        }
    }
     */
    //let replay_capacity = vec![10, 100, 1000, 10000];
    //let batch_size = vec![5, 10, 20, 50, 100];
    //let epoch_size = vec![1, 2, 16, 64, 124];

    let replay_capacity = vec![10000];
    let batch_size = vec![10];
    let epoch_size = vec![1];


    for rep in replay_capacity.clone() {
        for batch in batch_size.clone() {
            for epoch in epoch_size.clone() {
                if batch >= rep {
                    continue;
                }
                let model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
                deep_q_learning2::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
                    model.clone(),
                    100_000,
                    rep,
                    epoch,
                    0.999f32,
                    3e-3f32,
                    1.0f32,
                    1e-5f32,
                    batch,
                    &device,
                );

            }

        }
    }


    /*
    let replay_capacity = vec![10, 100, 1000, 10000];
    let batch_size = vec![5, 10, 20, 50, 100];

    for rep in replay_capacity.clone() {
        for batch in batch_size.clone() {
            if batch >= rep {
                continue;
            }
            let model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
            deep_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
                model.clone(),
                100_000,
                rep,
                0.999f32,
                3e-3f32,
                1.0f32,
                1e-5f32,
                batch,
                &device,
            );
        }
    }

     */
    /*

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

     */

    /*
    let num_episodes = vec![1_000, 10_000, 100_000, 1_000_000];
    for num_episode in num_episodes {
        let mut model = MyQMLP::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
        let mut value_model = MyQMLP::<MyAutodiffBackend>::new(
            &device,
            NUM_STATE_FEATURES,
            1,
        );
        model = reinforce_with_mean_baseline::<NUM_STATE_FEATURES, NUM_ACTIONS, _, MyAutodiffBackend, GameEnv>(
            model,
            value_model,
            num_episode,
            1e-4,
            0.999f32,
            1.0f32,
            1e-5f32,
            &device,
        );
    }

     */

}
