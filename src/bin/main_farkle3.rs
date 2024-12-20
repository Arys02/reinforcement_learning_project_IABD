extern crate IABD4_reinforcement_learning;

use burn::backend::Autodiff;

use IABD4_reinforcement_learning::environement::farkle_3::farkle_3::{
    Farkle3, NUM_ACTIONS, NUM_STATE_FEATURES,
};

use IABD4_reinforcement_learning::ml_core::mlp::MyQMLP;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::ddqn_with_prioritized_replay::deep_double_q_learning_per;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::deep_q_learning::deep_q_learning;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::double_deep_q_no_replay::deep_double_q_learning;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::ppo::ppo;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::reinforce::reinforce;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::reinforce_with_mean_baseline::reinforce_with_mean_baseline;

type GameEnv = Farkle3;

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
        500_000,
        5000,
        0.99,
        0.001f32,
        1e-5f32,
        1.0f32,
        32,
        20,
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

     */
}
