extern crate IABD4_reinforcement_learning;

use burn::backend::Autodiff;

use burn::module::AutodiffModule;
use burn::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use IABD4_reinforcement_learning::environement::environment_traits::ActionEnv;
use IABD4_reinforcement_learning::environement::environment_traits::BaseEnv;
use IABD4_reinforcement_learning::environement::environment_traits::DeepDiscreteActionsEnv;
use IABD4_reinforcement_learning::environement::farkle::farkle::Farkle;
use IABD4_reinforcement_learning::environement::farkle::farkle::{NUM_ACTIONS,
                                                                 NUM_STATE_FEATURES};
use IABD4_reinforcement_learning::ml_core::mlp::MyQMLP;

type GameEnv = Farkle;
use IABD4_reinforcement_learning::reinforcement_learning_functions::deep_reinforcement_learning_functions::episodic_semi_gradiant_sarsa::{episodic_semi_gradient_sarsa, epsilon_greedy_action};

type MyBackend = burn::backend::LibTorch;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = &Default::default();

    // Create the model
    let model = MyQMLP::<MyAutodiffBackend>::new(device,
                                                 NUM_STATE_FEATURES,
                                                 NUM_ACTIONS);

    // Train the model
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

    // Let's play some games (press enter to show the next game)
    let device = &Default::default();
    let mut env = GameEnv::default();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
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
        let mut s = String::new();
        std::io::stdin().read_line(&mut s).unwrap();
    }
}