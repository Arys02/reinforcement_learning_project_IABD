extern crate IABD4_reinforcement_learning;

use burn::backend::Autodiff;

use burn::module::AutodiffModule;
use burn::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use IABD4_reinforcement_learning::environement::environment_traits::{DeepDiscreteActionsEnv, Playable};
use IABD4_reinforcement_learning::environement::farkle::farkle::Farkle;
use IABD4_reinforcement_learning::environement::farkle::farkle::{NUM_ACTIONS,
                                                                 NUM_STATE_FEATURES};
use IABD4_reinforcement_learning::ml_core::mlp::MyQMLP;


fn main() {
    //Farkle::play_as_human();

    let start = std::time::Instant::now();
    let max = 10000;
    for i in 0..max{
        Farkle::play_as_random_ai()
    }
    println!("time for : {max} {:?}", start.elapsed());
    // 10 000 partie en 16.44 s
}