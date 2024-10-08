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
    let mut total_score = Vec::new();
    let max = 10000;
    for i in 0..max{
        total_score.push(Farkle::play_as_random_ai());
    }
    println!("time for : {max} {:?}", start.elapsed());

    //for 10 step, 10 000 parties : 3.07 sec
    //for 50 step, 10 000 parties : 14.14 sec

    let mut sum = [0usize; 2];

    // Nombre d'éléments dans le vecteur
    let len = total_score.len();

    // Parcourir le vecteur et accumuler les sommes
    for a in total_score.iter() {
        sum[0] += a[0];
        sum[1] += a[1];
    }

    // Calculer les moyennes
    let average = [
        sum[0] as f64 / len as f64,
        sum[1] as f64 / len as f64,
    ];

    println!("Les moyennes sont : {:?}", average);

}