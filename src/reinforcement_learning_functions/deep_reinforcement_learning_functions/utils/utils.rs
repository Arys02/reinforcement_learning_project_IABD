use burn::prelude::{Backend, Tensor};
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;
use rand::prelude::IteratorRandom;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn epsilon_greedy_action<
    B: Backend<FloatElem = f32, IntElem = i64>,
    const NUM_STATES_FEATURES: usize,
    const NUM_ACTIONS: usize,
>(
    q_s: &Tensor<B, 1>,
    mask_tensor: &Tensor<B, 1>,
    available_actions: impl Iterator<Item = usize>,
    epsilon: f32,
    rng: &mut impl Rng,
) -> usize {
    if rng.gen_range(0f32..=1f32) < epsilon {
        available_actions.choose(rng).unwrap()
    } else {
        let inverted_mask = mask_tensor
            .clone()
            .mul([-1f32; NUM_ACTIONS].into())
            .add([1f32; NUM_ACTIONS].into());
        let masked_q_s = (q_s.clone() * mask_tensor.clone())
            .add(inverted_mask.mul([f32::MIN; NUM_ACTIONS].into()));
        masked_q_s.clone().argmax(0).into_scalar() as usize
    }
}

pub fn soft_max_with_mask_action<
    B: Backend<FloatElem = f32, IntElem = i64>,
    const NUM_STATES_FEATURES: usize,
    const NUM_ACTIONS: usize,
>(
    X: &Tensor<B, 1>,
    M: &Tensor<B, 1>,
) -> usize {
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    // Étape 1 : Décaler X pour que toutes les valeurs soient positives
    let min_x = X.clone().min();
    let positive_x = X.clone().sub(min_x);

    // Étape 2 : Appliquer le masque
    let masked_positive_X = positive_x.mul(M.clone());

    // Étape 3 : Décaler masked_positive_X en soustrayant son maximum
    let max_masked_positive_X = masked_positive_X.clone().max();
    let negative_masked_X = masked_positive_X.sub(max_masked_positive_X);

    // Étape 4 : Calculer l'exponentielle
    let exp_X = negative_masked_X.exp();

    // Étape 5 : Appliquer le masque à nouveau
    let filtered_exp_X = exp_X.mul(M.clone());

    // Étape 6 : Calculer la somme des exponentielles filtrées
    let sum_filtered_exp_X = filtered_exp_X.clone().sum();

    // Étape 7 : Diviser les exponentielles filtrées par la somme pour obtenir les probabilités
    let output = filtered_exp_X.div(sum_filtered_exp_X);
    let soft_prob : Vec<f32>=  output.to_data().to_vec().unwrap();
    let mut dist_ = WeightedIndex::new(&soft_prob).unwrap();

    dist_.sample(&mut rng)
}

//source : https://burn.dev/blog/burn-rusty-approach-to-tensor-handling/

fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    log_softmax(tensor, dim).exp()
}

fn log_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    tensor.clone() - tensor.exp().sum_dim(dim).log()
}
