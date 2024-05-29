use ndarray::Array1;
use ndarray_rand::rand::SeedableRng;
use rand::prelude::StdRng;

pub fn get_random_value<A>(array : &Array1<A>, seed: u64) -> A {
    let mut rng = StdRng::seed_from_u64(seed);
    let rand_i = rng.get_range(0..array.len());
    return array.get(rand_i)
}
