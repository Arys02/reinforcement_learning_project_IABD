use std::collections::HashMap;

use ndarray_rand::rand::SeedableRng;
use rand::prelude::{IteratorRandom, StdRng};
use rand::Rng;

use crate::environement::environment::Environment;

/*
pub fn q_learning<E: Environment>(
    mut env: E,
    alpha: f64,
    epsilon: f64,
    gamma: f64,
    nb_iter: i32,
    seed: u64,
) -> i32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut Q = HashMap::new();

    for _ in 0..nb_iter {
        env.reset();

        while !env.is_terminal() {
            let state = env.state_id();
            let available_action = env.available_action();

            if Q.contains_key(&state) {
                let mut q_s = HashMap::new();

                for a in &available_action {
                    q_s.insert(*a, rng.gen::<f64>())
                }

                Q.insert(state, q_s);
            }

            let mut action : usize;
            if rng.gen::<f64>() < epsilon {
               action = E::available_actions(state).as_slice().unwrap().choose(&mut rng).unwrap()
            }
            else {
                let q_s : Vec<f64> = available_action.iter()
                    .map(|&a| *Q.get(&state).unwrap().get(&a).unwrap())
                    .collect();
            }
        }

    }

    return 0;

}

#[cfg(test)]
mod tests {
    use crate::environement::line_world::LineWorld;
    use crate::environement::grid_world::GridWorld;
    use super::*;


    #[test]
    fn monte_carlo_with_exploring_start_returns_correct_policy() {
        let lw = LineWorld::new();

        println!("stat ID :{:?}", lw.state_id());

        let policy = monte_carlo_with_exploring_start(lw, 0.999, 100, 10, 42);
        println!("{:?}", policy)
    }

    #[test]
    fn monte_carlo_with_exploring_start_returns_correct_policy_grid_world() {println!("gridworld : ");
        let gw= GridWorld::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = monte_carlo_with_exploring_start(gw, 0.999, 10, 10, 42);

        println!("{:?}", policy);
        assert_eq!(1, 1)
    }
}

 */