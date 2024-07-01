use std::collections::HashMap;

use ndarray::{Array, Array1, Array2, ArrayBase, OwnedRepr};
use ndarray_rand::rand::SeedableRng;
use rand::prelude::{IteratorRandom, StdRng};
use rand::Rng;

use crate::environement::environment::Environment;

pub fn policy_evaluation<E: Environment>(
    policy: Array2<usize>,
    gamma: f32,
    theta: f32,
) -> Array1<f32> {
    println!("{:?}", policy);
    let num_states = E::num_states();
    let num_actions = E::num_actions();
    let num_rewards = E::num_rewards();

    let mut V: Array1<f32> = Array1::zeros(num_states);

    let mut i = 0;

    loop {

       let mut delta: f32 = 0.;
        for s in 0..num_states {
            let mut v = V[s];
            let mut total = 0.;
            for a in 0..num_actions {
                let mut pi_s_a : f32  = policy[[s, a]] as f32;
                for s_p in 0..num_states {
                    for r in 0..num_rewards {
                        total += pi_s_a * E::get_transition_probability(s, a, s_p, r) * (E::get_reward(r) + gamma + V[s_p])
                    }
                }

            }
            V[s] = total;
            delta = delta.max((v - V[s]).abs())
        }
        println!("{:?}, delta : {:?}", i, delta);
        i+=1;
        if delta < theta {
            return V
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::environement::grid_world::GridWorld;
    use crate::environement::line_world::LineWorld;

    use super::*;

    #[test]
    fn policy_evaluation_line_world() {
        println!("start");


        let lw = LineWorld::new();
        println!("stat ID :{:?}", lw.state_id());

        let pi_right : Array2<usize>  = Array2::from_shape_vec((LineWorld::num_states(), 2), vec![
            0, 1,
            0, 1,
            0, 1,
            0, 1,
            0, 1,
        ]).unwrap();
        let pi_left : Array2<usize>  = Array2::from_shape_vec((LineWorld::num_states(), 2), vec![
            1, 0,
            1, 0,
            1, 0,
            1, 0,
            1, 0,
        ]).unwrap();


        let policy = policy_evaluation::<LineWorld>(pi_left, 0.999, 0.00001);


        println!("{:?}", policy);
        assert_eq!(1, 1)
    }
}