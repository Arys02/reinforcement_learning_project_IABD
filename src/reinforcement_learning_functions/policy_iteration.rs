use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::num_traits::abs;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::prelude::{SliceRandom, StdRng};

use crate::environement::environment::Environment;

pub fn policy_iteration<E: Environment>(
    mut env: E,
    gamma: f32,
    theta: f32,
    seed: u64,
) -> Vec<usize> {
    let num_states = E::num_states();
    let num_actions = E::num_actions();
    let num_rewards = E::num_rewards();


    let mut rng = StdRng::seed_from_u64(seed);

    let mut V = Array::random((num_states, 1), Uniform::new(0.0, 1.0)).into_raw_vec();

    for s in 0..num_states {
        if E::is_terminal_state(s) {
            V[s] = 0.;
        }
    }

    let mut pi = vec![];
    for i in 0..num_states {
        if let Some(&value) = E::available_actions(i).as_slice().unwrap().choose(&mut rng) {
            pi.push(value)
        }
    }
    println!("{:?}", pi);


    loop {
        loop {
            let mut delta: f32 = 0.;
            for s in 0..num_states {
                let a = pi[s];
                if s % 7 == 1 {
                    print!("8");
                }
                let mut v = V[s];
                let mut total = 0.;
                for s_p in 0..num_states {
                    for r in 0..num_rewards {
                        let p = env.get_transition_probability(s, a, s_p, r);
                        total += p * (E::get_reward(r) + gamma * V[s_p])
                    }
                }
                V[s] = total;
                delta = delta.max((v - V[s]).abs())
            }
            if delta < theta {
                break
            }
        }


        println!("{:?}", V);

        let mut policy_stable = true;

        for s in 0..num_states {
            if E::is_terminal_state(s) {
                continue
            }

            let old_action = pi[s];

            let mut best_a: Option<usize> = None;
            let mut best_action_score = -9999.;

            for a in 0..num_actions {
                let mut total = 0.;

                for s_p in 0..num_states {
                    for r_index in 0..num_rewards {
                        total += env.get_transition_probability(s, a, s_p, r_index) * (E::get_reward(r_index) + gamma * V[s_p])
                    }
                }

                if best_a.is_none() || total >= best_action_score {
                    best_a = Some(a);
                    best_action_score = total;
                }
            }
            pi[s] = best_a.unwrap();

            if pi[s] != old_action {
                policy_stable = false;
            }
        }
        if policy_stable {
            return pi
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::environement::line_world::LineWorld;
    use crate::environement::grid_world::GridWorld;

    use super::*;

    #[test]
    fn policy_iteration_line_world() {
        println!("start");


        let lw = LineWorld::new();
        println!("stat ID :{:?}", lw.state_id());

        let v = policy_iteration::<LineWorld>(lw, 0.999, 0.000001, 42);

        println!("{:?}", v);
        assert_eq!(1, 1)
    }

    #[test]
    fn policy_iteration_grid_world() {
        println!("start");
        let mut env = GridWorld::new();

        let v = policy_iteration::<GridWorld>(env, 0.999, 0.0001, 42);

        println!("{:?}", v);
        assert_eq!(1, 1)
    }
}