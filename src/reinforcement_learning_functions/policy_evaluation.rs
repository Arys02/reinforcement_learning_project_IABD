use ndarray::{Array1, Array2};

use crate::environement::environment_traits::Environment;

pub fn policy_evaluation<
    const NUM_STATES: usize,
    const NUM_ACTIONS: usize,
    const NUM_REWARDS: usize,
    Env: Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS>>
(
    policy: Array2<usize>,
    gamma: f32,
    theta: f32,
) -> Array1<f32> {
    println!("{:?}", policy);
    let num_states = Env::num_states();
    let num_actions = Env::num_actions();
    let num_rewards = Env::num_rewards();

    let mut V: Array1<f32> = Array1::zeros(num_states);

    let mut i = 0;

    loop {
        let mut delta: f32 = 0.;
        for s in 0..num_states {
            let v = V[s];
            let mut total = 0.;
            for a in 0..num_actions {
                let pi_s_a: f32 = policy[[s, a]] as f32;
                for s_p in 0..num_states {
                    for r in 0..num_rewards {
                        total += pi_s_a
                            * Env::build_transition_probability(s, a, s_p, r)
                            * (Env::get_reward(r) + gamma * V[s_p])
                    }
                }
            }
            V[s] = total;
            delta = delta.max((v - V[s]).abs())
        }
        println!("{:?}, delta : {:?}", i, delta);
        i += 1;
        if delta < theta {
            return V;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::environement::line_world::line_world;
    use crate::environement::line_world::line_world::LineWorld;
    use super::*;

    #[test]
    fn policy_evaluation_line_world() {
        const nb_states: usize = line_world::NUM_STATES;
        const nb_action: usize = line_world::NUM_ACTIONS;
        const nb_rewards: usize = line_world::NUM_REWARDS;
        println!("start");

        let lw = LineWorld::default();
        println!("stat ID :{:?}", lw.state_id());

        let pi_right: Array2<usize> = Array2::from_shape_vec(
            (LineWorld::num_states(), 2),
            vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        )
            .unwrap();
        let pi_left: Array2<usize> = Array2::from_shape_vec(
            (LineWorld::num_states(), 2),
            vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        )
            .unwrap();


        let policy = policy_evaluation::<nb_states, nb_action, nb_rewards, LineWorld>
            (pi_left, 0.999, 0.00001);

        println!("{:?}", policy);
        assert_eq!(1, 1)
    }
}
