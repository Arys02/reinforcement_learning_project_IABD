use std::collections::HashMap;

use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use rand::prelude::{IteratorRandom, StdRng};
use rand::Rng;
use ndarray_stats::QuantileExt;

use crate::environement::environment::Environment;

pub fn q_learning<E: Environment>(
    env: &mut E,
    alpha: f32,
    epsilon: f32,
    gamma: f32,
    nb_iter: usize,
    nb_step: usize,
    seed: u64,
) -> (HashMap<usize, usize>, HashMap<usize, HashMap<usize, f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut Q: HashMap<usize, HashMap<usize, f32>> = HashMap::new();

    for _ in 0..nb_iter {
        env.reset();
        let mut step = 0;

        while !env.is_terminal() && step < nb_step {
            let state = env.state_id();
            let available_action = env.available_action();

            if !Q.contains_key(&state) {
                let mut q_s = HashMap::new();

                for a in &available_action {
                    q_s.insert(*a, rng.gen::<f32>());
                }

                Q.insert(state, q_s);
            }

            let mut action : Option<usize> = None;

            if rng.gen::<f32>() < epsilon {
                let _ = action.insert(*E::available_actions(state).iter().choose(&mut rng).unwrap());
            } else {
                let q_s: Vec<f32> = available_action.iter().map(|&a| *Q.get(&state).unwrap().get(&a).unwrap()).collect();

                let best_a_index = q_s.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap();
                let _ = action.insert(best_a_index);
            }

            let prev_score: f32 = env.score();
            env.step(action.unwrap());
            let r = env.score() - prev_score;

            let state_p = env.state_id();
            let available_action_p = env.available_action();

            let mut target: f32;

            if env.is_terminal() {
                target = alpha * r;
            } else {
                if !Q.contains_key(&state_p) {
                    let mut tmp_q_s_p = HashMap::new();

                    //TODO check if in available works well
                    for action_p in &available_action_p {
                        tmp_q_s_p.insert(*action_p, rng.gen::<f32>());
                    }
                    Q.insert(state_p, tmp_q_s_p);
                }

                // équivalent de for comprehension en rust

                let q_s_p = available_action_p.iter().map(|&a_p| *Q.get(&state_p).unwrap().get(&a_p).unwrap()).collect();
                let array_q_s_p = Array::from_vec(q_s_p);
                //TODO check if it's working like that
                let max = array_q_s_p.max().unwrap();
                target = r + gamma * max;
            }
            let q_s_a = *Q.get(&state).unwrap().get(&action.unwrap()).unwrap();
            let new_value = (1. - alpha) * q_s_a + alpha * target;

            let _ = Q.get_mut(&state).unwrap().insert(action.unwrap(), new_value);
        }
    }

    let mut pi = HashMap::new();

    for (&s, actions) in &Q {
        let mut best_a: Option<usize> = None;
        let mut best_a_score = f32::MIN;

        for (&action, &a_score) in actions {
            if best_a.is_none() || best_a_score <= a_score {
                best_a = Some(action);
                best_a_score = a_score;
            }
        }
        pi.insert(s, best_a.unwrap());
    }
    println!("{:?}", pi);
    return (pi, Q);
}

#[cfg(test)]
mod tests {
    use crate::environement::grid_world::GridWorld;
    use crate::environement::line_world::LineWorld;
    use crate::environement::two_round_rps::TwoRoundRPS;
    use crate::reinforcement_learning_functions::sarsa::sarsa;

    use super::*;

    #[test]
    fn q_learning_policy_lineworld() {
        let mut lw = LineWorld::new();

        println!("stat ID :{:?}", lw.state_id());

        let policy = q_learning(&mut lw, 0.1, 0.1, 0.999, 10, 1000, 42);
        println!("{:?}", policy)
    }

    #[test]
    fn q_learning_grid_world() {
        println!("gridworld : ");
        let mut gw = GridWorld::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = q_learning(&mut gw, 0.1, 0.1, 0.999, 10, 1000, 42);

        let policy_to_play = policy.0;
        gw.play_strategy(policy_to_play);

        //println!("{:?}", policy);
        assert_eq!(1, 1)
    }
    #[test]
    fn q_learning_policy_two_round_rps() {
        let mut env = TwoRoundRPS::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = q_learning(&mut env, 0.1, 0.1, 0.999, 1000, 1000,  42);
        println!("{:?}", policy);
        env.reset();

        env.play_strategy(policy.0);

        assert_eq!(env.is_terminal() && env.score() == 1.0, true)
    }
}
