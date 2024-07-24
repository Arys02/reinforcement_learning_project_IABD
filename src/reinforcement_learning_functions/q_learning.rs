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

    log : (bool, &Vec<bool>),
) -> (HashMap<usize, usize>, HashMap<usize, HashMap<(usize, usize), f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut Q: HashMap<usize, HashMap<(usize, usize), f32>> = HashMap::new();

    for _ in 0..nb_iter {
        env.reset();
        let mut step = 0;

        while !env.is_terminal() && step < nb_step {
            let state = env.state_id();
            let available_action = env.available_action();

            if !Q.contains_key(&state) {
                let mut q_s = HashMap::new();

                for a_i in 0..available_action.len(){
                    q_s.insert((a_i, available_action[a_i]), rng.gen::<f32>());
                }

                Q.insert(state, q_s);
            }

            let mut action : Option<usize> = None;
            let mut action_i : Option<usize> = None;

            if rng.gen::<f32>() < epsilon {
                //insert dans action l'action
                let _ = action.insert(*env.available_action().iter().choose(&mut rng).unwrap());
            } else {
                let q_s: Vec<f32> = available_action.iter().enumerate().map(|(a_i, &a)| *Q.get(&state).unwrap().get(&(a_i, a)).unwrap()).collect();

                let best_a_index = q_s.iter().enumerate().max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap();
                let _ = action.insert(available_action[best_a_index]);
            }

            let _ = action_i.insert(available_action.iter().position(|&x| x == action.unwrap()).unwrap());

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
                    for action_i in 0..available_action_p.len() {
                        tmp_q_s_p.insert((action_i, available_action_p[action_i]), rng.gen::<f32>());
                    }
                    Q.insert(state_p, tmp_q_s_p);
                }

                // équivalent de for comprehension en rust

                let q_s_p = available_action_p.iter().enumerate().map(|(a_i_p, &a_p)| *Q.get(&state_p).unwrap().get(&(a_i_p, a_p)).unwrap()).collect();
                let array_q_s_p = Array::from_vec(q_s_p);
                //TODO check if it's working like that
                let max = array_q_s_p.max().unwrap();
                let max_i = array_q_s_p.iter().position(|&x| x == *max);
                target = r + gamma * max
            }
            let q_s_a = *Q.get(&state).unwrap().get(&(action_i.unwrap(), action.unwrap())).unwrap();
            let new_value = (1. - alpha) * q_s_a + alpha * target;

            let _ = Q.get_mut(&state).unwrap().insert((action_i.unwrap(), action.unwrap()), new_value);
        }
    }

    let mut pi = HashMap::new();

    for (&s, actions) in &Q {
        let mut best_a: Option<usize> = None;
        let mut best_a_score = f32::MIN;

        for (&(action_i, action), &a_score) in actions {
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
    use crate::environement::secret_env_0::SecretEnv0;
    use crate::environement::secret_env_1::SecretEnv1;
    use crate::environement::two_round_rps::TwoRoundRPS;
    use crate::reinforcement_learning_functions::sarsa::sarsa;

    use super::*;

    #[test]
    fn q_learning_policy_lineworld() {
        let mut lw = LineWorld::new();

        println!("stat ID :{:?}", lw.state_id());

        let policy = q_learning(&mut lw, 0.1, 0.1, 0.999, 10, 1000, 42, (false, &Vec::new()));
        println!("{:?}", policy)
    }

    #[test]
    fn q_learning_grid_world() {
        println!("gridworld : ");
        let mut gw = GridWorld::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = q_learning(&mut gw, 0.1, 0.1, 0.999, 10, 1000, 42,  (false, &Vec::new()));

        let policy_to_play = policy.0;
        gw.play_strategy(policy_to_play, false);

        //println!("{:?}", policy);
        assert_eq!(1, 1)
    }
    #[test]
    fn q_learning_policy_two_round_rps() {
        let mut env = TwoRoundRPS::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = q_learning(&mut env, 0.1, 0.1, 0.999, 1000, 1000,  42, (false, &Vec::new()));
        println!("{:?}", policy);
        env.reset();

        env.play_strategy(policy.0, false);

        assert_eq!(env.is_terminal() && env.score() == 1.0, true)
    }
    #[test]
    fn policy_iteration_env_0() {
        println!("start");
        let mut env = SecretEnv0::new();
        let policy = q_learning(&mut env, 0.1, 0.1, 0.999, 1000, 1000,  42, (false, &Vec::new()));
        println!("{:?}", policy);
        env.reset();

        env.play_strategy(policy.0, false);
        let score = env.score();
        let state = env.state_id();
        let best_score = SecretEnv0::get_reward(SecretEnv0::num_rewards() - 1);
        let is_terminal = env.is_terminal();

        let mut vec_score = Vec::new();
        for i in 0..SecretEnv0::num_rewards() {
            vec_score.push(SecretEnv0::get_reward(i))
        }

        assert_eq!(env.is_terminal() && env.score() > 0., true);
    }

    #[test]
    fn policy_iteration_env_1() {
        println!("start");
        let mut env = SecretEnv1::new();
        let policy = q_learning(&mut env, 0.1, 0.1, 0.999, 1000, 1000,  42, (false, &Vec::new()));
        println!("{:?}", policy);
        env.reset();

        env.play_strategy(policy.0, false);
        let score = env.score();
        let state = env.state_id();
        let best_score = SecretEnv0::get_reward(SecretEnv0::num_rewards() - 1);
        let is_terminal = env.is_terminal();

        let mut vec_score = Vec::new();
        for i in 0..SecretEnv0::num_rewards() {
            vec_score.push(SecretEnv0::get_reward(i))
        }

        assert_eq!(env.is_terminal() && env.score() > 15., true);
    }

}
