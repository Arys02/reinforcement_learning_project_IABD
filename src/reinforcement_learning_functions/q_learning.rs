use crate::environement::environment_traits::Environment;

use crate::environement::environment_traits::BaseEnv;
use crate::environement::environment_traits::ActionEnv;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_stats::QuantileExt;
use rand::prelude::{IteratorRandom, StdRng};
use rand::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

/// Executes the Q-Learning algorithm for a given environment.
///
/// This algorithm uses Q-Learning to find the optimal policy for the environment by iteratively
/// updating the action-value function `Q` based on episodes generated from interaction with the environment.
/// Q-Learning is an off-policy algorithm that updates the value of the current state-action pair using the
/// estimated optimal future value.
///
/// # Parameters
///
/// - `env`: A mutable reference to an environment that implements the `Environment` trait.
/// - `alpha`: The learning rate.
/// - `epsilon`: The exploration rate for the epsilon-greedy policy.
/// - `gamma`: The discount factor for future rewards.
/// - `nb_iter`: The number of iterations (episodes) to run the algorithm.
/// - `nb_step`: The maximum number of steps per episode.
/// - `seed`: The seed for the random number generator to ensure reproducibility.
/// - `log`: A tuple containing:
///     - `bool`: A flag to determine whether to log data.
///     - `&mut Vec<bool>`: A mutable reference to a vector to log whether the episode terminated.
///
/// # Returns
///
/// - `(HashMap<usize, usize>, HashMap<usize, HashMap<(usize, usize), f32>>)`:
///   - A HashMap representing the improved policy mapping states to actions.
///   - A HashMap representing the action-value function `Q`.
///
/// # Details
///
/// The `q_learning` function follows these steps:
///
/// 1. Initialize the random number generator with the provided seed.
/// 2. Initialize the action-value function `Q`.
/// 3. For each episode:
///     - Reset the environment.
///     - Initialize the step counter.
///     - Loop until the environment reaches a terminal state or the maximum number of steps is reached:
///         - Select an action using the epsilon-greedy policy.
///         - Execute the action and observe the reward and next state.
///         - Update the action-value function `Q` using the Bellman equation.
///     - Optionally log whether the episode terminated.
///
/// The Q-Learning algorithm updates the policy `pi` based on the learned action-value function `Q`
/// by selecting the action with the highest value for each state.

pub fn q_learning<
    const NUM_STATES: usize,
    const NUM_ACTIONS: usize,
    const NUM_REWARDS: usize,
    Env: Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS> + Debug
>(
    alpha: f32,
    epsilon: f32,
    gamma: f32,
    nb_iter: usize,
    nb_step: usize,
    seed: u64,
) -> (
    HashMap<usize, usize>,
    HashMap<usize, HashMap<(usize, usize), f32>>,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut Q: HashMap<usize, HashMap<(usize, usize), f32>> = HashMap::new();

    for _ in 0..nb_iter {
        let mut env = Env::default();
        let mut step = 0;

        while !env.is_terminal() && step < nb_step {
            step += 1;
            let state = env.state_id();
            let available_action = env.available_actions_ids().collect::<Vec<usize>>();

            if !Q.contains_key(&state) {
                let mut q_s = HashMap::new();

                for a_i in 0..available_action.len() {
                    q_s.insert((a_i, available_action[a_i]), rng.gen::<f32>());
                }

                Q.insert(state, q_s);
            }

            let mut action: Option<usize> = None;
            let mut action_i: Option<usize> = None;

            if rng.gen::<f32>() < epsilon {
                //insert dans action l'action
                let _ = action.insert(env.available_actions_ids().choose(&mut rng).unwrap());
            } else {
                let q_s: Vec<f32> = available_action
                    .iter()
                    .enumerate()
                    .map(|(a_i, &a)| *Q.get(&state).unwrap().get(&(a_i, a)).unwrap())
                    .collect();

                let best_a_index = q_s
                    .iter()
                    .enumerate()
                    .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap();
                let _ = action.insert(available_action[best_a_index]);
            }

            let _ = action_i.insert(
                available_action
                    .iter()
                    .position(|&x| x == action.unwrap())
                    .unwrap(),
            );

            let prev_score: f32 = env.score();
            env.step(action.unwrap());
            let r = env.score() - prev_score;

            let state_p = env.state_id();
            let available_action_p = env.available_actions_ids().collect::<Vec<usize>>();

            let target: f32;

            if env.is_terminal() {
                target = alpha * r;
            } else {
                if !Q.contains_key(&state_p) {
                    let mut tmp_q_s_p = HashMap::new();

                    //TODO check if in available works well
                    for action_i in 0..available_action_p.len() {
                        tmp_q_s_p
                            .insert((action_i, available_action_p[action_i]), rng.gen::<f32>());
                    }
                    Q.insert(state_p, tmp_q_s_p);
                }

                // équivalent de for comprehension en rust

                let q_s_p = available_action_p
                    .iter()
                    .enumerate()
                    .map(|(a_i_p, &a_p)| *Q.get(&state_p).unwrap().get(&(a_i_p, a_p)).unwrap())
                    .collect();
                let array_q_s_p = Array::from_vec(q_s_p);
                //TODO check if it's working like that
                let max = array_q_s_p.max().unwrap();
                target = r + gamma * max
            }
            let q_s_a = *Q
                .get(&state)
                .unwrap()
                .get(&(action_i.unwrap(), action.unwrap()))
                .unwrap();
            let new_value = (1. - alpha) * q_s_a + alpha * target;

            let _ = Q
                .get_mut(&state)
                .unwrap()
                .insert((action_i.unwrap(), action.unwrap()), new_value);
        }
    }

    let mut pi = HashMap::new();

    for (&s, actions) in &Q {
        let mut best_a: Option<usize> = None;
        let mut best_a_score = f32::MIN;

        for (&(_, action), &a_score) in actions {
            if best_a.is_none() || best_a_score <= a_score {
                best_a = Some(action);
                best_a_score = a_score;
            }
        }
        pi.insert(s, best_a.unwrap());
    }
    println!("{:?}", pi);
    (pi, Q)
}

#[cfg(test)]
mod tests {
    use crate::environement::grid_world::grid_world;
    use crate::environement::grid_world::grid_world::GridWorld;
    use crate::environement::line_world::{line_world};
    use crate::environement::line_world::line_world::LineWorld;
    use crate::environement::two_round_rps::two_round_rps;
    use crate::environement::two_round_rps::two_round_rps::TwoRoundRPS;
    use super::*;


    #[test]
    fn q_learning_policy_lineworld() {

        const nb_states: usize = line_world::NUM_STATES;
        const nb_action: usize = line_world::NUM_ACTIONS;
        const nb_rewards: usize = line_world::NUM_REWARDS;


        let policy = q_learning::<nb_states, nb_action, nb_rewards, LineWorld>
            (0.1, 0.1, 0.999, 10, 1000, 42);
        println!("{:?}", policy)
    }

    #[test]
    fn q_learning_grid_world() {
        println!("gridworld : ");
        let mut gw = GridWorld::default();

        println!("stat ID :{:?}", gw.state_id());
        const nb_states: usize = grid_world::NUM_STATES;
        const nb_action: usize = grid_world::NUM_ACTIONS;
        const nb_rewards: usize = grid_world::NUM_REWARDS;

        let policy = q_learning::<nb_states, nb_action, nb_rewards, GridWorld>
            (0.1, 0.1, 0.999, 10, 1000, 42);
        println!("{:?}", policy);


        let policy_to_play = policy.0;
        gw.play_strategy(policy_to_play, false);

        //println!("{:?}", policy);
        assert_eq!(1, 1)
    }
    #[test]
    fn q_learning_policy_two_round_rps() {
        let mut env = TwoRoundRPS::default();

        println!("stat ID :{:?}", env.state_id());
        const nb_states: usize = two_round_rps::NUM_STATES;
        const nb_action: usize = two_round_rps::NUM_ACTIONS;
        const nb_rewards: usize = two_round_rps::NUM_REWARDS;


        let policy = q_learning::<nb_states, nb_action, nb_rewards, TwoRoundRPS>
            (0.1, 0.1, 0.999, 10, 1000, 42);
        println!("{:?}", policy);

        env.reset();

        env.play_strategy(policy.0, false);

        assert_eq!(env.is_terminal() && env.score() == 1.0, true)
    }
    /*
    #[test]
    fn policy_iteration_env_0() {
        println!("start");
        let mut env = SecretEnv0::default();
        const nb_states: usize = SecretEnv0::num_states();
        const nb_action: usize = SecretEnv0::num_actions();
        const nb_rewards: usize = SecretEnv0::num_rewards();
        let mut lw = SecretEnv0::default();

        println!("stat ID :{:?}", lw.state_id());

        let policy = q_learning::<nb_states, nb_action, nb_rewards, SecretEnv0>
            (0.1, 0.1, 0.999, 10, 1000, 42);

        println!("{:?}", policy);

        env.reset();

        env.play_strategy(policy.0, false);
        let score = env.score();
        let state = env.state_id();
        let best_score = SecretEnv0::get_reward(SecretEnv0::num_rewards() - 1);
        let is_terminal = env.is_terminal();

        let mut vec_score = Vec::default();
        for i in 0..SecretEnv0::num_rewards() {
            vec_score.push(SecretEnv0::get_reward(i))
        }

        assert_eq!(env.is_terminal() && env.score() > 0., true);
    }

    #[test]
    fn policy_iteration_env_1() {
        println!("start");
        let mut env = SecretEnv1::default();
        const nb_states: usize = SecretEnv1::num_states();
        const nb_action: usize = SecretEnv1::num_actions();
        const nb_rewards: usize = SecretEnv1::num_rewards();

        let policy = q_learning::<nb_states, nb_action, nb_rewards, SecretEnv1>
            (0.1, 0.1, 0.999, 10, 1000, 42);

        env.reset();

        env.play_strategy(policy.0, false);
        let score = env.score();
        let state = env.state_id();
        let best_score = SecretEnv0::get_reward(SecretEnv0::num_rewards() - 1);
        let is_terminal = env.is_terminal();

        let mut vec_score = Vec::default();
        for i in 0..SecretEnv0::num_rewards() {
            vec_score.push(SecretEnv0::get_reward(i))
        }

        assert_eq!(env.is_terminal() && env.score() > 15., true);
    }

     */
}
