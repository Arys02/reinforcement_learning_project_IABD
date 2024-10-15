use crate::environement::environment_traits::Environment;
use ndarray::Array;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

/// Executes the Policy Iteration algorithm for a given environment.
///
/// This algorithm uses policy iteration to find the optimal policy for the environment by iteratively evaluating
/// and improving the policy until convergence. The algorithm alternates between policy evaluation and policy improvement
/// steps.
///
/// # Parameters
///
/// - `gamma`: The discount factor for future rewards.
/// - `theta`: The threshold for determining convergence in the policy evaluation step.
/// - `log`: A tuple containing:
///     - `bool`: A flag to determine whether to log data.
///     - `&mut Vec<usize>`: A mutable reference to a vector to log the number of iterations per evaluation step.
///     - `&mut usize`: A mutable reference to store the total number of policy improvement iterations.
///
/// # Returns
///
/// - `Vec<usize>`: The optimal policy mapping states to actions.
///
/// # Details
///
/// The `policy_iteration` function follows these steps:
///
/// 1. Initialize the value function `V` and the policy `pi` randomly.
/// 2. Loop until the policy is stable:
///     - **Policy Evaluation**:
///         - Iteratively evaluate the current policy `pi` by updating the value function `V` until the maximum change `delta` is below the threshold `theta`.
///     - **Policy Improvement**:
///         - For each state, find the action that maximizes the expected return and update the policy `pi`.
///         - Check if the policy has changed. If not, the policy is stable, and the algorithm terminates.
/// 3. Optionally log the number of iterations for each evaluation step and the total number of policy improvement iterations.
///
/// The Policy Iteration algorithm ensures convergence to the optimal policy by repeatedly improving the policy based on the evaluated value function.

pub fn policy_iteration<
    const NUM_STATES: usize,
    const NUM_ACTIONS: usize,
    const NUM_REWARDS: usize,
    Env: Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS>>
(gamma: f32, theta: f32) -> Vec<usize> {
    let num_states = NUM_STATES;
    let num_actions = NUM_ACTIONS;
    let num_rewards = NUM_REWARDS;

    let mut V = Array::random((num_states, 1), Uniform::new(0.0, 1.0)).into_raw_vec();

    let mut pi = vec![];
    for _ in 0..num_states {
        pi.push(0)
    }
    println!("pi : {:?}", pi);

    loop {
        loop {
            let mut delta: f32 = 0.;
            for s in 0..num_states {
                let a = pi[s];
                let v = V[s];
                let mut total = 0.;
                for s_p in 0..num_states {
                    for r in 0..num_rewards {
                        let p = Env::build_transition_probability(s, a, s_p, r);
                        total += p * (Env::get_reward(r) + gamma * V[s_p])
                    }
                }
                V[s] = total;
                delta = delta.max((v - V[s]).abs())
            }
            if delta < theta {
                break;
            }
        }

        println!("V : m{:?}", V);

        let mut policy_stable = true;

        for s in 0..num_states {
            let old_action = pi[s];

            let mut best_a: Option<usize> = None;
            let mut best_action_score = f32::MIN;

            for a in 0..num_actions {
                let mut total = 0.;

                for s_p in 0..num_states {
                    for r_index in 0..num_rewards {
                        total += Env::build_transition_probability(s, a, s_p, r_index)
                            * (Env::get_reward(r_index) + gamma * V[s_p])
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
            return pi;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(warnings)]

    use crate::environement::monty_hall_1::{monty_hall};
    use crate::environement::secret_env_0::SecretEnv0;
    use crate::environement::secret_env_1::SecretEnv1;
    use crate::environement::secret_env_2::SecretEnv2;
    use crate::environement::secret_env_3::SecretEnv3;
    use crate::environement::two_round_rps::{two_round_rps};
    use std::collections::HashMap;
    use crate::environement::grid_world::grid_world;
    use crate::environement::grid_world::grid_world::GridWorld;
    use crate::environement::line_world::line_world;
    use crate::environement::line_world::line_world::LineWorld;
    use crate::environement::monty_hall_1::monty_hall::MontyHall1;
    use crate::environement::two_round_rps::two_round_rps::TwoRoundRPS;
    use super::*;


    #[test]
    fn policy_iteration_line_world() {
        #![allow(warnings)]
        println!("start");
        const nb_states: usize = line_world::NUM_STATES;
        const nb_action: usize = line_world::NUM_ACTIONS;
        const nb_rewards: usize = line_world::NUM_REWARDS;


        let v = policy_iteration::<nb_states, nb_action, nb_rewards, LineWorld>(0.999, 0.000001);

        println!("{:?}", v);
        assert_eq!(1, 1)
    }

    #[test]
    fn policy_iteration_grid_world() {
        println!("start");
        const nb_states: usize = grid_world::NUM_STATES;
        const nb_action: usize = grid_world::NUM_ACTIONS;
        const nb_rewards: usize = grid_world::NUM_REWARDS;



        let v = policy_iteration::<nb_states, nb_action, nb_rewards, GridWorld>(0.999, 0.000001);

        let mut env = GridWorld::default();

        let mut policy = HashMap::default();
        for i in 0..v.len() {
            policy.insert(i, v[i]);
        }
        env.play_strategy(policy.clone(), false);

        println!("{:?}", v);

        assert_eq!(env.state_id(), 4)
    }
    #[test]
    fn policy_iteration_two_round_pfs() {
        println!("start");
        let mut env = TwoRoundRPS::default();
        const nb_states: usize = two_round_rps::NUM_STATES;
        const nb_action: usize = two_round_rps::NUM_ACTIONS;
        const nb_rewards: usize = two_round_rps::NUM_REWARDS;

        let v = policy_iteration::<nb_states, nb_action, nb_rewards, TwoRoundRPS>(0.999, 0.000001);

        let mut policy = HashMap::default();
        for i in 0..v.len() {
            policy.insert(i, v[i]);
        }

        env.play_strategy(policy.clone(), false);

        println!("{:?}", v);
        println!("{:?}", policy);
        assert_eq!(env.score(), 1.);
    }
    #[test]
    fn policy_iteration_monty_hall_1() {
        println!("start");
        let mut env = MontyHall1::default();

        const nb_states: usize = monty_hall::NUM_STATES;
        const nb_action: usize = monty_hall::NUM_ACTIONS;
        const nb_rewards: usize = monty_hall::NUM_REWARDS;

        let v = policy_iteration::<nb_states, nb_action, nb_rewards, MontyHall1>(0.999, 0.000001);

        let mut policy = HashMap::default();
        for i in 0..v.len() {
            policy.insert(i, v[i]);
        }

        println!("{:?}", policy);
        let nb_run: usize = 1000;

        let mut win: f32 = 0.;

        for _ in 0..nb_run {
            env.reset();
            env.play_strategy(policy.clone(), false);
            println!("score : {}", env.score());
            win += env.score();
        }

        let stat_win = win / (nb_run as f32);

        println!("win stat :  {}", stat_win);

        assert_eq!(stat_win > 0.5, true)
    }
    /*
    #[test]
    fn policy_iteration_env_0() {
        println!("start");
        let mut env = SecretEnv0::default();
        const nb_states: usize = SecretEnv0::num_states();
        const nb_action: usize = SecretEnv0::num_actions();
        const nb_rewards: usize = SecretEnv0::num_rewards();

        let v = policy_iteration::<nb_states, nb_action, nb_rewards, SecretEnv0>(0.999, 0.000001);


        let mut policy = HashMap::default();
        for i in 0..v.len() {
            policy.insert(i, v[i]);
        }

        env.play_strategy(policy.clone(), false);

        println!("{:?}", v);
        assert_eq!(1, 1)
    }

     */
}
