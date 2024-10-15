use crate::environement::environment_traits::Environment;
use ndarray_rand::rand::SeedableRng;
use rand::prelude::StdRng;
use rand::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

/// Executes the SARSA (State-Action-Reward-State-Action) algorithm for a given environment.
///
/// This algorithm uses SARSA to find the optimal policy for the environment by iteratively
/// updating the action-value function `Q` based on episodes generated from interaction with the environment.
/// SARSA is an on-policy algorithm that updates the value of the current state-action pair using the next state-action pair.
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
/// - `(HashMap<usize, usize>, HashMap<(usize, usize), (f32, usize)>)`:
///   - A HashMap representing the improved policy mapping states to actions.
///   - A HashMap representing the action-value function `Q`.
///
/// # Details
///
/// The `sarsa` function follows these steps:
///
/// 1. Initialize the random number generator with the provided seed.
/// 2. Initialize the action-value function `Q`.
/// 3. For each episode:
///     - Reset the environment.
///     - Initialize the step counter.
///     - Loop until the environment reaches a terminal state or the maximum number of steps is reached:
///         - **Action Selection**: Select an action using the epsilon-greedy policy.
///         - **Action Execution**: Execute the action and observe the reward and next state.
///         - **Next Action Selection**: Select the next action using the epsilon-greedy policy.
///         - **Q-Value Update**: Update the action-value function `Q` using the Bellman equation.
///     - Optionally log whether the episode terminated.
///
/// The SARSA algorithm updates the policy `pi` based on the learned action-value function `Q`
/// by selecting the action with the highest value for each state.
pub fn sarsa<
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
) -> (HashMap<usize, usize>, HashMap<(usize, usize), (f32, usize)>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut Q = HashMap::new();
    let mut pi = HashMap::new();

    for _ in 0..nb_iter {
        let mut env = Env::default();
        let mut step = 0;

        while !env.is_terminal() && step < nb_step {
            step += 1;
            let aa = env.available_actions_ids();
            let state = env.state_id();

            if !Q.contains_key(&(state, aa[0])) {
                for a in 0..aa.len() {
                    Q.insert((state, a), (rng.gen::<f32>(), aa[a]));
                }
            }

            let mut action_i: Option<usize> = None;
            if rng.gen::<f32>() < epsilon {
                //insert l'action dans Action (et pas l'index)
                let mut rng = rand::thread_rng();
                let random_i: usize = rng.gen_range(0..aa.len());
                let _ = action_i.insert(random_i);
            } else {
                let q_s: Vec<(f32, usize)> = aa
                    .iter()
                    .enumerate()
                    .map(|(i, _)| *Q.get(&(state, i)).unwrap())
                    .collect();

                let best_a_index = q_s
                    .iter()
                    .enumerate()
                    .max_by(|(_, (a, _)), (_, (b, _))| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap();
                let _ = action_i.insert(best_a_index);
            }

            let action = *aa.get(action_i.unwrap()).unwrap();

            let prev_score: f32 = env.score();
            env.step(action);
            let r = env.score() - prev_score;

            let state_p = env.state_id();
            let available_action_p = env.available_actions_ids();

            let target: f32;

            if env.is_terminal() {
                target = alpha * r
            } else {
                if !Q.contains_key(&(state_p, available_action_p[0])) {
                    for a in 0..available_action_p.len() {
                        Q.insert((state_p, a), (rng.gen::<f32>(), available_action_p[a]));
                    }
                }

                let mut action_i_p: Option<usize> = None;
                if rng.gen::<f32>() < epsilon {
                    //insert l'action dans Action (et pas l'index)
                    let mut rng = rand::thread_rng();
                    let random_i: usize = rng.gen_range(0..available_action_p.len());
                    let _ = action_i_p.insert(random_i);
                } else {
                    let q_s: Vec<(f32, usize)> = available_action_p
                        .iter()
                        .enumerate()
                        .map(|(i, _)| *Q.get(&(state_p, i)).unwrap())
                        .collect();

                    let best_a_index = q_s
                        .iter()
                        .enumerate()
                        .max_by(|&(_, (a, _)), &(_, (b, _))| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap();
                    let _ = action_i_p.insert(best_a_index);
                }

                let q_sp_ap = if Q.contains_key(&(state_p, action_i_p.unwrap())) {
                    *Q.get(&(state_p, action_i_p.unwrap())).unwrap()
                } else {
                    (0., 0)
                }
                    .0;
                let q_s_a = Q.get(&(state, action_i.unwrap())).unwrap().0;

                target = (1. - alpha) * q_s_a + alpha * (gamma * q_sp_ap + r);
            }

            Q.insert((state, action_i.unwrap()), (target, action));
        }
    }
    for s in 0..Env::num_states() {
        let mut best_a: Option<usize> = None;
        let mut best_a_score = f32::MIN;
        for action in 0..Env::num_actions() {
            if Q.contains_key(&(s, action)) {
                let a_score = Q[&(s, action)];
                if best_a.is_none() || best_a_score <= a_score.0 {
                    best_a = Some(action);
                    best_a_score = a_score.0;
                }
            }
        }
        if !best_a.is_none() {
            pi.insert(s, Q[&(s, best_a.unwrap())].1);
        }
    }

    return (pi, Q);
}

#[cfg(test)]
mod tests {
    use crate::environement::grid_world::grid_world;
    use crate::environement::grid_world::grid_world::GridWorld;
    use crate::environement::line_world::line_world;

    use crate::environement::line_world::line_world::LineWorld;
    use crate::environement::monty_hall_1::monty_hall;
    use crate::environement::monty_hall_1::monty_hall::MontyHall1;
    use crate::environement::two_round_rps::two_round_rps;
    use crate::environement::two_round_rps::two_round_rps::TwoRoundRPS;
    use super::*;

    #[test]
    fn sarsa_policy_lineworld() {
        let mut lw = LineWorld::default();

        println!("stat ID :{:?}", lw.state_id());
        const nb_states: usize = line_world::NUM_STATES;
        const nb_action: usize = line_world::NUM_ACTIONS;
        const nb_rewards: usize = line_world::NUM_REWARDS;

        let policy = sarsa::<nb_states, nb_action, nb_rewards, LineWorld>
            (0.1, 0.1, 0.999, 1000, 1000, 42);
        println!("{:?}", policy);
        lw.play_strategy(policy.0, false);
        assert_eq!(lw.is_terminal() && lw.score() == 1.0, true);
    }

    #[test]
    fn sarsa_policy_gridworld() {
        const nb_states: usize = grid_world::NUM_STATES;
        const nb_action: usize = grid_world::NUM_ACTIONS;
        const nb_rewards: usize = grid_world::NUM_REWARDS;

        let mut env = GridWorld::default();
        println!("stat ID :{:?}", env.state_id());


        let policy = sarsa::<nb_states, nb_action, nb_rewards, GridWorld>
            (0.1, 0.1, 0.999, 1000, 1000, 42);

        println!("{:?}", policy);
        env.reset();

        env.play_strategy(policy.0, false);

        assert_eq!(env.is_terminal() && env.score() == 3.0, true)
    }

    #[test]
    fn sarsa_policy_two_round_rps() {
        let mut env = TwoRoundRPS::default();

        println!("stat ID :{:?}", env.state_id());

        const nb_states: usize = two_round_rps::NUM_STATES;
        const nb_action: usize = two_round_rps::NUM_ACTIONS;
        const nb_rewards: usize = two_round_rps::NUM_REWARDS;

        let policy = sarsa::<nb_states, nb_action, nb_rewards, TwoRoundRPS>
            (0.1, 0.1, 0.999, 1000, 1000, 42);

        env.reset();

        env.play_strategy(policy.0, false);

        assert_eq!(env.is_terminal() && env.score() == 1.0, true)
    }

    #[test]
    fn sarsa_monty_hall_1() {
        println!("Monty Hall 1: ");
        let mut env = MontyHall1::default();
        const nb_states: usize = monty_hall::NUM_STATES;
        const nb_action: usize = monty_hall::NUM_ACTIONS;
        const nb_rewards: usize = monty_hall::NUM_REWARDS;


        let policy = sarsa::<nb_states, nb_action, nb_rewards, MontyHall1>
            (0.1, 0.1, 0.999, 1000, 1000, 42);


        println!("{:?}", policy);
        let nb_run: usize = 1000;

        let mut win: f32 = 0.;

        for _ in 0..nb_run {
            env.reset();
            env.play_strategy(policy.0.clone(), false);
            win += env.score();
        }

        let stat_win = win / (nb_run as f32);

        println!("win stat :  {}", stat_win);

        assert_eq!(stat_win > 0.6, true)
    }
}
