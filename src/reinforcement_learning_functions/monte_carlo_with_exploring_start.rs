use crate::environement::environment_traits::Environment;

use crate::environement::environment_traits::BaseEnv;
use crate::environement::environment_traits::ActionEnv;
use ndarray_rand::rand::SeedableRng;
use rand::prelude::{IteratorRandom, StdRng};
use rand::Rng;
use std::collections::HashMap;
use std::fmt::Debug;

/// Executes the Monte Carlo with Exploring Starts algorithm for a given environment.
///
/// This algorithm uses an on-policy Monte Carlo control method with exploring starts
/// to improve the policy `pi` based on episodes generated with a random start. The algorithm
/// iteratively updates the action-value function `Q` and the policy `pi` using returns from the generated episodes.
///
/// # Parameters
///
/// - `env`: A mutable reference to an environment that implements the `Environment` trait.
/// - `gamma`: The discount factor for future rewards.
/// - `nb_iter`: The number of iterations (episodes) to run the algorithm.
/// - `max_steps`: The maximum number of steps per episode.
/// - `seed`: The seed for the random number generator to ensure reproducibility.
/// - `log`: A tuple containing:
///     - `bool`: A flag to determine whether to log data.
///     - `&mut Vec<f32>`: A mutable reference to a vector to log the return (G).
///     - `&mut Vec<usize>`: A mutable reference to a vector to log the trajectory size.
///     - `&mut Vec<bool>`: A mutable reference to a vector to log whether the episode terminated.
///
/// # Returns
///
/// - `HashMap<usize, usize>`: The improved policy mapping states to actions.
///
/// # Details
///
/// The `monte_carlo_with_exploring_start` function follows these steps:
///
/// 1. Initialize the random number generator with the provided seed.
/// 2. Initialize the policy `pi`, action-value function `Q`, and returns tracker.
/// 3. For each episode:
///     - Reset the environment to a random state using the provided seed.
///     - Generate an episode following a policy with exploring starts:
///         - For the first action, select a random action.
///         - For subsequent actions, follow the current policy `pi`.
///     - Track the trajectory of state-action-reward tuples.
///     - Calculate the return `G` and update `Q` and `pi` using the returns observed in the episode.
///     - Optionally log the return, trajectory size, and termination status.
///
/// Exploring starts ensure that the agent explores different actions at the beginning of each episode,
/// which helps in discovering better policies. The target policy `pi` is iteratively improved based on the observed returns.

pub fn monte_carlo_with_exploring_start<
    const NUM_STATES: usize,
    const NUM_ACTIONS: usize,
    const NUM_REWARDS: usize,
    Env: Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS> + Debug
>(
    gamma: f32,
    nb_iter: i32,
    max_steps: i32,
    mut seed: u64,
) -> HashMap<usize, usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pi = HashMap::new();
    let mut Q = HashMap::new();
    let mut returns = HashMap::new();

    for _ in 0..nb_iter {
        let mut env = Env::default();
        seed += 1;
        env.reset_random_state(seed);

        let mut trajectory = Vec::new();
        let mut step_count = 0;
        let mut is_first_action = true;

        while !env.is_terminal() && step_count < max_steps {
            let state = env.state_id();
            let available_action = env.available_actions_ids()
                .collect::<Vec<usize>>();

            if !pi.contains_key(&state) {
                pi.insert(state, *available_action.iter().choose(&mut rng).unwrap());
            }

            let a: usize;
            if is_first_action {
                //a = get_random_value(&available_action, &rng);
                a = *available_action.iter().choose(&mut rng).unwrap();
                is_first_action = false
            } else {
                a = pi.get(&state).unwrap().to_owned()
            }

            let prev_score = env.score();
            env.step(a);
            let r = env.score() - prev_score;
            trajectory.push((state, a, r, available_action));
            step_count += 1;
        }

        let mut g = 0.;

        for (t, &(s, a, r, ref aa)) in trajectory.iter().enumerate().rev() {
            g = gamma * g + r;
            //println!("{}", g);

            if trajectory[..t]
                .iter()
                .all(|&(si, ai, _, _)| si != s || ai != a)
            {
                returns.entry((s, a)).or_insert(Vec::new()).push(g);
                let average_return =
                    returns[&(s, a)].iter().sum::<f32>() / returns[&(s, a)].len() as f32;
                Q.insert((s, a), average_return);

                let best_action = aa
                    .iter()
                    .max_by(|&&action1, &&action2| {
                        let q_val1 = Q.get(&(s, action1)).copied().unwrap_or_else(|| rng.gen());
                        let q_val2 = Q.get(&(s, action2)).copied().unwrap_or_else(|| rng.gen());

                        q_val1
                            .partial_cmp(&q_val2)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied()
                    .unwrap();

                pi.insert(s, best_action);
            }
        }
    }
    pi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environement::grid_world::grid_world;
    use crate::environement::grid_world::grid_world::GridWorld;
    use crate::environement::line_world::line_world::LineWorld;
    use crate::environement::line_world::line_world;
    use crate::environement::monty_hall_1::monty_hall;
    use crate::environement::monty_hall_1::monty_hall::MontyHall1;
    use crate::environement::two_round_rps::two_round_rps;
    use crate::environement::two_round_rps::two_round_rps::TwoRoundRPS;

    #[test]
    fn monte_carlo_with_exploring_start_line_world() {
        const nb_states: usize = line_world::NUM_STATES;
        const nb_action: usize = line_world::NUM_ACTIONS;
        const nb_rewards: usize = line_world::NUM_REWARDS;

        let lw = LineWorld::default();

        println!("stat ID :{:?}", lw.state_id());

        let policy = monte_carlo_with_exploring_start::
        <nb_states, nb_action, nb_rewards, LineWorld>(0.999, 100, 10, 42);
        let mut gw2 = LineWorld::default();
        gw2.play_strategy(policy.clone(), false);
        assert_eq!(gw2.state_id(), 40)
    }

    #[test]
    fn monte_carlo_with_exploring_start_grid_world() {
        println!("gridworld : ");
        const nb_states: usize = grid_world::NUM_STATES;
        const nb_action: usize = grid_world::NUM_ACTIONS;
        const nb_rewards: usize = grid_world::NUM_REWARDS;
        let gw = GridWorld::default();

        println!("stat ID :{:?}", gw.state_id());
        let policy = monte_carlo_with_exploring_start::<nb_states, nb_action, nb_rewards, GridWorld>
            (0.999, 100, 10, 42);


        println!("{:?}", policy);

        let mut gw2 = GridWorld::default();
        gw2.play_strategy(policy.clone(), false);
        assert_eq!(gw2.state_id(), 40)
    }

    #[test]
    fn monte_carlo_with_exploring_start_two_round_rps() {
        println!("Rock paper scissors : ");
        let mut env = TwoRoundRPS::default();

        println!("stat ID :{:?}", env.state_id());
        const nb_states: usize = two_round_rps::NUM_STATES;
        const nb_action: usize = two_round_rps::NUM_ACTIONS;
        const nb_rewards: usize = two_round_rps::NUM_REWARDS;

        let policy = monte_carlo_with_exploring_start::<nb_states, nb_action, nb_rewards, TwoRoundRPS>
            (0.999, 100, 10, 42);


        println!("{:?}", policy);

        for _ in 0..5 {
            env.reset();
            env.play_strategy(policy.clone(), false);
            println!("score : {}", env.score())
        }

        assert_eq!(env.score(), 1.)
    }

    #[test]
    fn monte_carlo_with_exploring_start_monty_hall_1() {
        println!("Monty Hall 1: ");
        let mut env = MontyHall1::default();

        println!("stat ID :{:?}", env.state_id());
        const nb_states: usize = monty_hall::NUM_STATES;
        const nb_action: usize = monty_hall::NUM_ACTIONS;
        const nb_rewards: usize = monty_hall::NUM_REWARDS;

        let policy = monte_carlo_with_exploring_start::<nb_states, nb_action, nb_rewards, MontyHall1>
            (0.999, 100, 10, 42);



        println!("{:?}", policy);
        let nb_run: usize = 1000;

        let mut win: f32 = 0.;

        for _ in 0..nb_run {
            env.reset();
            env.play_strategy(policy.clone(), false);
            win += env.score();
        }

        let stat_win = win / (nb_run as f32);

        println!("win stat :  {}", stat_win);

        assert_eq!(stat_win > 0.5, true)
    }
}
