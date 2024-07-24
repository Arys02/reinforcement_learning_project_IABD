use std::collections::HashMap;
use std::error::Error;
use ndarray_rand::rand::SeedableRng;
use rand::prelude::{IteratorRandom, StdRng};
use rand::Rng;
use serde::Serialize;
use crate::environement::environment::Environment;


fn write_csv(path: &str,
             name: &str,
             g_vec: Vec<f32>,
             size_trajectoire_vec: Vec<usize>,
             is_terminal_vec: Vec<bool>) -> Result<(), Box<dyn Error>>
{
    #[derive(Serialize)]
    struct Record<'a> {
        key: &'a str,
        t1: f32,
        t2: usize,
        t4: bool,
    }
    ;
    let mut wtr = csv::Writer::from_path(path).unwrap();

    let len = g_vec.len();

    for i in 0..len {
        let record = Record {
            key: name,
            t1: g_vec[i],
            t2: size_trajectoire_vec[i],
            t4: is_terminal_vec[i],
        };

        wtr.serialize(&record)?;
    }

    wtr.flush().unwrap();
    Ok(())
}

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

pub fn monte_carlo_with_exploring_start<E: Environment>(
    mut env: &mut E,
    gamma: f32,
    nb_iter: i32,
    max_steps: i32,
    mut seed: u64,
    log : (bool, &mut Vec<f32>, &mut Vec<usize>, &mut Vec<bool>),
) -> HashMap<usize, usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pi = HashMap::new();
    let mut Q = HashMap::new();
    let mut returns = HashMap::new();

    for _ in 0..nb_iter {
        seed += 1;
        env.reset_random_state(seed);

        let mut trajectory = Vec::new();
        let mut step_count = 0;
        let mut is_first_action = true;

        while !env.is_terminal() && step_count < max_steps {
            let state = env.state_id();
            let available_action = env.available_action();
            let available_action_vec = available_action.to_vec();
            //println!("state : {:?}", state);
            //println!("available action : {:?}", available_action);

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

            if trajectory[..t].iter().all(|&(si, ai, _, _)| si != s || ai != a) {
                returns.entry((s, a)).or_insert(Vec::new()).push(g);
                let average_return = returns[&(s, a)].iter().sum::<f32>() / returns[&(s, a)].len() as f32;
                Q.insert((s, a), average_return);

                let best_action = aa.iter().max_by(|&&action1, &&action2| {
                    let q_val1 = Q.get(&(s, action1)).copied().unwrap_or_else(|| rng.gen());
                    let q_val2 = Q.get(&(s, action2)).copied().unwrap_or_else(|| rng.gen());

                    q_val1.partial_cmp(&q_val2).unwrap_or(std::cmp::Ordering::Equal)
                }).copied().unwrap();

                pi.insert(s, best_action);
            }
        }
        if log.0 {
            &log.1.push(g);
            log.2.push(trajectory.len());
            log.3.push(env.is_terminal())
        }
    }
    return pi;
}

#[cfg(test)]
mod tests {
    use crate::environement::grid_world::GridWorld;
    use crate::environement::line_world::LineWorld;
    use crate::environement::monty_hall_1::MontyHall1;
    use crate::environement::secret_env_0::SecretEnv0;
    use crate::environement::secret_env_1::SecretEnv1;
    use crate::environement::secret_env_2::SecretEnv2;
    use crate::environement::secret_env_3::SecretEnv3;
    use crate::environement::two_round_rps::TwoRoundRPS;
    use crate::reinforcement_learning_functions::monte_carlo_on_policy::monte_carlo_on_policy;
    use super::*;

    fn test_env_policy<E: Environment>(mut env: &mut E, label: &str) -> u64 {
        let mut env_test = E::new();

        let mut g = Vec::new();
        let mut t_size = Vec::new();
        let mut is_terminal = Vec::new();

        use std::time::Instant;
        let now = Instant::now();
        let policy_map = monte_carlo_with_exploring_start(env, 0.999, 1000, 1000,  42, (true, &mut g, &mut t_size, &mut is_terminal));
        let elapsed = now.elapsed();

        let path = format!("record/monte_carlo_es_{}.csv", label);
        write_csv(path.as_str(),
                  label,
                  g,
                  t_size,
                  is_terminal).expect("TODO: panic message");

        env_test.play_strategy(policy_map.clone(), false);
        return elapsed.as_millis() as u64;
    }

    #[test]
    fn monte_carlo_es_all_env() {
        let mut lineworld = LineWorld::new();
        println!("lineworld,{}", test_env_policy(&mut lineworld, "lineworld"));

        let mut gridworld = GridWorld::new();
        println!("gridworld,{}", test_env_policy(&mut gridworld, "gridworld"));

        let mut monty_hall = MontyHall1::new();
        println!("montyhall,{}", test_env_policy(&mut monty_hall, "montyhall"));

        let mut two_round_rps = TwoRoundRPS::new();
        println!("tworoundrps,{}", test_env_policy(&mut two_round_rps, "tworoundrps"));

        let mut secret_env0 = SecretEnv0::new();
        println!("secretenv0,{}", test_env_policy(&mut secret_env0, "secretenv0"));

        let mut secret_env1 = SecretEnv1::new();
        println!("secretenv1,{}", test_env_policy(&mut secret_env1, "secretenv1"));

        let mut secret_env2 = SecretEnv2::new();
        println!("secretenv2,{}", test_env_policy(&mut secret_env2, "secretenv2"));

        let mut secret_env3 = SecretEnv3::new();
        println!("secretenv3,{}", test_env_policy(&mut secret_env3, "secretenv3"));
    }
    #[test]
    fn monte_carlo_with_exploring_start_returns_correct_policy() {
        let mut lw = LineWorld::new();

        println!("stat ID :{:?}", lw.state_id());

        let policy = monte_carlo_with_exploring_start(&mut lw, 0.999, 100, 10, 42, (false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));
        let mut gw2 = LineWorld::new();
        gw2.play_strategy(policy.clone(), false);
        assert_eq!(gw2.state_id(), 40)
    }

    #[test]
    fn monte_carlo_with_exploring_start_returns_correct_policy_grid_world() {
        println!("gridworld : ");
        let mut gw = GridWorld::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = monte_carlo_with_exploring_start(&mut gw, 0.999, 10000, 1000, 42,(false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));

        println!("{:?}", policy);

        let mut gw2 = GridWorld::new();
        gw2.play_strategy(policy.clone(), false);
        assert_eq!(gw2.state_id(), 40)
    }

    #[test]
    fn monte_carlo_with_exploring_start_two_round_rps() {
        println!("Rock paper scissors : ");
        let mut env = TwoRoundRPS::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = monte_carlo_with_exploring_start(&mut env, 0.999, 10000, 1000, 42, (false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));

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
        let mut env = MontyHall1::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = monte_carlo_with_exploring_start(&mut env, 0.999, 10000, 10000, 42, (false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));

        println!("{:?}", policy);
        let nb_run: usize = 1000;

        let mut win: f32 = 0.;

        for _ in 0..nb_run {
            env.reset();
            env.play_strategy(policy.clone(), false);
            //println!("score : {}", env.score());
            win += env.score();
        }

        let stat_win = win / (nb_run as f32);

        println!("win stat :  {}", stat_win);

        assert_eq!(stat_win > 0.5 , true)
    }
}