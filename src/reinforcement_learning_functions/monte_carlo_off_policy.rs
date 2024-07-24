extern crate csv;
extern crate serde;

use crate::environement::environment::Environment;
use std::collections::HashMap;
use std::error::Error;
use csv::Writer;
use ndarray_rand::rand::SeedableRng;
use rand::distributions::WeightedIndex;
use rand::prelude::{Distribution, IteratorRandom, StdRng};
use rand::Rng;

use serde::Serialize;

fn write_csv(path: &str,
                 name: &str,
                 g_vec: Vec<f32>,
                 size_trajectoire_vec: Vec<usize>,
                 w_vec: Vec<f32>, is_terminal_vec:
                 Vec<bool>) -> Result<(), Box<dyn Error>>
{
    #[derive(Serialize)]
    struct Record<'a> {
        key: &'a str,
        t1: f32,
        t2: usize,
        t3: f32,
        t4: bool,
    }
    ;
    //let mut data = HashMap::new();
    let mut wtr = csv::Writer::from_path(path).unwrap();
    //let mut wtr = Writer::from_writer(vec![]);


    //data.insert(name, (g_vec, size_trajectoire_vec, w_vec, is_terminal_vec));

    //wtr.write_record(&["env", "g", "trajectoir_size", "w_vec"]).unwrap();

    let len = g_vec.len();

    for i in 0..len {

        //wtr.write_record(&[key, t1[i], *t2[i], *t3[i], *t4[i]]);
        let record = Record {
            key: name,
            t1: g_vec[i],
            t2: size_trajectoire_vec[i],
            t3: w_vec[i],
            t4: is_terminal_vec[i],
        };

        wtr.serialize(&record)?;
    }
    //let data = String::from_utf8(wtr.into_inner()?)?;

    wtr.flush().unwrap();
    Ok(())
}

/// Executes the Monte Carlo Off-Policy algorithm for a given environment.
///
/// This algorithm uses an off-policy Monte Carlo control method to improve the policy `pi`
/// based on episodes generated using a behavior policy. The behavior policy is typically
/// an epsilon-greedy policy. The algorithm iteratively updates the action-value function `Q`
/// and the policy `pi`.
///
/// # Parameters
///
/// - `env`: A mutable reference to an environment that implements the `Environment` trait.
/// - `gamma`: The discount factor for future rewards.
/// - `nb_iter`: The number of iterations (episodes) to run the algorithm.
/// - `max_steps`: The maximum number of steps per episode.
/// - `epsilon`: The epsilon parameter for the epsilon-greedy behavior policy.
/// - `seed`: The seed for the random number generator to ensure reproducibility.
/// - `log`: A tuple containing:
///     - `bool`: A flag to determine whether to log data.
///     - `&mut Vec<f32>`: A mutable reference to a vector to log the return (G).
///     - `&mut Vec<usize>`: A mutable reference to a vector to log the trajectory size.
///     - `&mut Vec<f32>`: A mutable reference to a vector to log the importance sampling ratio (W).
///     - `&mut Vec<bool>`: A mutable reference to a vector to log whether the episode terminated.
///
/// # Returns
///
/// - `HashMap<usize, usize>`: The improved policy mapping states to actions.
///
/// # Details
///
/// The `monte_carlo_off_policy` function follows these steps:
///
/// 1. Initialize the random number generator with the provided seed.
/// 2. Initialize the policy `pi`, action-value function `Q`, and counter `C`.
/// 3. For each episode:
///     - Reset the environment.
///     - Generate an episode following the epsilon-greedy behavior policy.
///     - Track the trajectory of state-action-reward tuples.
///     - Calculate the return `G` and update `Q` and `pi` using importance sampling weights.
///     - Optionally log the return, trajectory size, importance sampling ratio, and termination status.
///
/// The behavior policy selects actions based on a probability distribution influenced by the epsilon parameter,
/// balancing exploration and exploitation. The target policy `pi` is iteratively improved based on the observed returns.

pub fn monte_carlo_off_policy<E: Environment>(
    mut env: &mut E,
    gamma: f32,
    nb_iter: i32,
    max_steps: i32,
    epsilon: f32,
    mut seed: u64,
    //     check, g        size_traj     w          env.ister
    mut log: (bool, &mut Vec<f32>, &mut Vec<usize>, &mut Vec<f32>, &mut Vec<bool>),
) -> HashMap<usize, usize> {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut pi = HashMap::new();
    let mut Q: HashMap<(usize, usize), f32> = HashMap::new();
    let mut C: HashMap<(usize, usize), f32> = HashMap::new();


    //loop forever for each episode
    for i in 0..nb_iter {
        seed += 1;
        env.reset();

        let mut trajectory = Vec::new();
        let mut step_count = 0;

        let mut b = HashMap::new();
        // generat an episode from S0 following pi
        while !env.is_terminal() && step_count < max_steps {
            let state = env.state_id();
            let available_action = env.available_action();

            if !Q.contains_key(&(state, available_action[0])) {
                let mut max_val = f32::MIN;
                let mut max_i = 0;
                for a in 0..available_action.len() {
                    let new_val = rng.gen::<f32>();

                    if new_val > max_val {
                        max_val = new_val;
                        max_i = a
                    }

                    Q.insert((state, a), new_val);
                    C.insert((state, a), 0.);
                }
                pi.insert(state, available_action[max_i]);
            }

            let mut tmp_action_vector = Vec::with_capacity(available_action.len());
            let mut i_max: usize = 0;
            let mut val_max: f32 = 0.;
            for i in 0..available_action.len() {
                let val = rng.gen::<f32>();
                if val > val_max {
                    val_max = val;
                    i_max = i;
                }
                tmp_action_vector.push(rng.gen::<f32>());
            }

            for i in 0..available_action.len() {
                if i == i_max {
                    tmp_action_vector[i] = 1. - epsilon + epsilon / (available_action.len()
                        as f32)
                } else {
                    tmp_action_vector[i] = epsilon / (available_action.len() as f32)
                }
            }


            let mut dist = WeightedIndex::new(&tmp_action_vector).unwrap();
            let action = dist.sample(&mut rng);

            b.insert(state, tmp_action_vector.clone());

            let next_action = available_action[action];
            let prev_score = env.score();
            env.step(next_action);
            let r = env.score() - prev_score;

            trajectory.push((state, action, r, available_action.clone()));
            step_count += 1;
        }

        let mut g = 0.;
        let mut w = 1.;

        for (t, &(s, a, r, ref aa)) in trajectory.iter().enumerate().rev() {
            let av_vec = aa.to_vec();

            g = gamma * g + r;
            C.insert((s, a), C.get(&(s, a)).unwrap() + w);
            Q.insert((s, a), Q.get(&(s, a)).unwrap() + (w / C.get(&(s, a)).unwrap()) * (g - Q
                .get(&(s, a)).unwrap()));

            let best_action = aa.iter().max_by(|&&action1, &&action2| {
                let q_val1 = Q.get(&(s, action1)).copied().unwrap_or_else(|| rng.gen());
                let q_val2 = Q.get(&(s, action2)).copied().unwrap_or_else(|| rng.gen());

                q_val1.partial_cmp(&q_val2).unwrap_or(std::cmp::Ordering::Equal)
            }).copied().unwrap();

            pi.insert(s, best_action);
            if best_action != a {
                break;
            }

            w = w * (1. / b.get(&s).unwrap()[a]);

            //     check, g        size_traj     w          env.ister
            if log.0 {
                &log.1.push(g);
                log.2.push(trajectory.len());
                log.3.push(w);
                log.4.push(env.is_terminal())
            }
        }
    }
    let mut result: HashMap<usize, usize> = HashMap::new();
    for (key, values) in pi {
        result.insert(key, values);
    }

    return result;
}

#[cfg(test)]
mod tests {
    use ndarray_stats::histogram::Grid;
    use crate::environement::grid_world::GridWorld;
    use crate::environement::line_world::LineWorld;
    use crate::environement::monty_hall_1::MontyHall1;
    use crate::environement::secret_env_0::SecretEnv0;
    use crate::environement::secret_env_1::SecretEnv1;
    use crate::environement::secret_env_2::SecretEnv2;
    use crate::environement::secret_env_3::SecretEnv3;
    use crate::environement::two_round_rps::TwoRoundRPS;

    use super::*;

    fn build_policy(map: &HashMap<usize, Vec<f32>>) -> HashMap<usize, usize> {
        let mut result: HashMap<usize, usize> = HashMap::new();
        for (key, values) in map {
            if let Some((max_index, _)) = values.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                result.insert(*key, max_index);
            }
        }

        result
    }

    fn test_env_policy<E: Environment>(mut env: &mut E, label: &str) -> u64 {
        let mut env_test = E::new();

        let mut g = Vec::new();
        let mut t_size = Vec::new();
        let mut w = Vec::new();
        let mut is_terminal = Vec::new();

        use std::time::Instant;
        let now = Instant::now();
        let policy_map = monte_carlo_off_policy(env, 0.999, 1000, 1000, 0.4, 42, (true, &mut g, &mut t_size, &mut w, &mut is_terminal));
        let elapsed = now.elapsed();

        let path = format!("record/monte_carlo_off_{}.csv", label);
        write_csv(path.as_str(),
                  label,
                  g,
                  t_size,
                  w,
                  is_terminal).expect("TODO: panic message");

        env_test.play_strategy(policy_map.clone(), false);
        return elapsed.as_millis() as u64;
    }
    #[test]
    fn monte_carlo_off_all_env() {
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
    fn monte_carlo_off_policy_lineworld() {
        let mut lw = LineWorld::new();


        println!("stat ID :{:?}", lw.state_id());

        let mut g = Vec::new();
        let mut t_size = Vec::new();
        let mut w = Vec::new();
        let mut is_terminal = Vec::new();

        use std::time::Instant;
        let now = Instant::now();
        let policy_map = monte_carlo_off_policy(&mut lw, 0.999, 1000, 1000, 0.4, 42, (true, &mut g, &mut t_size, &mut w, &mut is_terminal));
        let elapsed = now.elapsed();


        write_csv("record/monte_carlo_off_lineworld.csv",
                  "lineworld",
                  g,
                  t_size,
                  w,
                  is_terminal).expect("TODO: panic message");


        //let policy: HashMap<usize, usize> = build_policy(&policy_map);
        let mut gw2 = LineWorld::new();
        gw2.play_strategy(policy_map.clone(), false);
        assert_eq!(gw2.state_id(), 4)
    }

    #[test]
    fn monte_carlo_off_policy_grid_world() {
        println!("gridworld : ");
        let mut gw = GridWorld::new();

        let policy = monte_carlo_off_policy(&mut gw, 0.999, 10000, 1000, 0.1, 42, (false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));


        println!("{:?}", policy);

        //let policy = build_policy(&policy);
        let mut gw2 = GridWorld::new();
        gw2.play_strategy(policy.clone(), false);
        assert_eq!(gw2.state_id(), 40)
    }
    #[test]
    fn monte_carlo_off_policy_rps() {
        println!("twoRoundRPS: ");
        let mut gw = TwoRoundRPS::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = monte_carlo_off_policy(&mut gw, 0.999, 10000, 1000, 0.1, 42, (false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));

        println!("{:?}", policy);

        //let policy = build_policy(&policy);
        let mut gw2 = TwoRoundRPS::new();
        gw2.play_strategy(policy.clone(), false);
        assert_eq!(gw2.score(), 1.)
    }
    #[test]
    fn monte_carlo_off_policy_monty_hall_1() {
        println!("Monty Hall 1: ");
        let mut env = MontyHall1::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = monte_carlo_off_policy(&mut env, 0.999, 10000, 1000, 0.1, 42, (false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));


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
}