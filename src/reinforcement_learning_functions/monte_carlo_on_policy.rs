use std::collections::HashMap;
use std::error::Error;
use ndarray_rand::rand::SeedableRng;
use rand::distributions::WeightedIndex;
use rand::prelude::{Distribution, IteratorRandom, StdRng};
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

pub fn monte_carlo_on_policy<E: Environment>(
    mut env: &mut E,
    gamma: f32,
    nb_iter: i32,
    max_steps: i32,
    epsilon: f32,
    mut seed: u64,
    mut log: (bool, &mut Vec<f32>, &mut Vec<usize>, &mut Vec<bool>),
) -> HashMap<usize, usize> {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut pi = HashMap::new();
    let mut Q = HashMap::new();
    let mut returns = HashMap::new();

    for _ in 0..nb_iter {
        seed += 1;
        env.reset();

        let mut trajectory = Vec::new();
        let mut step_count = 0;
        // generat an episode from S0 following pi
        while !env.is_terminal() && step_count < max_steps {
            let state = env.state_id();
            let available_action = env.available_action();
            let vec_aa = available_action.to_vec();

            if !pi.contains_key(&state) {
                let mut tmp_action_vector = Vec::with_capacity(available_action.len());
                let mut i_max: usize = 0;
                let mut val_max: f32 = 0.;
                for i in 0..available_action.len() {
                    let val = rng.gen::<f32>();
                    if val > val_max {
                        val_max = val;
                        i_max = i;
                    }
                    tmp_action_vector.push(rng.gen::<f32>())
                }

                for i in 0..available_action.len() {
                    if i == i_max {
                        tmp_action_vector[i] = 1. - epsilon + epsilon / (available_action.len()
                            as f32)
                    } else {
                        tmp_action_vector[i] = epsilon / (available_action.len() as f32)
                    }
                }


                pi.insert(state, (tmp_action_vector, available_action.clone()));
            }

            let actions = pi.get(&state).unwrap().to_owned().0;
            let mut dist = WeightedIndex::new(actions).unwrap();
            let action = dist.sample(&mut rng);


            let prev_score = env.score();
            env.step(*available_action.clone().get(action).unwrap());
            let r = env.score() - prev_score;

            trajectory.push((state, action, r, available_action.clone()));
            step_count += 1;
        }

        let mut g = 0.;

        for (t, &(s, a, r, ref aa)) in trajectory.iter().enumerate().rev() {
            let av_vec = aa.to_vec();

            g = gamma * g + r;
            //unless the pair St, At appear
            if trajectory[..t].iter().all(|&(si, ai, _, _)| si != s || ai != a) {
                returns.entry((s, a)).or_insert(Vec::new()).push(g);
                let average_return = returns[&(s, a)].iter().sum::<f32>() / returns[&(s, a)].len() as f32;

                Q.insert((s, a), average_return);

                let best_action = aa.iter().max_by(|&&action1, &&action2| {
                    let q_val1 = Q.get(&(s, action1)).copied().unwrap_or_else(|| rng.gen());
                    let q_val2 = Q.get(&(s, action2)).copied().unwrap_or_else(|| rng.gen());

                    q_val1.partial_cmp(&q_val2).unwrap_or(std::cmp::Ordering::Equal)
                }).copied().unwrap();


                let mut tmp_action_vector = Vec::with_capacity(aa.len());
                for i in 0..aa.len() {
                    if i == best_action {
                        tmp_action_vector.push(1. - epsilon + (epsilon / (aa.len() as f32)))
                    } else {
                        tmp_action_vector.push(epsilon / (aa.len() as f32))
                    }
                }

                pi.insert(s, (tmp_action_vector.clone(), aa.clone()));
            }
            if log.0 {
                &log.1.push(g);
                log.2.push(trajectory.len());
                log.3.push(env.is_terminal())
            }
        }
    }
    let mut result: HashMap<usize, usize> = HashMap::new();
    for (state, values) in pi {
        if let Some((max_index, _)) = values.0.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            let available_actions = values.1;
            let action_index = available_actions[max_index];
            result.insert(state, action_index);
        }
    }


    return result;
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
    use crate::reinforcement_learning_functions::monte_carlo_off_policy::monte_carlo_off_policy;
    use crate::reinforcement_learning_functions::monte_carlo_with_exploring_start::monte_carlo_with_exploring_start;

    use super::*;

    fn test_env_policy<E: Environment>(mut env: &mut E, label: &str) -> u64 {
        let mut env_test = E::new();

        let mut g = Vec::new();
        let mut t_size = Vec::new();
        let mut is_terminal = Vec::new();

        use std::time::Instant;
        let now = Instant::now();
        let policy_map = monte_carlo_on_policy(env, 0.999, 1000, 1000, 0.4, 42, (true, &mut g, &mut t_size, &mut is_terminal));
        let elapsed = now.elapsed();

        let path = format!("record/monte_carlo_on_{}.csv", label);
        write_csv(path.as_str(),
                  label,
                  g,
                  t_size,
                  is_terminal).expect("TODO: panic message");

        env_test.play_strategy(policy_map.clone(), false);
        return elapsed.as_millis() as u64;
    }

    #[test]
    fn monte_carlo_on_all_env() {
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
    fn build_policy(map: &HashMap<usize, Vec<f32>>) -> HashMap<usize, usize> {
        let mut result: HashMap<usize, usize> = HashMap::new();
        for (key, values) in map {
            if let Some((max_index, _)) = values.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                result.insert(*key, max_index);
            }
        }

        result
    }

    #[test]
    fn monte_carlo_on_policy_lineworld() {
        let mut lw = LineWorld::new();

        println!("stat ID :{:?}", lw.state_id());

        let policy_map = monte_carlo_on_policy(&mut lw, 0.999, 10000, 10000, 0.4, 42, (false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));

        println!("{:?}", policy_map);
        let mut gw2 = LineWorld::new();
        gw2.play_strategy(policy_map.clone(), false);
        assert_eq!(gw2.state_id(), 4)
    }

    #[test]
    fn monte_carlo_on_policy_grid_world() {
        println!("gridworld : ");
        let mut gw = GridWorld::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = monte_carlo_on_policy(&mut gw, 0.999, 10000, 1000, 0.1, 42, (false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));

        println!("{:?}", policy);
        let mut gw2 = GridWorld::new();
        gw2.play_strategy(policy.clone(), false);
        assert_eq!(gw2.state_id(), 40)
    }

    #[test]
    fn monte_carlo_on_policy_monty_hall_1() {
        println!("Monty Hall 1: ");
        let mut env = MontyHall1::new();

        println!("stat ID :{:?}", env.state_id());

        let policy= monte_carlo_on_policy(&mut env, 0.999, 10000, 1000, 0.1,42, (false, &mut Vec::new(), &mut Vec::new(), &mut Vec::new()));


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

        assert_eq!(stat_win > 0.5 , true)
    }


}