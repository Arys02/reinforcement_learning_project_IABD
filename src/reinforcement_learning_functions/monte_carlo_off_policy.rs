use std::collections::HashMap;

use ndarray_rand::rand::SeedableRng;
use rand::distributions::WeightedIndex;
use rand::prelude::{Distribution, IteratorRandom, StdRng};
use rand::Rng;

use crate::environement::environment::Environment;

pub fn monte_carlo_off_policy<E: Environment>(
    mut env: &mut E,
    gamma: f32,
    nb_iter: i32,
    max_steps: i32,
    epsilon: f32,
    mut seed: u64,
    log : bool,
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

            if !Q.contains_key(&(state, available_action[0])){
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

            w = w * (1. / b.get(&s).unwrap()[a])
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
    use crate::environement::grid_world::GridWorld;
    use crate::environement::line_world::LineWorld;
    use crate::environement::monty_hall_1::MontyHall1;
    use crate::environement::two_round_rps::TwoRoundRPS;
    use crate::reinforcement_learning_functions::monte_carlo_on_policy::monte_carlo_on_policy;

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

    #[test]
    fn monte_carlo_off_policy_lineworld() {
        let mut lw = LineWorld::new();

        println!("stat ID :{:?}", lw.state_id());

        let policy_map = monte_carlo_off_policy(&mut lw, 0.999, 1000, 1000, 0.4, 42, false);

        //let policy: HashMap<usize, usize> = build_policy(&policy_map);
        println!("{:?}", policy_map);
    }

    #[test]
    fn monte_carlo_off_policy_grid_world() {
        println!("gridworld : ");
        let mut gw = GridWorld::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = monte_carlo_off_policy(&mut gw, 0.999, 10000, 1000, 0.1, 42, false);

        println!("{:?}", policy);

        //let policy = build_policy(&policy);
        let mut gw2 = GridWorld::new();
        gw2.play_strategy(policy.clone());
        assert_eq!(gw2.state_id(), 40)
    }
    #[test]
    fn monte_carlo_off_policy_rps() {
        println!("twoRoundRPS: ");
        let mut gw = TwoRoundRPS::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = monte_carlo_off_policy(&mut gw, 0.999, 10000, 1000, 0.1, 42, false);

        println!("{:?}", policy);

        //let policy = build_policy(&policy);
        let mut gw2 = TwoRoundRPS::new();
        gw2.play_strategy(policy.clone());
        assert_eq!(gw2.score(), 1.)
    }
    #[test]
    fn monte_carlo_off_policy_monty_hall_1() {
        println!("Monty Hall 1: ");
        let mut env = MontyHall1::new();

        println!("stat ID :{:?}", env.state_id());

        let policy= monte_carlo_off_policy(&mut env, 0.999, 10000, 1000, 0.1,42, false);


        println!("{:?}", policy);
        let nb_run: usize = 1000;

        let mut win: f32 = 0.;

        for _ in 0..nb_run {
            env.reset();
            env.play_strategy(policy.clone());
            println!("score : {}", env.score());
            win += env.score();
        }

        let stat_win = win / (nb_run as f32);

        println!("win stat :  {}", stat_win);

        assert_eq!(stat_win > 0.5 , true)
    }
}