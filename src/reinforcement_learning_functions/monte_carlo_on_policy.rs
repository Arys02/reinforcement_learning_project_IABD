use std::collections::HashMap;

use ndarray_rand::rand::SeedableRng;
use rand::distributions::WeightedIndex;
use rand::prelude::{Distribution, IteratorRandom, StdRng};
use rand::Rng;

use crate::environement::environment::Environment;

pub fn monte_carlo_on_policy<E: Environment>(
    mut env: &mut E,
    gamma: f32,
    nb_iter: i32,
    max_steps: i32,
    epsilon: f32,
    mut seed: u64,
) -> HashMap<usize, Vec<f32>> {
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


                pi.insert(state, tmp_action_vector);
            }

            let actions = pi.get(&state).unwrap().to_owned();
            let mut dist = WeightedIndex::new(actions).unwrap();
            let action = dist.sample(&mut rng);


            let prev_score = env.score();
            env.step(action);
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

                pi.insert(s, tmp_action_vector.clone());
            }
        }
    }


    return pi;
}

#[cfg(test)]
mod tests {
    use crate::environement::grid_world::GridWorld;
    use crate::environement::line_world::LineWorld;

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
    fn monte_carlo_on_policy_policy() {
        let mut lw = LineWorld::new();

        println!("stat ID :{:?}", lw.state_id());

        let policy_map = monte_carlo_on_policy(&mut lw, 0.999, 10000, 10000, 0.4, 42);

        let policy: HashMap<usize, usize> = build_policy(&policy_map);
        println!("{:?}", policy_map);
        println!("{:?}", policy)
    }

    #[test]
    fn monte_carlo_on_policy_grid_world() {
        println!("gridworld : ");
        let mut gw = GridWorld::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = monte_carlo_on_policy(&mut gw, 0.999, 10000, 1000, 0.1, 42);

        println!("{:?}", policy);

        let policy = build_policy(&policy);
        let mut gw2 = GridWorld::new();
        gw2.play_strategy(policy.clone());
        assert_eq!(gw2.state_id(), 40)
    }
}