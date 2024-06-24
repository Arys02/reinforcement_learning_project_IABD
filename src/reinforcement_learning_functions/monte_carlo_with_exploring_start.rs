use std::collections::HashMap;

use ndarray_rand::rand::SeedableRng;
use rand::prelude::{IteratorRandom, StdRng};
use rand::Rng;

use crate::environement::environment::Environment;

pub fn monte_carlo_with_exploring_start(
    mut env: impl Environment,
    gamma: f64,
    nb_iter: i32,
    max_steps: i32,
    seed: u64,
) -> HashMap<usize, usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pi = HashMap::new();
    let mut Q = HashMap::new();
    let mut returns = HashMap::new();

    for _ in 0..nb_iter {
        env.reset_random_state(seed);

        let mut trajectory = Vec::new();
        let mut step_count = 0;
        let mut is_first_action = true;

        while !env.is_terminal() && step_count < max_steps {
            let state = env.state_id();
            let available_action = env.available_action();

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
                let average_return = returns[&(s, a)].iter().sum::<f64>() / returns[&(s, a)].len() as f64;
                Q.insert((s, a), average_return);

                let best_action = aa.iter().max_by(|&&action1, &&action2| {
                    let q_val1 = Q.get(&(s, action1)).copied().unwrap_or_else(|| rng.gen());
                    let q_val2 = Q.get(&(s, action2)).copied().unwrap_or_else(|| rng.gen());

                    q_val1.partial_cmp(&q_val2).unwrap_or(std::cmp::Ordering::Equal)
                }).copied().unwrap();

                pi.insert(s, best_action);
            }
        }

    }
    return pi
}

#[cfg(test)]
mod tests {
    use crate::environement::line_world::LineWorld;
    use super::*;


    #[test]
    fn monte_carlo_with_exploring_start_returns_correct_policy() {
        let lw = LineWorld::new();

        let policy = monte_carlo_with_exploring_start(lw, 0.999, 10, 10, 42);
        println!("{:?}", policy)
    }
}