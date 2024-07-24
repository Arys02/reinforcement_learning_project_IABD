use std::collections::HashMap;

use ndarray_rand::rand::SeedableRng;
use rand::prelude::{IteratorRandom, StdRng};
use rand::Rng;

use crate::environement::environment::Environment;

pub fn monte_carlo_with_exploring_start<E: Environment>(
    mut env: &mut E,
    gamma: f32,
    nb_iter: i32,
    max_steps: i32,
    mut seed: u64,
    log : (bool, &Vec<f32>, &Vec<usize>, &Vec<bool>),
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
    }
    return pi;
}

#[cfg(test)]
mod tests {
    use crate::environement::grid_world::GridWorld;
    use crate::environement::line_world::LineWorld;
    use crate::environement::monty_hall_1::MontyHall1;
    use crate::environement::two_round_rps::TwoRoundRPS;

    use super::*;

    #[test]
    fn monte_carlo_with_exploring_start_returns_correct_policy() {
        let mut lw = LineWorld::new();

        println!("stat ID :{:?}", lw.state_id());

        let policy = monte_carlo_with_exploring_start(&mut lw, 0.999, 100, 10, 42, (false, &Vec::new(), &Vec::new(), &Vec::new()));
        let mut gw2 = LineWorld::new();
        gw2.play_strategy(policy.clone(), false);
        assert_eq!(gw2.state_id(), 40)
    }

    #[test]
    fn monte_carlo_with_exploring_start_returns_correct_policy_grid_world() {
        println!("gridworld : ");
        let mut gw = GridWorld::new();

        println!("stat ID :{:?}", gw.state_id());

        let policy = monte_carlo_with_exploring_start(&mut gw, 0.999, 10000, 1000, 42,(false, &Vec::new(), &Vec::new(), &Vec::new()));

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

        let policy = monte_carlo_with_exploring_start(&mut env, 0.999, 10000, 1000, 42, (false, &Vec::new(), &Vec::new(), &Vec::new()));

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

        let policy = monte_carlo_with_exploring_start(&mut env, 0.999, 10000, 10000, 42, (false, &Vec::new(), &Vec::new(), &Vec::new()));

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