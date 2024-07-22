use std::collections::HashMap;

use ndarray_rand::rand::SeedableRng;
use rand::prelude::{Distribution, IteratorRandom, StdRng};
use rand::Rng;

use crate::environement::environment::Environment;

pub fn sarsa<E: Environment>(
    mut env: &mut E,
    alpha: f32,
    epsilon: f32,
    gamma: f32,
    nb_iter: usize,
    nb_step: usize,
    seed: u64,
) -> (HashMap<usize, usize>, HashMap<(usize, usize), f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut Q = HashMap::new();

    let mut pi = HashMap::new();

    //INIT
    for s in 0..E::num_states() {
        for a in 0..E::available_actions(s).len() {
            if E::is_terminal_state(s) {
                Q.insert((s, a), 0.);
            } else {
                Q.insert((s, a), rng.gen::<f32>());
            }
        }
    }


    for _ in 0..nb_iter {
        env.reset();
        let mut step = 0;


        while !env.is_terminal() && step < nb_step{
            step +=1;
            let available_action = env.available_action();
            let state = env.state_id();

            let mut action_i: Option<usize> = None;
            if rng.gen::<f32>() < epsilon {
                //insert l'action dans Action (et pas l'index)
                let mut rng = rand::thread_rng();
                let random_i: usize = rng.gen_range(0..available_action.len());
                let _ = action_i.insert(random_i);
            } else {
                let q_s: Vec<f32> = available_action.iter().enumerate().map(|(i, &a)| *Q.get(&
                (state,
                                                                                           i))
                    .unwrap())
                    .collect();

                let best_a_index = q_s.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap();
                let _ = action_i.insert(best_a_index);
            }

            let action = *available_action.get(action_i.unwrap()).unwrap();

            let prev_score: f32 = env.score();
            env.step(action);
            let r = env.score() - prev_score;

            let state_p = env.state_id();
            let available_action_p = env.available_action();

            let mut target: f32;

            if env.is_terminal() {
                target = alpha * r
            } else {

                let mut action_i_p: Option<usize> = None;
                if rng.gen::<f32>() < epsilon {
                    //insert l'action dans Action (et pas l'index)
                    let mut rng = rand::thread_rng();
                    let random_i: usize = rng.gen_range(0..available_action_p.len());
                    let _ = action_i_p.insert(random_i);
                } else {
                    let q_s: Vec<f32> = available_action_p.iter().enumerate().map(|(i, &a)|  *Q
                        .get(&(state_p,
                                                                                         a))
                        .unwrap_or(&0.))
                        .collect();

                    let best_a_index = q_s.iter().enumerate().max_by(|&(_, a), &(_, b)| a.partial_cmp
                    (b).unwrap()).map(|(index, _)| index).unwrap();
                    let _ = action_i_p.insert(best_a_index);
                }


                let q_sp_ap = *Q.get(&(state_p, action_i_p.unwrap())).unwrap();
                let q_s_a = *Q.get(&(state, action_i.unwrap())).unwrap();


                target = (1. - alpha) * q_s_a + alpha * (gamma * q_sp_ap + r);
            }

            Q.insert((state, action), target);
        }
    }
    for s in 0..E::num_states() {
        if E::is_terminal_state(s) { continue; }
        let mut best_a: Option<usize> = None;
        let mut best_a_score = f32::MIN;
        for (action) in E::available_actions(s) {
            let a_score = Q[&(s, action)];
            if best_a.is_none() || best_a_score <= a_score {
                best_a = Some(action);
                best_a_score = a_score;
            }
        }
        pi.insert(s, best_a.unwrap());
    }


    println!("{:?}", Q);
    println!("{:?}", pi);
    return (pi, Q);
}

#[cfg(test)]
mod tests {
    use crate::environement::grid_world::GridWorld;
    use crate::environement::line_world::LineWorld;
    use crate::environement::monty_hall_1::MontyHall1;
    use crate::environement::two_round_rps::TwoRoundRPS;

    use super::*;

    #[test]
    fn sarsa_policy_lineworld() {
        let mut lw = LineWorld::new();

        println!("stat ID :{:?}", lw.state_id());

        let policy = sarsa(&mut lw, 0.1, 0.1, 0.999, 1000,1000,  42);
        println!("{:?}", policy);
        lw.play_strategy(policy.0);
        assert_eq!(lw.is_terminal() && lw.score() == 1.0, true);
    }

    #[test]
    fn sarsa_policy_gridworld() {
        let mut env = GridWorld::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = sarsa(&mut env, 0.1, 0.1, 0.999, 100000,100000, 42);
        println!("{:?}", policy);
        env.reset();

        env.play_strategy(policy.0);

        assert_eq!(env.is_terminal() && env.score() == 3.0, true)
    }

    #[test]
    fn sarsa_policy_two_round_rps() {
        let mut env = TwoRoundRPS::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = sarsa(&mut env, 0.1, 0.1, 0.999, 100, 1000, 42);
        println!("{:?}", policy);
        env.reset();

        env.play_strategy(policy.0);

        assert_eq!(env.is_terminal() && env.score() == 1.0, true)
    }

    #[test]
    fn sarsa_monty_hall_1() {
        println!("Monty Hall 1: ");
        let mut env = MontyHall1::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = sarsa(&mut env, 0.1, 0.1, 0.999, 1000, 1000,  42);


        println!("{:?}", policy);
        let nb_run: usize = 1000;

        let mut win: f32 = 0.;

        for _ in 0..nb_run {
            env.reset();
            env.play_strategy(policy.0.clone());
            win += env.score();
        }

        let stat_win = win / (nb_run as f32);

        println!("win stat :  {}", stat_win);

        assert_eq!(stat_win > 0.6, true)
    }
}