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

    log : (bool, &Vec<bool>),
) -> (HashMap<usize, usize>, HashMap<(usize, usize), (f32, usize)>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut Q = HashMap::new();

    let mut pi = HashMap::new();


    for _ in 0..nb_iter {
        env.reset();
        let mut step = 0;


        while !env.is_terminal() && step < nb_step {
            step += 1;
            let aa = env.available_action();
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
                let q_s: Vec<(f32, usize)> = aa.iter().enumerate().map(|(i, &a)| *Q.get(&
                (state, i))
                    .unwrap())
                    .collect();

                let best_a_index = q_s.iter().enumerate().max_by(|(_, (a, _)), (_, (b, _))| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap();
                let _ = action_i.insert(best_a_index);
            }

            let action = *aa.get(action_i.unwrap()).unwrap();

            let prev_score: f32 = env.score();
            env.step(action);
            let r = env.score() - prev_score;

            let state_p = env.state_id();
            let available_action_p = env.available_action();



            let mut target: f32;

            if env.is_terminal() {
                target = alpha * r
            } else
            {
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
                    let q_s: Vec<(f32, usize)> = available_action_p.iter().enumerate().map(|(i, &a)| *Q
                        .get(&(state_p,
                               i))
                        .unwrap())
                        .collect();

                    let best_a_index = q_s.iter().enumerate().max_by(|&(_, (a, _)), &(_, (b, _))| a.partial_cmp
                    (b).unwrap()).map(|(index, _)| index).unwrap();
                    let _ = action_i_p.insert(best_a_index);
                }

                let q_sp_ap = if Q.contains_key(&(state_p, action_i_p.unwrap())) {
                    *Q.get(&(state_p, action_i_p.unwrap())).unwrap()
                } else { (0., 0) }.0;
                let q_s_a = Q.get(&(state, action_i.unwrap())).unwrap().0;

                target = (1. - alpha) * q_s_a + alpha * (gamma * q_sp_ap + r);
            }

            Q.insert((state, action_i.unwrap()), (target, action));
        }
    }
    for s in 0..E::num_states() {
        let mut best_a: Option<usize> = None;
        let mut best_a_score = f32::MIN;
        for action in 0..E::num_actions() {
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

        let policy = sarsa(&mut lw, 0.1, 0.1, 0.999, 1000, 1000, 42, (false, &Vec::new()));
        println!("{:?}", policy);
        lw.play_strategy(policy.0, false);
        assert_eq!(lw.is_terminal() && lw.score() == 1.0, true);
    }

    #[test]
    fn sarsa_policy_gridworld() {
        let mut env = GridWorld::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = sarsa(&mut env, 0.1, 0.1, 0.999, 10000, 10000, 42, (false, &Vec::new()));
        println!("{:?}", policy);
        env.reset();

        env.play_strategy(policy.0, false);

        assert_eq!(env.is_terminal() && env.score() == 3.0, true)
    }

    #[test]
    fn sarsa_policy_two_round_rps() {
        let mut env = TwoRoundRPS::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = sarsa(&mut env, 0.1, 0.1, 0.999, 100, 1000, 42, (false, &Vec::new()));
        println!("{:?}", policy);
        env.reset();

        env.play_strategy(policy.0, false);

        assert_eq!(env.is_terminal() && env.score() == 1.0, true)
    }

    #[test]
    fn sarsa_monty_hall_1() {
        println!("Monty Hall 1: ");
        let mut env = MontyHall1::new();

        println!("stat ID :{:?}", env.state_id());

        let policy = sarsa(&mut env, 0.1, 0.1, 0.999, 1000, 1000, 42, (false, &Vec::new()));


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