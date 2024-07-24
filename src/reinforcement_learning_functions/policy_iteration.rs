use std::error::Error;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::num_traits::abs;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::prelude::{SliceRandom, StdRng};
use serde::Serialize;
use crate::environement::environment::Environment;


fn write_csv(path: &str,
             name: &str,
             delta: Vec<usize>,
             stable: usize
) -> Result<(), Box<dyn Error>>
{
    #[derive(Serialize)]
    struct Record<'a> {
        key: &'a str,
        t1: usize,
        t2: usize,
    }
    ;
    let mut wtr = csv::Writer::from_path(path).unwrap();

    let len = delta.len();

    for i in 0..len {
        let record = Record {
            key: name,
            t1: delta[i],
            t2: stable
        };

        wtr.serialize(&record)?;
    }

    wtr.flush().unwrap();
    Ok(())
}

pub fn policy_iteration<E: Environment>(
    gamma: f32,
    theta: f32,
    mut log: (bool, &mut Vec<usize>, &mut usize),
) -> Vec<usize> {
    let num_states = E::num_states();
    let num_actions = E::num_actions();
    let num_rewards = E::num_rewards();


    let mut V = Array::random((num_states, 1), Uniform::new(0.0, 1.0)).into_raw_vec();



    let mut pi = vec![];
    for i in 0..num_states {
            pi.push(0)
    }
    println!("pi : {:?}", pi);




    let mut i = 0;
    loop {
        let mut j = 0;
        loop {
            let mut delta: f32 = 0.;
            for s in 0..num_states {
                let a = pi[s];
                let mut v = V[s];
                let mut total = 0.;
                for s_p in 0..num_states {
                    for r in 0..num_rewards {
                        let p = E::build_transition_probability(s, a, s_p, r);
                        total += p * (E::get_reward(r) + gamma * V[s_p])
                    }
                }
                V[s] = total;
                delta = delta.max((v - V[s]).abs())
            }
            if delta < theta {
                break
            }
        }
        if log.0 {
            log.1.push(j);
            j +=1;
        }


        println!("V : m{:?}", V);

        let mut policy_stable = true;

        for s in 0..num_states {
            let old_action = pi[s];

            let mut best_a: Option<usize> = None;
            let mut best_action_score = f32::MIN;

            for a in 0..num_actions {
                let mut total = 0.;

                for s_p in 0..num_states {
                    for r_index in 0..num_rewards {
                        total += E::build_transition_probability(s, a, s_p, r_index) * (E::get_reward(r_index) + gamma * V[s_p])
                    }
                }

                if best_a.is_none() || total >= best_action_score {
                    best_a = Some(a);
                    best_action_score = total;
                }
            }
            pi[s] = best_a.unwrap();

            if pi[s] != old_action {
                policy_stable = false;
            }
        }
        if policy_stable {
            if log.0 {
                *log.2 = i;
            }

            return pi
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(warnings)]

    use std::collections::HashMap;
    use crate::environement::line_world::LineWorld;
    use crate::environement::grid_world::GridWorld;
    use crate::environement::monty_hall_1::MontyHall1;
    use crate::environement::secret_env_0::SecretEnv0;
    use crate::environement::secret_env_1::SecretEnv1;
    use crate::environement::secret_env_2::SecretEnv2;
    use crate::environement::secret_env_3::SecretEnv3;
    use crate::environement::two_round_rps::TwoRoundRPS;

    use super::*;

    fn test_env_policy<E: Environment>(mut env: &mut E, label: &str) -> u64 {
        let mut delta = Vec::new();
        let mut value = 0;

        use std::time::Instant;
        let now = Instant::now();
        let v = policy_iteration::<E>(0.999, 0.001, (true, &mut delta, &mut value));
        let elapsed = now.elapsed();
        let path = format!("record/policy_iteration_{}.csv", label);
        write_csv(path.as_str(),
                  label,
                  delta, value).expect("TODO: panic message");

        return elapsed.as_millis() as u64;
    }

    #[test]
    fn policy_iteration_all_env() {
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
    fn policy_iteration_some_env()  {

    }

    #[test]
    fn policy_iteration_line_world() {
        #![allow(warnings)]
        println!("start");


        let lw = LineWorld::new();
        println!("stat ID :{:?}", lw.state_id());

        let mut a : usize = 0;
        let v = policy_iteration::<LineWorld>(0.999, 0.000001, (false, &mut Vec::new(), &mut a));

        println!("{:?}", v);
        assert_eq!(1, 1)
    }


    #[test]
    fn policy_iteration_grid_world() {
        println!("start");
        let mut env = GridWorld::new();
        let mut a : usize = 0;

        let v = policy_iteration::<GridWorld>( 0.999, 0.0001, (false, &mut Vec::new(), &mut a));

        let mut policy = HashMap::new();
        for i in 0..v.len(){
            policy.insert(i, v[i]);
        }
        env.play_strategy(policy.clone(), false);

        println!("{:?}", v);

        assert_eq!(env.state_id(), 4)
    }
    #[test]
    fn policy_iteration_two_round_pfs() {
        println!("start");
        let mut env = TwoRoundRPS::new();



        let mut a : usize = 0;
        let v = policy_iteration::<TwoRoundRPS>( 0.999, 0.0001, (false, &mut Vec::new(), &mut a));

        let mut policy = HashMap::new();
        for i in 0..v.len(){
            policy.insert(i, v[i]);
        }

        env.play_strategy(policy.clone(), false);

        println!("{:?}", v);
        println!("{:?}", policy);
        assert_eq!(env.score(), 1.);
    }
    #[test]
    fn policy_iteration_monty_hall_1() {
        println!("start");
        let mut env = MontyHall1::new();


        let mut a : usize = 0;
        let v = policy_iteration::<MontyHall1>( 0.999, 0.0001, (false, &mut Vec::new(), &mut a));

        let mut policy = HashMap::new();
        for i in 0..v.len(){
            policy.insert(i, v[i]);
        }

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
    #[test]
    fn policy_iteration_env_0() {
        println!("start");
        let mut env = SecretEnv0::new();


        let mut a : usize = 0;
        let v = policy_iteration::<SecretEnv0>( 0.999, 0.001,(false, &mut Vec::new(), &mut a));

        let mut policy = HashMap::new();
        for i in 0..v.len(){
            policy.insert(i, v[i]);
        }

        env.play_strategy(policy.clone(), false);

        println!("{:?}", v);
        assert_eq!(1, 1)
    }
}