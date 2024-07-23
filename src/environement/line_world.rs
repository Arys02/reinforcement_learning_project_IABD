extern crate rand;

use std::collections::HashMap;

use ndarray::{array, Array1, Array4, ArrayBase, Ix4, OwnedRepr};
use rand::{Rng, SeedableRng};
use rand::prelude::StdRng;

use crate::environement::environment::Environment;

pub struct LineWorld {
    agent_pos: usize,

    transition_probability_matrix: ArrayBase<OwnedRepr<f32>, Ix4>,
}

impl LineWorld {
    fn build_transition_matrix() -> ArrayBase<OwnedRepr<f32>, Ix4> {
        let mut transition_probability_matrix = Array4::zeros((Self::num_states(), Self::num_actions(), Self::num_states(), Self::num_rewards()));

        for s in 0..Self::num_states() {
            for a in 0..Self::num_actions() {
                for s_p in 0..Self::num_states() {
                    for r in 0..Self::num_rewards() {
                        if s < Self::num_states() && s_p == s + 1 && a == 1 && Self::get_reward(r) == 0. && (s == 1 || s == 2) {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                        if s > 0 && s_p == s - 1 && a == 0 && Self::get_reward(r) == 0. && (s == 2 || s == 3) {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                    }
                }
            }
        }
        transition_probability_matrix[[3, 1, 4, 2]] = 1.0;
        transition_probability_matrix[[1, 0, 0, 0]] = 1.0;


        return transition_probability_matrix;
    }
}

impl Environment for LineWorld {
    fn new() -> Self {
        LineWorld {
            agent_pos: 2,
            transition_probability_matrix: Self::build_transition_matrix(),
        }
    }

    fn state_id(&self) -> usize {
        self.agent_pos
    }


    fn from_random_state() -> Self {
        let mut rng = rand::thread_rng();
        let agent_pos_: usize = rng.gen_range(1..4);
        return LineWorld {
            agent_pos: agent_pos_,
            transition_probability_matrix: Self::build_transition_matrix(),
        };
    }

    fn reset(&mut self) {
        self.agent_pos = 2
    }

    fn num_states() -> usize {
        5
    }

    fn num_actions() -> usize {
        2
    }

    fn num_rewards() -> usize {
        3
    }

    fn get_reward(i: usize) -> f32 {
        match i {
            0 => -1.,
            1 => 0.,
            2 => 1.,
            _ => panic!("reward is out of range")
        }
    }

    fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        let tm = Self::build_transition_matrix();
        return tm[[s, a, s_p, r]];
    }


    fn get_transition_probability(&mut self, s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        return self.transition_probability_matrix[[s, a, s_p, r]];
    }

    fn reset_random_state(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let agent_pos_: usize = rng.gen_range(1..4);
        self.agent_pos = agent_pos_
    }


    fn available_action(&self) -> Array1<usize> {
        match self.agent_pos {
            1 | 2 | 3 => array![0, 1],
            _ => array![]
        }
    }

    fn available_action_delete(&self) {
        todo!()
    }

    fn is_terminal(&self) -> bool {
        match self.agent_pos {
            1 | 2 | 3 => false,
            _ => true
        }
    }

    fn is_forbidden(&self, action: usize) -> bool {
        todo!()
    }

    fn step(&mut self, action: usize) {
        assert_eq!(self.is_terminal(), false);
        assert_eq!(self.available_action().iter().any(|&x| x == action), true);

        match action {
            0 => self.agent_pos -= 1,
            1 => self.agent_pos += 1,
            _ => {}
        }
    }

    fn delete(&mut self) {
        todo!()
    }

    fn score(&self) -> f32 {
        match self.agent_pos {
            0 => -1.0,
            4 => 1.0,
            _ => 0.0
        }
    }

    fn display(&self) {
        for i in 0..5 {
            match self.agent_pos {
                x if x == i => print!("X"),
                _ => print!("_")
            }
        }
        println!()
    }

    fn play_strategy(&mut self, strategy: HashMap<usize, usize>) {
        self.display();
        loop {
            if self.is_terminal() {
                println!("Terminal, OVER");
                break;
            }
            let action = strategy.get(&self.agent_pos);
            if action.is_none() {
                println!("Action not found.");
                break;
            }
            self.step(*action.unwrap());
            self.display();
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_random_state() {
        let seed: u64 = 42;
        let rng = StdRng::seed_from_u64(seed).gen_range(1..4);
        let mut lw = LineWorld::new();
        lw.reset_random_state(seed);
        println!("position : {}, rng : {}", lw.agent_pos, rng);

        assert_eq!(lw.agent_pos, rng, "With seed {}, expected pos {}", seed, lw.agent_pos)
    }

    #[test]
    fn test_new() {
        let lw = LineWorld::new();
        assert_eq!(lw.agent_pos, 2, "should be 2, {} find instead", lw.agent_pos)
    }

    #[test]
    fn test_available_action() {
        let lw = LineWorld::new();
        let lw2 = LineWorld { agent_pos: 0, transition_probability_matrix: LineWorld::build_transition_matrix() };
        let lw3 = LineWorld { agent_pos: 4, transition_probability_matrix: LineWorld::build_transition_matrix() };

        assert_eq!(lw.available_action(), array![0, 1], "should be [0, 1], found [] instead");
        assert_eq!(lw2.available_action(), array![], "should be [], found [0, 1] instead");
        assert_eq!(lw3.available_action(), array![], "should be [], found [0, 1] instead");
        assert_eq!(lw.available_action().iter().any(|&x| x == 0), true)
    }

    #[test]
    fn test_line_world() {
        let mut lw = LineWorld::new();

        lw.display();
        lw.step(0);
        lw.display();
        lw.step(1);
        lw.display();
        lw.step(1);
        lw.display();

        assert_eq!(lw.state_id(), 3)
    }
}