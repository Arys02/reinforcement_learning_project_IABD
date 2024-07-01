extern crate rand;

use std::ops::Add;

use ndarray::{array, Array1, Array4, ArrayBase, Ix4, OwnedRepr};
use ndarray_rand::rand::SeedableRng;
use rand::prelude::StdRng;
use rand::Rng;

use crate::environement::environment::Environment;

pub struct GridWorld {
    agent_pos: i32,

    transition_probability_matrix: ArrayBase<OwnedRepr<f64>, Ix4>,
}

impl GridWorld {
    fn build_transition_matrix() -> ArrayBase<OwnedRepr<f64>, Ix4> {
        let mut transition_probability_matrix = Array4::zeros((Self::num_states(), Self::num_actions(), Self::num_states(), Self::num_rewards()));

        for s in 0..Self::num_states() {
            for a in 0..Self::num_actions() {
                for s_p in 0..Self::num_states() {
                    for r in 0..Self::num_rewards() {
                        if s > 0 && s_p == s - 1 && a == 0 && Self::get_reward(r) == 0. && s % 5 != 0 && s != 0 && s != 4 && s != 24 {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                        if s < Self::num_states() && s_p == s + 1 && a == 1 && Self::get_reward(r) == 0. && s % 5 == 4 {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }

                        if s < Self::num_states() - 5 && s_p == s + 5 && a == 2 && Self::get_reward(r) == 0. && s != 4 && s < 20 {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }

                        if s > 5 && s_p == s - 5 && a == 3 && Self::get_reward(r) == 0. && s != 24 && s > 4 {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                    }
                }
            }
        }
        transition_probability_matrix[[3, 1, 4, 0]] = 1.0;
        transition_probability_matrix[[9, 3, 4, 0]] = 1.0;

        transition_probability_matrix[[23, 1, 24, 3]] = 1.0;
        transition_probability_matrix[[19, 2, 24, 3]] = 1.0;

        return transition_probability_matrix
    }
}

impl Environment for GridWorld {
    fn new() -> Self {
        GridWorld {
            agent_pos: 0,
            transition_probability_matrix: Self::build_transition_matrix()
        }
    }

    fn state_id(&self) -> usize {
        self.agent_pos as usize
    }

    fn from_random_state() -> Self {
        let mut rng = rand::thread_rng();
        //TODO update for better values

        let mut agent_pos: i32 = rng.gen_range(0..24);

        while agent_pos == 4 || agent_pos == 24 {
            agent_pos = rng.gen_range(0..24);
        }

        return GridWorld {
            agent_pos,
            transition_probability_matrix: Self::build_transition_matrix()
        }
    }

    fn reset(&mut self) {
        self.agent_pos = 0;
    }

    fn num_states() -> usize {
        25
    }

    fn num_actions() -> usize {
        4
    }

    fn num_rewards() -> usize {
        4
    }

    fn get_reward(i: usize) -> f32 {
        match i {
            0 => -3.,
            1 => -1.,
            2 => 0.,
            3 => 1.,
            _ => panic!("reward out of range")
        }
    }

    fn get_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        Self::get_transition_probability(s, a, s_p, r)
    }

    fn reset_random_state(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut agent_pos: i32 = rng.gen_range(0..24);
        while agent_pos == 4 || agent_pos == 24 {
            agent_pos = rng.gen_range(0..24);
        }
        self.agent_pos = agent_pos
    }

    fn available_action(&self) -> Array1<usize> {
        let mut array_vec: Vec<usize> = Vec::new();
        if self.agent_pos % 5 != 0 {
            array_vec.push(0);
        }
        if self.agent_pos % 5 != 4 {
            array_vec.push(1);
        }
        if self.agent_pos < 20 {
            array_vec.push(2)
        }
        if self.agent_pos > 5 {
            array_vec.push(3)
        }

        //    [g, d, b, h]
        Array1::from(array_vec)
        //return array![0, 1, 2, 3]
    }

    fn available_action_delete(&self) {
        todo!()
    }

    fn is_terminal(&self) -> bool {
        match self.agent_pos {
            4 | 24 => true,
            x if x < 0 => true,
            x if x > 24 => true,
            _ => false
        }
    }

    fn is_forbidden(&self, action: usize) -> bool {
        if self.agent_pos % 5 == 0 && action == 0 {
            return true
        }
        if (self.agent_pos % 5) == 4 && action == 1 {
            return true
        }
        if self.agent_pos >= 20 && action == 2 {
            return true
        }
        if self.agent_pos < 5 && action == 3 {
            return true
        }
        false
    }

    fn step(&mut self, action: usize) {
        assert_eq!(self.is_terminal(), false);
        assert_eq!(self.available_action().iter().any(|&x| x == action), true);

        match action {
            0 => self.agent_pos -= 1,
            1 => self.agent_pos += 1,
            2 => self.agent_pos += 5,
            3 => self.agent_pos -= 5,
            _ => {}
        }
    }

    fn delete(&mut self) {
        todo!()
    }

    fn score(&self) -> f64 {
        match self.agent_pos {
            4 => -3.0,
            24 => 3.0,
            0..=23 => 0.0,
            _ => -1.0
        }
    }

    fn display(&self) {
        for j in 0..5 {
            for i in 0..5 {
                match self.agent_pos {
                    x if x % 5 == i && (x / 5 == j) => print!("X"),
                    _ => print!("_")
                }
            }
            println!()
        }
        println!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_random_state() {
        let seed: u64 = 42;
        let rng = StdRng::seed_from_u64(seed).gen_range(0..24);
        let mut lw = GridWorld::new();
        lw.reset_random_state(seed);
        println!("position : {}, rng : {}", lw.agent_pos, rng);

        assert_eq!(lw.agent_pos, rng, "With seed {}, expected pos {}", seed, lw.agent_pos)
    }

    #[test]
    fn test_new() {
        let lw = GridWorld::new();
        assert_eq!(lw.agent_pos, 0, "should be 0, {} find instead", lw.agent_pos)
    }

    #[test]
    fn test_available_action() {
        let gw = GridWorld::new();

        assert_eq!(gw.available_action(), array![1, 2], "should be [1, 2], found [] instead");
    }

    #[test]
    fn test_line_world() {
        let mut gw = GridWorld::new();

        gw.display();
        gw.step(1);
        gw.display();
        gw.step(2);
        gw.display();
        gw.step(0);
        gw.display();
        gw.step(3);
        gw.display();


        assert_eq!(gw.state_id(), 0)
    }
}
