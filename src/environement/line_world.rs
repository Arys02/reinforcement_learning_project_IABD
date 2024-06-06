extern crate rand;

use ndarray::{array, Array1};
use rand::{Rng, SeedableRng};
use rand::prelude::StdRng;
use crate::environement::environement::Environement;

pub struct LineWorld {
    agent_pos: i32,
}

impl Environement for LineWorld {
    fn reset_random_state(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let agent_pos_: i32 = rng.gen_range(1..4);
        self.agent_pos = agent_pos_
    }

    fn new() -> Self {
        LineWorld { agent_pos: 2 }
    }


    fn available_action(&self) -> Array1<i32> {
        match self.agent_pos {
            1 | 2 | 3 => array![0, 1],
            _ => array![]
        }
    }


    fn is_terminal(&self) -> bool {
        match self.agent_pos {
            1 | 2 | 3 => false,
            _ => true
        }
    }

    fn state_id(&self) -> i32 {
        self.agent_pos
    }

    fn step(&mut self, action: i32) {
        assert_eq!(self.is_terminal(), false);
        assert_eq!(self.available_action().iter().any(|&x| x == action), true);

        match action {
            0 => self.agent_pos -= 1,
            1 => self.agent_pos += 1,
            _ => {}
        }
    }

    fn score(&self) -> f64 {
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
        let lw2 = LineWorld { agent_pos: 0 };
        let lw3 = LineWorld { agent_pos: 4 };

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

        assert_eq!(lw.state_id(), 1)
    }
}