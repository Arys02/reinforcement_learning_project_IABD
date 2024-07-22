extern crate rand;

use std::collections::HashMap;

use ndarray::{array, Array1, Array4, ArrayBase, Ix4, OwnedRepr};
use ndarray_rand::rand::SeedableRng;
use rand::prelude::StdRng;
use rand::Rng;

use crate::environement::environment::Environment;

pub struct MontyHall1{
    state: usize,
    transition_probability_matrix: ArrayBase<OwnedRepr<f32>, Ix4>,
    agent_pos : usize,
    door_win : usize
}

impl MontyHall1 {
    fn build_transition_matrix() -> ArrayBase<OwnedRepr<f32>, Ix4> {
        let mut transition_probability_matrix = Array4::zeros((Self::num_states(), Self::num_actions(), Self::num_states(), Self::num_rewards()));

        transition_probability_matrix[[0, 0, 1, 0]] = 1.0;
        transition_probability_matrix[[0, 1, 2, 0]] = 1.0;
        transition_probability_matrix[[0, 2, 3, 0]] = 1.0;

        transition_probability_matrix[[1, 3, 4, 1]] = 1./ 3.;
        transition_probability_matrix[[1, 3, 4, 0]] = 2./ 3.;
        transition_probability_matrix[[1, 4, 5, 1]] = 2./ 3.;
        transition_probability_matrix[[1, 4, 5, 0]] = 1./ 3.;


        return transition_probability_matrix;
    }
}

impl Environment for MontyHall1{
    fn new() -> Self {
        //let mut rng = StdRng::seed_from_u64(42);

        let mut rng = rand::thread_rng();
        let mut dore_win : usize = rng.gen_range(1..=3);
        MontyHall1 {
            state: 0,
            agent_pos: 0,
            door_win: dore_win,
            transition_probability_matrix: Self::build_transition_matrix(),
        }
    }

    fn state_id(&self) -> usize {
        self.state as usize
    }

    fn available_actions(state: usize) -> Array1<usize> {
        if state == 0 {
            array![0, 1, 2]
        }
        else {
            array![3, 4]
        }
    }

    fn is_terminal_state(state: usize) -> bool {
        match state {
            4 => true,
            5 => true,
            _ => false
        }
    }

    fn from_random_state() -> Self {
        let mut rng = rand::thread_rng();
        let mut agent_pos: usize = rng.gen_range(0..=3);
        let mut door_win : usize = rng.gen_range(1..=3);

        return MontyHall1{
            state: agent_pos,
            agent_pos,
            door_win,
            transition_probability_matrix: Self::build_transition_matrix(),
        };
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        let mut dore_win : usize = rng.gen_range(1..=3);
        self.door_win = dore_win;
        self.state = 0;
    }

    fn num_states() -> usize {
        6
    }

    fn num_actions() -> usize {
        5
    }

    fn num_rewards() -> usize {
        2
    }

    fn get_reward(i: usize) -> f32 {
        match i {
            0 => 0.,
            1 => 1.,
            _ => panic!("reward out of range")
        }
    }

    fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        let matrix = Self::build_transition_matrix();
        return matrix[[s, a, s_p, r]];
    }
    fn get_transition_probability(&mut self, s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        return self.transition_probability_matrix[[s, a, s_p, r]];
    }

    fn reset_random_state(&mut self, seed: u64) {
        let mut rng = rand::thread_rng();
        let agent_pos: usize = rng.gen_range(0..=3);
        let dore_win : usize = rng.gen_range(1..=3);
        self.door_win = dore_win;
        self.state = agent_pos
    }

    fn available_action(&self) -> Array1<usize> {
        return Self::available_actions(self.state);
    }

    fn available_action_delete(&self) {
        todo!()
    }

    fn is_terminal(&self) -> bool {
        Self::is_terminal_state(self.state)
    }

    fn is_forbidden(&self, action: usize) -> bool {
        if self.state > 0 && vec![0, 1, 2].contains(&action) {
            return true;
        }
        false
    }

    fn step(&mut self, action: usize) {
        assert_eq!(self.is_terminal(), false);
        assert_eq!(self.available_action().iter().any(|&x| x == action), true);

        if self.state == 0 {
            self.state = action + 1;
            self.agent_pos = action + 1;
        }
        else {
            self.state = if action == 3 { 4 } else {
                if self.agent_pos != self.door_win {
                    self.agent_pos = self.door_win
                }
                else {
                    self.agent_pos += 1 % 3
                }
               5
            }
        }

    }

    fn delete(&mut self) {
        todo!()
    }

    fn score(&self) -> f32 {
        if self.state < 0 {
            return 0.
        }
        if self.agent_pos == self.door_win {
            return 1.
        }
        return 0.
    }

    fn display(&self) {

        println!("--------------");
        if self.state == 0 {
            println!("   |    1    |");
            println!("   |    2    |");
            println!("   |    3    |");
        }
        if self.state == 1 {
            let mut rng = rand::thread_rng();
            let rand : usize = rng.gen_range(0..=1);
            if rand == 0 {
                println!("-> |    1    |");
                println!("   |   goat  |");
                println!("   |    3    |");
            }
            else {
                println!("-> |    1    |");
                println!("   |    2    |");
                println!("   |   goat  |");
            }
        }
        if self.state == 2 {
            let mut rng = rand::thread_rng();
            let rand : usize = rng.gen_range(0..=1);
            if rand == 0 {
                println!("   |   goat  |");
                println!("-> |    2    |");
                println!("   |    3    |");
            }
            else {
                println!("   |   goat  |");
                println!("-> |    2    |");
                println!("   |    3    |");
            }
        }
        if self.state == 3 {
            let mut rng = rand::thread_rng();
            let rand : usize = rng.gen_range(0..=1);
            if rand == 0 {
                println!("   |   goat  |");
                println!("-> |    2    |");
                println!("   |    3    |");
            }
            else {
                println!("   |    1    |");
                println!("-> |    2    |");
                println!("   |   goat  |");
            }
        }
        if self.state > 3 {
            for i in 1..=3 {
                if self.agent_pos == i {
                    print!("-> ")
                }
                else {
                    print!("   ")
                }
                if self.door_win == i {
                    print!("| treasure |")
                }
                else {
                    print!("|   goat   |");
                }
                println!();
            }

            if self.door_win == self.agent_pos {
                println!("Victory");
            }
            println!();
        }
        println!();
    }

    fn play_strategy(&mut self, strategy: HashMap<usize, usize>) {
        //self.display();
        loop {
            if self.is_terminal() {
                //println!("Terminal, OVER");
                break;
            }
            let action = strategy.get(&self.state);
            if action.is_none() {
                //println!("Action not found.");
                break;
            }
            self.step(*action.unwrap());
            //self.display();
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::distributions::WeightedError::TooMany;
    use crate::environement::two_round_rps;
    use super::*;

    #[test]
    fn test_new() {
        let lw = MontyHall1::new();
        assert_eq!(lw.state, 0, "should be 0, {} find instead", lw.state)
    }

    #[test]
    fn test_available_action() {
        let gw = MontyHall1::new();

        assert_eq!(gw.available_action(), array![0, 1, 2], "should be [0, 1, 2], found [] instead");
    }

    #[test]
    fn test_monty_hall_1() {
        let mut gw = MontyHall1::new();

        gw.display();
        gw.step(1);
        gw.display();
        gw.step(3);
        gw.display();
        println!("{}", gw.state_id());
        println!("winning door {}", gw.door_win);
        println!("agent pos {}", gw.agent_pos);


        assert_eq!(gw.state_id(), 4)
    }


}
