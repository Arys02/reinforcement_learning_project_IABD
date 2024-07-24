extern crate rand;

use std::collections::HashMap;

use ndarray::{array, Array1, Array4, ArrayBase, Ix4, OwnedRepr};
use ndarray_rand::rand::SeedableRng;
use rand::prelude::StdRng;
use rand::Rng;

use crate::environement::environment::Environment;

/// The `GridWorld` struct represents a grid-based environment where an agent can navigate a 5x5 grid.
/// The agent starts at the top-left corner of the grid and can move left, right, up, or down.
///
/// Characteristics of `GridWorld`:
/// - **Grid Size**: The grid consists of 5 rows and 5 columns.
/// - **Starting Position**: The agent typically starts at the first row, first column (top-left corner).
/// - **Actions**: The agent has 4 possible actions: move left, move right, move up, and move down.
/// - **Terminal States and Rewards**:
///   - Reaching the last cell of the first row (top-right corner) results in a terminal state with a reward of -3.
///   - Reaching the last cell of the last row (bottom-right corner) results in a terminal state with a reward of 1.
///   - Attempting to move outside the grid boundaries results in a terminal state with a score of -1.
///
/// The `GridWorld` struct implements the `Environment` trait, providing methods for managing the agent's state,
/// executing actions, and calculating rewards and transition probabilities.
pub struct GridWorld {
    agent_pos: usize,

    transition_probability_matrix: ArrayBase<OwnedRepr<f32>, Ix4>,
}

impl GridWorld {
    fn build_transition_matrix() -> ArrayBase<OwnedRepr<f32>, Ix4> {
        let mut transition_probability_matrix = Array4::zeros((Self::num_states(), Self::num_actions(), Self::num_states(), Self::num_rewards()));

        for s in 0..Self::num_states() {
            for a in 0..Self::num_actions() {
                for s_p in 0..Self::num_states() {
                    for r in 0..Self::num_rewards() {
                        // cas aller à gauche valide
                        if s > 0 && s % 7 >= 2 && s % 7 <= 5 && s < 40 && s > 7 && s_p == s - 1 && a == 0 && Self::get_reward(r) == 0. && s != 12
                        {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                        // cas aller à gauche out of bound
                        if s > 0 && s % 7 == 1 && s < 41 && s > 7 && s_p == s - 1 && a == 0 && Self::get_reward(r) == -1.
                        {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                        // cas aller à droite valide
                        if s > 0 && s % 7 >= 1 && s % 7 <= 4 && s < 40 && s > 7 && s_p == s + 1 && a == 1 && Self::get_reward(r) == 0. && s != 12
                        {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                        // cas aller à droite out of bound
                        if s > 0 && s % 7 == 5 && s < 40 && s > 7 && s_p == s + 1 && a == 1 && Self::get_reward(r) == -1. && s != 12
                        {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                        // cas aller en bas valide
                        if s > 0 && s % 7 >= 1 && s % 7 <= 5 && s < 34 && s > 7 && s_p == s + 7 && a == 2 && Self::get_reward(r) == 0. && s != 12 && s != 40
                        {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                        // cas aller à bas out of bound
                        if s > 0 && s < 40 && s > 35 && s_p == s + 7 && a == 2 && Self::get_reward(r) == -1.
                        {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                        // cas aller en haut valide
                        if s > 0 && s % 7 >= 1 && s % 7 <= 5 && s < 40 && s > 14 && s_p == s - 7 && a == 3 && Self::get_reward(r) == 0.
                        {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                        // cas aller à haut out of bound
                        if s > 0 && s < 7 && s > 12 && s_p == s - 7 && a == 3 && Self::get_reward(r) == -1.
                        {
                            transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                        }
                    }
                }
            }
        }
        transition_probability_matrix[[4, 1, 5, 0]] = 1.0;
        transition_probability_matrix[[19, 3, 12, 0]] = 1.0;

        transition_probability_matrix[[39, 1, 40, 3]] = 1.0;
        transition_probability_matrix[[33, 2, 40, 3]] = 1.0;

        return transition_probability_matrix
    }
}

impl Environment for GridWorld {
    fn new() -> Self {
        GridWorld {
            agent_pos: 8,
            transition_probability_matrix: Self::build_transition_matrix()
        }
    }

    fn state_id(&self) -> usize {
        self.agent_pos as usize
    }


    fn from_random_state() -> Self {
        let mut rng = rand::thread_rng();
        //TODO update for better values

        let mut agent_pos: usize = rng.gen_range(8..40);

        while agent_pos == 12 || agent_pos % 7 == 0 || agent_pos % 7 == 6 {
            agent_pos = rng.gen_range(8..40);
        }

        return GridWorld {
            agent_pos,
            transition_probability_matrix: Self::build_transition_matrix()
        }
    }

    fn reset(&mut self) {
        self.agent_pos = 8;
    }

    fn num_states() -> usize {
        49
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

    fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        let matrix = Self::build_transition_matrix();
        return matrix[[s, a, s_p, r]]
    }
    fn get_transition_probability(&mut self, s: usize, a: usize, s_p: usize, r: usize) -> f32 {
        return self.transition_probability_matrix[[s, a, s_p, r]]
    }

    fn reset_random_state(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut agent_pos: usize = rng.gen_range(8..40);
        while agent_pos == 12 || agent_pos % 7 == 0 || agent_pos % 7 == 6 {
            agent_pos = rng.gen_range(8..40);
        }
        self.agent_pos = agent_pos
    }

    fn available_action(&self) -> Array1<usize> {
        return array![0, 1, 2, 3];
    }

    fn available_action_delete(&self) {
        todo!()
    }

    fn is_terminal(&self) -> bool {
        match self.agent_pos {
            x if x < 8 => true,
            x if x > 40 => true,
            x if x % 7 == 0 => true,
            x if x % 7 == 6 => true,
            12 | 40 => true,
            _ => false
        }
    }

    fn is_forbidden(&self, action: usize) -> bool {
        if self.agent_pos % 7 == 0 && action == 0 {
            return true
        }
        if (self.agent_pos % 7) == 6 && action == 1 {
            return true
        }
        if self.agent_pos >= 40 && action == 2 {
            return true
        }
        if self.agent_pos < 7 && action == 3 {
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
            2 => self.agent_pos += 7,
            3 => self.agent_pos -= 7,
            _ => {}
        }
    }

    fn delete(&mut self) {
        todo!()
    }

    fn score(&self) -> f32 {
        match self.agent_pos {
            12 => -3.0,
            40 => 3.0,
            x if x > 7 && x < 40 && x % 7 >= 1 && x % 7 <= 5 => 0.0,
            _ => -1.0
        }
    }

    fn display(&self) {
        for j in 0..7 {
            for i in 0..7 {
                if self.agent_pos % 7 == i && (self.agent_pos / 7 == j) {
                    print!("X")
                } else if (i + 7 * j) % 7 == 0 {
                    print!("|")
                } else if (i + 7 * j) % 7 == 6 {
                    print!("|")
                } else if (i + 7 * j) < 7 {
                    print!("_")
                } else if (i + 7 * j) > 40 {
                    print!("_")
                } else {
                    print!(".")
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
        let rng = StdRng::seed_from_u64(seed).gen_range(8..40);
        let mut lw = GridWorld::new();
        lw.reset_random_state(seed);
        println!("position : {}, rng : {}", lw.agent_pos, rng);

        assert_eq!(lw.agent_pos, rng, "With seed {}, expected pos {}", seed, lw.agent_pos)
    }

    #[test]
    fn test_new() {
        let lw = GridWorld::new();
        assert_eq!(lw.agent_pos, 8, "should be 8, {} find instead", lw.agent_pos)
    }

    #[test]
    fn test_available_action() {
        let gw = GridWorld::new();

        assert_eq!(gw.available_action(), array![0, 1, 2, 3], "should be [1, 2], found [] instead");
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


        assert_eq!(gw.state_id(), 8)
    }

    #[test]
    fn test_grid_world_strategy() {
        let strategy: HashMap<usize, usize> = HashMap::from([
            (8, 1),
            (9, 2),
            (16, 2),
            (23, 2),
            (30, 1),
            (31, 1),
            (32, 1),
            (33, 2),
        ]);
        let mut gw = GridWorld::new();
        gw.play_strategy(strategy, false);

        assert_eq!(gw.state_id(), 40)
    }
}
