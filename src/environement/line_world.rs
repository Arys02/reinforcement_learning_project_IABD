pub mod line_world {
    extern crate rand;

    use ndarray::{array, Array1, Array4, ArrayBase, Ix4, OwnedRepr};
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::environement::environment_traits::{BaseEnv, Environment};
    pub const NUM_ACTIONS: usize = 5;
    pub const NUM_STATES: usize = 2;
    pub const NUM_REWARDS: usize = 3;

    // The `LineWorld` struct represents a linear environment where an agent can navigate along a vector of fixed size.
    /// The agent starts at the middle of the vector and can move left or right.
    ///
    /// Characteristics of `LineWorld`:
    /// - **Vector Size**: The environment is a vector of 5 positions.
    /// - **Starting Position**: The agent starts at the position 2 (middle of the vector).
    /// - **Actions**: The agent has 2 possible actions: move left and move right.
    /// - **Terminal States and Rewards**:
    ///   - Reaching position 4 results in a terminal state with a reward of 1.
    ///   - Reaching position 0 results in a terminal state with a reward of -1.
    ///
    /// The `LineWorld` struct implements the `Environment` trait, providing methods for managing the agent's state,
    /// executing actions, and calculating rewards and transition probabilities.
    ///
    #[derive(Clone, Debug)]
    pub struct LineWorld {
        agent_pos: usize,
    }

    impl LineWorld {
        fn build_transition_matrix() -> ArrayBase<OwnedRepr<f32>, Ix4> {
            let mut transition_probability_matrix = Array4::zeros((
                Self::num_states(),
                Self::num_actions(),
                Self::num_states(),
                Self::num_rewards(),
            ));

            for s in 0..Self::num_states() {
                for a in 0..Self::num_actions() {
                    for s_p in 0..Self::num_states() {
                        for r in 0..Self::num_rewards() {
                            if s < Self::num_states()
                                && s_p == s + 1
                                && a == 1
                                && Self::get_reward(r) == 0.
                                && (s == 1 || s == 2)
                            {
                                transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                            }
                            if s > 0
                                && s_p == s - 1
                                && a == 0
                                && Self::get_reward(r) == 0.
                                && (s == 2 || s == 3)
                            {
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

    impl Default for LineWorld {
        fn default() -> Self {
            LineWorld {
                agent_pos: 2,
            }
        }
    }

    impl BaseEnv for LineWorld {
        fn is_terminal(&self) -> bool {
            match self.agent_pos {
                1 | 2 | 3 => false,
                _ => true,
            }
        }
        fn score(&self) -> f32 {
            match self.agent_pos {
                0 => -1.0,
                4 => 1.0,
                _ => 0.0,
            }
        }
        fn reset(&mut self) {
            self.agent_pos = 2
        }
    }

    impl Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS> for LineWorld {
        fn state_id(&self) -> usize {
            self.agent_pos
        }

        fn from_random_state() -> Self {
            let mut rng = rand::thread_rng();
            let agent_pos_: usize = rng.gen_range(1..4);
            LineWorld {
                agent_pos: agent_pos_,
            }
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
                _ => panic!("reward is out of range"),
            }
        }

        fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
            let tm = Self::build_transition_matrix();
            return tm[[s, a, s_p, r]];
        }


        fn reset_random_state(&mut self, seed: u64) {
            let mut rng = StdRng::seed_from_u64(seed);
            let agent_pos_: usize = rng.gen_range(1..4);
            self.agent_pos = agent_pos_
        }

        fn available_actions_ids(&self) -> Array1<usize> {
            match self.agent_pos {
                1 | 2 | 3 => array![0, 1],
                _ => array![],
            }
        }

        fn available_action_delete(&self) {
            todo!()
        }



        fn step(&mut self, action: usize) {
            assert_eq!(self.is_terminal(), false);
            assert_eq!(self.available_actions_ids().iter().any(|&x| x == action), true);

            match action {
                0 => self.agent_pos -= 1,
                1 => self.agent_pos += 1,
                _ => {}
            }
        }

        fn delete(&mut self) {
            todo!()
        }



        fn display(&self) {
            for i in 0..5 {
                match self.agent_pos {
                    x if x == i => print!("X"),
                    _ => print!("_"),
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
            let mut lw = LineWorld::default();
            lw.reset_random_state(seed);
            println!("position : {}, rng : {}", lw.agent_pos, rng);

            assert_eq!(
                lw.agent_pos, rng,
                "With seed {}, expected pos {}",
                seed, lw.agent_pos
            )
        }

        #[test]
        fn test_new() {
            let lw = LineWorld::default();
            assert_eq!(
                lw.agent_pos, 2,
                "should be 2, {} find instead",
                lw.agent_pos
            )
        }

        #[test]
        fn test_available_action() {}

        #[test]
        fn test_line_world() {
            let mut lw = LineWorld::default();

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
}