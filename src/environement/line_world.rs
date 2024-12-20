pub mod line_world {
    extern crate rand;

    use ndarray::{Array4, ArrayBase, Ix4, OwnedRepr};
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::environement::environment_traits::{
        ActionEnv, BaseEnv, DeepDiscreteActionsEnv, Environment,
    };
    pub const NUM_ACTIONS: usize = 2;
    pub const NUM_STATES: usize = 5;
    pub const NUM_STATE_FEATURES: usize = 5;
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
            LineWorld { agent_pos: 2 }
        }
    }

    impl BaseEnv for LineWorld {
        fn get_name(&self) -> String {
            "LineWorld".to_string()
        }

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

    impl ActionEnv<NUM_ACTIONS> for LineWorld {
        fn available_actions_ids(&self) -> impl Iterator<Item = usize> {
            match self.agent_pos {
                1 | 2 | 3 => 0..=1,
                _ => 0..=0,
            }
        }
        fn step(&mut self, action: usize) {
            assert_eq!(self.is_terminal(), false);
            assert_eq!(self.available_actions_ids().any(|x| x == action), true);

            match action {
                0 => self.agent_pos -= 1,
                1 => self.agent_pos += 1,
                _ => {}
            }
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

        fn available_action_delete(&self) {
            todo!()
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

    impl DeepDiscreteActionsEnv<NUM_STATES, NUM_ACTIONS> for LineWorld {
        fn state_description(&self) -> [f32; NUM_STATES] {
            std::array::from_fn(|idx| if self.agent_pos == idx { 1.0 } else { 0.0 })
        }

        fn action_mask(&self) -> [f32; NUM_ACTIONS] {
            std::array::from_fn(|idx| {
                if (self.agent_pos == 1 || self.agent_pos == 2 || self.agent_pos == 3) {
                    1.0
                } else {
                    0.0
                }
            })
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn run_with_player() {}

        #[test]
        fn count_run() {}
    }
}
