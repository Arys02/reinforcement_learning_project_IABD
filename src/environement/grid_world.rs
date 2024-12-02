pub mod grid_world {
    use ndarray::{Array4, ArrayBase, Ix4, OwnedRepr};
    use ndarray_rand::rand::SeedableRng;
    use rand::prelude::StdRng;
    use rand::Rng;

    use crate::environement::environment_traits::{
        ActionEnv, BaseEnv, DeepDiscreteActionsEnv, Environment,
    };
    pub const NUM_ACTIONS: usize = 4;
    pub const NUM_STATES: usize = 49;
    pub const NUM_STATE_FEATURES: usize = 49;
    pub const NUM_REWARDS: usize = 4;

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
    ///
    #[derive(Clone, Debug)]
    pub struct GridWorld {
        agent_pos: usize,
    }

    impl GridWorld {
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
                            // cas aller à gauche valide
                            if s > 0
                                && s % 7 >= 2
                                && s % 7 <= 5
                                && s < 40
                                && s > 7
                                && s_p == s - 1
                                && a == 0
                                && Self::get_reward(r) == 0.
                                && s != 12
                            {
                                transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                            }
                            // cas aller à gauche out of bound
                            if s > 0
                                && s % 7 == 1
                                && s < 41
                                && s > 7
                                && s_p == s - 1
                                && a == 0
                                && Self::get_reward(r) == -1.
                            {
                                transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                            }
                            // cas aller à droite valide
                            if s > 0
                                && s % 7 >= 1
                                && s % 7 <= 4
                                && s < 40
                                && s > 7
                                && s_p == s + 1
                                && a == 1
                                && Self::get_reward(r) == 0.
                                && s != 12
                            {
                                transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                            }
                            // cas aller à droite out of bound
                            if s > 0
                                && s % 7 == 5
                                && s < 40
                                && s > 7
                                && s_p == s + 1
                                && a == 1
                                && Self::get_reward(r) == -1.
                                && s != 12
                            {
                                transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                            }
                            // cas aller en bas valide
                            if s > 0
                                && s % 7 >= 1
                                && s % 7 <= 5
                                && s < 34
                                && s > 7
                                && s_p == s + 7
                                && a == 2
                                && Self::get_reward(r) == 0.
                                && s != 12
                                && s != 40
                            {
                                transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                            }
                            // cas aller à bas out of bound
                            if s > 0
                                && s < 40
                                && s > 35
                                && s_p == s + 7
                                && a == 2
                                && Self::get_reward(r) == -1.
                            {
                                transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                            }
                            // cas aller en haut valide
                            if s > 0
                                && s % 7 >= 1
                                && s % 7 <= 5
                                && s < 40
                                && s > 14
                                && s_p == s - 7
                                && a == 3
                                && Self::get_reward(r) == 0.
                            {
                                transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                            }
                            // cas aller à haut out of bound
                            if s > 0
                                && s < 7
                                && s > 12
                                && s_p == s - 7
                                && a == 3
                                && Self::get_reward(r) == -1.
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

            return transition_probability_matrix;
        }
    }

    impl Default for GridWorld {
        fn default() -> Self {
            GridWorld { agent_pos: 8 }
        }
    }

    impl BaseEnv for GridWorld {
        fn get_name(&self) -> String {
            String::from("gridworld")
        }

        fn is_terminal(&self) -> bool {
            match self.agent_pos {
                x if x < 8 => true,
                x if x > 40 => true,
                x if x % 7 == 0 => true,
                x if x % 7 == 6 => true,
                12 | 40 => true,
                _ => false,
            }
        }

        fn score(&self) -> f32 {
            match self.agent_pos {
                12 => -3.0,
                40 => 3.0,
                x if x > 7 && x < 40 && x % 7 >= 1 && x % 7 <= 5 => 0.0,
                _ => -1.0,
            }
        }

        fn reset(&mut self) {
            self.agent_pos = 8;
        }
    }

    impl ActionEnv<NUM_ACTIONS> for GridWorld {
        fn available_actions_ids(&self) -> impl Iterator<Item = usize> {
            0..4
        }

        fn step(&mut self, action: usize) {
            assert_eq!(self.is_terminal(), false);
            assert_eq!(self.available_actions_ids().any(|x| x == action), true);

            match action {
                0 => self.agent_pos -= 1,
                1 => self.agent_pos += 1,
                2 => self.agent_pos += 7,
                3 => self.agent_pos -= 7,
                _ => {}
            }
        }
    }

    impl DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> for GridWorld {
        fn state_description(&self) -> [f32; NUM_STATE_FEATURES] {
            std::array::from_fn(|idx| if self.agent_pos == idx { 1.0 } else { 0.0 })
        }

        fn action_mask(&self) -> [f32; NUM_ACTIONS] {
            std::array::from_fn(|idx| {
                if self.agent_pos > 7
                    && self.agent_pos < 40
                    && self.agent_pos % 7 != 0
                    && self.agent_pos % 7 != 6
                    && self.agent_pos != 12
                {
                    1.0
                } else {
                    0.0
                }
            })
        }
    }
    impl Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS> for GridWorld {
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

            GridWorld { agent_pos }
        }

        fn num_states() -> usize {
            NUM_STATES
        }

        fn num_actions() -> usize {
            NUM_ACTIONS
        }

        fn num_rewards() -> usize {
            NUM_REWARDS
        }

        fn get_reward(i: usize) -> f32 {
            match i {
                0 => -3.,
                1 => -1.,
                2 => 0.,
                3 => 1.,
                _ => panic!("reward out of range"),
            }
        }

        fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
            let matrix = Self::build_transition_matrix();
            matrix[[s, a, s_p, r]]
        }

        fn reset_random_state(&mut self, seed: u64) {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut agent_pos: usize = rng.gen_range(8..40);
            while agent_pos == 12 || agent_pos % 7 == 0 || agent_pos % 7 == 6 {
                agent_pos = rng.gen_range(8..40);
            }
            self.agent_pos = agent_pos
        }

        fn available_action_delete(&self) {
            todo!()
        }

        fn delete(&mut self) {
            todo!()
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
        fn run_with_player() {}

        #[test]
        fn count_run() {}
    }
}
