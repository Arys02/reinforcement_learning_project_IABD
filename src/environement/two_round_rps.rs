pub mod two_round_rps {
    extern crate rand;

    use ndarray::{Array4, ArrayBase, Ix4, OwnedRepr};
    use ndarray_rand::rand::SeedableRng;
    use rand::prelude::StdRng;
    use rand::Rng;

    use crate::environement::environment_traits::{ActionEnv, BaseEnv, Environment};

    pub const NUM_ACTIONS: usize = 19;
    pub const NUM_STATES: usize = 3;
    pub const NUM_REWARDS: usize = 3;
    /// The `TwoRoundRPS` struct represents an environment where an agent plays a two-round game of Rock-Paper-Scissors (RPS) against an opponent with a particular strategy.
    /// The opponent plays randomly in the first round but in the second round, it always mirrors the agent's choice from the first round.
    ///
    /// Characteristics of `TwoRoundRPS`:
    /// - **Rounds**: The game consists of two rounds of Rock-Paper-Scissors.
    /// - **Opponent Strategy**:
    ///   - In the first round, the opponent's choice is random.
    ///   - In the second round, the opponent plays the exact choice made by the agent in the first round.
    /// - **Rewards**:
    ///   - Winning a round results in a reward of +1.
    ///   - Losing a round results in a reward of -1.
    ///   - A tie results in a reward of 0.
    ///
    /// The `TwoRoundRPS` struct implements the `Environment` trait, providing methods for managing the agent's state,
    /// executing actions, and calculating rewards and transition probabilities.
    ///
    #[derive(Clone, Debug)]
    pub struct TwoRoundRPS {
        agent_pos: usize,
    }

    impl TwoRoundRPS {
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
                            if s == 0 && s_p < 10 {
                                if s_p == 1 && Self::get_reward(r) == 0. && a == 0 {
                                    transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                }
                                if s_p == 2 && Self::get_reward(r) == -1. && a == 0 {
                                    transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                }
                                if s_p == 3 && Self::get_reward(r) == 1. && a == 0 {
                                    transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                }
                                if s_p == 4 && Self::get_reward(r) == 1. && a == 1 {
                                    transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                }
                                if s_p == 5 && Self::get_reward(r) == 0. && a == 1 {
                                    transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                }
                                if s_p == 6 && Self::get_reward(r) == -1. && a == 1 {
                                    transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                }
                                if s_p == 7 && Self::get_reward(r) == -1. && a == 2 {
                                    transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                }
                                if s_p == 8 && Self::get_reward(r) == 1. && a == 2 {
                                    transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                }
                                if s_p == 9 && Self::get_reward(r) == 0. && a == 2 {
                                    transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                }
                            }
                            if s > 0 && s_p > 9 {
                                if vec![1, 2, 3].contains(&s) {
                                    if s_p == 10 && Self::get_reward(r) == 0. && a == 0 {
                                        transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                    }
                                    if s_p == 13 && Self::get_reward(r) == 1. && a == 1 {
                                        transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                    }
                                    if s_p == 16 && Self::get_reward(r) == -1. && a == 2 {
                                        transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                    }
                                }
                                if vec![4, 5, 6].contains(&s) {
                                    if s_p == 11 && Self::get_reward(r) == -1. && a == 0 {
                                        transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                    }
                                    if s_p == 14 && Self::get_reward(r) == 0. && a == 1 {
                                        transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                    }
                                    if s_p == 17 && Self::get_reward(r) == 1. && a == 2 {
                                        transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                    }
                                }
                                if vec![7, 8, 9].contains(&s) {
                                    if s_p == 12 && Self::get_reward(r) == 1. && a == 0 {
                                        transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                    }
                                    if s_p == 15 && Self::get_reward(r) == -1. && a == 1 {
                                        transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                    }
                                    if s_p == 18 && Self::get_reward(r) == 0. && a == 2 {
                                        transition_probability_matrix[[s, a, s_p, r]] = 1.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return transition_probability_matrix;
        }
    }

    impl Default for TwoRoundRPS {
        fn default() -> Self {
            TwoRoundRPS {
                agent_pos: 0,
            }
        }
    }

    impl BaseEnv for TwoRoundRPS {
        fn is_terminal(&self) -> bool {
            match self.agent_pos {
                x if x > 9 => true,
                _ => false,
            }
        }

        fn score(&self) -> f32 {
            match self.agent_pos {
                n if vec![2, 6, 7, 11, 15, 16].contains(&n) => -1.,
                n if vec![0, 1, 5, 9, 10, 14, 18].contains(&n) => 0.,
                n if vec![3, 4, 8, 12, 13, 17].contains(&n) => 1.,
                _ => 0.,
            }
        }
        fn reset(&mut self) {
            self.agent_pos = 0;
        }
    }

    impl ActionEnv<NUM_ACTIONS> for TwoRoundRPS {
        fn available_actions_ids(&self) -> impl Iterator<Item=usize> {
            if self.agent_pos > 18 {
                0..0
            } else {
                0..3
            }
        }
        fn step(&mut self, action: usize) {
            assert_eq!(self.is_terminal(), false);
            assert_eq!(self.available_actions_ids().any(|x| x == action), true);

            if self.agent_pos == 0 {
                let mut rng = rand::thread_rng();

                let ia_move = rng.gen_range(0..=2);
                self.agent_pos = action * 3 + ia_move + 1;
            } else {
                let ia_move = (self.agent_pos - 1) / 3;
                self.agent_pos = (action * 3 + ia_move + 1) + 9;
            };
        }


    }

    impl Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS> for TwoRoundRPS {
        fn state_id(&self) -> usize {
            self.agent_pos as usize
        }

        fn from_random_state() -> Self {
            let mut rng = rand::thread_rng();
            let agent_pos: usize = rng.gen_range(0..=9);

            return TwoRoundRPS {
                agent_pos,
            };
        }


        fn num_states() -> usize {
            19
        }

        fn num_actions() -> usize {
            3
        }

        fn num_rewards() -> usize {
            3
        }

        fn get_reward(i: usize) -> f32 {
            match i {
                0 => -1.,
                1 => 0.,
                2 => 1.,
                _ => panic!("reward out of range"),
            }
        }

        fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
            let matrix = Self::build_transition_matrix();
            return matrix[[s, a, s_p, r]];
        }


        fn reset_random_state(&mut self, seed: u64) {
            let mut rng = StdRng::seed_from_u64(seed);
            let agent_pos: usize = rng.gen_range(0..=9);
            self.agent_pos = agent_pos
        }



        fn available_action_delete(&self) {
            todo!()
        }



        fn delete(&mut self) {
            todo!()
        }


        fn display(&self) {
            if vec![1, 2, 3, 10, 11, 12].contains(&self.agent_pos) {
                print!("P | ")
            };
            if vec![4, 5, 6, 13, 14, 15].contains(&self.agent_pos) {
                print!("F | ")
            };
            if vec![7, 8, 9, 16, 17, 18].contains(&self.agent_pos) {
                print!("S | ")
            };

            if self.agent_pos % 3 == 0 && self.agent_pos != 0 {
                print!("S")
            }
            if self.agent_pos % 3 == 1 {
                print!("P")
            }
            if self.agent_pos % 3 == 2 {
                print!("F")
            }
            println!();
        }
    }

    #[cfg(test)]
    mod tests {
        #![allow(warnings)]
        use super::*;

        #[test]
        #[allow(warnings)]
        fn test_from_random_state() {
            let seed: u64 = 42;
            let rng = StdRng::seed_from_u64(seed).gen_range(0..=9);
            let mut lw = TwoRoundRPS::default();
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
            let lw = TwoRoundRPS::default();
            assert_eq!(
                lw.agent_pos, 0,
                "should be 8, {} find instead",
                lw.agent_pos
            )
        }

        #[test]
        fn test_available_action() {
            let gw = TwoRoundRPS::default();

            assert_eq!(
                gw.available_actions_ids().collect::<Vec<_>>(),
                vec![0, 1, 2],
                "should be [0, 1, 2], found [] instead"
            );
        }

        #[test]
        fn test_two_round_world() {
            let mut gw = TwoRoundRPS::default();

            gw.display();
            gw.step(1);
            gw.display();
            gw.step(2);
            gw.display();
            println!("{}", gw.state_id());

            assert_eq!(gw.state_id(), 17)
        }

        #[test]
        fn test_two_round_strategy() {
            let strategy: std::collections::HashMap<usize, usize> = std::collections::HashMap::from([(0, 1), (4, 2), (5, 2), (6, 2)]);
            let mut gw = TwoRoundRPS::default();
            gw.play_strategy(strategy, false);

            assert_eq!(gw.state_id(), 17)
        }
    }
}