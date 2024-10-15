pub mod monty_hall {
    extern crate rand;

    use ndarray::{array, Array1, Array4, ArrayBase, Ix4, OwnedRepr};
    use rand::Rng;

    use crate::environement::environment_traits::Environment;


    pub const NUM_ACTIONS: usize = 5;
    pub const NUM_STATES: usize = 2;
    pub const NUM_REWARDS: usize = 3;
    /// The `MontyHall1` struct represents the Monty Hall problem environment where an agent acts as a contestant.
    /// The agent makes two successive decisions in a scenario involving three doors: A, B, and C.
    ///
    /// Characteristics of `MontyHall1`:
    /// - **Initial State**: The environment starts with a hidden randomly selected winning door among the three doors.
    /// - **Actions**:
    ///   1. The agent first chooses one of the three doors.
    ///   2. One of the remaining two unchosen doors, which is not the winning door, is revealed and removed.
    ///   3. The agent then decides to either stick with the initial choice or switch to the remaining door.
    /// - **Terminal States and Rewards**:
    ///   - If the final chosen door is the winning door, the agent receives a reward of 1.0.
    ///   - If the final chosen door is not the winning door, the agent receives a reward of 0.0.
    ///
    /// The `MontyHall1` struct implements the `Environment` trait, providing methods for managing the agent's state,
    /// executing actions, and calculating rewards and transition probabilities.
    ///
    #[derive(Clone, Debug)]
    pub struct MontyHall1 {
        state: usize,
        agent_pos: usize,
        door_win: usize,
    }

    impl MontyHall1 {
        fn build_transition_matrix() -> ArrayBase<OwnedRepr<f32>, Ix4> {
            let mut transition_probability_matrix = Array4::zeros((
                Self::num_states(),
                Self::num_actions(),
                Self::num_states(),
                Self::num_rewards(),
            ));

            transition_probability_matrix[[0, 0, 1, 0]] = 1.0;
            transition_probability_matrix[[0, 1, 2, 0]] = 1.0;
            transition_probability_matrix[[0, 2, 3, 0]] = 1.0;

            transition_probability_matrix[[1, 3, 4, 1]] = 1. / 3.;
            transition_probability_matrix[[1, 3, 4, 0]] = 2. / 3.;
            transition_probability_matrix[[1, 4, 5, 1]] = 2. / 3.;
            transition_probability_matrix[[1, 4, 5, 0]] = 1. / 3.;

            return transition_probability_matrix;
        }
    }

    impl Default for MontyHall1 {
        fn default() -> Self {
            //let mut rng = StdRng::seed_from_u64(42);
            let mut rng = rand::thread_rng();
            let dore_win: usize = rng.gen_range(1..=3);
            MontyHall1 {
                state: 0,
                agent_pos: 0,
                door_win: dore_win,
            }
        }
    }

    impl Environment<NUM_STATES, NUM_ACTIONS, NUM_REWARDS> for MontyHall1 {
        fn state_id(&self) -> usize {
            self.state as usize
        }

        fn from_random_state() -> Self {
            let mut rng = rand::thread_rng();
            let agent_pos: usize = rng.gen_range(0..=3);
            let door_win: usize = rng.gen_range(1..=3);

            MontyHall1 {
                state: agent_pos,
                agent_pos,
                door_win,
            }
        }

        fn reset(&mut self) {
            let mut rng = rand::thread_rng();
            let dore_win: usize = rng.gen_range(1..=3);
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
                _ => panic!("reward out of range"),
            }
        }

        fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32 {
            let matrix = Self::build_transition_matrix();
            return matrix[[s, a, s_p, r]];
        }

        fn reset_random_state(&mut self, _seed: u64) {
            let mut rng = rand::thread_rng();
            let agent_pos: usize = rng.gen_range(0..=3);
            let dore_win: usize = rng.gen_range(1..=3);
            self.door_win = dore_win;
            self.state = agent_pos
        }

        fn available_actions_ids(&self) -> Array1<usize> {
            if self.state == 0 {
                array![0, 1, 2]
            } else {
                array![3, 4]
            }
        }

        fn available_action_delete(&self) {
            todo!()
        }

        fn is_terminal(&self) -> bool {
            match self.state {
                4 => true,
                5 => true,
                _ => false,
            }
        }

        fn step(&mut self, action: usize) {
            assert_eq!(self.is_terminal(), false);
            assert_eq!(self.available_actions_ids().iter().any(|&x| x == action), true);

            if self.state == 0 {
                self.state = action + 1;
                self.agent_pos = action + 1;
            } else {
                self.state = if action == 3 {
                    4
                } else {
                    if self.agent_pos != self.door_win {
                        self.agent_pos = self.door_win
                    } else {
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
            if self.agent_pos == self.door_win {
                return 1.;
            }
            0.
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
                let rand: usize = rng.gen_range(0..=1);
                if rand == 0 {
                    println!("-> |    1    |");
                    println!("   |   goat  |");
                    println!("   |    3    |");
                } else {
                    println!("-> |    1    |");
                    println!("   |    2    |");
                    println!("   |   goat  |");
                }
            }
            if self.state == 2 {
                let mut rng = rand::thread_rng();
                let rand: usize = rng.gen_range(0..=1);
                if rand == 0 {
                    println!("   |   goat  |");
                    println!("-> |    2    |");
                    println!("   |    3    |");
                } else {
                    println!("   |   goat  |");
                    println!("-> |    2    |");
                    println!("   |    3    |");
                }
            }
            if self.state == 3 {
                let mut rng = rand::thread_rng();
                let rand: usize = rng.gen_range(0..=1);
                if rand == 0 {
                    println!("   |   goat  |");
                    println!("-> |    2    |");
                    println!("   |    3    |");
                } else {
                    println!("   |    1    |");
                    println!("-> |    2    |");
                    println!("   |   goat  |");
                }
            }
            if self.state > 3 {
                for i in 1..=3 {
                    if self.agent_pos == i {
                        print!("-> ")
                    } else {
                        print!("   ")
                    }
                    if self.door_win == i {
                        print!("| treasure |")
                    } else {
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
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_new() {
            let lw = MontyHall1::default();
            assert_eq!(lw.state, 0, "should be 0, {} find instead", lw.state)
        }

        #[test]
        fn test_available_action() {
            let gw = MontyHall1::default();

            assert_eq!(
                gw.available_actions_ids(),
                array![0, 1, 2],
                "should be [0, 1, 2], found [] instead"
            );
        }

        #[test]
        fn test_monty_hall_1() {
            let mut gw = MontyHall1::default();

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
}