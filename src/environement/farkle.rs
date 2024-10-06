pub mod farkle {
    use crate::environement::environment_traits::DeepDiscreteActionsEnv;
    use rand::prelude::IteratorRandom;
    use rand::Rng;
    use std::fmt::Display;

    pub const NUM_STATE_FEATURES: usize = 36;
    pub const NUM_ACTIONS: usize = 12;

    #[derive(Clone)]
    pub struct Farkle {
        pub board: [u8; 6],
        pub player: u8,
        pub remaining_dice: u8,
        pub total_score: [usize; 2],
        pub round: u8,
        pub score: f32,
        pub is_game_over: bool,
        pub reroll_allowed: bool
    }

    impl Farkle {
        fn roll(&mut self) {
            self.board = [0u8; 6];
            let mut rng = rand::thread_rng();
            for _ in 0..self.remaining_dice as usize {
                let n: usize = rng.gen_range(1..=6);
                self.board[n] += 1;
            }
        }

        fn getScore(&mut self, action: usize) -> f32 {
            //[   1,  11, 111, 222, 333, 444,   5,  55, 555, 666, roll, stop ]
            //[   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11 ]
            match action {
                0 => 100.,
                1 => 200.,
                2 => 1000.,
                3 => 200.,
                4 => 300.,
                5 => 400.,
                6 => 50.,
                7 => 100.,
                8 => 500.,
                9 => 600.,
                _ => panic!("Invalid action {}", action),
            }
        }
    }

    impl Default for Farkle {
        fn default() -> Self {
            Self {
                board: [0u8; 6],
                player: 0,
                remaining_dice: 0,
                total_score: [0; 2],
                round: 0,
                score: 0.0,
                is_game_over: false,
                reroll_allowed: false,
            }
        }
    }

    impl DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> for Farkle {
        fn state_description(&self) -> [f32; NUM_STATE_FEATURES] {
            let mut output = [0f32; NUM_STATE_FEATURES];
            let mut v: Vec<f32> = Vec::with_capacity(36);
            v.append(&mut vec![1., 2., 3.]);
            let mut j = 0;
            for i in 0..6 {
                if self.board[i] != 0 {
                    for _ in 0..self.board[i] {
                        output[j * 6 + i] = 1.0;
                        j += 1
                    }
                }
            }
            output
        }

        fn available_actions_ids(&self) -> impl Iterator<Item=usize> {
            //[   1,  11, 111, 222, 333, 444,   5,  55, 555, 666, roll, stop ]
            //[   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11 ]
            let mut v = vec![None; NUM_ACTIONS];
            if self.board[0] > 0 { v[0] = Some(0) };
            if self.board[0] > 1 { v[1] = Some(1) };
            if self.board[0] > 2 { v[2] = Some(2) };
            if self.board[1] > 2 { v[3] = Some(3) };
            if self.board[2] > 2 { v[4] = Some(4) };
            if self.board[3] > 2 { v[5] = Some(5) };
            if self.board[4] > 0 { v[6] = Some(6) };
            if self.board[4] > 1 { v[7] = Some(7) };
            if self.board[4] > 2 { v[8] = Some(8) };
            if self.board[5] > 0 { v[9] = Some(9) };

            let is_farkle = v.iter().all(|&x| x.is_none());

            if !is_farkle {
                v.insert(11, Some(11));
            }
            if !self.reroll_allowed {
                v.insert(10, Some(10));
            }

            v.into_iter().filter_map(|x| x)
        }

        fn action_mask(&self) -> [f32; NUM_ACTIONS] {
            let mut mask = [0.0; NUM_ACTIONS];
            for idx in self.available_actions_ids() {
                mask[idx] = 1.0;
            }
            mask
        }

        fn step(&mut self, action: usize) {
            if self.is_game_over {
                panic!("Trying to play while Game is Over");
            }

            if !self.available_actions_ids().any(|a| a == action) {
                panic!("Action non disponible : {}", action);
            }

            if self.round >= 10 {
                self.is_game_over = true;
                self.score = (self.total_score[0] - self.total_score[1]) as f32;
                return;
            }

            //stop and get the points
            if action == 11 {
                self.total_score[self.player as usize] += self.score as usize;
                self.score = 0.;
                self.remaining_dice = 6;

                self.roll();
                self.reroll_allowed = false;
                if self.player == 0 {
                    self.player = 1;
                    // random move
                    let mut rng = rand::thread_rng();
                    let random_action = self.available_actions_ids().choose(&mut rng).unwrap();
                    self.step(random_action);
                } else {
                    self.round += 1;
                    self.player = 0;
                }
            }

            //reroll
            else if action == 10 {
                self.roll();
                self.reroll_allowed = false;
                let available_actions: Vec<usize> = self.available_actions_ids().collect();
                //farkle
                if available_actions.is_empty() {
                    self.score = 0.;
                    //change player
                    if self.player == 0 {
                        self.player = 1;
                        // random move
                        let mut rng = rand::thread_rng();
                        let random_action = self.available_actions_ids().choose(&mut rng).unwrap();
                        self.step(random_action);
                    } else {
                        self.player = 0;
                        self.round += 1;
                    }
                }
            }
            //[   1,  11, 111, 222, 333, 444,   5,  55, 555, 666, roll, stop ]
            //[   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11 ]
            else if action < 10 {
                self.score += self.getScore(action);

                match action {
                    0 => {
                        self.remaining_dice -= 1;
                        self.board[0] -= 1
                    }
                    1 => {
                        self.remaining_dice -= 2;
                        self.board[0] -= 2
                    }
                    2 => {
                        self.remaining_dice -= 3;
                        self.board[0] -= 3
                    }
                    3 => {
                        self.remaining_dice -= 3;
                        self.board[1] -= 3
                    }
                    4 => {
                        self.remaining_dice -= 3;
                        self.board[2] -= 3
                    }
                    5 => {
                        self.remaining_dice -= 3;
                        self.board[3] -= 3
                    }
                    6 => {
                        self.remaining_dice -= 1;
                        self.board[4] -= 1
                    }
                    7 => {
                        self.remaining_dice -= 2;
                        self.board[4] -= 2
                    }
                    8 => {
                        self.remaining_dice -= 3;
                        self.board[4] -= 3
                    }
                    9 => {
                        self.remaining_dice -= 3;
                        self.board[5] -= 3
                    }
                    _ => {
                        panic!("Invalid action : {}", action);
                    }
                }
                self.reroll_allowed = true
            }
        }

        fn is_game_over(&self) -> bool {
            self.is_game_over
        }

        fn score(&self) -> f32 {
            self.score
        }

        fn reset(&mut self) {
            self.board = [0u8; 6];
            self.player = 0;
            self.remaining_dice = 0;
            self.total_score = [0; 2];
            self.round = 0;
            self.score = 0.0;
            self.is_game_over = false;
            self.reroll_allowed = false;
        }
    }

    impl Display for Farkle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            todo!()
            /*
            Ok(())
             */
        }
    }
}