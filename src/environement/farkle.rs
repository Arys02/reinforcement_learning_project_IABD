pub mod farkle{
    use crate::environement::environment_traits::DeepDiscreteActionsEnv;
    use rand::prelude::IteratorRandom;
    use std::fmt::Display;
    use colored::*;

    const DICE_ART: [&str; 6] = [
        "┌─────┐\n│     │\n│  ●  │\n│     │\n└─────┘",
        "┌─────┐\n│●    │\n│     │\n│    ●│\n└─────┘",
        "┌─────┐\n│●    │\n│  ●  │\n│    ●│\n└─────┘",
        "┌─────┐\n│●   ●│\n│     │\n│●   ●│\n└─────┘",
        "┌─────┐\n│●   ●│\n│  ●  │\n│●   ●│\n└─────┘",
        "┌─────┐\n│●   ●│\n│●   ●│\n│●   ●│\n└─────┘",
    ];

    pub const NUM_STATE_FEATURES: usize = 36;
    pub const NUM_ACTIONS: usize = 12;

    #[derive(Clone)]
    pub struct Farkle{
        pub board: [u8; 6],
        pub remaining_dice: u8,
        pub player_score: usize,
        pub ai_score: usize,
        pub round: u8,
        pub score: f32,
        pub is_game_over: bool,
        pub is_player_turn: bool,
    }

    impl Default for Farkle{
        fn default() -> Self {
            Self {
                board: [0u8; 6],
                remaining_dice: 0,
                player_score: 0,
                ai_score: 0,
                round: 0,
                score: 0.0,
                is_game_over: false,
                is_player_turn: true,
            }
        }
    }

    impl DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> for Farkle{
        fn state_description(&self) -> [f32; NUM_STATE_FEATURES] {
            let mut output = [0f32; NUM_STATE_FEATURES];
            let mut v: Vec<f32> = Vec::with_capacity(36);
            v.append(&mut vec![1., 2., 3.]);
            let mut j = 0;
            for (i, &value) in self.board.iter().enumerate() {
                if self.board[i] != 0 {
                    for _  in 0..self.board[i] {
                        output[j * 6 + i] = 1.0;
                        j += 1
                    }
                }
            }
            output
        }

        fn available_actions_ids(&self) -> impl Iterator<Item = usize> {
            //[   1,  11, 111, 222, 333, 444,   5,  55, 555, 666, roll, stop ]
            //[   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11 ]
            let mut v = vec![None; NUM_ACTIONS];
            if self.board[0] > 0 {v.insert(0, Some(0))};
            if self.board[0] > 1 {v.insert(1, Some(1))};
            if self.board[0] > 2 {v.insert(2, Some(2))};
            if self.board[1] > 2 {v.insert(3, Some(3))};
            if self.board[2] > 2 {v.insert(4, Some(4))};
            if self.board[3] > 2 {v.insert(5, Some(5))};
            if self.board[4] > 0 {v.insert(6, Some(6))};
            if self.board[4] > 1 {v.insert(7, Some(7))};
            if self.board[4] > 2 {v.insert(8, Some(8))};
            if self.board[5] > 0 {v.insert(9, Some(9))};

            let is_farkle = v.iter().all(|&x| x == None);
            if is_farkle {
                panic!("Trying to get action while farkle")
            }
            if !is_farkle {
                v.insert(10, Some(10));
                v.insert(11, Some(11));
            }

            v.into_iter().filter_map(|x| x)

        }

        fn action_mask(&self) -> [f32; NUM_ACTIONS] {
            todo!();
            /*
            let x = self.available_actions_ids();
            std::array::from_fn(|idx| if x[idx] != None { 1.0 } else { 0.0 })

             */
        }

        fn step(&mut self, action: usize) {
            if self.is_game_over {
                panic!("Trying to play while Game is Over");
            }

            if action >= 12 {
                panic!("Invalid action : {}", action);
            }
            // should i check if the action is valid with available action ?

            //stop and get the points
            if action == 11 {

            }

            //reroll
            if action == 10 {

            }


            todo!();
            /*

            self.board[action] = self.player as f32 + 1.0;

            let row = action / 3;
            let col = action % 3;

            // check line, column and diagonals
            if self.board[row * 3] == self.board[row * 3 + 1]
                && self.board[row * 3 + 1] == self.board[row * 3 + 2]
                || self.board[col] == self.board[col + 3]
                    && self.board[col + 3] == self.board[col + 6]
                || self.board[0] == self.board[4]
                    && self.board[4] == self.board[8]
                    && self.board[0] == self.board[action]
                || self.board[2] == self.board[4]
                    && self.board[4] == self.board[6]
                    && self.board[2] == self.board[action]
            {
                self.is_game_over = true;
                self.score = if self.player == 0 { 1.0 } else { -1.0 };
                return;
            }

            // check if board is full
            if self.board.iter().all(|&val| val != 0.0) {
                self.is_game_over = true;
                self.score = 0.0;
                return;
            }

            if self.player == 0 {
                self.player = 1;

                // random move
                let mut rng = rand::thread_rng();
                let random_action = self.available_actions_ids().choose(&mut rng).unwrap();
                self.step(random_action);
            } else {
                self.player = 0;
            }
             */
        }

        fn is_game_over(&self) -> bool {

            todo!();
            //self.is_game_over
        }

        fn score(&self) -> f32 {

            todo!()
            //self.score
        }

        fn reset(&mut self) {
            todo!()
            /*
            self.board = [0f32; NUM_ACTIONS];
            self.player = 0;
            self.score = 0.0;
            self.is_game_over = false;

             */
        }
    }

    impl Display for Farkle{
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(
                f,
                "{}",
                "=== Farkle Game ===".bold().underline().blue()
            )?;
            writeln!(f, "Round: {}", self.round)?;
            writeln!(f, "Player Score: {}", self.player_score)?;
            writeln!(f, "AI Score: {}", self.ai_score)?;
            writeln!(f, "Game Over: {}", self.is_game_over)?;
            writeln!(f, "Remaining Dice: {}", self.remaining_dice)?;
            writeln!(f, "Current Round Score: {:.2}", self.score)?;
            writeln!(
                f,
                "Current Turn: {}",
                if self.is_player_turn {
                    "Player".green()
                } else {
                    "AI".red()
                }
            )?;
            writeln!(f, "Dice:")?;

            // Define ASCII Art for Dice Faces 1 through 6
            const DICE_ART: [&str; 6] = [
                // Die face 1
                "┌─────┐\n│     │\n│  ●  │\n│     │\n└─────┘",
                // Die face 2
                "┌─────┐\n│●    │\n│     │\n│    ●│\n└─────┘",
                // Die face 3
                "┌─────┐\n│●    │\n│  ●  │\n│    ●│\n└─────┘",
                // Die face 4
                "┌─────┐\n│●   ●│\n│     │\n│●   ●│\n└─────┘",
                // Die face 5
                "┌─────┐\n│●   ●│\n│  ●  │\n│●   ●│\n└─────┘",
                // Die face 6
                "┌─────┐\n│●   ●│\n│●   ●│\n│●   ●│\n└─────┘",
            ];

            // Expand the board counts into a list of individual die faces
            let mut dice_faces: Vec<u8> = Vec::new();
            for (face_index, &count) in self.board.iter().enumerate() {
                for _ in 0..count {
                    // face_index 0 => die face 1, ..., face_index 5 => die face 6
                    dice_faces.push(face_index as u8 + 1);
                }
            }

            // Check if there are no dice rolled
            if dice_faces.is_empty() {
                writeln!(f, "No dice rolled.")?;
                return Ok(());
            }

            // Ensure that the total number of dice does not exceed 6
            if dice_faces.len() > 6 {
                writeln!(f, "Error: More than 6 dice are rolled!")?;
                return Ok(());
            }

            // Prepare lines for ASCII art
            let mut dice_lines = vec![String::new(); 5]; // Each die has 5 lines

            for &die in &dice_faces {
                // Get the ASCII art for the current die face
                let face = DICE_ART[(die - 1) as usize];
                let face_lines: Vec<&str> = face.split('\n').collect();

                // Append each line of the die's ASCII art to the corresponding dice_lines
                for (i, line) in face_lines.iter().enumerate() {
                    dice_lines[i].push_str(line);
                    dice_lines[i].push_str("  "); // Space between dice
                }
            }

            // Write the collected ASCII art lines
            for line in dice_lines {
                writeln!(f, "{}", line)?;
            }

            Ok(())
        }
    }
}
