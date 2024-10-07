/** TODO
  hot dice : when a player take all 6 dice, he can play again
  the state vectorized should be able to give every information to be able to get back

*/
use crate::environement::farkle::farkle::Farkle;

pub mod farkle {
    use crate::environement::environment_traits::{DeepDiscreteActionsEnv, Playable};
    use rand::prelude::IteratorRandom;
    use rand::Rng;
    use std::fmt::Display;
    use colored::*;
    use std::io::Write;
    use std::io;

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

    #[derive(Clone, Debug)]
    pub struct Farkle {
        pub board: [u8; 6],
        pub player: u8,
        pub remaining_dice: u8,
        pub total_score: [usize; 2],
        pub round: u8,
        pub score: f32,
        pub is_game_over: bool,
        pub reroll_allowed: bool,
    }

    impl Farkle {
        fn roll(&mut self) {
            self.board = [0u8; 6];
            let mut rng = rand::thread_rng();
            for _ in 0..self.remaining_dice as usize {
                let n: usize = rng.gen_range(0..6);
                self.board[n] += 1;
            }
            self.reroll_allowed = false;
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

        fn endTurn(&mut self) {
            self.total_score[self.player as usize] += self.score as usize;
            self.score = 0.;
            self.remaining_dice = 6;

            self.roll();
            while self.available_actions_ids().count() == 0 {
                self.roll();
            }
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

        fn get_user_choice(&self) -> usize {
            loop {
                // Prompt the user
                println!("Please enter your choice (1-4):");

                // Create a mutable String to store the input
                let mut option_choice = String::new();

                // Read the input from the user
                io::stdin()
                    .read_line(&mut option_choice)
                    .expect("Failed to read line");

                // Trim the input to remove any trailing newline characters
                let option_choice = option_choice.trim();

                // Attempt to parse the input to a usize
                match option_choice.parse::<usize>() {
                    Ok(num) => {
                        // Collect available actions into a Vec
                        let available_actions: Vec<usize> = self.available_actions_ids().collect();

                        // Check if the input number is within the valid range
                        if num >= 1 && num <= available_actions.len() {
                            // Map the display number to the actual action ID
                            let action_id = available_actions[num - 1];
                            return action_id;
                        } else {
                            // Invalid choice; inform the user
                            println!(
                                "Invalid choice: {}. Please enter a number between 1 and {}.",
                                num,
                                available_actions.len()
                            );
                        }
                    }
                    Err(_) => {
                        // Parsing failed; inform the user
                        println!(
                            "Invalid input: '{}'. Please enter a valid number.",
                            option_choice
                        );
                    }
                }
            }
        }

    }

    impl Default for Farkle {
        fn default() -> Self {
            let mut farkle = Self {
                board: [0u8; 6],
                player: 0,
                remaining_dice: 6,
                total_score: [0; 2],
                round: 0,
                score: 0.0,
                is_game_over: false,
                reroll_allowed: false,
            };
            farkle.roll();
            farkle
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
            if self.board[5] > 2 { v[9] = Some(9) };

            if self.reroll_allowed {
                v.insert(10, Some(10));
            }

            if !v.iter().all(|&x| x.is_none()) {
                v.insert(11, Some(11));
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
            let aa: Vec<usize> = self.available_actions_ids().collect();
            println!("player {:?} with action : {:?} playes {action} on \n{:?} ", self.player, aa, self);
            if self.is_game_over {
                panic!("Trying to play while Game is Over");
            }

            if !self.available_actions_ids().any(|a| a == action) {
                panic!("Action unavailable : {}", action);
            }

            if self.round >= 10 {
                self.is_game_over = true;
                self.score = (self.total_score[0] as f32 - self.total_score[1] as f32);
                return;
            }

            //stop and get the points
            if action == 11 {
                self.endTurn()
            }

            //reroll
            else if action == 10 {
                self.roll();
                self.reroll_allowed = false;
                let available_actions: Vec<usize> = self.available_actions_ids().collect();
                //farkle
                if available_actions.is_empty() {
                    println!("FARKELED");
                    self.score = 0.;
                    self.endTurn();
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
            self.remaining_dice = 6;
            self.total_score = [0; 2];
            self.round = 0;
            self.score = 0.0;
            self.is_game_over = false;
            self.reroll_allowed = false;

            self.roll();
            while self.available_actions_ids().count() == 0 {
                self.roll();
            }
        }
    }

    impl Display for Farkle {
        //use std::fmt;
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(
                f,
                "{}",
                "=== Farkle Game ===".bold().underline().blue()
            )?;
            writeln!(f, "Round: {}", self.round)?;
            writeln!(f, "Player Score: {}", self.total_score[0])?;
            writeln!(f, "AI Score: {}", self.total_score[1])?;
            writeln!(f, "Game Over: {}", self.is_game_over)?;
            writeln!(f, "Remaining Dice: {}", self.remaining_dice)?;
            writeln!(f, "Current Round Score: {:.2}", self.score)?;
            writeln!(
                f,
                "Current Turn: {}",
                if self.player == 0 {
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

            writeln!(f, "\nAvailable Actions:")?;
            let available_actions: Vec<usize> = self.available_actions_ids().collect();
            for (index, action_id) in available_actions.iter().enumerate() {
                let display_number = index + 1;
                let description = if *action_id < ACTION_DESCRIPTIONS.len() {
                    ACTION_DESCRIPTIONS[*action_id]
                } else {
                    "Unknown action"
                };
                writeln!(f, "{}: {}", display_number, description)?;
            }

            Ok(())
        }
    }

    const ACTION_DESCRIPTIONS: [&str; 12] = [
        "Take 1 die of 1",
        "Take 2 dice of 1",
        "Take 3 dice of 1",
        "Take 3 dice of 2",
        "Take 3 dice of 3",
        "Take 3 dice of 4",
        "Take 1 die of 5",
        "Take 2 dice of 5",
        "Take 3 dice of 5",
        "Take 3 die of 6",
        "Roll again",
        "Stop and bank score",
    ];

    fn clear_screen() {
        print!("{}[2J{}[1;1H", 27 as char, 27 as char);
        use std::io::{self, Write};
        io::stdout().flush().unwrap();
    }


    /// Trait that defines the capability for a game to be played interactively with a human player.
    /// Implementing this trait allows a game to be run in an interactive mode,
    /// utilizing all the provided methods to facilitate a complete game session.
    ///

    impl Playable for Farkle {
        /// Initiates a full game of Farkle to be played with a human player.
        ///
        /// The objective is to utilize all existing methods to manage the game state and interactions.
        /// This function should:
        /// - Initialize the game and any necessary variables.
        /// - Enter a game loop that continues until the game is over.
        /// - Display the current game state to the player at each turn.
        /// - Prompt the player for input and handle their chosen actions.
        /// - Update the game state based on the player's actions and game rules.
        /// - Handle the AI player's turns, if applicable.
        /// - Conclude the game by displaying the final scores and the winner.
        ///
        /// **Note:** This method is currently unimplemented and needs to be filled out
        /// to allow a real game of Farkle to be played interactively.
        ///
        ///
        fn play_with_human(&mut self) {
            writeln!(std::io::stdout(), "Welcome to the Land of Farkle\nYour adventure begins now")
                .expect("Failed to write welcome message");
            self.reset(); // self is the current instance of Farkle
            while !self.is_game_over()
            {
                clear_screen();
                println!("{}", self);
                let option_choice = self.get_user_choice();
                self.step(option_choice);
            }

            println!("Game Over!");
            println!("Final Scores: {:?}", self.total_score);
        }
        fn play_with_random_ai() {
            todo!()
        }
    }




    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn test_default_initialization() {
            let game = Farkle::default();
            assert_eq!(game.board, [0u8; 6]);
            assert_eq!(game.player, 0);
            assert_eq!(game.remaining_dice, 0);
            assert_eq!(game.total_score, [0; 2]);
            assert_eq!(game.round, 0);
            assert_eq!(game.score, 0.0);
            assert!(!game.is_game_over);
            assert!(!game.reroll_allowed);
        }
    }
    #[test]
    fn test_available_actions_ids() {
        let mut game = Farkle::default();
        game.board = [3, 0, 2, 0, 1, 0]; // Trois '1'

        let actions: Vec<usize> = game.available_actions_ids().collect();
        assert_eq!(actions, vec![0, 1, 2, 6, 11]); // Actions pour 1, 11, 111, stop

        game.reroll_allowed = true;
        let actions: Vec<usize> = game.available_actions_ids().collect();
        assert_eq!(actions, vec![0, 1, 2, 6, 10, 11]); // 'roll' n'est pas disponible

        game.reroll_allowed = false;
        game.board = [0, 0, 0, 0, 0, 0];
        let actions: Vec<usize> = game.available_actions_ids().collect();
        assert!(actions.is_empty()); // Aucune action disponible
    }


    #[test]
    fn test_step_action_scoring() {
        let mut game = Farkle::default();
        game.board = [1, 0, 0, 0, 0, 0];
        game.remaining_dice = 1;

        // Action pour '1' (index 0)
        game.step(0);
        assert_eq!(game.score, 100.0);
        assert_eq!(game.remaining_dice, 0);
        assert_eq!(game.board, [0, 0, 0, 0, 0, 0]);
        assert!(game.reroll_allowed);
        println!("{:?}", game);

        game.reroll_allowed = false;

        // Comme 'reroll_allowed' est vrai, 'roll' n'est pas disponible
        let actions: Vec<usize> = game.available_actions_ids().collect();
        //assert_eq!(actions, vec![]); // Seulement 'stop' est disponible

        // Action 'stop' (index 11)
        game.step(11);
        assert_eq!(game.total_score[0], 100);
        assert_eq!(game.score, 0.0);
        assert_eq!(game.player, 1); // Changement de joueur
    }


    #[test]
    fn test_roll() {
        /*
        let mut game = Farkle::default();
        println!("{:?}", game);
        game.step(game.available_actions_ids().nth(0).unwrap());
        println!("{:?}", game);
        game.roll();
        println!("{:?}", game);

         */

        // Utiliser un RNG avec une graine fixe pour des résultats reproductibles

    }
}

