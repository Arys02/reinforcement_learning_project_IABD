pub mod farkle_2 {
    use crate::environement::environment_traits::{ActionEnv, BaseEnv, DeepDiscreteActionsEnv, Playable};
    use colored::*;
    use rand::prelude::IteratorRandom;
    use rand::Rng;
    use std::collections::{HashMap, HashSet};
    use std::fmt::Display;
    use std::io;
    use std::io::Write;

    pub const NUM_STATE_FEATURES: usize = 36;
    pub const NUM_ACTIONS: usize = 268;

    pub const MAX_GAME: usize = 50;
    pub const MAX_SCORE: usize = 5000;

    /**
    Rules for action
    Action < DICE_ACTION_VALUE.len() => Action then roll
    Action >= DICE_ACTION_VALUE.len() => Action then bank
    **/
    pub const DICE_ACTION_VALUE: [(&'static str, f32); 134] = [("555666", 1100.), ("111222",
                                                                                   1200.), ("1555", 600.), ("123456", 1500.), ("22225", 450.), ("12222", 500.), ("111333", 1300.), ("133335", 750.), ("5", 50.), ("333335", 1250.), ("224466", 1000.), ("3333", 600.), ("1111", 2000.), ("55555", 2000.), ("222222", 1600.), ("4445", 450.), ("115555", 1200.), ("11222", 400.), ("2225", 250.), ("1222", 300.), ("111111", 8000.), ("444444", 3200.), ("11555", 700.), ("112233", 1000.), ("335566", 1000.), ("222", 200.), ("112266", 1000.), ("1155", 300.), ("111444", 1400.), ("223344", 1000.), ("2222", 400.), ("444666", 1000.), ("444455", 900.), ("22222", 800.), ("111155", 2100.), ("114444", 1000.), ("144445", 950.), ("1", 100.), ("115", 250.), ("155", 200.), ("444445", 1650.), ("144444", 1700.), ("113335", 550.), ("11111", 4000.), ("111", 1000.), ("11115", 2050.), ("14445", 550.), ("133355", 500.), ("44455", 500.), ("1333", 400.), ("222555", 700.), ("115566", 1000.), ("11444", 600.), ("666666", 4800.), ("333666", 900.), ("111555", 1500.), ("55", 100.), ("13335", 450.), ("333355", 700.), ("66666", 2400.), ("555555", 4000.), ("155666", 800.), ("333444", 700.), ("33333", 1200.), ("113355", 1000.), ("4444", 800.), ("223355", 1000.), ("133333", 1300.), ("11666", 800.), ("113344", 1000.), ("223366", 1000.), ("155555", 2100.), ("334455", 1000.), ("113333", 800.), ("14444", 900.), ("3335", 350.), ("6666", 1200.), ("566666", 2450.), ("112225", 450.), ("11333", 500.), ("222444", 600.), ("156666", 1350.), ("114455", 1000.), ("333", 300.), ("122225", 550.), ("56666", 1250.), ("116666", 1400.), ("13333", 700.), ("44445", 850.), ("15666", 750.), ("112255", 1000.), ("444", 400.), ("222255", 500.), ("1666", 700.), ("115666", 850.), ("222225", 850.), ("333555", 800.), ("1115", 1050.), ("33355", 400.), ("333333", 2400.), ("112244", 1000.), ("225566", 1000.), ("334466", 1000.), ("444555", 900.), ("114445", 650.), ("5555", 1000.), ("166666", 2500.), ("144455", 600.), ("16666", 1300.), ("222666", 800.), ("222333", 500.), ("11155", 1100.), ("55666", 700.), ("111666", 1600.), ("556666", 1300.), ("22255", 300.), ("11", 200.), ("114466", 1000.), ("15", 150.), ("555", 500.), ("113366", 1000.), ("1444", 500.), ("122255", 400.), ("15555", 1100.), ("12225", 350.), ("33335", 650.), ("5666", 650.), ("111115", 4050.), ("112222", 600.), ("224455", 1000.), ("666", 600.), ("445566", 1000.), ("122222", 900.), ("44444", 1600.)
    ];


    #[derive(Clone, Debug)]
    pub struct Farkle2 {
        pub board: [u8; 6],
        pub player: u8,
        pub remaining_dice: u8,
        pub action_counts: [[usize; 6]; NUM_ACTIONS],
        pub total_score: [usize; 2],
        pub round: u8,
        pub score: f32,
        pub is_game_over: bool,
    }

    impl Farkle2 {
        fn roll(&mut self) {
            self.board = [0u8; 6];
            let mut rng = rand::thread_rng();
            for _ in 0..self.remaining_dice as usize {
                let n: usize = rng.gen_range(0..6);
                self.board[n] += 1;
            }
        }


        //function used to generate all action possible
        #[warn(dead_code)]
        fn getAllAction() -> HashSet<(String, i32)> {
            let mut hset: HashSet<(String, i32)> = HashSet::new();
            let mut hmap: HashMap<String, i32> = HashMap::new();

            let un_v = vec![0, 100, 200, 1000, 2000, 4000, 8000];
            let un = vec!["", "1", "11", "111", "1111", "11111", "111111"];

            let deux_v = vec![0, 200, 400, 800, 1600];
            let deux = vec!["", "222", "2222", "22222", "222222"];

            let trois_v = vec![0, 300, 600, 1200, 2400];
            let trois = vec!["", "333", "3333", "33333", "333333"];

            let quatre_v = vec![0, 400, 800, 1600, 3200];
            let quatre = vec!["", "444", "4444", "44444", "444444"];

            let cinq_v = vec![0, 50, 100, 500, 1000, 2000, 4000];
            let cinq = vec!["", "5", "55", "555", "5555", "55555", "555555"];

            let six_v = vec![0, 600, 1200, 2400, 4800];
            let six = vec!["", "666", "6666", "66666", "666666"];

            for (i6, s6) in six.iter().enumerate() {
                for (i5, s5) in cinq.iter().enumerate() {
                    for (i4, s4) in quatre.iter().enumerate() {
                        for (i3, s3) in trois.iter().enumerate() {
                            for (i2, s2) in deux.iter().enumerate() {
                                for (i1, s1) in un.iter().enumerate() {
                                    let together = format!("{}{}{}{}{}{}", s1, s2, s3, s4, s5, s6);
                                    if together.len() <= 6 && together != "" {
                                        let mut tog: Vec<char> = together.chars().collect();
                                        tog.sort();
                                        //v'ec.push(tog.into_iter().collect::<String>());
                                        let value = un_v[i1] + deux_v[i2] + trois_v[i3] +
                                            quatre_v[i4] + cinq_v[i5] + six_v[i6];
                                        hmap.insert(tog.iter().collect::<String>(), value);
                                        hset.insert((tog.into_iter().collect::<String>(), value));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let un = vec!["", "11"];
            let deux = vec!["", "22"];
            let trois = vec!["", "33"];
            let quatre = vec!["", "44"];
            let cinq = vec!["", "55"];
            let six = vec!["", "66"];

            for s1 in six.iter() {
                for s2 in cinq.iter() {
                    for s3 in quatre.iter() {
                        for s4 in trois.iter() {
                            for s5 in deux.iter() {
                                for s6 in un.iter() {
                                    let together = format!("{}{}{}{}{}{}", s1, s2, s3, s4, s5, s6);
                                    if together.len() == 6 && together != "" {
                                        let mut tog: Vec<char> = together.chars().collect();
                                        tog.sort();
                                        hmap.insert(tog.iter().collect::<String>(), 1000);
                                        hset.insert((tog.into_iter().collect::<String>(), 1000));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            hset.insert((String::from("123456"), 1500));
            hmap.insert(String::from("123456"), 1500);


            hset
        }


        fn getActionScore(&mut self, action_id: usize) -> f32 {
            if action_id >= DICE_ACTION_VALUE.len() {
                DICE_ACTION_VALUE[action_id - DICE_ACTION_VALUE.len()].1
            } else {
                DICE_ACTION_VALUE[action_id].1
            }
        }

        fn endTurn(&mut self) {
            self.total_score[self.player as usize] += self.score as usize;
            self.score = 0.;
            self.remaining_dice = 6;

            self.roll();
            while self.available_actions_ids().count() < 1 {
                //this is where first roll farkle logic will go
                self.roll();
            }
            if self.player == 0 {
                self.player = 1;
                // random move
                let mut rng = rand::thread_rng();
                while self.player == 1 && !self.is_game_over
                {
                    let random_action = self.available_actions_ids().choose(&mut rng).unwrap();
                    self.step(random_action);
                }
            } else {
                self.round += 1;
                self.player = 0;
            }
        }

        fn get_user_choice(&self) -> usize {
            loop {
                //why is this 1 - 4?
                println!("Please enter your choice :");

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

    impl Default for Farkle2 {
        fn default() -> Self {
            let mut actions_count: [[usize; 6]; NUM_ACTIONS] = [[0usize; 6]; NUM_ACTIONS];
            for (i, (str, _)) in DICE_ACTION_VALUE.iter().enumerate() {
                for char in str.chars() {
                    match char {
                        '1' => actions_count[i][0] += 1,
                        '2' => actions_count[i][1] += 1,
                        '3' => actions_count[i][2] += 1,
                        '4' => actions_count[i][3] += 1,
                        '5' => actions_count[i][4] += 1,
                        '6' => actions_count[i][5] += 1,
                        _ => {}
                    }
                }
            }


            let mut farkle = Self {
                board: [0u8; 6],
                player: 0,
                remaining_dice: 6,
                action_counts: actions_count,
                total_score: [0; 2],
                round: 0,
                score: 0.0,
                is_game_over: false,
            };
            farkle.roll();
            //todo remove to add the right score
            while farkle.available_actions_ids().count() == 0 {
                farkle.roll();
            }
            farkle
        }
    }

    impl ActionEnv<NUM_ACTIONS> for Farkle2 {
        fn available_actions_ids(&self) -> impl Iterator<Item=usize> {
            let mut v = vec![None; NUM_ACTIONS];
            for i in 0..DICE_ACTION_VALUE.len() {
                let mut check = true;
                for (j, &dice_action_count) in self.action_counts[i].iter().enumerate() {
                    if dice_action_count > self.board[j] as usize {
                        check = false;
                        break;
                    }
                }
                if check {
                    v[i] = Some(i);
                    v[i + DICE_ACTION_VALUE.len()] = Some(i + DICE_ACTION_VALUE.len());
                }
            }

            v.into_iter().filter_map(|x| x)
        }
        fn step(&mut self, action: usize) {
            #[cfg(feature = "print")]
            println!("{}\n IA selected : {action}", self);


            if self.is_game_over {
                panic!("Trying to play while Game is Over");
            }

            if !self.available_actions_ids().any(|a| a == action) {
                panic!("Action unavailable : {}", action);
            }

            self.score += self.getActionScore(action);

            if self.total_score[0] >= MAX_SCORE || self.total_score[1] >= MAX_SCORE {
                //println!("{:?}", self);
                self.is_game_over = true;
                self.score = self.total_score[0] as f32 - self.total_score[1] as f32;
                return;
            }

            let action_result = if action < DICE_ACTION_VALUE.len() {
                DICE_ACTION_VALUE[action].0.chars()
            } else {
                DICE_ACTION_VALUE[action - DICE_ACTION_VALUE.len()].0.chars()
            };

            for chars in action_result {
                match chars {
                    '1' => {
                        self.remaining_dice -= 1;
                        self.board[0] -= 1
                    }
                    '2' => {
                        self.remaining_dice -= 1;
                        self.board[1] -= 1
                    }
                    '3' => {
                        self.remaining_dice -= 1;
                        self.board[2] -= 1
                    }
                    '4' => {
                        self.remaining_dice -= 1;
                        self.board[3] -= 1
                    }
                    '5' => {
                        self.remaining_dice -= 1;
                        self.board[4] -= 1
                    }
                    '6' => {
                        self.remaining_dice -= 1;
                        self.board[5] -= 1
                    }
                    _ => {}
                }
            }

            if self.remaining_dice == 0 {
                self.remaining_dice = 6;
                self.roll();
                while self.available_actions_ids().count() == 0 {
                    self.roll();
                }
            } else {
                //reroll
                if action < 134 {
                    self.roll();
                    let available_actions: Vec<usize> = self.available_actions_ids().collect();
                    //farkle
                    if available_actions.is_empty() {
                        self.score = 0.;
                        self.endTurn();
                    }
                } else {
                    self.endTurn()
                }
            }
        }
    }

    impl BaseEnv for Farkle2 {
        fn get_name(&self) -> String {
            String::from("farkle2")
        }

        fn is_terminal(&self) -> bool {
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

            self.roll();
            while self.available_actions_ids().count() == 0 {
                self.roll();
            }
        }
    }

    impl DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> for Farkle2 {
        fn state_description(&self) -> [f32; NUM_STATE_FEATURES] {
            let mut output = [0f32; NUM_STATE_FEATURES];
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
        fn action_mask(&self) -> [f32; NUM_ACTIONS] {
            let mut mask = [0.0; NUM_ACTIONS];
            for idx in self.available_actions_ids() {
                mask[idx] = 1.0;
            }
            mask
        }
    }


    impl Display for Farkle2 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(
                f,
                "{}",
                "=== Farkle Game ===".bold().underline().blue()
            )?;
            writeln!(f, "Round: {}", self.round)?;
            writeln!(f, "{} {}", "Player Score: ".green(), self.total_score[0])?;
            writeln!(f, "{} {}", "AI Score: ".red(), self.total_score[1])?;
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

            // Now enter the loop to process sorted actions
            for (index, &action_id) in available_actions.iter().enumerate() {
                let display_number = index + 1;

                if action_id < 134 {
                    // Pass the original dice value to the formatting function
                    writeln!(f, "{}: Take {} and reroll", display_number,
                             format_dice_choices_for_print(DICE_ACTION_VALUE[action_id].0))?;
                }
                else  {
                    writeln!(f, "{}: Take {} and bank", display_number,
                             format_dice_choices_for_print(DICE_ACTION_VALUE[action_id -
                                 DICE_ACTION_VALUE.len()].0))?;
                }
            }
            Ok(())
        }
    }

    fn format_dice_choices_for_print(dice_value: &str) -> String {
        let mut chars: Vec<char> = dice_value.chars().collect();

        // Sort characters by their numeric value
        chars.sort();

        // Format the result based on the length of the string
        match chars.len() {
            0 => String::new(),
            1 => chars[0].to_string(),
            _ => {
                let last_char = chars.pop().unwrap().to_string(); // Get the last character
                let other_chars: Vec<String> = chars.into_iter().map(|c| c.to_string()).collect();
                format!("{} and {}", other_chars.join(", "), last_char)
            }
        }
    }


    /// Trait that defines the capability for a game to be played interactively with a human player.
    /// Implementing this trait allows a game to be run in an interactive mode,
    /// utilizing all the provided methods to facilitate a complete game session.
    ///

    impl Playable for Farkle2 {
        /*
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

         */




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
        fn play_as_human() {
            fn clear_screen() {
                print!("{}[2J{}[1;1H", 27 as char, 27 as char);
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
            let mut env: Farkle2 = Farkle2::default();

            writeln!(std::io::stdout(), "Welcome to the Land of Farkle\nYour adventure begins now")
                .expect("Failed to write welcome message");
            env.reset(); // self is the current instance of Farkle
            while !env.is_terminal()
            {
                //clear_screen();
                println!("{}", env);
                let option_choice = env.get_user_choice();
                env.step(option_choice);
            }

            println!("Game Over!");
            println!("Final Scores: {:?}", env.total_score);
        }

        fn play_as_random_ai() -> [usize; 2] {
            let mut env: Farkle2 = Farkle2::default();
            while !env.is_game_over {
                let aa = env.available_actions_ids().choose(&mut rand::thread_rng()).unwrap();
                env.step(aa);
            }
            env.total_score
        }
    }


    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_default_initialization() {
            let game = Farkle2::default();
            assert_eq!(game.board, [0u8; 6]);
            assert_eq!(game.player, 0);
            assert_eq!(game.remaining_dice, 0);
            assert_eq!(game.total_score, [0; 2]);
            assert_eq!(game.round, 0);
            assert_eq!(game.score, 0.0);
            assert!(!game.is_game_over);
        }

        #[test]
        fn play_with_human_test() {
            Farkle2::play_as_human();
        }
        #[test]
        fn test_available_actions_ids() {
            let mut game = Farkle2::default();
            game.board = [1, 0, 3, 0, 0, 1]; // Trois '1'

            let actions: Vec<usize> = game.available_actions_ids().collect();
            println!("Available Actions:{:?} which is : {:?}", actions, actions.iter().map(|x|
                { if *x < 135 { DICE_ACTION_VALUE[*x].0 } else { "" } }).collect::<Vec<&str>>());

        }


        #[test]
        fn test_step_action_scoring() {
            let mut game = Farkle2::default();
            game.board = [1, 0, 0, 0, 0, 0];
            game.remaining_dice = 1;

            // Action pour '1' (index 0)
            game.step(0);
            assert_eq!(game.score, 100.0);
            assert_eq!(game.remaining_dice, 0);
            assert_eq!(game.board, [0, 0, 0, 0, 0, 0]);
            println!("{:?}", game);


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
        fn test_actions() {
            let x = Farkle2::getAllAction();
            println!("{:?}\n{:?}", x, x.len())
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
}

