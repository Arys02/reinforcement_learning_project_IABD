use std::collections::HashMap;
use std::{io, print};
use std::io::Write;
use ndarray::Array1;

/// The `Environment` trait defines the common interface for environments used in reinforcement learning.
/// It includes methods for managing the environment's state, executing actions, and retrieving rewards and transition probabilities.
pub trait Environment<const NUM_STATES: usize, const NUM_ACTIONS: usize, const NUM_REWARDS:
usize> : Default + Clone + BaseEnv {
    /// Creates a new instance of the environment.

    fn state_id(&self) -> usize;
    /// Creates a new instance of the environment from a random state.
    fn from_random_state() -> Self;
    /// Resets the environment to its initial state.

    fn num_states() -> usize;
    /// Returns the total number of actions available in the environment.
    fn num_actions() -> usize;
    /// Returns the total number of possible rewards in the environment.
    fn num_rewards() -> usize;
    /// Retrieves the reward value for a given index.
    ///
    /// # Parameters
    /// - `i`: The index of the reward.
    ///
    /// # Returns
    /// The reward value as a `f32`.
    fn get_reward(i: usize) -> f32;

    /// Builds the transition probability for a given state-action-next state-reward tuple.
    ///
    /// # Parameters
    /// - `s`: The current state.
    /// - `a`: The action taken.
    /// - `s_p`: The next state.
    /// - `r`: The reward received.
    ///
    /// # Returns
    /// The transition probability as a `f32`.
    fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32;


    /// Resets the environment to a random state using a specified seed.
    ///
    /// # Parameters
    /// - `seed`: The seed for the random state generator.
    fn reset_random_state(&mut self, seed: u64);

    /// Retrieves the available actions in the current state as an array.
    ///
    /// # Returns
    /// An `Array1` containing the available actions.
    fn available_actions_ids(&self) -> Array1<usize>;

    /// Deletes available actions (presumably from some internal state or structure).
    fn available_action_delete(&self);


    //remove

    /// Executes a step in the environment with the given action.
    ///
    /// # Parameters
    /// - `action`: The action to execute.
    fn step(&mut self, action: usize);

    /// Deletes the environment (presumably some cleanup operation).
    fn delete(&mut self);


    /// Displays the current state of the environment (for debugging or visualization).
    fn display(&self);

    /// Plays a strategy within the environment, optionally printing the environment's state.
    ///
    /// # Parameters
    /// - `strategy`: A `HashMap` where the key is the state identifier and the value is the action to take.
    /// - `print`: A boolean indicating whether to print the environment's state after each step.
    fn play_strategy(&mut self, strategy: HashMap<usize, usize>, print: bool) {
        if print {
            self.display();
        }
        let mut check_loop = HashMap::new();
        loop {
            if check_loop.contains_key(&self.state_id()) {
                println!("Boucle");
                break;
            }
            check_loop.insert(self.state_id(), true);

            if self.is_terminal() {
                break;
            }
            let action = strategy.get(&self.state_id());
            if action.is_none() {
                break;
            }

            self.step(*action.unwrap());
            if print {
                self.display();
            }
        }
        println!("score {}, state {}", self.score(), self.state_id());
    }
}

pub trait TabularEnv<const NUM_STATES: usize, const NUM_ACTIONS: usize, const NUM_REWARDS: usize> :
Default + Clone + BaseEnv +
ActionEnv<NUM_ACTIONS>{

}

pub trait BaseEnv : Default + Clone {
    fn is_terminal(&self) -> bool;
    fn score(&self) -> f32;
    fn reset(&mut self);

}

pub trait ActionEnv<const NUM_ACTIONS: usize>: Default + Clone {
    fn available_actions_ids(&self) -> impl Iterator<Item=usize>;

    fn step(&mut self, action: usize);
}

pub trait DeepDiscreteActionsEnv<const NUM_STATES_FEATURES: usize, const NUM_ACTIONS: usize>:
Default + Clone + ActionEnv<NUM_ACTIONS> + BaseEnv
{
    fn state_description(&self) -> [f32; NUM_STATES_FEATURES];
    fn action_mask(&self) -> [f32; NUM_ACTIONS];

}

/// Trait that defines the capability for a game to be played interactively with a human player.
/// Implementing this trait allows a game to be run in an interactive mode,
/// utilizing all the provided methods to facilitate a complete game session.
pub trait Playable: Default + Clone {
    /// Starts an interactive game session with a human player.
    ///
    /// This function should orchestrate the game flow, handling user input and game state updates.
    /// It should leverage all the existing methods to allow a real game of Farkle to be played,
    /// including initializing the game state, processing player actions, and determining game outcomes.
    fn play_as_human();
    fn play_as_random_ai() -> [usize; 2];
}