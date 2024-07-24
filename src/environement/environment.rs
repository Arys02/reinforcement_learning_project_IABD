use std::collections::HashMap;

use ndarray::Array1;

pub trait Environment {
    fn new() -> Self;

    fn state_id(&self) -> usize;

    fn from_random_state() -> Self;

    fn reset(&mut self);

    fn num_states() -> usize;

    fn num_actions() -> usize;

    fn num_rewards() -> usize;

    fn get_reward(i: usize) -> f32;
    fn build_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32;

    fn get_transition_probability(&mut self, s: usize, a: usize, s_p: usize, r: usize) -> f32;

    fn reset_random_state(&mut self, seed: u64);

    fn available_action(&self) -> Array1<usize>;

    fn available_action_delete(&self);

    fn is_terminal(&self) -> bool;

    fn is_forbidden(&self, state: usize) -> bool;

    fn step(&mut self, action: usize);

    fn delete(&mut self);

    fn score(&self) -> f32;

    fn display(&self);
    fn play_strategy(&mut self, strategy: HashMap<usize, usize>, print:bool) {
        if print {
            self.display();
        }
        let mut check_loop = HashMap::new();
        loop {
            if check_loop.contains_key(&self.state_id()){
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
    }
}
