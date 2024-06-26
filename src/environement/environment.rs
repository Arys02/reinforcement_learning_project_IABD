use ndarray::Array1;

pub trait Environment {
    fn new() -> Self;

    fn state_id(&self) -> usize;

    fn available_actions(i: usize) -> Array1<usize>;
    fn is_terminal_state(state: usize) -> bool;

    fn from_random_state() -> Self;

    fn reset(&mut self);

    fn num_states() -> usize;

    fn num_actions() -> usize;

    fn num_rewards() -> usize;

    fn get_reward(i: usize) -> f32;

    fn get_transition_probability(s: usize, a: usize, s_p: usize, r: usize) -> f32;

    fn reset_random_state(&mut self, seed: u64);

    fn available_action(&self) -> Array1<usize>;

    fn available_action_delete(&self);

    fn is_terminal(&self) -> bool;

    fn is_forbidden(&self, state: usize) -> bool;

    fn step(&mut self, action: usize);

    fn delete(&mut self);

    fn score(&self) -> f64;

    fn display(&self);
}
