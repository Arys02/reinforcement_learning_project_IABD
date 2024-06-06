use ndarray::Array1;

pub trait Environement {
    fn reset_random_state(&mut self, seed: u64);

    fn new() -> Self;

    fn available_action(&self) -> Array1<i32>;

    fn is_terminal(&self) -> bool;

    fn state_id(&self) -> i32;

    fn step(&mut self, action: i32);

    fn score(&self) -> f64;

    fn display(&self);
}
