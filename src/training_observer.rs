// src/training_observer.rs

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::time::Duration;
use serde::Serialize;

pub enum TrainingEvent{
    HyperparametersLogged {
        hyperparameters: Hyperparameters,
    },

    LoggingSummary {
        episodes: usize,
        total_score: f32,
        average_score_per_episode: f32,
        average_steps_per_episode: f32,
        average_loss: f32,
        epsilon: f32,
        win_count: usize,
        best_score: f32,
        average_time: f32,
        epoch: usize,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Hyperparameters {
    pub num_episodes: usize,
    pub replay_capacity: usize,
    pub gamma: f32,
    pub alpha: f32,
    pub start_epsilon: f32,
    pub final_epsilon: f32,
    pub batch_size: usize,
    pub log_interval: usize,
}

pub trait TrainingObserver{
    fn on_event(&mut self, event: &TrainingEvent);
}
