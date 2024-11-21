// src/logger.rs

use crate::training_observer::{Hyperparameters, TrainingEvent, TrainingObserver};
use burn::tensor::backend::Backend;
use chrono::prelude::*;
use csv::Writer;
use serde::Serialize;
use std::fs::{create_dir_all, File};
use std::io::BufWriter;
use std::os::raw::c_int;

#[derive(Serialize)]
pub struct LogRecord {
    pub event_type: String,
    pub total_score: Option<f32>,
    pub epsilon: Option<f32>,
    pub average_loss: Option<f32>,
    pub timestamp: String,
    pub average_score_per_episode: Option<f32>, // Average score per episode
    pub average_steps_per_episode: Option<f32>, // Average steps per episode
    pub win_count: Option<usize>,
    pub best_score: Option<f32>,
    pub average_time: Option<f32>,
    pub epoch: Option<usize>
}

pub struct Logger {
    pub hyperparameters: Option<Hyperparameters>,
    csv_writer: Writer<BufWriter<File>>,
    metrics_file_path: String,
    hyperparameters_file_path: String,
}

impl Logger {
    pub fn new(model_name: &str) -> Self {
        let now: DateTime<Local> = Local::now();
        let datetime_str = now.format("%Y%m%d_%H%M%S").to_string();

        let model_dir_path = format!("logs/{}", model_name);
        let metrics_dir_path = format!("logs/{}/metrics", model_name);
        let hyperparameters_dir_path = format!("logs/{}/hyperparameters", model_name);

        create_dir_all(&model_dir_path).expect("Failed to create directory");
        create_dir_all(&metrics_dir_path).expect("Failed to create directory");
        create_dir_all(&hyperparameters_dir_path).expect("Failed to create directory");

        let metrics_file_name = format!("{}_{}.csv", model_name, datetime_str);
        let metrics_file_path = format!("{}/{}", metrics_dir_path, metrics_file_name);

        let hyperparameters_file_name = format!("{}_{}_hyperparameters.json", model_name, datetime_str);
        let hyperparameters_file_path = format!("{}/{}", hyperparameters_dir_path, hyperparameters_file_name);

        let metrics_file = File::create(&metrics_file_path).expect("Failed to create log file");
        let writer = BufWriter::new(metrics_file);
        let mut csv_writer = Writer::from_writer(writer);

        // csv_writer
        //     .write_record(&[
        //         "event_type",
        //         "total_score",
        //         "epsilon",
        //         "average_loss",
        //         "timestamp",
        //         "average_score_per_episode",
        //         "average_steps_per_episode",
        //         "win_count",
        //         "best_score",
        //         "average_time",
        //         "epoch",
        //     ])
        //     .expect("Failed to write CSV headers");

        Logger {
            hyperparameters: None,
            csv_writer,
            metrics_file_path,
            hyperparameters_file_path,
        }
    }

    pub fn close(&mut self) {
        self.csv_writer.flush().expect("Failed to flush CSV writer");
    }
}

impl TrainingObserver for Logger {
    fn on_event(&mut self, event: &TrainingEvent) {
        let timestamp = Local::now().to_rfc3339();

        match event {
            TrainingEvent::HyperparametersLogged { hyperparameters } => {
                self.hyperparameters = Some(hyperparameters.clone());

                let json_output = serde_json::json!({
                    "timestamp": timestamp,
                    "hyperparameters": hyperparameters
                });

                // Convert to pretty-printed JSON string
                let json_string = serde_json::to_string_pretty(&json_output)
                    .expect("Failed to serialize hyperparameters");

                // Write to JSON file
                std::fs::write(&self.hyperparameters_file_path, json_string)
                    .expect("Failed to write hyperparameters JSON file");
            },


            TrainingEvent::LoggingSummary {
                episodes: _,
                total_score,
                average_score_per_episode,
                average_steps_per_episode,
                average_loss,
                epsilon,
                win_count,
                best_score,
                average_time,
                epoch,
            } => {
                let record = LogRecord {
                    event_type: "LoggingSummary".to_string(),
                    total_score: Some(*total_score),
                    epsilon: Some(*epsilon),
                    average_loss: Some(*average_loss),
                    timestamp: timestamp.clone(),
                    average_score_per_episode: Some(*average_score_per_episode),
                    average_steps_per_episode: Some(*average_steps_per_episode),
                    win_count: Some(*win_count),
                    best_score: Some(*best_score),
                    average_time: Some(*average_time),
                    epoch: Some(*epoch),

                };

                self.csv_writer
                    .serialize(record)
                    .expect("Failed to write LoggingSummary record");
            }
        }
    }
}
