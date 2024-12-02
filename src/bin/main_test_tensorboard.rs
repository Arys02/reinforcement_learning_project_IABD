extern crate IABD4_reinforcement_learning;

use std::collections::HashMap;
use tensorboard_rs::summary_writer::SummaryWriter;

fn main() {

    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    for n_iter in 0..100 {
        let mut map = HashMap::new();
        map.insert("x^2".to_string(), (n_iter as f32) * (n_iter as f32));
        writer.add_scalars("data/scalar_group", &map, n_iter);
    }
    for n_iter in 0..100 {
        let mut map = HashMap::new();
        map.insert("x1".to_string(), (n_iter as f32));
        writer.add_scalars("data2/scalar_group", &map, n_iter);
    }
    writer.flush();


}