use crate::{models::simple_cnn::{SimpleCnn, SimpleCnnConfig}, config::ModelConfigYaml};
use burn::prelude::*;

use burn::prelude::*;

pub fn create_model<B: Backend>(
    config: &ModelConfigYaml,
    num_classes: usize,
    device: &B::Device,
) -> SimpleCnn<B> {
    let num_classes = if config.num_classes == 0 {num_classes} else {config.num_classes};
    match config.model_type {
        ModelType::SimpleCnn => {
            SimpleCnnConfig::new(num_classes, config.hidden_size)
                .with_dropout(config.dropout)
                .init(device)
        }
        _ => todo!("ResNet models not yet implemented"),
    }
}
