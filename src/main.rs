mod model;
mod data;
mod dataset;
mod training;

use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};

fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let config = TrainingConfig::new(
        ModelConfig::new(101, 512),
        AdamConfig::new(),
    );
    let artifact_dir = "./artifacts";

    // 訓練実行
    training::train::<MyAutodiffBackend>(
        artifact_dir,
        config,
        device.clone(),
    );
}

