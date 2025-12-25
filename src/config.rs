use burn::config::Config;
use std::path::PathBuf;

#[derive(Config, Debug)]
pub struct AppConfig {
    pub task: TaskType,
    pub dataset: DatasetConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
}

#[derive(Config, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Classification,
    Segmentation,
    Detection,
    Generation,
}

#[derive(Config, Debug)]
pub struct DatasetConfig {
    pub path: PathBuf,
    #[config(default = 224)]
    pub image_size: usize,
    #[config(default = "train")]
    pub train_split: String,
    #[config(default = "val")]
    pub val_split: String,
}

#[derive(Config, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    SimpleCnn,
    ResNet18,
    ResNet34,
    ResNet50,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub model_type: ModelType,
    #[config(default = 0)] // 0 = auto detect from dataset
    pub num_classes: usize,
    #[config(default = false)]
    pub pretrained: bool,
    #[config(default = 512)]
    pub hidden_size: usize,
    #[config(default = 0.5)]
    pub dropout: f64,
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
}

impl AppConfig {
    pub fn load_yaml(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
