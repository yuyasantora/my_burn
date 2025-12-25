use serde::Deserialize;
use std::path::PathBuf;

// --- アプリ設定はserdeを使用 ---
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub task: TaskType,
    pub dataset: DatasetConfig,
    pub model: ModelConfigYaml,
    pub training: TrainingConfigYaml,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    Classification,
    Segmentation,
    Detection,
    Generation,
}

#[derive(Debug, Deserialize)]
pub struct DatasetConfig {
    pub path: PathBuf,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_train_split")]
    pub train_split: String,
    #[serde(default = "default_val_split")]
    pub val_split: String,
}

fn default_image_size() -> usize { 224 }
fn default_train_split() -> String { "train".to_string() }
fn default_val_split() -> String { "val".to_string() }

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    SimpleCnn,
    ResNet18,
    ResNet34,
    ResNet50,
}

// YAMLから読み込む設定
#[derive(Debug, Deserialize)]
pub struct ModelConfigYaml {
    #[serde(rename = "type")]
    pub model_type: ModelType,
    #[serde(default)]
    pub num_classes: usize, // 0 = auto
    #[serde(default)]
    pub pretrained: bool,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
}

fn default_hidden_size() -> usize { 512 }
fn default_dropout() -> f64 { 0.5 }

#[derive(Debug, Deserialize)]
pub struct TrainingConfigYaml {
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_epochs() -> usize { 10 }
fn default_batch_size() -> usize { 32 }
fn default_learning_rate() -> f64 { 1e-4 }
fn default_num_workers() -> usize { 4 }
fn default_seed() -> u64 { 42 }

impl AppConfig {
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
