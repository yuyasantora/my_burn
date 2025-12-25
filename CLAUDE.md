# My Burn - 汎用画像学習フレームワーク

## プロジェクト概要

Burn (Rust ML フレームワーク) を使用した汎用画像学習ツール。
設定ファイル (YAML) で任意のデータセット・モデル・タスクを指定して訓練可能。

---

## アーキテクチャ

```
src/
├── main.rs              # CLI エントリポイント
├── config.rs            # 設定構造体（YAML読み込み）
├── data/
│   ├── mod.rs
│   ├── batcher.rs       # バッチ処理
│   ├── transforms.rs    # 画像変換（リサイズ、正規化）
│   └── loader.rs        # 汎用データローダー
├── models/
│   ├── mod.rs
│   ├── registry.rs      # モデル選択・生成
│   ├── simple_cnn.rs    # シンプルCNN
│   └── resnet.rs        # ResNet (Phase 2)
└── training/
    ├── mod.rs
    └── trainer.rs       # 汎用訓練ループ
```

---

## 設定ファイル形式 (config.yaml)

```yaml
task: classification
dataset:
  path: ./data/my_dataset
  image_size: 224
  train_split: train  # default
  val_split: val      # default
model:
  type: simplecnn  # simplecnn, resnet18, resnet34, resnet50
  num_classes: 0   # 0 = auto detect
  hidden_size: 512
  dropout: 0.5
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.0001
  num_workers: 4
  seed: 42
```

データセット構造:
```
dataset/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```

---

## 実装ステップ

### Step 1: 設定ファイル対応 [完了]

- [x] Cargo.toml 更新 (serde, serde_yaml, clap 追加)
- [x] src/config.rs 作成

**src/config.rs の構造:**
- `AppConfig` - 全体設定
- `TaskType` - classification, segmentation, detection, generation
- `DatasetConfig` - path, image_size, train_split, val_split
- `ModelType` - SimpleCnn, ResNet18, ResNet34, ResNet50
- `ModelConfigYaml` - model_type, num_classes, hidden_size, dropout
- `TrainingConfigYaml` - epochs, batch_size, learning_rate, num_workers, seed

### Step 2: データローダー汎化 [未完了]

作成ファイル:
- src/data/mod.rs
- src/data/batcher.rs (ClassificationBatcher - image_size パラメータ化)
- src/data/transforms.rs (image_to_tensor 関数)
- src/data/loader.rs (create_classification_loaders, count_classes)

削除ファイル:
- src/data.rs (旧)
- src/dataset.rs (旧)

**src/data/mod.rs:**
```rust
pub mod batcher;
pub mod transforms;
pub mod loader;

pub use batcher::*;
pub use transforms::*;
pub use loader::*;
```

**src/data/transforms.rs:**
```rust
use burn::prelude::*;
use image::{imageops::FilterType, RgbImage};

pub fn image_to_tensor<B: Backend>(
    pixels: &[u8],
    width: u32,
    height: u32,
    target_size: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let img = RgbImage::from_raw(width, height, pixels.to_vec())
        .expect("Failed to create image");
    let resized = image::imageops::resize(
        &img, target_size as u32, target_size as u32, FilterType::Triangle,
    );
    let resized_pixels: Vec<u8> = resized.into_raw();
    let data = TensorData::new(resized_pixels, [target_size, target_size, 3]);
    let tensor: Tensor<B, 3> = Tensor::from_data(data.convert::<B::FloatElem>(), device);
    tensor.permute([2, 0, 1]) / 255.0
}
```

**src/data/batcher.rs:**
```rust
use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::vision::{Annotation, ImageDatasetItem, PixelDepth},
    },
    prelude::*,
};
use crate::data::transforms::image_to_tensor;

#[derive(Clone)]
pub struct ClassificationBatcher {
    pub image_size: usize,
}

impl ClassificationBatcher {
    pub fn new(image_size: usize) -> Self {
        Self { image_size }
    }
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, ImageDatasetItem, ClassificationBatch<B>> for ClassificationBatcher {
    fn batch(&self, items: Vec<ImageDatasetItem>, device: &B::Device) -> ClassificationBatch<B> {
        let images: Vec<Tensor<B, 3>> = items
            .iter()
            .map(|item| {
                let pixels: Vec<u8> = item.image.iter()
                    .map(|p| match p {
                        PixelDepth::U8(v) => *v,
                        _ => 0u8,
                    })
                    .collect();
                image_to_tensor::<B>(
                    &pixels,
                    item.image_width as u32,
                    item.image_height as u32,
                    self.image_size,
                    device,
                )
            })
            .collect();

        let targets: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| {
                if let Annotation::Label(y) = item.annotation {
                    Tensor::<B, 1, Int>::from_data(
                        [(y as i64).elem::<B::IntElem>()],
                        device,
                    )
                } else {
                    panic!("Expected label annotation")
                }
            })
            .collect();

        let images = Tensor::stack(images, 0);
        let targets = Tensor::cat(targets, 0);
        ClassificationBatch { images, targets }
    }
}
```

**src/data/loader.rs:**
```rust
use burn::data::{dataloader::DataLoaderBuilder, dataset::vision::ImageFolderDataset};
use std::path::Path;
use crate::config::DatasetConfig;
use crate::data::ClassificationBatcher;

pub fn count_classes(path: &Path) -> usize {
    std::fs::read_dir(path)
        .expect("Failed to read dataset directory")
        .filter(|entry| entry.as_ref().map(|e| e.path().is_dir()).unwrap_or(false))
        .count()
}

pub fn create_classification_loaders<B: burn::prelude::Backend>(
    config: &DatasetConfig,
    batch_size: usize,
    num_workers: usize,
    seed: u64,
) -> (
    burn::data::dataloader::DataLoader<ImageFolderDataset, B>,
    burn::data::dataloader::DataLoader<ImageFolderDataset, B>,
    usize,
) {
    let train_path = config.path.join(&config.train_split);
    let val_path = config.path.join(&config.val_split);
    let num_classes = count_classes(&train_path);
    let batcher = ClassificationBatcher::new(config.image_size);

    let train_dataset = ImageFolderDataset::new_classification(&train_path)
        .expect("Failed to load train dataset");
    let val_dataset = ImageFolderDataset::new_classification(&val_path)
        .expect("Failed to load val dataset");

    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .build(train_dataset);

    let val_loader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(val_dataset);

    (train_loader, val_loader, num_classes)
}
```

### Step 3: モデル汎化 [未完了]

作成ファイル:
- src/models/mod.rs
- src/models/simple_cnn.rs (現在の model.rs を移行・パラメータ化)
- src/models/registry.rs (モデル選択)

削除ファイル:
- src/model.rs (旧)

**src/models/mod.rs:**
```rust
pub mod simple_cnn;
pub mod registry;

pub use simple_cnn::*;
pub use registry::*;
```

**src/models/simple_cnn.rs:**
現在の model.rs を移行。image_size に応じた動的なサイズ計算を追加。

**src/models/registry.rs:**
```rust
use crate::config::{ModelType, ModelConfigYaml};
use crate::models::simple_cnn::{SimpleCnn, SimpleCnnConfig};
use burn::prelude::*;

pub fn create_model<B: Backend>(
    config: &ModelConfigYaml,
    num_classes: usize,
    device: &B::Device,
) -> SimpleCnn<B> {
    let num_classes = if config.num_classes == 0 { num_classes } else { config.num_classes };

    match config.model_type {
        ModelType::SimpleCnn => {
            SimpleCnnConfig::new(num_classes, config.hidden_size)
                .with_dropout(config.dropout)
                .init(device)
        }
        _ => todo!("ResNet models not yet implemented"),
    }
}
```

### Step 4: 訓練ロジック整理 [未完了]

作成ファイル:
- src/training/mod.rs
- src/training/trainer.rs

移行元:
- src/training.rs (旧)

### Step 5: CLI対応 [未完了]

**src/main.rs:**
```rust
mod config;
mod data;
mod models;
mod training;

use clap::Parser;
use std::path::PathBuf;
use burn::backend::{Autodiff, Wgpu};
use crate::config::AppConfig;

#[derive(Parser)]
#[command(name = "my_burn")]
#[command(about = "Universal image training tool")]
struct Cli {
    #[arg(short, long)]
    config: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let config = AppConfig::load(&cli.config).expect("Failed to load config");

    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    // 設定に基づいて訓練実行
    training::run::<MyAutodiffBackend>(&config, device);
}
```

### Step 6: ビルド確認 [未完了]

```bash
cargo build
cargo run -- --config config.yaml
```

---

## 使用例

```bash
# 設定ファイルで実行
cargo run -- --config config.yaml

# サンプル config.yaml
task: classification
dataset:
  path: ./food-101/images
  image_size: 64
model:
  type: simplecnn
training:
  epochs: 10
  batch_size: 32
```

---

## 今後の拡張 (Phase 2+)

- ResNet18/34/50 実装
- セグメンテーション (U-Net)
- 物体検出 (YOLOX)
- COCO/VOC アノテーション対応
