use burn::data::{dataloader::{DataLoader, DataLoaderBuilder}, dataset::vision::ImageFolderDataset};
use std::sync::Arc;
use crate::data::ClassificationBatch;
use burn::prelude::*;
use std::path::Path;
use crate::config::DatasetConfig;
use crate::data::ClassificationBatcher;

pub fn count_classes(path: &Path) -> usize {
    std::fs::read_dir(path)
        .expect("Failed to read dataset directory")
        .filter(|entry| entry.as_ref().map(|e| e.path().is_dir()).unwrap_or(false))
        .count()
}

pub fn create_classification_loaders<B: Backend>(
    config: &DatasetConfig,
    batch_size: usize,
    num_workers: usize,
    seed: u64,
) -> (
    Arc<dyn DataLoader<B, ClassificationBatch<B>>>,
    Arc<dyn DataLoader<B, ClassificationBatch<B>>>,
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
