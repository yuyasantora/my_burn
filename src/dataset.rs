use burn::data::dataset::vision::ImageFolderDataset;
use std::path::Path;

const DATASET_PATH: &str = "C:/Users/yuya.oohara/Desktop/AI/burn/my_food101/food-101/food-101/images";

// train/testの読み込み
pub trait Food101Loader {
    fn food101_train() -> Self;
    fn food101_test() -> Self;
}

impl Food101Loader for ImageFolderDataset {
    fn food101_train() -> Self {
        let path = Path::new(DATASET_PATH);
        Self::new_classification(path).expect("Failed to load Food101 train dataset")
    }

    fn food101_test() -> Self {
        let path = Path::new(DATASET_PATH);
        Self::new_classification(path).expect("Failed to load Food101 test dataset")
    }
}