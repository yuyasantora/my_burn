use crate::{
    data::{Food101Batch, Food101Batcher},
    dataset::Food101Loader,
    model::{Model, ModelConfig}
};

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::ImageFolderDataset},
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep
    }
};

// 分類用順伝播
impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}


// 訓練ループ
impl<B: AutodiffBackend> TrainStep<Food101Batch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: Food101Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

// 検証ループ
impl<B: Backend> ValidStep<Food101Batch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: Food101Batch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

// 学習の設定
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
      pub optimizer: AdamConfig,
      #[config(default = 10)]
      pub num_epochs: usize,
      #[config(default = 32)]
      pub batch_size: usize,
      #[config(default = 4)]
      pub num_workers: usize,
      #[config(default = 42)]
      pub seed: u64,
      #[config(default = 1.0e-4)]
      pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

// 訓練関数
pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config.save(format!("{artifact_dir}/config.json")).expect("Config should be saved successfully");
    B::seed(&device, config.seed);

    let batcher = Food101Batcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::food101_train());
    
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::food101_test());
    
    // 学習器の設定
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    // 学習実行
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // モデル保存
    model_trained
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");


}