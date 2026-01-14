use crate::config::AppConfig;
use crate::data::{ClassificationBatch, create_classification_loaders};
use crate::models::{SimpleCnn, create_model};

use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::AccuracyMetric,
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};

// 分類用順伝播
impl<B: Backend> SimpleCnn<B> {
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

// 訓練ステップ
impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>, ClassificationOutput<B>> for SimpleCnn<B> {
    fn step(&self, batch: ClassificationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

// 検証ステップ
impl<B: Backend> ValidStep<ClassificationBatch<B>, ClassificationOutput<B>> for SimpleCnn<B> {
    fn step(&self, batch: ClassificationBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

// 汎用訓練関数
pub fn run<B: AutodiffBackend>(config: &AppConfig, device: B::Device) {
    let artifact_dir = "artifacts";
    create_artifact_dir(artifact_dir);

    B::seed(&device, config.training.seed);

    // データローダー作成
    let (train_loader, val_loader, num_classes) = create_classification_loaders::<B>(
        &config.dataset,
        config.training.batch_size,
        config.training.num_workers,
        config.training.seed,
    );

    // モデル作成
    let model = create_model::<B>(&config.model, num_classes, &device);

    // 学習器の設定
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.training.epochs)
        .summary()
        .build(
            model,
            AdamConfig::new().init(),
            config.training.learning_rate,
        );

    // 学習実行
    let model_trained = learner.fit(train_loader, val_loader);

    // モデル保存
    model_trained
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
