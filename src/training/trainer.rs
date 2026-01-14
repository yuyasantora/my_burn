use crate::{
    data::{batcher::ClassificationBathcer, crate_classification_loaders, ClassificationBatch},
    model::{SimpleCnn, SimpleCnnConfig},
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
    },
};

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        image: Tensor<B, 4>,
        targets: Tensor<B, 3>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(image);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> Trainstep for Model<B> {
    type Input = ClassificationBatch<B>;
    type Oupput = ClassificationOutput<B>;

    fn step(&self, batch: ClassificationBatch<B>) -> TraniOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backend(), item)
    }
}
