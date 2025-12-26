use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::vision::{Annotation, ImageDatasetItem, PixelDepth},
    },
    prelude::*,
};

use crate::data::transforms::image_to_tensor;

#[derive(Clone, Debug)]
pub struct ClassificationBatcher {
    pub image_size: usize
}

impl ClassificationBatcher {
    pub fn new(image_size: usize) -> Self {
        Self { image_size }
    }
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>
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