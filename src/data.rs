use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::vision::{Annotation, ImageDatasetItem, PixelDepth},
    },
    prelude::*,
};

pub const IMAGE_SIZE: usize = 64;

#[derive(Clone, Default)]
pub struct Food101Batcher {}

#[derive(Clone, Debug)]
pub struct Food101Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, ImageDatasetItem, Food101Batch<B>> for Food101Batcher {
    fn batch(&self, items: Vec<ImageDatasetItem>, device: &B::Device) -> Food101Batch<B> {
        let images: Vec<Tensor<B, 3>> = items
            .iter()
            .map(|item| {
                // PixelDepth -> u8 に変換
                let pixels: Vec<u8> = item.image.iter()
                    .map(|p| match p {
                        PixelDepth::U8(v) => *v,
                        _ => 0u8,
                    })
                    .collect();

                let data = TensorData::new(pixels, [item.image_height, item.image_width, 3]);
                data.convert::<B::FloatElem>()
            })
            .map(|data| Tensor::<B, 3>::from_data(data, device))
            .map(|tensor| tensor.permute([2, 0, 1])) // [H,W,C] -> [C,H,W]
            .map(|tensor| tensor / 255)
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

        Food101Batch { images, targets }
    }
}
