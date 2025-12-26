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
    let resized_img = image::imageops::resize(
        &img, target_size as u32, target_size as u32, FilterType::Triangle,
    );
    let resized_pixels: Vec<u8> = resized_img.into_raw();
    let data = TensorData::new(resized_pixels, [target_size, target_size, 3]);
    let tensor: Tensor<B, 3> = Tensor::from_data(data.convert::<B::FloatElem>(), device);
    tensor.permute([2, 0, 1]) / 255.0
}

