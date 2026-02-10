//! Image conversion utilities between image-rs and CubeCL tensors.
//!
//! This module provides functionality to convert between image-rs image types
//! and CubeCL tensor representations for GPU processing.

use image::{DynamicImage, GenericImageView, ImageBuffer};

/// Represents an image as a CubeCL tensor for GPU processing.
///
/// The tensor is stored in HWC (Height, Width, Channels) format.
pub struct ImageTensor {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub data: Vec<f32>,
}

impl ImageTensor {
    /// Create a new ImageTensor from raw data.
    pub fn new(width: u32, height: u32, channels: u32, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            (width * height * channels) as usize,
            "Data length must match dimensions"
        );
        Self {
            width,
            height,
            channels,
            data,
        }
    }

    /// Convert from image-rs DynamicImage to ImageTensor.
    pub fn from_dynamic_image(img: &DynamicImage) -> Self {
        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();
        let channels = 4;

        let data: Vec<f32> = rgba
            .pixels()
            .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
            .collect();

        Self {
            width,
            height,
            channels,
            data,
        }
    }

    /// Convert from ImageTensor to image-rs DynamicImage.
    pub fn to_dynamic_image(&self) -> DynamicImage {
        match self.channels {
            1 => {
                // Grayscale
                let buffer: Vec<u8> = self
                    .data
                    .iter()
                    .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                    .collect();
                let img_buffer =
                    ImageBuffer::from_raw(self.width, self.height, buffer)
                        .expect("Failed to create image buffer");
                DynamicImage::ImageLuma8(img_buffer)
            }
            3 => {
                // RGB
                let buffer: Vec<u8> = self
                    .data
                    .iter()
                    .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                    .collect();
                let img_buffer =
                    ImageBuffer::from_raw(self.width, self.height, buffer)
                        .expect("Failed to create image buffer");
                DynamicImage::ImageRgb8(img_buffer)
            }
            4 => {
                // RGBA
                let buffer: Vec<u8> = self
                    .data
                    .iter()
                    .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                    .collect();
                let img_buffer =
                    ImageBuffer::from_raw(self.width, self.height, buffer)
                        .expect("Failed to create image buffer");
                DynamicImage::ImageRgba8(img_buffer)
            }
            _ => panic!("Unsupported channel count: {}", self.channels),
        }
    }

    /// Get pixel value at (x, y, channel).
    pub fn get_pixel(&self, x: u32, y: u32, c: u32) -> f32 {
        let idx = ((y * self.width + x) * self.channels + c) as usize;
        self.data[idx]
    }

    /// Set pixel value at (x, y, channel).
    pub fn set_pixel(&mut self, x: u32, y: u32, c: u32, value: f32) {
        let idx = ((y * self.width + x) * self.channels + c) as usize;
        self.data[idx] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_tensor_creation() {
        let data = vec![0.0; 100 * 100 * 3];
        let tensor = ImageTensor::new(100, 100, 3, data);
        assert_eq!(tensor.width, 100);
        assert_eq!(tensor.height, 100);
        assert_eq!(tensor.channels, 3);
    }

    #[test]
    fn test_pixel_access() {
        let data = vec![0.0; 10 * 10 * 3];
        let mut tensor = ImageTensor::new(10, 10, 3, data);
        tensor.set_pixel(5, 5, 0, 1.0);
        assert_eq!(tensor.get_pixel(5, 5, 0), 1.0);
    }

    #[test]
    fn test_dynamic_image_conversion() {
        let img = DynamicImage::new_rgb8(10, 10);
        let tensor = ImageTensor::from_dynamic_image(&img);
        assert_eq!(tensor.width, 10);
        assert_eq!(tensor.height, 10);
        assert_eq!(tensor.channels, 4); // Converted to RGBA
        
        let reconstructed = tensor.to_dynamic_image();
        assert_eq!(reconstructed.dimensions(), (10, 10));
    }
}
