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

    /// Create ImageTensor from raw HWC (Height, Width, Channels) format data.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `channels` - Number of channels
    /// * `hwc_data` - Data in HWC format where pixels are stored as [h][w][c]
    ///
    /// # Example
    ///
    /// ```rust
    /// use cubecv::ImageTensor;
    ///
    /// let hwc_data = vec![0.0; 10 * 10 * 3];
    /// let tensor = ImageTensor::from_hwc_data(10, 10, 3, hwc_data);
    /// ```
    pub fn from_hwc_data(width: u32, height: u32, channels: u32, hwc_data: Vec<f32>) -> Self {
        Self::new(width, height, channels, hwc_data)
    }

    /// Create ImageTensor from raw CHW (Channels, Height, Width) format data.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `channels` - Number of channels
    /// * `chw_data` - Data in CHW format where pixels are stored as [c][h][w]
    ///
    /// # Example
    ///
    /// ```rust
    /// use cubecv::ImageTensor;
    ///
    /// let chw_data = vec![0.0; 3 * 10 * 10];
    /// let tensor = ImageTensor::from_chw_data(10, 10, 3, chw_data);
    /// ```
    pub fn from_chw_data(width: u32, height: u32, channels: u32, chw_data: Vec<f32>) -> Self {
        assert_eq!(
            chw_data.len(),
            (width * height * channels) as usize,
            "Data length must match dimensions"
        );

        let mut hwc_data = vec![0.0; (width * height * channels) as usize];

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let chw_idx = (c * height * width + h * width + w) as usize;
                    let hwc_idx = ((h * width + w) * channels + c) as usize;
                    hwc_data[hwc_idx] = chw_data[chw_idx];
                }
            }
        }

        Self {
            width,
            height,
            channels,
            data: hwc_data,
        }
    }

    /// Convert to HWC (Height, Width, Channels) format data.
    ///
    /// Returns a vector in HWC format where pixels are stored as [h][w][c].
    /// This is the native format of ImageTensor, so this is a clone operation.
    pub fn to_hwc_data(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Convert to CHW (Channels, Height, Width) format data.
    ///
    /// Returns a vector in CHW format where pixels are stored as [c][h][w].
    /// This format is commonly used in deep learning frameworks like PyTorch.
    pub fn to_chw_data(&self) -> Vec<f32> {
        let mut chw_data = vec![0.0; self.data.len()];

        for c in 0..self.channels {
            for h in 0..self.height {
                for w in 0..self.width {
                    let hwc_idx = ((h * self.width + w) * self.channels + c) as usize;
                    let chw_idx = (c * self.height * self.width + h * self.width + w) as usize;
                    chw_data[chw_idx] = self.data[hwc_idx];
                }
            }
        }

        chw_data
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

    #[test]
    fn test_hwc_chw_conversion() {
        // Create a simple 2x2x3 tensor with known values
        // HWC: [R0,G0,B0, R1,G1,B1, R2,G2,B2, R3,G3,B3]
        let hwc_data = vec![
            1.0, 2.0, 3.0,  // pixel (0,0)
            4.0, 5.0, 6.0,  // pixel (1,0)
            7.0, 8.0, 9.0,  // pixel (0,1)
            10.0, 11.0, 12.0, // pixel (1,1)
        ];
        
        let tensor = ImageTensor::from_hwc_data(2, 2, 3, hwc_data.clone());
        
        // Convert to CHW
        let chw_data = tensor.to_chw_data();
        
        // CHW should be: [R0,R1,R2,R3, G0,G1,G2,G3, B0,B1,B2,B3]
        let expected_chw = vec![
            1.0, 4.0, 7.0, 10.0,  // R channel
            2.0, 5.0, 8.0, 11.0,  // G channel
            3.0, 6.0, 9.0, 12.0,  // B channel
        ];
        
        assert_eq!(chw_data, expected_chw);
        
        // Convert back to HWC
        let tensor2 = ImageTensor::from_chw_data(2, 2, 3, chw_data);
        let hwc_data2 = tensor2.to_hwc_data();
        
        assert_eq!(hwc_data, hwc_data2);
    }

    #[test]
    fn test_from_chw_data() {
        // Test creating tensor from CHW data
        let chw_data = vec![
            1.0, 2.0, 3.0, 4.0,  // Channel 0
            5.0, 6.0, 7.0, 8.0,  // Channel 1
            9.0, 10.0, 11.0, 12.0, // Channel 2
        ];
        
        let tensor = ImageTensor::from_chw_data(2, 2, 3, chw_data);
        
        assert_eq!(tensor.width, 2);
        assert_eq!(tensor.height, 2);
        assert_eq!(tensor.channels, 3);
        
        // Verify pixel values
        assert_eq!(tensor.get_pixel(0, 0, 0), 1.0);
        assert_eq!(tensor.get_pixel(0, 0, 1), 5.0);
        assert_eq!(tensor.get_pixel(0, 0, 2), 9.0);
    }

    #[test]
    fn test_hwc_data_is_native_format() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = ImageTensor::from_hwc_data(2, 1, 3, data.clone());
        
        // to_hwc_data should return the same data since it's native format
        assert_eq!(tensor.to_hwc_data(), data);
    }
}
