//! Gaussian blur operations.

use crate::convert::ImageTensor;

/// Generate a Gaussian kernel.
fn generate_gaussian_kernel(size: u32, sigma: f32) -> Vec<f32> {
    let mut kernel = vec![0.0; (size * size) as usize];
    let radius = (size / 2) as i32;
    let mut sum = 0.0;
    
    for y in 0..size {
        for x in 0..size {
            let dy = y as i32 - radius;
            let dx = x as i32 - radius;
            let dist_sq = (dx * dx + dy * dy) as f32;
            let value = (-dist_sq / (2.0 * sigma * sigma)).exp();
            kernel[(y * size + x) as usize] = value;
            sum += value;
        }
    }
    
    // Normalize
    for v in &mut kernel {
        *v /= sum;
    }
    
    kernel
}

/// Apply Gaussian blur to an image.
///
/// # Arguments
///
/// * `input` - Input ImageTensor
/// * `sigma` - Gaussian kernel standard deviation (controls blur strength)
///
/// # Returns
///
/// A new ImageTensor with the blur applied.
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::ops::gaussian_blur;
///
/// let blurred = gaussian_blur(&input_tensor, 2.0);
/// ```
pub fn gaussian_blur(input: &ImageTensor, sigma: f32) -> ImageTensor {
    // CPU implementation
    let kernel_size = (sigma * 3.0).ceil() as u32 * 2 + 1;
    let kernel_size = kernel_size.min(31); // Limit kernel size
    let kernel = generate_gaussian_kernel(kernel_size, sigma);
    
    let width = input.width;
    let height = input.height;
    let channels = input.channels;
    let radius = (kernel_size / 2) as i32;
    
    let mut output_data = vec![0.0; (width * height * channels) as usize];
    
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let mut sum = 0.0;
                
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let src_y = y as i32 + ky as i32 - radius;
                        let src_x = x as i32 + kx as i32 - radius;
                        
                        // Clamp to image boundaries
                        if src_y >= 0 && src_y < height as i32 && 
                           src_x >= 0 && src_x < width as i32 {
                            let pixel = input.get_pixel(src_x as u32, src_y as u32, c);
                            let k_weight = kernel[(ky * kernel_size + kx) as usize];
                            sum += pixel * k_weight;
                        }
                    }
                }
                
                let idx = ((y * width + x) * channels + c) as usize;
                output_data[idx] = sum;
            }
        }
    }
    
    ImageTensor::new(width, height, channels, output_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel_generation() {
        let kernel = generate_gaussian_kernel(3, 1.0);
        assert_eq!(kernel.len(), 9);
        
        // Sum should be approximately 1.0
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gaussian_blur() {
        // Create a simple image with a white pixel in the center
        let mut data = vec![0.0; 5 * 5 * 3];
        let center_idx = ((2 * 5 + 2) * 3) as usize;
        data[center_idx] = 1.0;
        data[center_idx + 1] = 1.0;
        data[center_idx + 2] = 1.0;
        
        let input = ImageTensor::new(5, 5, 3, data);
        let output = gaussian_blur(&input, 1.0);
        
        assert_eq!(output.width, 5);
        assert_eq!(output.height, 5);
        assert_eq!(output.channels, 3);
        
        // Center pixel should still be bright but neighbors should also have some value
        let center_val = output.get_pixel(2, 2, 0);
        let neighbor_val = output.get_pixel(2, 1, 0);
        
        assert!(center_val > neighbor_val);
        assert!(neighbor_val > 0.0);
    }
}
