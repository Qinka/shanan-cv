//! Image filtering operations.

use crate::convert::ImageTensor;

/// Apply median filter for noise reduction.
///
/// # Arguments
///
/// * `input` - Input ImageTensor
/// * `kernel_size` - Size of the filter window (must be odd)
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::imageproc::median_filter;
///
/// let filtered = median_filter(&noisy_img, 3);
/// ```
pub fn median_filter(input: &ImageTensor, kernel_size: u32) -> ImageTensor {
    assert!(kernel_size % 2 == 1, "Kernel size must be odd");
    
    let width = input.width;
    let height = input.height;
    let channels = input.channels;
    let radius = (kernel_size / 2) as i32;
    let mut output_data = vec![0.0; (width * height * channels) as usize];
    
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let mut values = Vec::new();
                
                for ky in -(radius)..=radius {
                    for kx in -(radius)..=radius {
                        let ny = (y as i32 + ky).clamp(0, height as i32 - 1) as u32;
                        let nx = (x as i32 + kx).clamp(0, width as i32 - 1) as u32;
                        values.push(input.get_pixel(nx, ny, c));
                    }
                }
                
                // Find median
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = values[values.len() / 2];
                
                let idx = ((y * width + x) * channels + c) as usize;
                output_data[idx] = median;
            }
        }
    }
    
    ImageTensor::new(width, height, channels, output_data)
}

/// Apply bilateral filter for edge-preserving smoothing.
///
/// # Arguments
///
/// * `input` - Input ImageTensor
/// * `kernel_size` - Size of the filter window (must be odd)
/// * `sigma_spatial` - Spatial sigma for Gaussian kernel
/// * `sigma_range` - Range sigma for intensity difference
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::imageproc::bilateral_filter;
///
/// let filtered = bilateral_filter(&img, 5, 1.0, 0.1);
/// ```
pub fn bilateral_filter(
    input: &ImageTensor,
    kernel_size: u32,
    sigma_spatial: f32,
    sigma_range: f32,
) -> ImageTensor {
    assert!(kernel_size % 2 == 1, "Kernel size must be odd");
    
    let width = input.width;
    let height = input.height;
    let channels = input.channels;
    let radius = (kernel_size / 2) as i32;
    let mut output_data = vec![0.0; (width * height * channels) as usize];
    
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let center_val = input.get_pixel(x, y, c);
                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                
                for ky in -(radius)..=radius {
                    for kx in -(radius)..=radius {
                        let ny = (y as i32 + ky).clamp(0, height as i32 - 1) as u32;
                        let nx = (x as i32 + kx).clamp(0, width as i32 - 1) as u32;
                        let neighbor_val = input.get_pixel(nx, ny, c);
                        
                        // Spatial weight
                        let spatial_dist = ((kx * kx + ky * ky) as f32).sqrt();
                        let spatial_weight = 
                            (-spatial_dist * spatial_dist / (2.0 * sigma_spatial * sigma_spatial)).exp();
                        
                        // Range weight
                        let range_dist = (center_val - neighbor_val).abs();
                        let range_weight = 
                            (-range_dist * range_dist / (2.0 * sigma_range * sigma_range)).exp();
                        
                        let weight = spatial_weight * range_weight;
                        sum += neighbor_val * weight;
                        weight_sum += weight;
                    }
                }
                
                let idx = ((y * width + x) * channels + c) as usize;
                output_data[idx] = sum / weight_sum;
            }
        }
    }
    
    ImageTensor::new(width, height, channels, output_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_filter() {
        // Create a simple noisy image
        let data = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,  // Center pixel is noise
            0.0, 0.0, 0.0,
        ];
        
        let input = ImageTensor::new(3, 3, 1, data);
        let output = median_filter(&input, 3);
        
        // Center pixel should be filtered to median (0.0)
        assert!(output.get_pixel(1, 1, 0) < 0.5);
    }

    #[test]
    fn test_bilateral_filter_preserves_dimensions() {
        let data = vec![0.5; 10 * 10 * 3];
        let input = ImageTensor::new(10, 10, 3, data);
        let output = bilateral_filter(&input, 3, 1.0, 0.1);
        
        assert_eq!(output.width, 10);
        assert_eq!(output.height, 10);
        assert_eq!(output.channels, 3);
    }
}
