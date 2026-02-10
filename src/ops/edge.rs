//! Edge detection operations using Sobel filter.

use crate::convert::ImageTensor;

/// Apply Sobel edge detection to an image.
///
/// # Arguments
///
/// * `input` - Input ImageTensor (should be grayscale for best results)
///
/// # Returns
///
/// A new ImageTensor with edge magnitude values.
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::ops::{grayscale, sobel_edge_detection};
///
/// let gray = grayscale(&input_tensor);
/// let edges = sobel_edge_detection(&gray);
/// ```
pub fn sobel_edge_detection(input: &ImageTensor) -> ImageTensor {
    // CPU implementation
    // Convert to grayscale if needed
    let grayscale_data = if input.channels == 1 {
        input.data.clone()
    } else {
        // Convert to grayscale first
        let mut gray = Vec::with_capacity((input.width * input.height) as usize);
        for y in 0..input.height {
            for x in 0..input.width {
                let r = input.get_pixel(x, y, 0);
                let g = input.get_pixel(x, y, 1);
                let b = input.get_pixel(x, y, 2);
                let gray_val = 0.299 * r + 0.587 * g + 0.114 * b;
                gray.push(gray_val);
            }
        }
        gray
    };
    
    let width = input.width;
    let height = input.height;
    let mut output_data = vec![0.0; (width * height) as usize];
    
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Sobel operators
            let top_left = grayscale_data[((y - 1) * width + (x - 1)) as usize];
            let top = grayscale_data[((y - 1) * width + x) as usize];
            let top_right = grayscale_data[((y - 1) * width + (x + 1)) as usize];
            let left = grayscale_data[(y * width + (x - 1)) as usize];
            let right = grayscale_data[(y * width + (x + 1)) as usize];
            let bottom_left = grayscale_data[((y + 1) * width + (x - 1)) as usize];
            let bottom = grayscale_data[((y + 1) * width + x) as usize];
            let bottom_right = grayscale_data[((y + 1) * width + (x + 1)) as usize];
            
            // Gx: horizontal gradient
            let gx = -top_left + top_right
                   - 2.0 * left + 2.0 * right
                   - bottom_left + bottom_right;
            
            // Gy: vertical gradient
            let gy = -top_left - 2.0 * top - top_right
                   + bottom_left + 2.0 * bottom + bottom_right;
            
            // Magnitude
            let magnitude = (gx * gx + gy * gy).sqrt();
            output_data[(y * width + x) as usize] = magnitude;
        }
    }
    
    ImageTensor::new(width, height, 1, output_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobel_edge_detection() {
        // Create a simple step edge (vertical)
        let mut data = vec![0.0; 5 * 5];
        for y in 0..5 {
            for x in 0..5 {
                if x < 2 {
                    data[(y * 5 + x) as usize] = 0.0;
                } else {
                    data[(y * 5 + x) as usize] = 1.0;
                }
            }
        }
        
        let input = ImageTensor::new(5, 5, 1, data);
        let output = sobel_edge_detection(&input);
        
        assert_eq!(output.width, 5);
        assert_eq!(output.height, 5);
        assert_eq!(output.channels, 1);
        
        // Edge should be detected around x=2
        let edge_val = output.get_pixel(2, 2, 0);
        let non_edge_val = output.get_pixel(0, 2, 0);
        
        assert!(edge_val > non_edge_val);
    }

    #[test]
    fn test_sobel_rgb_conversion() {
        // Test that RGB images are converted to grayscale first
        let mut data = vec![0.0; 3 * 3 * 3];
        // Set middle column to white
        for y in 0..3 {
            for c in 0..3 {
                data[((y * 3 + 1) * 3 + c) as usize] = 1.0;
            }
        }
        
        let input = ImageTensor::new(3, 3, 3, data);
        let output = sobel_edge_detection(&input);
        
        assert_eq!(output.channels, 1);
    }
}
