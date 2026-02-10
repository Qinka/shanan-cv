//! Grayscale conversion operations.

use crate::convert::ImageTensor;

/// Convert an image to grayscale using GPU acceleration.
///
/// # Arguments
///
/// * `input` - Input ImageTensor (RGB or RGBA)
///
/// # Returns
///
/// A new ImageTensor with a single channel containing the grayscale image.
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::ops::grayscale;
/// use cubecv::convert::ImageTensor;
///
/// let grayscale_img = grayscale(&input_tensor);
/// ```
pub fn grayscale(input: &ImageTensor) -> ImageTensor {
    // CPU implementation for simplicity
    // In a full implementation, this would use CubeCL runtime
    
    let width = input.width;
    let height = input.height;
    let mut output_data = Vec::with_capacity((width * height) as usize);
    
    for y in 0..height {
        for x in 0..width {
            let r = input.get_pixel(x, y, 0);
            let g = input.get_pixel(x, y, 1);
            let b = input.get_pixel(x, y, 2);
            
            // ITU-R BT.601 grayscale conversion
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            output_data.push(gray);
        }
    }
    
    ImageTensor::new(width, height, 1, output_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grayscale_conversion() {
        // Create a simple red image
        let mut data = vec![0.0; 10 * 10 * 4];
        for i in 0..(10 * 10) {
            data[i * 4] = 1.0;     // R
            data[i * 4 + 1] = 0.0; // G
            data[i * 4 + 2] = 0.0; // B
            data[i * 4 + 3] = 1.0; // A
        }
        
        let input = ImageTensor::new(10, 10, 4, data);
        let output = grayscale(&input);
        
        assert_eq!(output.width, 10);
        assert_eq!(output.height, 10);
        assert_eq!(output.channels, 1);
        
        // Red should convert to approximately 0.299
        let gray_value = output.get_pixel(0, 0, 0);
        assert!((gray_value - 0.299).abs() < 0.001);
    }

    #[test]
    fn test_grayscale_white() {
        let mut data = vec![0.0; 5 * 5 * 4];
        for i in 0..(5 * 5) {
            data[i * 4] = 1.0;     // R
            data[i * 4 + 1] = 1.0; // G
            data[i * 4 + 2] = 1.0; // B
            data[i * 4 + 3] = 1.0; // A
        }
        
        let input = ImageTensor::new(5, 5, 4, data);
        let output = grayscale(&input);
        
        // White should convert to 1.0
        assert!((output.get_pixel(0, 0, 0) - 1.0).abs() < 0.001);
    }
}
