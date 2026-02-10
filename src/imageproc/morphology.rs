//! Morphological operations (erosion, dilation).

use crate::convert::ImageTensor;

/// Apply erosion morphological operation.
///
/// Erosion erodes away the boundaries of regions of foreground pixels.
///
/// # Arguments
///
/// * `input` - Input grayscale ImageTensor
/// * `kernel_size` - Size of the structuring element (must be odd)
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::imageproc::erode;
///
/// let eroded = erode(&grayscale_img, 3);
/// ```
pub fn erode(input: &ImageTensor, kernel_size: u32) -> ImageTensor {
    assert_eq!(input.channels, 1, "Erosion requires grayscale image");
    assert!(kernel_size % 2 == 1, "Kernel size must be odd");
    
    let width = input.width;
    let height = input.height;
    let radius = (kernel_size / 2) as i32;
    let mut output_data = vec![0.0; (width * height) as usize];
    
    for y in 0..height {
        for x in 0..width {
            let mut min_val: f32 = 1.0;
            
            for ky in -(radius)..=radius {
                for kx in -(radius)..=radius {
                    let ny = y as i32 + ky;
                    let nx = x as i32 + kx;
                    
                    if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                        let val = input.get_pixel(nx as u32, ny as u32, 0);
                        min_val = min_val.min(val);
                    }
                }
            }
            
            output_data[(y * width + x) as usize] = min_val;
        }
    }
    
    ImageTensor::new(width, height, 1, output_data)
}

/// Apply dilation morphological operation.
///
/// Dilation expands the boundaries of regions of foreground pixels.
///
/// # Arguments
///
/// * `input` - Input grayscale ImageTensor
/// * `kernel_size` - Size of the structuring element (must be odd)
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::imageproc::dilate;
///
/// let dilated = dilate(&grayscale_img, 3);
/// ```
pub fn dilate(input: &ImageTensor, kernel_size: u32) -> ImageTensor {
    assert_eq!(input.channels, 1, "Dilation requires grayscale image");
    assert!(kernel_size % 2 == 1, "Kernel size must be odd");
    
    let width = input.width;
    let height = input.height;
    let radius = (kernel_size / 2) as i32;
    let mut output_data = vec![0.0; (width * height) as usize];
    
    for y in 0..height {
        for x in 0..width {
            let mut max_val: f32 = 0.0;
            
            for ky in -(radius)..=radius {
                for kx in -(radius)..=radius {
                    let ny = y as i32 + ky;
                    let nx = x as i32 + kx;
                    
                    if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                        let val = input.get_pixel(nx as u32, ny as u32, 0);
                        max_val = max_val.max(val);
                    }
                }
            }
            
            output_data[(y * width + x) as usize] = max_val;
        }
    }
    
    ImageTensor::new(width, height, 1, output_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erode() {
        // Create a simple image with a white square
        let mut data = vec![0.0; 5 * 5];
        for y in 1..4 {
            for x in 1..4 {
                data[(y * 5 + x)] = 1.0;
            }
        }
        
        let input = ImageTensor::new(5, 5, 1, data);
        let output = erode(&input, 3);
        
        // After erosion, only the center pixel should remain white
        assert_eq!(output.get_pixel(2, 2, 0), 1.0);
        assert!(output.get_pixel(1, 1, 0) < 1.0);
    }

    #[test]
    fn test_dilate() {
        // Create a simple image with a single white pixel
        let mut data = vec![0.0; 5 * 5];
        data[2 * 5 + 2] = 1.0;
        
        let input = ImageTensor::new(5, 5, 1, data);
        let output = dilate(&input, 3);
        
        // After dilation, neighboring pixels should be white
        assert_eq!(output.get_pixel(2, 2, 0), 1.0);
        assert_eq!(output.get_pixel(2, 1, 0), 1.0);
        assert_eq!(output.get_pixel(1, 2, 0), 1.0);
    }
}
