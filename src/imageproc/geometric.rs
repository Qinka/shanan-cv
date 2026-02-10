//! Geometric transformations.

use crate::convert::ImageTensor;

/// Resize image using bilinear interpolation.
///
/// # Arguments
///
/// * `input` - Input ImageTensor
/// * `new_width` - Target width
/// * `new_height` - Target height
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::imageproc::resize_bilinear;
///
/// let resized = resize_bilinear(&img, 256, 256);
/// ```
pub fn resize_bilinear(input: &ImageTensor, new_width: u32, new_height: u32) -> ImageTensor {
    let channels = input.channels;
    let mut output_data = vec![0.0; (new_width * new_height * channels) as usize];
    
    let x_ratio = input.width as f32 / new_width as f32;
    let y_ratio = input.height as f32 / new_height as f32;
    
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = x as f32 * x_ratio;
            let src_y = y as f32 * y_ratio;
            
            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(input.width - 1);
            let y1 = (y0 + 1).min(input.height - 1);
            
            let dx = src_x - x0 as f32;
            let dy = src_y - y0 as f32;
            
            for c in 0..channels {
                let v00 = input.get_pixel(x0, y0, c);
                let v10 = input.get_pixel(x1, y0, c);
                let v01 = input.get_pixel(x0, y1, c);
                let v11 = input.get_pixel(x1, y1, c);
                
                // Bilinear interpolation
                let v0 = v00 * (1.0 - dx) + v10 * dx;
                let v1 = v01 * (1.0 - dx) + v11 * dx;
                let val = v0 * (1.0 - dy) + v1 * dy;
                
                let idx = ((y * new_width + x) * channels + c) as usize;
                output_data[idx] = val;
            }
        }
    }
    
    ImageTensor::new(new_width, new_height, channels, output_data)
}

/// Rotate image by specified angle (in degrees).
///
/// # Arguments
///
/// * `input` - Input ImageTensor
/// * `angle_degrees` - Rotation angle in degrees (positive = counter-clockwise)
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::imageproc::rotate;
///
/// let rotated = rotate(&img, 45.0);
/// ```
pub fn rotate(input: &ImageTensor, angle_degrees: f32) -> ImageTensor {
    let width = input.width;
    let height = input.height;
    let channels = input.channels;
    let mut output_data = vec![0.0; (width * height * channels) as usize];
    
    let angle_rad = angle_degrees.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    
    for y in 0..height {
        for x in 0..width {
            // Translate to origin
            let tx = x as f32 - cx;
            let ty = y as f32 - cy;
            
            // Rotate (inverse)
            let src_x = tx * cos_a + ty * sin_a + cx;
            let src_y = -tx * sin_a + ty * cos_a + cy;
            
            if src_x >= 0.0 && src_x < width as f32 && src_y >= 0.0 && src_y < height as f32 {
                let x0 = src_x.floor() as u32;
                let y0 = src_y.floor() as u32;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);
                
                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;
                
                for c in 0..channels {
                    let v00 = input.get_pixel(x0, y0, c);
                    let v10 = input.get_pixel(x1, y0, c);
                    let v01 = input.get_pixel(x0, y1, c);
                    let v11 = input.get_pixel(x1, y1, c);
                    
                    let v0 = v00 * (1.0 - dx) + v10 * dx;
                    let v1 = v01 * (1.0 - dx) + v11 * dx;
                    let val = v0 * (1.0 - dy) + v1 * dy;
                    
                    let idx = ((y * width + x) * channels + c) as usize;
                    output_data[idx] = val;
                }
            }
        }
    }
    
    ImageTensor::new(width, height, channels, output_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_bilinear() {
        let data = vec![1.0; 10 * 10 * 3];
        let input = ImageTensor::new(10, 10, 3, data);
        let output = resize_bilinear(&input, 20, 20);
        
        assert_eq!(output.width, 20);
        assert_eq!(output.height, 20);
        assert_eq!(output.channels, 3);
        
        // Values should be close to 1.0
        assert!((output.get_pixel(10, 10, 0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rotate() {
        let data = vec![0.5; 10 * 10 * 3];
        let input = ImageTensor::new(10, 10, 3, data);
        let output = rotate(&input, 90.0);
        
        assert_eq!(output.width, 10);
        assert_eq!(output.height, 10);
        assert_eq!(output.channels, 3);
    }
}
