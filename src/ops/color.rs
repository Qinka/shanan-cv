//! Color space transformation operations.

use crate::convert::ImageTensor;

/// Convert RGB to HSV color space.
///
/// # Arguments
///
/// * `input` - Input ImageTensor in RGB format
///
/// # Returns
///
/// A new ImageTensor in HSV format where all values are normalized to [0, 1]:
/// - H (Hue) is normalized from [0, 360°] to [0, 1]
/// - S (Saturation) is in range [0, 1]
/// - V (Value) is in range [0, 1]
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::ops::rgb_to_hsv;
///
/// let hsv = rgb_to_hsv(&rgb_tensor);
/// ```
pub fn rgb_to_hsv(input: &ImageTensor) -> ImageTensor {
    assert!(input.channels >= 3, "Input must have at least 3 channels (RGB)");
    
    let width = input.width;
    let height = input.height;
    let mut output_data = Vec::with_capacity((width * height * 3) as usize);
    
    for y in 0..height {
        for x in 0..width {
            let r = input.get_pixel(x, y, 0);
            let g = input.get_pixel(x, y, 1);
            let b = input.get_pixel(x, y, 2);
            
            let max_val = r.max(g.max(b));
            let min_val = r.min(g.min(b));
            let delta = max_val - min_val;
            
            // Hue
            let h = if delta == 0.0 {
                0.0
            } else if max_val == r {
                60.0 * (((g - b) / delta) % 6.0)
            } else if max_val == g {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };
            
            let h = if h < 0.0 { h + 360.0 } else { h };
            
            // Saturation
            let s = if max_val == 0.0 { 0.0 } else { delta / max_val };
            
            // Value
            let v = max_val;
            
            output_data.push(h / 360.0); // Normalize to [0, 1]
            output_data.push(s);
            output_data.push(v);
        }
    }
    
    ImageTensor::new(width, height, 3, output_data)
}

/// Convert HSV to RGB color space.
///
/// # Arguments
///
/// * `input` - Input ImageTensor in HSV format where H, S, and V are all normalized to [0, 1]
///
/// # Returns
///
/// A new ImageTensor in RGB format with values in range [0, 1].
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::ops::hsv_to_rgb;
///
/// let rgb = hsv_to_rgb(&hsv_tensor);
/// ```
pub fn hsv_to_rgb(input: &ImageTensor) -> ImageTensor {
    assert_eq!(input.channels, 3, "Input must have 3 channels (HSV)");
    
    let width = input.width;
    let height = input.height;
    let mut output_data = Vec::with_capacity((width * height * 3) as usize);
    
    for y in 0..height {
        for x in 0..width {
            let h = input.get_pixel(x, y, 0) * 360.0; // Denormalize from [0, 1]
            let s = input.get_pixel(x, y, 1);
            let v = input.get_pixel(x, y, 2);
            
            let c = v * s;
            let x_val = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
            let m = v - c;
            
            let (r, g, b) = if h < 60.0 {
                (c, x_val, 0.0)
            } else if h < 120.0 {
                (x_val, c, 0.0)
            } else if h < 180.0 {
                (0.0, c, x_val)
            } else if h < 240.0 {
                (0.0, x_val, c)
            } else if h < 300.0 {
                (x_val, 0.0, c)
            } else {
                (c, 0.0, x_val)
            };
            
            output_data.push(r + m);
            output_data.push(g + m);
            output_data.push(b + m);
        }
    }
    
    ImageTensor::new(width, height, 3, output_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_hsv_pure_colors() {
        // Test pure red (1, 0, 0) -> (0°, 1, 1)
        let data = vec![1.0, 0.0, 0.0];
        let input = ImageTensor::new(1, 1, 3, data);
        let output = rgb_to_hsv(&input);
        
        assert_eq!(output.channels, 3);
        let h = output.get_pixel(0, 0, 0);
        let s = output.get_pixel(0, 0, 1);
        let v = output.get_pixel(0, 0, 2);
        
        assert!((h - 0.0).abs() < 0.01); // Hue should be ~0
        assert!((s - 1.0).abs() < 0.01); // Saturation should be 1
        assert!((v - 1.0).abs() < 0.01); // Value should be 1
    }

    #[test]
    fn test_hsv_to_rgb_pure_colors() {
        // Test (0°, 1, 1) -> pure red (1, 0, 0)
        let data = vec![0.0, 1.0, 1.0]; // H normalized to [0,1]
        let input = ImageTensor::new(1, 1, 3, data);
        let output = hsv_to_rgb(&input);
        
        let r = output.get_pixel(0, 0, 0);
        let g = output.get_pixel(0, 0, 1);
        let b = output.get_pixel(0, 0, 2);
        
        assert!((r - 1.0).abs() < 0.01);
        assert!(g.abs() < 0.01);
        assert!(b.abs() < 0.01);
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        // Test that RGB -> HSV -> RGB preserves the color
        let data = vec![0.5, 0.3, 0.7];
        let input = ImageTensor::new(1, 1, 3, data.clone());
        
        let hsv = rgb_to_hsv(&input);
        let rgb = hsv_to_rgb(&hsv);
        
        for i in 0..3 {
            let original = input.get_pixel(0, 0, i);
            let converted = rgb.get_pixel(0, 0, i);
            assert!((original - converted).abs() < 0.01, 
                    "Channel {} mismatch: {} vs {}", i, original, converted);
        }
    }

    #[test]
    fn test_grayscale_to_hsv() {
        // Grayscale (R=G=B) should have S=0
        let data = vec![0.5, 0.5, 0.5];
        let input = ImageTensor::new(1, 1, 3, data);
        let output = rgb_to_hsv(&input);
        
        let s = output.get_pixel(0, 0, 1);
        assert!(s.abs() < 0.01); // Saturation should be 0
    }

    #[test]
    fn test_hsv_manipulation_pipeline() {
        // Test modifying HSV values and converting back to RGB
        let data = vec![0.5, 0.3, 0.7, 0.2, 0.8, 0.4];
        let input = ImageTensor::new(2, 1, 3, data);
        
        // Convert to HSV
        let mut hsv = rgb_to_hsv(&input);
        
        // Boost saturation by 1.5x (clamp to 1.0)
        for y in 0..hsv.height {
            for x in 0..hsv.width {
                let s = hsv.get_pixel(x, y, 1);
                hsv.set_pixel(x, y, 1, (s * 1.5).min(1.0));
            }
        }
        
        // Convert back to RGB
        let output = hsv_to_rgb(&hsv);
        
        // Verify dimensions are preserved
        assert_eq!(output.width, 2);
        assert_eq!(output.height, 1);
        assert_eq!(output.channels, 3);
        
        // Verify RGB values are still in valid range
        for pixel in output.data.iter() {
            assert!(*pixel >= 0.0 && *pixel <= 1.0, 
                   "RGB value {} out of range [0, 1]", pixel);
        }
    }
}
