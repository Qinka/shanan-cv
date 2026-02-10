//! Statistical operations on images.

use crate::convert::ImageTensor;

/// Compute histogram of image intensities.
///
/// # Arguments
///
/// * `input` - Input grayscale ImageTensor
/// * `bins` - Number of histogram bins
///
/// # Returns
///
/// Vector of histogram values (normalized to sum to 1.0)
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::imageproc::histogram;
///
/// let hist = histogram(&grayscale_img, 256);
/// ```
pub fn histogram(input: &ImageTensor, bins: usize) -> Vec<f32> {
    assert_eq!(input.channels, 1, "Histogram requires grayscale image");
    
    let mut hist = vec![0u32; bins];
    let total_pixels = (input.width * input.height) as f32;
    
    for &val in &input.data {
        let bin = ((val.clamp(0.0, 1.0) * (bins - 1) as f32).round() as usize).min(bins - 1);
        hist[bin] += 1;
    }
    
    // Normalize
    hist.iter().map(|&count| count as f32 / total_pixels).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        // Create image with known distribution
        let mut data = vec![0.0; 100];
        for i in 0..50 {
            data[i] = 0.0;
        }
        for i in 50..100 {
            data[i] = 1.0;
        }
        
        let input = ImageTensor::new(10, 10, 1, data);
        let hist = histogram(&input, 2);
        
        assert_eq!(hist.len(), 2);
        assert!((hist[0] - 0.5).abs() < 0.01);
        assert!((hist[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_histogram_normalization() {
        let data = vec![0.5; 100];
        let input = ImageTensor::new(10, 10, 1, data);
        let hist = histogram(&input, 10);
        
        let sum: f32 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
