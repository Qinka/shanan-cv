//! Segmentation mask visualization.

use crate::convert::ImageTensor;

/// Draw a segmentation mask overlay on an image.
///
/// # Arguments
///
/// * `image` - Input ImageTensor (will be modified in place)
/// * `mask` - Segmentation mask (grayscale, values in [0, 1])
/// * `color` - RGB color for the mask overlay
/// * `alpha` - Transparency of the overlay (0.0 = transparent, 1.0 = opaque)
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::draw::draw_segmentation_mask;
///
/// let mask = segment_image(&img);
/// draw_segmentation_mask(&mut img, &mask, [0.0, 1.0, 0.0], 0.5);
/// ```
pub fn draw_segmentation_mask(
    image: &mut ImageTensor,
    mask: &ImageTensor,
    color: [f32; 3],
    alpha: f32,
) {
    assert_eq!(mask.channels, 1, "Mask must be grayscale");
    assert_eq!(
        (image.width, image.height),
        (mask.width, mask.height),
        "Image and mask dimensions must match"
    );
    assert!(
        image.channels >= 3,
        "Image must have at least 3 channels (RGB)"
    );

    for y in 0..image.height {
        for x in 0..image.width {
            let mask_val = mask.get_pixel(x, y, 0);
            
            if mask_val > 0.5 {
                // Apply colored overlay with alpha blending
                for c in 0..3 {
                    let orig_val = image.get_pixel(x, y, c);
                    let blended = orig_val * (1.0 - alpha) + color[c as usize] * alpha;
                    image.set_pixel(x, y, c, blended);
                }
            }
        }
    }
}

/// Draw a multi-class segmentation mask with different colors.
///
/// # Arguments
///
/// * `image` - Input ImageTensor (will be modified in place)
/// * `mask` - Segmentation mask with class indices (grayscale, values = class IDs)
/// * `colors` - Color palette for each class
/// * `alpha` - Transparency of the overlay
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::draw::draw_multiclass_segmentation;
///
/// let colors = vec![
///     [1.0, 0.0, 0.0],  // Class 0: red
///     [0.0, 1.0, 0.0],  // Class 1: green
///     [0.0, 0.0, 1.0],  // Class 2: blue
/// ];
/// draw_multiclass_segmentation(&mut img, &mask, &colors, 0.5);
/// ```
pub fn draw_multiclass_segmentation(
    image: &mut ImageTensor,
    mask: &ImageTensor,
    colors: &[[f32; 3]],
    alpha: f32,
) {
    assert_eq!(mask.channels, 1, "Mask must be grayscale");
    assert_eq!(
        (image.width, image.height),
        (mask.width, mask.height),
        "Image and mask dimensions must match"
    );
    assert!(
        image.channels >= 3,
        "Image must have at least 3 channels (RGB)"
    );

    for y in 0..image.height {
        for x in 0..image.width {
            let class_id = (mask.get_pixel(x, y, 0) * (colors.len() as f32 - 1.0)).round() as usize;
            
            if class_id > 0 && class_id < colors.len() {
                let color = colors[class_id];
                
                for c in 0..3 {
                    let orig_val = image.get_pixel(x, y, c);
                    let blended = orig_val * (1.0 - alpha) + color[c as usize] * alpha;
                    image.set_pixel(x, y, c, blended);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_segmentation_mask() {
        let mut img = ImageTensor::new(10, 10, 3, vec![0.5; 10 * 10 * 3]);
        let mask = ImageTensor::new(10, 10, 1, vec![1.0; 10 * 10]);
        
        draw_segmentation_mask(&mut img, &mask, [1.0, 0.0, 0.0], 0.5);
        
        // Check that overlay was applied
        let val = img.get_pixel(5, 5, 0);
        assert!(val > 0.5); // Should be blend of 0.5 and 1.0
    }

    #[test]
    fn test_multiclass_segmentation() {
        let mut img = ImageTensor::new(10, 10, 3, vec![0.0; 10 * 10 * 3]);
        let mut mask_data = vec![0.0; 10 * 10];
        
        // Set different regions to different classes
        // Class 0 (background): 0.0
        // Class 1: 0.5  
        // Class 2: 1.0
        for i in 0..50 {
            mask_data[i] = 0.5; // Class 1
        }
        for i in 50..100 {
            mask_data[i] = 1.0; // Class 2
        }
        
        let mask = ImageTensor::new(10, 10, 1, mask_data);
        let colors = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        
        draw_multiclass_segmentation(&mut img, &mask, &colors, 0.5);
        
        // Verify different colors were applied
        // First half should be red (class 1)
        assert!(img.get_pixel(0, 0, 0) > 0.0); // Red channel
        // Second half should be green (class 2)
        assert!(img.get_pixel(9, 9, 1) > 0.0); // Green channel
    }
}
