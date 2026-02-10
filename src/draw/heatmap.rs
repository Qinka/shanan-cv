//! Heatmap visualization for attention maps, saliency maps, etc.

use crate::convert::ImageTensor;

/// Apply a colormap heatmap to a grayscale image.
///
/// # Arguments
///
/// * `heatmap` - Input grayscale heatmap (values in [0, 1])
/// * `colormap` - Colormap to use ("jet", "hot", "viridis")
///
/// # Returns
///
/// RGB ImageTensor with heatmap colors applied
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::draw::apply_heatmap;
///
/// let attention_map = compute_attention(&img);
/// let heatmap = apply_heatmap(&attention_map, "jet");
/// ```
pub fn apply_heatmap(heatmap: &ImageTensor, colormap: &str) -> ImageTensor {
    assert_eq!(heatmap.channels, 1, "Heatmap must be grayscale");
    
    let width = heatmap.width;
    let height = heatmap.height;
    let mut output_data = vec![0.0; (width * height * 3) as usize];
    
    for y in 0..height {
        for x in 0..width {
            let val = heatmap.get_pixel(x, y, 0).clamp(0.0, 1.0);
            let color = match colormap {
                "jet" => jet_colormap(val),
                "hot" => hot_colormap(val),
                "viridis" => viridis_colormap(val),
                _ => jet_colormap(val),
            };
            
            let idx = ((y * width + x) * 3) as usize;
            output_data[idx] = color[0];
            output_data[idx + 1] = color[1];
            output_data[idx + 2] = color[2];
        }
    }
    
    ImageTensor::new(width, height, 3, output_data)
}

/// Overlay a heatmap on an RGB image with alpha blending.
///
/// # Arguments
///
/// * `image` - Base RGB image (will be modified in place)
/// * `heatmap` - Grayscale heatmap (values in [0, 1])
/// * `colormap` - Colormap to use
/// * `alpha` - Transparency of heatmap overlay
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::draw::overlay_heatmap;
///
/// overlay_heatmap(&mut img, &attention_map, "jet", 0.5);
/// ```
pub fn overlay_heatmap(
    image: &mut ImageTensor,
    heatmap: &ImageTensor,
    colormap: &str,
    alpha: f32,
) {
    assert_eq!(heatmap.channels, 1, "Heatmap must be grayscale");
    assert!(
        image.channels >= 3,
        "Image must have at least 3 channels (RGB)"
    );
    assert_eq!(
        (image.width, image.height),
        (heatmap.width, heatmap.height),
        "Image and heatmap dimensions must match"
    );
    
    for y in 0..image.height {
        for x in 0..image.width {
            let val = heatmap.get_pixel(x, y, 0).clamp(0.0, 1.0);
            let color = match colormap {
                "jet" => jet_colormap(val),
                "hot" => hot_colormap(val),
                "viridis" => viridis_colormap(val),
                _ => jet_colormap(val),
            };
            
            for c in 0..3 {
                let orig_val = image.get_pixel(x, y, c);
                let blended = orig_val * (1.0 - alpha) + color[c as usize] * alpha;
                image.set_pixel(x, y, c, blended);
            }
        }
    }
}

// Jet colormap (blue -> cyan -> green -> yellow -> red)
fn jet_colormap(val: f32) -> [f32; 3] {
    let v = val * 4.0;
    
    let r = (v - 1.5).clamp(0.0, 1.0).min((3.5 - v).clamp(0.0, 1.0));
    let g = (v - 0.5).clamp(0.0, 1.0).min((2.5 - v).clamp(0.0, 1.0));
    let b = (v + 0.5).clamp(0.0, 1.0).min((1.5 - v).clamp(0.0, 1.0));
    
    [r, g, b]
}

// Hot colormap (black -> red -> yellow -> white)
fn hot_colormap(val: f32) -> [f32; 3] {
    let r = (val * 3.0).min(1.0);
    let g = ((val - 0.33) * 3.0).clamp(0.0, 1.0);
    let b = ((val - 0.67) * 3.0).clamp(0.0, 1.0);
    
    [r, g, b]
}

// Viridis colormap (purple -> blue -> green -> yellow)
fn viridis_colormap(val: f32) -> [f32; 3] {
    // Simplified viridis approximation
    let r = (0.282 + 0.718 * val).clamp(0.0, 1.0);
    let g = (0.004 + 0.996 * val.powf(0.8)).clamp(0.0, 1.0);
    let b = (0.541 - 0.341 * val).clamp(0.0, 1.0);
    
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_heatmap() {
        let heatmap = ImageTensor::new(10, 10, 1, vec![0.5; 10 * 10]);
        let result = apply_heatmap(&heatmap, "jet");
        
        assert_eq!(result.width, 10);
        assert_eq!(result.height, 10);
        assert_eq!(result.channels, 3);
    }

    #[test]
    fn test_overlay_heatmap() {
        let mut img = ImageTensor::new(10, 10, 3, vec![0.0; 10 * 10 * 3]);
        let heatmap = ImageTensor::new(10, 10, 1, vec![1.0; 10 * 10]);
        
        overlay_heatmap(&mut img, &heatmap, "hot", 0.5);
        
        // Check that overlay was applied
        assert!(img.get_pixel(5, 5, 0) > 0.0);
    }

    #[test]
    fn test_colormaps() {
        // Test that colormaps produce valid RGB values
        for val in [0.0, 0.5, 1.0] {
            let jet = jet_colormap(val);
            let hot = hot_colormap(val);
            let viridis = viridis_colormap(val);
            
            for &color in &[jet, hot, viridis] {
                assert!(color[0] >= 0.0 && color[0] <= 1.0);
                assert!(color[1] >= 0.0 && color[1] <= 1.0);
                assert!(color[2] >= 0.0 && color[2] <= 1.0);
            }
        }
    }
}
