//! Text rendering utilities.

use crate::convert::ImageTensor;

/// Draw text on an image (simple implementation).
///
/// # Arguments
///
/// * `image` - Input ImageTensor (will be modified in place)
/// * `text` - Text to draw
/// * `x` - X coordinate of text position
/// * `y` - Y coordinate of text position
/// * `color` - RGB color values [r, g, b] in range [0, 1]
/// * `scale` - Text scale factor
///
/// # Note
///
/// This is a simplified implementation. For production use, consider using
/// a proper text rendering library like rusttype with imageproc.
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::draw::draw_text;
///
/// draw_text(&mut img, "Person: 0.95", 10, 10, [1.0, 1.0, 1.0], 1.0);
/// ```
pub fn draw_text(
    image: &mut ImageTensor,
    text: &str,
    x: u32,
    y: u32,
    color: [f32; 3],
    scale: f32,
) {
    // Simple text rendering using a basic 5x7 bitmap font
    // For each character, draw a simple representation
    
    let char_width = (5.0 * scale) as u32;
    let char_height = (7.0 * scale) as u32;
    let spacing = (2.0 * scale) as u32;
    
    for (i, ch) in text.chars().enumerate() {
        let char_x = x + i as u32 * (char_width + spacing);
        
        // Draw a simple filled rectangle for each character (placeholder)
        // In production, use proper font rendering
        draw_char_simple(image, ch, char_x, y, color, char_width, char_height);
    }
}

fn draw_char_simple(
    image: &mut ImageTensor,
    _ch: char,
    x: u32,
    y: u32,
    color: [f32; 3],
    width: u32,
    height: u32,
) {
    // Simple filled rectangle as placeholder
    for dy in 0..height {
        for dx in 0..width {
            let px = x + dx;
            let py = y + dy;
            
            if px < image.width && py < image.height {
                for c in 0..3.min(image.channels) {
                    image.set_pixel(px, py, c, color[c as usize]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_text() {
        let mut img = ImageTensor::new(100, 100, 3, vec![0.0; 100 * 100 * 3]);
        draw_text(&mut img, "Test", 10, 10, [1.0, 1.0, 1.0], 1.0);
        
        // Check that text area has been modified
        assert!(img.get_pixel(10, 10, 0) > 0.0);
    }
}
