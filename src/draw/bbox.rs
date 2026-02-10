//! Bounding box drawing utilities.

use crate::convert::ImageTensor;

/// Represents a bounding box for object detection.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// X coordinate of top-left corner
    pub x: u32,
    /// Y coordinate of top-left corner
    pub y: u32,
    /// Width of the box
    pub width: u32,
    /// Height of the box
    pub height: u32,
    /// Optional label for the box
    pub label: Option<&'static str>,
    /// Optional confidence score
    pub confidence: Option<f32>,
}

impl BoundingBox {
    /// Create a new bounding box.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            label: None,
            confidence: None,
        }
    }

    /// Set the label for this bounding box.
    pub fn with_label(mut self, label: &'static str) -> Self {
        self.label = Some(label);
        self
    }

    /// Set the confidence score for this bounding box.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// Draw a bounding box on an image.
///
/// # Arguments
///
/// * `image` - Input ImageTensor (will be modified in place)
/// * `bbox` - Bounding box to draw
/// * `color` - RGB color values [r, g, b] in range [0, 1]
/// * `thickness` - Line thickness in pixels
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::draw::{draw_bbox, BoundingBox};
///
/// let bbox = BoundingBox::new(10, 10, 50, 50)
///     .with_label("person")
///     .with_confidence(0.95);
/// draw_bbox(&mut img, &bbox, [1.0, 0.0, 0.0], 2);
/// ```
pub fn draw_bbox(
    image: &mut ImageTensor,
    bbox: &BoundingBox,
    color: [f32; 3],
    thickness: u32,
) {
    let x1 = bbox.x;
    let y1 = bbox.y;
    let x2 = (bbox.x + bbox.width).min(image.width - 1);
    let y2 = (bbox.y + bbox.height).min(image.height - 1);

    // Ensure RGB or RGBA
    assert!(
        image.channels >= 3,
        "Image must have at least 3 channels (RGB)"
    );

    // Draw horizontal lines
    for t in 0..thickness {
        // Top line
        if y1 + t < image.height {
            for x in x1..=x2 {
                if x < image.width {
                    for c in 0..3 {
                        image.set_pixel(x, y1 + t, c, color[c as usize]);
                    }
                }
            }
        }
        
        // Bottom line
        if y2 >= t && y2 - t < image.height {
            for x in x1..=x2 {
                if x < image.width {
                    for c in 0..3 {
                        image.set_pixel(x, y2 - t, c, color[c as usize]);
                    }
                }
            }
        }
    }

    // Draw vertical lines
    for t in 0..thickness {
        // Left line
        if x1 + t < image.width {
            for y in y1..=y2 {
                if y < image.height {
                    for c in 0..3 {
                        image.set_pixel(x1 + t, y, c, color[c as usize]);
                    }
                }
            }
        }
        
        // Right line
        if x2 >= t && x2 - t < image.width {
            for y in y1..=y2 {
                if y < image.height {
                    for c in 0..3 {
                        image.set_pixel(x2 - t, y, c, color[c as usize]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_bbox() {
        let mut img = ImageTensor::new(100, 100, 3, vec![0.0; 100 * 100 * 3]);
        let bbox = BoundingBox::new(10, 10, 30, 30);
        
        draw_bbox(&mut img, &bbox, [1.0, 0.0, 0.0], 2);
        
        // Check that top-left corner has red color
        assert_eq!(img.get_pixel(10, 10, 0), 1.0);
        assert_eq!(img.get_pixel(10, 10, 1), 0.0);
        assert_eq!(img.get_pixel(10, 10, 2), 0.0);
    }

    #[test]
    fn test_bbox_with_label() {
        let bbox = BoundingBox::new(0, 0, 10, 10)
            .with_label("test")
            .with_confidence(0.95);
        
        assert_eq!(bbox.label, Some("test"));
        assert_eq!(bbox.confidence, Some(0.95));
    }
}
