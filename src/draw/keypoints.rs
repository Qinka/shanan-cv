//! Keypoint visualization for pose estimation and landmark detection.

use crate::convert::ImageTensor;

/// Represents a 2D keypoint.
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    /// X coordinate
    pub x: u32,
    /// Y coordinate
    pub y: u32,
    /// Optional confidence score
    pub confidence: Option<f32>,
    /// Optional keypoint ID or index
    pub id: Option<usize>,
}

impl Keypoint {
    /// Create a new keypoint.
    pub fn new(x: u32, y: u32) -> Self {
        Self {
            x,
            y,
            confidence: None,
            id: None,
        }
    }

    /// Set confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set keypoint ID.
    pub fn with_id(mut self, id: usize) -> Self {
        self.id = Some(id);
        self
    }
}

/// Draw keypoints on an image.
///
/// # Arguments
///
/// * `image` - Input ImageTensor (will be modified in place)
/// * `keypoints` - List of keypoints to draw
/// * `color` - RGB color for keypoints
/// * `radius` - Radius of keypoint circles
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::draw::{draw_keypoints, Keypoint};
///
/// let keypoints = vec![
///     Keypoint::new(50, 50).with_confidence(0.9),
///     Keypoint::new(100, 100).with_confidence(0.95),
/// ];
/// draw_keypoints(&mut img, &keypoints, [1.0, 0.0, 0.0], 3);
/// ```
pub fn draw_keypoints(
    image: &mut ImageTensor,
    keypoints: &[Keypoint],
    color: [f32; 3],
    radius: u32,
) {
    assert!(
        image.channels >= 3,
        "Image must have at least 3 channels (RGB)"
    );

    for kp in keypoints {
        draw_circle(image, kp.x, kp.y, radius, color);
    }
}

/// Draw connections between keypoints (skeleton).
///
/// # Arguments
///
/// * `image` - Input ImageTensor (will be modified in place)
/// * `keypoints` - List of keypoints
/// * `connections` - List of (start_idx, end_idx) pairs defining skeleton
/// * `color` - RGB color for connections
/// * `thickness` - Line thickness
///
/// # Example
///
/// ```rust,ignore
/// use cubecv::draw::{draw_skeleton, Keypoint};
///
/// let keypoints = vec![
///     Keypoint::new(50, 50),
///     Keypoint::new(60, 80),
///     Keypoint::new(70, 110),
/// ];
/// let connections = vec![(0, 1), (1, 2)];
/// draw_skeleton(&mut img, &keypoints, &connections, [0.0, 1.0, 0.0], 2);
/// ```
pub fn draw_skeleton(
    image: &mut ImageTensor,
    keypoints: &[Keypoint],
    connections: &[(usize, usize)],
    color: [f32; 3],
    thickness: u32,
) {
    assert!(
        image.channels >= 3,
        "Image must have at least 3 channels (RGB)"
    );

    for &(start_idx, end_idx) in connections {
        if start_idx < keypoints.len() && end_idx < keypoints.len() {
            let start = &keypoints[start_idx];
            let end = &keypoints[end_idx];
            
            draw_line(image, start.x, start.y, end.x, end.y, color, thickness);
        }
    }
}

fn draw_circle(image: &mut ImageTensor, cx: u32, cy: u32, radius: u32, color: [f32; 3]) {
    let r_sq = (radius * radius) as i32;
    
    for dy in -(radius as i32)..=(radius as i32) {
        for dx in -(radius as i32)..=(radius as i32) {
            if dx * dx + dy * dy <= r_sq {
                let x = (cx as i32 + dx) as u32;
                let y = (cy as i32 + dy) as u32;
                
                if x < image.width && y < image.height {
                    for c in 0..3 {
                        image.set_pixel(x, y, c, color[c as usize]);
                    }
                }
            }
        }
    }
}

fn draw_line(
    image: &mut ImageTensor,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    color: [f32; 3],
    thickness: u32,
) {
    // Bresenham's line algorithm
    let dx = (x1 as i32 - x0 as i32).abs();
    let dy = -(y1 as i32 - y0 as i32).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    
    let mut x = x0 as i32;
    let mut y = y0 as i32;
    
    loop {
        // Draw thick line by drawing circles at each point
        for t in 0..thickness {
            let px = (x + t as i32) as u32;
            let py = y as u32;
            if px < image.width && py < image.height {
                for c in 0..3 {
                    image.set_pixel(px, py, c, color[c as usize]);
                }
            }
        }
        
        if x == x1 as i32 && y == y1 as i32 {
            break;
        }
        
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_keypoints() {
        let mut img = ImageTensor::new(100, 100, 3, vec![0.0; 100 * 100 * 3]);
        let keypoints = vec![Keypoint::new(50, 50)];
        
        draw_keypoints(&mut img, &keypoints, [1.0, 0.0, 0.0], 3);
        
        // Check that keypoint was drawn
        assert_eq!(img.get_pixel(50, 50, 0), 1.0);
    }

    #[test]
    fn test_draw_skeleton() {
        let mut img = ImageTensor::new(100, 100, 3, vec![0.0; 100 * 100 * 3]);
        let keypoints = vec![Keypoint::new(10, 10), Keypoint::new(20, 20)];
        let connections = vec![(0, 1)];
        
        draw_skeleton(&mut img, &keypoints, &connections, [0.0, 1.0, 0.0], 1);
        
        // Check that line was drawn
        assert!(img.get_pixel(15, 15, 1) > 0.0);
    }
}
