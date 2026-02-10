//! Drawing utilities for visualization of deep learning results.
//!
//! This module provides functions for drawing bounding boxes, text,
//! segmentation masks, and other visualizations commonly used in
//! computer vision and deep learning applications.

pub mod bbox;
pub mod text;
pub mod segmentation;
pub mod keypoints;
pub mod heatmap;

pub use bbox::{draw_bbox, BoundingBox};
pub use text::draw_text;
pub use segmentation::{draw_segmentation_mask, draw_multiclass_segmentation};
pub use keypoints::{draw_keypoints, draw_skeleton, Keypoint};
pub use heatmap::{apply_heatmap, overlay_heatmap};
