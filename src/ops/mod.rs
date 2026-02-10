//! Image processing operations using CubeCL.
//!
//! This module provides GPU-accelerated image processing operations.

pub mod grayscale;
pub mod blur;
pub mod edge;
pub mod color;

pub use grayscale::grayscale;
pub use blur::gaussian_blur;
pub use edge::sobel_edge_detection;
pub use color::{rgb_to_hsv, hsv_to_rgb};
