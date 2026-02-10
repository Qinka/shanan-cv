//! Image processing operations inspired by imageproc library.
//!
//! This module provides additional image processing functionality
//! commonly used in computer vision applications.

pub mod morphology;
pub mod filter;
pub mod geometric;
pub mod stats;

pub use morphology::{erode, dilate};
pub use filter::{median_filter, bilateral_filter};
pub use geometric::{resize_bilinear, rotate};
pub use stats::histogram;
