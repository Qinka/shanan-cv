//! # CubeCV - High-Performance Computer Vision Library
//! 
//! CubeCV is a computer vision library that leverages CubeCL for GPU-accelerated
//! parallel processing of images. It provides efficient implementations of common
//! computer vision operations on image-rs types.
//!
//! ## Features
//!
//! - GPU-accelerated image processing using CubeCL
//! - Seamless integration with image-rs types
//! - High-performance parallel operations
//! - HWC/CHW format conversions for deep learning frameworks
//! - imageproc-inspired operations (morphology, filtering, etc.)
//! - Visualization tools for deep learning results
//!
//! ## Example
//!
//! ```rust,ignore
//! use cubecv::prelude::*;
//! use image::DynamicImage;
//!
//! let img = image::open("input.jpg").unwrap();
//! let processed = grayscale(&img);
//! ```

pub mod convert;
pub mod ops;
pub mod imageproc;
pub mod draw;
pub mod prelude;

pub use convert::ImageTensor;

#[cfg(test)]
mod tests {
    // Basic module structure test - no assertions needed as the module compiles
}
