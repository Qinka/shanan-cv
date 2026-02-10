# CubeCV

A high-performance parallel computing computer vision library using Rust and CubeCL.

CubeCV provides GPU-accelerated image processing operations that work seamlessly with [image-rs](https://github.com/image-rs/image) types, powered by the [CubeCL](https://github.com/tracel-ai/cubecl) compute library.

## Features

- 🚀 **High Performance**: Designed for GPU acceleration using CubeCL
- 🖼️ **image-rs Integration**: Seamless conversion between `DynamicImage` and internal tensor format
- 🎨 **Rich Operations**: Common computer vision operations out of the box
- 📦 **Easy to Use**: Simple, idiomatic Rust API
- 🔄 **Format Conversions**: HWC ↔ CHW tensor format conversions for deep learning frameworks
- 🛠️ **imageproc Features**: Morphology, filtering, geometric transformations, and statistics
- 🎯 **DL Visualization**: Tools for visualizing detection, segmentation, and pose estimation results

## Supported Operations

### Core Operations
- **Grayscale**: Convert RGB/RGBA images to grayscale using ITU-R BT.601 weights
- **RGB ↔ HSV**: Bidirectional color space transformations
- **Gaussian Blur**: Smooth images with configurable sigma
- **Edge Detection**: Sobel edge detection filter

### Format Conversions
- **HWC ↔ CHW**: Convert between Height-Width-Channel and Channel-Height-Width formats
- **Image-rs Integration**: Seamless conversion to/from `DynamicImage`

### imageproc-inspired Operations
- **Morphological**: Erosion and dilation
- **Filtering**: Median filter, bilateral filter
- **Geometric**: Bilinear resize, rotation
- **Statistics**: Histogram computation

### Deep Learning Visualization
- **Bounding Boxes**: Draw detection boxes with labels and confidence
- **Segmentation**: Single and multi-class mask overlays
- **Keypoints**: Draw keypoints and skeletal connections
- **Heatmaps**: Apply colormaps (jet, hot, viridis) to attention/saliency maps
- **Text**: Simple text rendering

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
cubecv = "0.1"
image = "0.25"
```

## Quick Start

```rust
use cubecv::prelude::*;
use image;

fn main() {
    // Load an image using image-rs
    let img = image::open("input.jpg").unwrap();
    
    // Convert to CubeCV tensor
    let tensor = ImageTensor::from_dynamic_image(&img);
    
    // Apply operations
    let gray = grayscale(&tensor);
    let blurred = gaussian_blur(&tensor, 2.0);
    let edges = sobel_edge_detection(&gray);
    
    // Convert back to image-rs
    let output = edges.to_dynamic_image();
    output.save("edges.jpg").unwrap();
}
```

## Examples

### Basic Operations

```rust
use cubecv::prelude::*;

// Create a tensor from raw data
let data = vec![0.0; 100 * 100 * 3];
let tensor = ImageTensor::new(100, 100, 3, data);

// Grayscale conversion
let gray = grayscale(&tensor);

// Gaussian blur
let blurred = gaussian_blur(&tensor, 1.5);

// Edge detection
let edges = sobel_edge_detection(&gray);
```

### Color Space Transformations

```rust
use cubecv::prelude::*;

// Convert to HSV
let hsv = rgb_to_hsv(&rgb_tensor);

// Modify in HSV space (e.g., boost saturation)
let mut modified_hsv = hsv.clone();
for y in 0..modified_hsv.height {
    for x in 0..modified_hsv.width {
        let s = modified_hsv.get_pixel(x, y, 1);
        modified_hsv.set_pixel(x, y, 1, (s * 1.5).min(1.0));
    }
}

// Convert back to RGB
let enhanced = hsv_to_rgb(&modified_hsv);
```

### Processing Pipeline

```rust
use cubecv::prelude::*;

// Chain multiple operations
let result = {
    let gray = grayscale(&input);
    let blurred = gaussian_blur(&gray, 2.0);
    sobel_edge_detection(&blurred)
};
```

### HWC/CHW Format Conversions

```rust
use cubecv::prelude::*;

// Create from CHW format (PyTorch style)
let chw_data = vec![/* channel-first data */];
let tensor = ImageTensor::from_chw_data(width, height, channels, chw_data);

// Convert to CHW for deep learning frameworks
let chw_output = tensor.to_chw_data();

// HWC is the native format
let hwc_data = tensor.to_hwc_data();
```

### Deep Learning Visualization

```rust
use cubecv::prelude::*;

// Draw bounding boxes
let bbox = BoundingBox::new(10, 10, 50, 50)
    .with_label("person")
    .with_confidence(0.95);
draw_bbox(&mut img, &bbox, [1.0, 0.0, 0.0], 2);

// Overlay segmentation mask
draw_segmentation_mask(&mut img, &mask, [0.0, 1.0, 0.0], 0.5);

// Draw keypoints and skeleton
let keypoints = vec![Keypoint::new(50, 50), Keypoint::new(100, 100)];
draw_keypoints(&mut img, &keypoints, [1.0, 0.0, 0.0], 3);

let connections = vec![(0, 1)];
draw_skeleton(&mut img, &keypoints, &connections, [0.0, 1.0, 0.0], 2);

// Apply heatmap
let heatmap_colored = apply_heatmap(&attention_map, "jet");
overlay_heatmap(&mut img, &attention_map, "viridis", 0.5);
```

## Running Examples

The library includes example programs demonstrating various features:

```bash
# Basic operations example
cargo run --example basic

# Image processing pipeline
cargo run --example pipeline

# Advanced features (HWC/CHW, imageproc, visualization)
cargo run --example advanced
```

## API Documentation

### `ImageTensor`

The core data structure representing an image as a tensor.

```rust
pub struct ImageTensor {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub data: Vec<f32>,
}
```

**Methods:**
- `new(width, height, channels, data)`: Create from raw data
- `from_dynamic_image(&DynamicImage)`: Convert from image-rs
- `to_dynamic_image()`: Convert to image-rs
- `from_hwc_data(...)`: Create from HWC format data
- `from_chw_data(...)`: Create from CHW format data
- `to_hwc_data()`: Export to HWC format
- `to_chw_data()`: Export to CHW format
- `get_pixel(x, y, c)`: Get pixel value
- `set_pixel(x, y, c, value)`: Set pixel value

### Core Operations

All operations take an `&ImageTensor` and return a new `ImageTensor`.

**Basic Operations:**
- `grayscale(&ImageTensor) -> ImageTensor`
- `gaussian_blur(&ImageTensor, sigma: f32) -> ImageTensor`
- `sobel_edge_detection(&ImageTensor) -> ImageTensor`
- `rgb_to_hsv(&ImageTensor) -> ImageTensor`
- `hsv_to_rgb(&ImageTensor) -> ImageTensor`

**Morphological Operations:**
- `erode(&ImageTensor, kernel_size: u32) -> ImageTensor`
- `dilate(&ImageTensor, kernel_size: u32) -> ImageTensor`

**Filtering:**
- `median_filter(&ImageTensor, kernel_size: u32) -> ImageTensor`
- `bilateral_filter(&ImageTensor, kernel_size: u32, sigma_spatial: f32, sigma_range: f32) -> ImageTensor`

**Geometric Transformations:**
- `resize_bilinear(&ImageTensor, new_width: u32, new_height: u32) -> ImageTensor`
- `rotate(&ImageTensor, angle_degrees: f32) -> ImageTensor`

**Statistics:**
- `histogram(&ImageTensor, bins: usize) -> Vec<f32>`

### Visualization Tools

**Bounding Boxes:**
- `draw_bbox(&mut ImageTensor, &BoundingBox, color: [f32; 3], thickness: u32)`
- `BoundingBox::new(x, y, width, height).with_label(...).with_confidence(...)`

**Segmentation:**
- `draw_segmentation_mask(&mut ImageTensor, &mask, color: [f32; 3], alpha: f32)`
- `draw_multiclass_segmentation(&mut ImageTensor, &mask, colors: &[[f32; 3]], alpha: f32)`

**Keypoints:**
- `draw_keypoints(&mut ImageTensor, keypoints: &[Keypoint], color: [f32; 3], radius: u32)`
- `draw_skeleton(&mut ImageTensor, keypoints: &[Keypoint], connections: &[(usize, usize)], color: [f32; 3], thickness: u32)`

**Heatmaps:**
- `apply_heatmap(&ImageTensor, colormap: &str) -> ImageTensor`
- `overlay_heatmap(&mut ImageTensor, &heatmap, colormap: &str, alpha: f32)`
- Supported colormaps: "jet", "hot", "viridis"

**Text:**
- `draw_text(&mut ImageTensor, text: &str, x: u32, y: u32, color: [f32; 3], scale: f32)`

## Performance Considerations

The current implementation uses CPU-based processing as a foundation. Future versions will leverage CubeCL's GPU acceleration capabilities for improved performance on large images.

## Architecture

CubeCV is structured into several modules:

- **`convert`**: Image format conversions (image-rs ↔ tensor, HWC ↔ CHW)
- **`ops`**: Core image processing operations
  - `grayscale`: Grayscale conversion
  - `blur`: Gaussian blur filtering
  - `edge`: Edge detection (Sobel)
  - `color`: Color space transformations (RGB ↔ HSV)
- **`imageproc`**: Additional image processing operations
  - `morphology`: Erosion and dilation
  - `filter`: Median and bilateral filtering
  - `geometric`: Resize and rotation
  - `stats`: Histogram computation
- **`draw`**: Visualization tools for deep learning
  - `bbox`: Bounding box drawing
  - `segmentation`: Mask overlay
  - `keypoints`: Keypoint and skeleton visualization
  - `heatmap`: Heatmap and colormap application
  - `text`: Text rendering
- **`prelude`**: Convenient re-exports of commonly used items

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Built with [CubeCL](https://github.com/tracel-ai/cubecl) for compute operations
- Uses [image-rs](https://github.com/image-rs/image) for image I/O
- Inspired by OpenCV and other computer vision libraries

