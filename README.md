# CubeCV

A high-performance parallel computing computer vision library using Rust and CubeCL.

CubeCV provides GPU-accelerated image processing operations that work seamlessly with [image-rs](https://github.com/image-rs/image) types, powered by the [CubeCL](https://github.com/tracel-ai/cubecl) compute library.

## Features

- 🚀 **High Performance**: Designed for GPU acceleration using CubeCL
- 🖼️ **image-rs Integration**: Seamless conversion between `DynamicImage` and internal tensor format
- 🎨 **Rich Operations**: Common computer vision operations out of the box
- 📦 **Easy to Use**: Simple, idiomatic Rust API

## Supported Operations

### Color Conversions
- **Grayscale**: Convert RGB/RGBA images to grayscale using ITU-R BT.601 weights
- **RGB ↔ HSV**: Bidirectional color space transformations

### Filtering
- **Gaussian Blur**: Smooth images with configurable sigma
- **Edge Detection**: Sobel edge detection filter

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

## Running Examples

The library includes example programs demonstrating various features:

```bash
# Basic operations example
cargo run --example basic

# Image processing pipeline
cargo run --example pipeline
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
- `get_pixel(x, y, c)`: Get pixel value
- `set_pixel(x, y, c, value)`: Set pixel value

### Operations

All operations take an `&ImageTensor` and return a new `ImageTensor`.

- `grayscale(&ImageTensor) -> ImageTensor`
- `gaussian_blur(&ImageTensor, sigma: f32) -> ImageTensor`
- `sobel_edge_detection(&ImageTensor) -> ImageTensor`
- `rgb_to_hsv(&ImageTensor) -> ImageTensor`
- `hsv_to_rgb(&ImageTensor) -> ImageTensor`

## Performance Considerations

The current implementation uses CPU-based processing as a foundation. Future versions will leverage CubeCL's GPU acceleration capabilities for improved performance on large images.

## Architecture

CubeCV is structured into several modules:

- **`convert`**: Image format conversions (image-rs ↔ tensor)
- **`ops`**: Image processing operations
  - `grayscale`: Grayscale conversion
  - `blur`: Gaussian blur filtering
  - `edge`: Edge detection (Sobel)
  - `color`: Color space transformations
- **`prelude`**: Convenient re-exports

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Built with [CubeCL](https://github.com/tracel-ai/cubecl) for compute operations
- Uses [image-rs](https://github.com/image-rs/image) for image I/O
- Inspired by OpenCV and other computer vision libraries

