//! Basic example demonstrating image processing operations with CubeCV.
//!
//! This example creates synthetic images and applies various operations.

use cubecv::prelude::*;

fn main() {
    println!("CubeCV - Basic Image Processing Example\n");

    // Create a synthetic RGB image (checkerboard pattern)
    let width = 100;
    let height = 100;
    let mut data = vec![0.0; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Create a checkerboard pattern
            if (x / 10 + y / 10) % 2 == 0 {
                data[idx] = 1.0;     // R
                data[idx + 1] = 1.0; // G
                data[idx + 2] = 1.0; // B
            } else {
                data[idx] = 0.2;     // R
                data[idx + 1] = 0.2; // G
                data[idx + 2] = 0.8; // B
            }
        }
    }

    let input = ImageTensor::new(width, height, 3, data);
    println!("Created {}x{} checkerboard image with {} channels", 
             input.width, input.height, input.channels);

    // 1. Grayscale conversion
    println!("\n1. Converting to grayscale...");
    let gray = grayscale(&input);
    println!("   Output: {}x{} with {} channel", gray.width, gray.height, gray.channels);

    // 2. Gaussian blur
    println!("\n2. Applying Gaussian blur (sigma=2.0)...");
    let blurred = gaussian_blur(&input, 2.0);
    println!("   Output: {}x{} with {} channels", blurred.width, blurred.height, blurred.channels);

    // 3. Edge detection (on grayscale)
    println!("\n3. Applying Sobel edge detection...");
    let edges = sobel_edge_detection(&gray);
    println!("   Output: {}x{} with {} channel", edges.width, edges.height, edges.channels);

    // 4. Color space conversion
    println!("\n4. Converting RGB to HSV...");
    let hsv = rgb_to_hsv(&input);
    println!("   Output: {}x{} in HSV color space", hsv.width, hsv.height);

    println!("\n5. Converting HSV back to RGB...");
    let rgb_back = hsv_to_rgb(&hsv);
    println!("   Output: {}x{} in RGB color space", rgb_back.width, rgb_back.height);

    // Verify roundtrip accuracy
    let mut max_diff = 0.0_f32;
    for i in 0..input.data.len() {
        let diff = (input.data[i] - rgb_back.data[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("   RGB->HSV->RGB roundtrip max difference: {:.6}", max_diff);

    // Convert back to DynamicImage for demonstration
    let dynamic_img = input.to_dynamic_image();
    println!("\n6. Converted back to DynamicImage: {:?}", dynamic_img.color());

    println!("\n✓ All operations completed successfully!");
}
