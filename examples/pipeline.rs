//! Example demonstrating image processing pipeline.
//!
//! This example shows how to chain multiple operations together.

use cubecv::prelude::*;

fn main() {
    println!("CubeCV - Image Processing Pipeline Example\n");

    // Create a test image with a gradient
    let width = 256;
    let height = 256;
    let mut data = vec![0.0; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Create a radial gradient
            let dx = x as f32 - width as f32 / 2.0;
            let dy = y as f32 - height as f32 / 2.0;
            let dist = (dx * dx + dy * dy).sqrt();
            let max_dist = width as f32 / 2.0;
            let intensity = 1.0 - (dist / max_dist).min(1.0);

            data[idx] = intensity;           // R
            data[idx + 1] = intensity * 0.5; // G
            data[idx + 2] = intensity * 0.8; // B
        }
    }

    let input = ImageTensor::new(width, height, 3, data);
    println!("Created {}x{} radial gradient image", width, height);

    // Processing pipeline
    println!("\nProcessing Pipeline:");

    // Step 1: Convert to HSV and modify saturation
    println!("1. Converting to HSV and boosting saturation...");
    let mut hsv = rgb_to_hsv(&input);
    
    // Boost saturation
    for y in 0..hsv.height {
        for x in 0..hsv.width {
            let s = hsv.get_pixel(x, y, 1);
            hsv.set_pixel(x, y, 1, (s * 1.5).min(1.0));
        }
    }
    
    // Step 2: Convert back to RGB
    println!("2. Converting back to RGB...");
    let enhanced = hsv_to_rgb(&hsv);

    // Step 3: Apply blur for smoothing
    println!("3. Applying Gaussian blur for smoothing...");
    let smoothed = gaussian_blur(&enhanced, 1.5);

    // Step 4: Convert to grayscale
    println!("4. Converting to grayscale...");
    let gray = grayscale(&smoothed);

    // Step 5: Detect edges
    println!("5. Detecting edges with Sobel filter...");
    let edges = sobel_edge_detection(&gray);

    // Calculate some statistics
    let edge_mean: f32 = edges.data.iter().sum::<f32>() / edges.data.len() as f32;
    let edge_max = edges.data.iter().cloned().fold(0.0_f32, f32::max);
    
    println!("\nEdge Detection Statistics:");
    println!("  Mean edge strength: {:.4}", edge_mean);
    println!("  Max edge strength: {:.4}", edge_max);

    // Find strongest edge pixel
    let mut max_pos = (0, 0);
    let mut max_val = 0.0_f32;
    for y in 0..edges.height {
        for x in 0..edges.width {
            let val = edges.get_pixel(x, y, 0);
            if val > max_val {
                max_val = val;
                max_pos = (x, y);
            }
        }
    }
    
    println!("  Strongest edge at: ({}, {})", max_pos.0, max_pos.1);

    println!("\n✓ Pipeline completed successfully!");
}
