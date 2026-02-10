//! Advanced features example: HWC/CHW conversions, imageproc operations, and visualization.

use cubecv::prelude::*;

fn main() {
    println!("CubeCV - Advanced Features Example\n");

    // ===== Part 1: HWC/CHW Conversions =====
    println!("=== Part 1: HWC/CHW Format Conversions ===");
    
    // Create data in CHW format (common in PyTorch)
    let chw_data = vec![
        // R channel
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        // G channel
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5,
        // B channel
        0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2,
    ];
    
    let img = ImageTensor::from_chw_data(4, 2, 3, chw_data);
    println!("Created image from CHW data: {}x{}x{}", img.width, img.height, img.channels);
    
    // Convert to HWC and back
    let hwc_data = img.to_hwc_data();
    let chw_data_back = img.to_chw_data();
    println!("HWC data length: {}, CHW data length: {}", hwc_data.len(), chw_data_back.len());
    
    // ===== Part 2: imageproc Operations =====
    println!("\n=== Part 2: imageproc-inspired Operations ===");
    
    let data = vec![0.5; 50 * 50 * 3];
    let mut test_img = ImageTensor::new(50, 50, 3, data);
    
    // Convert to grayscale for morphology
    let gray = grayscale(&test_img);
    println!("Converted to grayscale: {}x{}", gray.width, gray.height);
    
    // Morphological operations
    println!("\n2a. Morphological Operations:");
    let eroded = erode(&gray, 3);
    println!("  Applied erosion with kernel size 3");
    
    let dilated = dilate(&gray, 3);
    println!("  Applied dilation with kernel size 3");
    
    // Filtering
    println!("\n2b. Image Filtering:");
    let median_filtered = median_filter(&test_img, 3);
    println!("  Applied median filter ({}x{})", median_filtered.width, median_filtered.height);
    
    let bilateral = bilateral_filter(&test_img, 5, 1.0, 0.1);
    println!("  Applied bilateral filter ({}x{})", bilateral.width, bilateral.height);
    
    // Geometric transformations
    println!("\n2c. Geometric Transformations:");
    let resized = resize_bilinear(&test_img, 100, 100);
    println!("  Resized to {}x{}", resized.width, resized.height);
    
    let rotated = rotate(&test_img, 45.0);
    println!("  Rotated by 45 degrees");
    
    // Statistics
    println!("\n2d. Statistical Operations:");
    let hist = histogram(&gray, 10);
    println!("  Computed histogram with {} bins", hist.len());
    let sum: f32 = hist.iter().sum();
    println!("  Histogram sum (should be 1.0): {:.4}", sum);
    
    // ===== Part 3: Deep Learning Visualizations =====
    println!("\n=== Part 3: Deep Learning Visualization ===");
    
    let mut viz_img = ImageTensor::new(200, 200, 3, vec![0.2; 200 * 200 * 3]);
    
    // Draw bounding boxes
    println!("\n3a. Bounding Box Drawing:");
    let bbox1 = BoundingBox::new(20, 20, 60, 80)
        .with_label("person")
        .with_confidence(0.95);
    draw_bbox(&mut viz_img, &bbox1, [1.0, 0.0, 0.0], 2);
    println!("  Drew bbox at ({}, {}) size {}x{}", 
             bbox1.x, bbox1.y, bbox1.width, bbox1.height);
    
    let bbox2 = BoundingBox::new(100, 50, 50, 70)
        .with_label("car")
        .with_confidence(0.88);
    draw_bbox(&mut viz_img, &bbox2, [0.0, 1.0, 0.0], 2);
    println!("  Drew bbox at ({}, {}) size {}x{}", 
             bbox2.x, bbox2.y, bbox2.width, bbox2.height);
    
    // Draw text
    println!("\n3b. Text Rendering:");
    draw_text(&mut viz_img, "Person", 22, 15, [1.0, 1.0, 1.0], 1.0);
    println!("  Drew text label");
    
    // Segmentation mask
    println!("\n3c. Segmentation Overlay:");
    let mut seg_img = ImageTensor::new(100, 100, 3, vec![0.3; 100 * 100 * 3]);
    let mut mask_data = vec![0.0; 100 * 100];
    // Create a circular mask
    for y in 0..100 {
        for x in 0..100 {
            let dx = x as f32 - 50.0;
            let dy = y as f32 - 50.0;
            if (dx * dx + dy * dy).sqrt() < 30.0 {
                mask_data[(y * 100 + x) as usize] = 1.0;
            }
        }
    }
    let mask = ImageTensor::new(100, 100, 1, mask_data);
    draw_segmentation_mask(&mut seg_img, &mask, [0.0, 1.0, 0.0], 0.5);
    println!("  Drew circular segmentation mask");
    
    // Keypoints
    println!("\n3d. Keypoint Visualization:");
    let mut kp_img = ImageTensor::new(150, 150, 3, vec![0.2; 150 * 150 * 3]);
    let keypoints = vec![
        Keypoint::new(50, 50).with_confidence(0.98),
        Keypoint::new(70, 60).with_confidence(0.95),
        Keypoint::new(90, 70).with_confidence(0.92),
    ];
    draw_keypoints(&mut kp_img, &keypoints, [1.0, 0.0, 0.0], 3);
    println!("  Drew {} keypoints", keypoints.len());
    
    // Skeleton
    let connections = vec![(0, 1), (1, 2)];
    draw_skeleton(&mut kp_img, &keypoints, &connections, [0.0, 1.0, 0.0], 2);
    println!("  Drew skeleton with {} connections", connections.len());
    
    // Heatmap
    println!("\n3e. Heatmap Visualization:");
    let heatmap_data = (0..100)
        .flat_map(|y| {
            (0..100).map(move |x| {
                let dx = x as f32 - 50.0;
                let dy = y as f32 - 50.0;
                let dist = (dx * dx + dy * dy).sqrt();
                (1.0 - (dist / 50.0)).max(0.0)
            })
        })
        .collect();
    let heatmap = ImageTensor::new(100, 100, 1, heatmap_data);
    
    let heatmap_colored = apply_heatmap(&heatmap, "jet");
    println!("  Applied jet colormap to heatmap ({}x{})", 
             heatmap_colored.width, heatmap_colored.height);
    
    let heatmap_hot = apply_heatmap(&heatmap, "hot");
    println!("  Applied hot colormap to heatmap");
    
    let heatmap_viridis = apply_heatmap(&heatmap, "viridis");
    println!("  Applied viridis colormap to heatmap");
    
    // Overlay heatmap
    let mut overlay_img = ImageTensor::new(100, 100, 3, vec![0.5; 100 * 100 * 3]);
    overlay_heatmap(&mut overlay_img, &heatmap, "jet", 0.5);
    println!("  Overlaid heatmap on image with alpha=0.5");
    
    println!("\n✓ All advanced features demonstrated successfully!");
}
