import numpy as np
import tifffile
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import filters, segmentation, measure
from mpl_toolkits.mplot3d import Axes3D

def create_depth_map(image_3d):
    grad_z = np.gradient(image_3d, axis=0)
    depth_map = np.cumsum(np.abs(grad_z), axis=0)
    return depth_map

def smooth_depth_map(depth_map, sigma=1):
    smoothed_depth_map = ndimage.gaussian_filter(depth_map, sigma=sigma)
    return smoothed_depth_map

def segment_with_depth_map(image_3d, percentile=75, **kwargs):
    # Create depth map
    depth_map = create_depth_map(image_3d)
    sigma=.8
    
    # Calculate threshold based on depth_map
    smoothed_depth_map = smooth_depth_map(depth_map, sigma=sigma)
    
    threshold = np.percentile(smoothed_depth_map, percentile)
    
    # Simple thresholding
    segmentation = smoothed_depth_map > threshold
    
    # Connected component analysis
    labeled, num_features = ndimage.label(segmentation)
    
    return labeled, depth_map, num_features

def normalize_to_uint8(array):
    min_val, max_val = np.min(array), np.max(array)
    normalized = (array - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)

def visualize_slices(original, segmented, depth, slice_index):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].imshow(original[slice_index], cmap='gray')
    axes[0].set_title(f'Original - Slice {slice_index}')
    axes[0].axis('off')
    
    axes[1].imshow(segmented[slice_index], cmap='gray')
    axes[1].set_title(f'Segmented - Slice {slice_index}')
    axes[1].axis('off')
    
    im = axes[2].imshow(depth[slice_index], cmap='jet')
    axes[2].set_title(f'Depth Map - Slice {slice_index}')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], label='Depth')
    
    plt.tight_layout()
    plt.show()

def visualize_3d_depth(depth_map):
    fig = plt.figure(figsize=(20, 6))
    
    # 1. Visualize middle slice
    ax1 = fig.add_subplot(131)
    middle_slice = depth_map.shape[0] // 2
    im1 = ax1.imshow(depth_map[middle_slice], cmap='viridis')
    ax1.set_title(f'Depth Map - Middle Slice (Z={middle_slice})')
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, label='Depth')
    
    # 2. Visualize maximum intensity projection
    ax2 = fig.add_subplot(132)
    max_projection = np.max(depth_map, axis=0)
    im2 = ax2.imshow(max_projection, cmap='viridis')
    ax2.set_title('Depth Map - Maximum Intensity Projection')
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, label='Depth')
    
    # 3. 3D surface plot of the middle slice
    ax3 = fig.add_subplot(133, projection='3d')
    x, y = np.meshgrid(np.arange(depth_map.shape[2]), np.arange(depth_map.shape[1]))
    
    # Create a mask to show only significant features
    slice_to_plot = depth_map[middle_slice]
    mask = slice_to_plot > np.percentile(slice_to_plot, 95)
    
    # Plot the surface
    surf = ax3.plot_surface(x, y, slice_to_plot * mask, cmap='viridis', 
                            linewidth=0, antialiased=False)
    
    ax3.set_title(f'3D Depth Map - Middle Slice (Z={middle_slice})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Depth')
    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    ax3.zaxis.set_ticklabels([])
    
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5, label='Depth')
    
    plt.tight_layout()
    plt.show()

def visualize_multiple_slices(depth_map, num_slices=5):
    total_slices = depth_map.shape[0]
    step = total_slices // num_slices
    
    fig, axes = plt.subplots(1, num_slices, figsize=(4*num_slices, 4))
    for i, ax in enumerate(axes):
        slice_index = i * step
        im = ax.imshow(depth_map[slice_index], cmap='viridis')
        ax.set_title(f'Slice {slice_index}')
        ax.axis('off')
        fig.colorbar(im, ax=ax, label='Depth')
    
    plt.tight_layout()
    plt.show()

def analyze_segmentation(segmented, num_features):
    """Analyze the segmentation results."""
    volumes = ndimage.sum(np.ones_like(segmented), segmented, index=range(1, num_features + 1))
    centroids = ndimage.center_of_mass(segmented, segmented, index=range(1, num_features + 1))
    
    print(f"\nSegmentation Analysis:")
    print(f"Number of segmented features: {num_features}")
    print(f"Largest segment volume: {np.max(volumes):.2f}")
    print(f"Smallest segment volume: {np.min(volumes):.2f}")
    print(f"Mean segment volume: {np.mean(volumes):.2f}")

    return volumes, centroids

if __name__ == "__main__":
    image_path = "tomographic_image.tiff"
    
    # Load the 3D image
    image_3d = tifffile.imread(image_path)
    
    print(f"Image shape: {image_3d.shape}")
    print(f"Image dtype: {image_3d.dtype}")
    
    # Analyze image intensities
    min_intensity, max_intensity = np.min(image_3d), np.max(image_3d)
    mean_intensity, median_intensity = np.mean(image_3d), np.median(image_3d)
    
    print(f"Image intensity range: {min_intensity} to {max_intensity}")
    print(f"Mean intensity: {mean_intensity:.2f}")
    print(f"Median intensity: {median_intensity:.2f}")
    
    # Perform segmentation with depth map refinement
    segmentation, depth_map, num_features = segment_with_depth_map(image_3d, method='threshold')
    
    print(f"\nSegmentation shape: {segmentation.shape}")
    print(f"Segmentation dtype: {segmentation.dtype}")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth map dtype: {depth_map.dtype}")
    
    # Analyze segmentation results
    volumes, centroids = analyze_segmentation(segmentation, num_features)
    
    # Normalize images for visualization
    original_norm = normalize_to_uint8(image_3d)
    segmentation_norm = normalize_to_uint8(segmentation)
    depth_map_norm = normalize_to_uint8(depth_map)
    
    # Visualize a middle slice
    middle_slice = image_3d.shape[0] // 2
    visualize_slices(original_norm, segmentation_norm, depth_map_norm, middle_slice)

    # Visualize 3D depth map
    visualize_3d_depth(depth_map)
    
    # Visualize multiple slices of the depth map
    visualize_multiple_slices(depth_map, num_slices=5)
    
    # Save the results
    print("\nSaving segmentation and depth map results...")
    tifffile.imwrite("segmentation_3d.tiff", segmentation_norm)
    tifffile.imwrite("depth_map_3d.tiff", depth_map_norm)
    print("Segmentation saved as 'segmentation_3d.tiff'")
    print("Depth map saved as 'depth_map_3d.tiff'")

    print("\nProcessing complete.")
