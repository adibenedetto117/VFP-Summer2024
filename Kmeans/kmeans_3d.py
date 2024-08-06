import numpy as np
import tifffile
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def kmeans_3d_clustering(image_3d, n_clusters=3, threshold=None):
    print(f"Image shape: {image_3d.shape}")
    print(f"Image dtype: {image_3d.dtype}")

    # Apply thresholding if specified
    if threshold is not None:
        image_3d = np.where(image_3d > threshold, image_3d, 0)

    # Create a 3D grid of coordinates
    z, y, x = np.meshgrid(np.arange(image_3d.shape[0]),
                          np.arange(image_3d.shape[1]),
                          np.arange(image_3d.shape[2]),
                          indexing='ij')

    # Combine intensity and coordinates
    features = np.column_stack([image_3d.ravel(),
                                z.ravel(), y.ravel(), x.ravel()])

    print(f"Features shape: {features.shape}")

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    # Reshape labels back to 3D
    segmentation = labels.reshape(image_3d.shape)

    return segmentation

def normalize_to_uint8(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized = (array - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)

def visualize_slices(original, thresholded, segmented, slice_index):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].imshow(original[slice_index], cmap='gray')
    axes[0].set_title(f'Original - Slice {slice_index}')
    axes[0].axis('off')
    
    axes[1].imshow(thresholded[slice_index], cmap='gray')
    axes[1].set_title(f'Thresholded - Slice {slice_index}')
    axes[1].axis('off')
    
    axes[2].imshow(segmented[slice_index], cmap='gray')
    axes[2].set_title(f'Segmented - Slice {slice_index}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "tomographic_image.tiff"
    
    # Load the 3D image
    image_3d = tifffile.imread(image_path)
    
    # Analyze image intensities
    min_intensity = np.min(image_3d)
    max_intensity = np.max(image_3d)
    mean_intensity = np.mean(image_3d)
    median_intensity = np.median(image_3d)
    
    print(f"Image intensity range: {min_intensity} to {max_intensity}")
    print(f"Mean intensity: {mean_intensity:.2f}")
    print(f"Median intensity: {median_intensity:.2f}")
    
    # Calculate percentiles for threshold suggestions
    percentiles = [25, 50, 75, 90, 95, 99]
    thresholds = np.percentile(image_3d, percentiles)
    
    print("\nSuggested threshold options:")
    for p, t in zip(percentiles, thresholds):
        print(f"{p}th percentile: {t:.2f}")
    
    # You can choose one of these thresholds or input your own
    threshold = thresholds[4] 
    print(f"\nUsing threshold: {threshold:.2f}")
    
    # Create thresholded image
    thresholded_3d = np.where(image_3d > threshold, image_3d, 0)
    
    # Perform segmentation
    segmentation = kmeans_3d_clustering(thresholded_3d, n_clusters=2)
    
    # Normalize images for visualization
    original_norm = normalize_to_uint8(image_3d)
    thresholded_norm = normalize_to_uint8(thresholded_3d)
    segmentation_norm = normalize_to_uint8(segmentation)
    
    # Visualize a middle slice
    middle_slice = image_3d.shape[0] // 2
    visualize_slices(original_norm, thresholded_norm, segmentation_norm, middle_slice)
    # Save the entire 3D segmentation
    
    print("SAVING")
    tifffile.imwrite("segmentation_kmeans_3d.tiff", segmentation_norm)