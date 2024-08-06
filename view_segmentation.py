import tifffile
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

# Configuration variables
IMAGE_PATH = 'tomographic_image_88.tiff'  # Path to the original image file
MASK_PATH = 'MASK_FULL_88.tiff'  # Path to the ground truth mask file
SEGMENTED_OUTPUT_PATH = 'segmented_output_Full.tiff'  # Path to the segmented output file

# User-defined variables
SLICE_INDICES = [900, 1000, 1200]  # Indices of slices to visualize
THRESHOLD = 0.15  # Threshold for converting to black and white

def flatten_segmented_output(segmented_output):
    num_chunks, chunk_size, height, width, channels = segmented_output.shape
    flattened_output = segmented_output.reshape(num_chunks * chunk_size, height, width, channels)
    return flattened_output

def apply_threshold_to_slice(slice_data, threshold=0.5):
    """ Apply a simple threshold to a slice to convert it to black and white. """
    bw_slice = np.where(slice_data > threshold, 0, 255).astype(np.uint8)
    return bw_slice

def visualize_comparison(image_path, mask_path, segmented_output_path, slice_indices, threshold):
    # Load the original image, ground truth mask, and segmented output
    image = tifffile.imread(image_path)
    mask = tifffile.imread(mask_path)
    segmented_output = tifffile.imread(segmented_output_path)
    
    # Print shapes of the loaded data
    print(f"Shape of the original image: {image.shape}")
    print(f"Shape of the ground truth mask: {mask.shape}")
    print(f"Shape of the segmented output: {segmented_output.shape}")
    
    # Plot the slices
    num_slices = len(slice_indices)
    fig, axes = plt.subplots(3, num_slices, figsize=(15, 10))
    
    for i, slice_index in enumerate(slice_indices):
        if slice_index >= image.shape[0] or slice_index >= mask.shape[0] or slice_index >= segmented_output.shape[0]:
            print(f"Slice index {slice_index} is out of range.")
            continue

        # Extract slices from the original image, ground truth mask, and segmented output
        image_slice = image[slice_index]
        mask_slice = mask[slice_index]
        segmented_slice = segmented_output[slice_index]
        
        # Print slice statistics
        print(f"Slice {slice_index} statistics:")
        print(f"Min value: {np.min(segmented_slice)}")
        print(f"Max value: {np.max(segmented_slice)}")
        print(f"Mean value: {np.mean(segmented_slice)}")
        
        # Apply histogram equalization to enhance contrast
        segmented_slice_eq = exposure.equalize_hist(segmented_slice)
        
        # Apply thresholding to the equalized segmented slice
        segmented_slice_bw = apply_threshold_to_slice(segmented_slice_eq, threshold)
        
        # Print slice information
        print(f"Visualizing slice {slice_index}:")
        print(f"Shape of image slice: {image_slice.shape}")
        print(f"Shape of mask slice: {mask_slice.shape}")
        print(f"Shape of segmented slice: {segmented_slice.shape}")

        # Original image slice
        axes[0, i].imshow(np.squeeze(image_slice), cmap='gray')
        axes[0, i].set_title(f'Original Image Slice {i}')
        axes[0, i].axis('off')
        
        # Ground truth mask slice
        axes[1, i].imshow(np.squeeze(mask_slice), cmap='gray')
        axes[1, i].set_title(f'Ground Truth Mask Slice {i}')
        axes[1, i].axis('off')
        
        # Segmented output slice
        axes[2, i].imshow(segmented_slice_bw, cmap='gray')
        axes[2, i].set_title(f'Segmented Output Slice {i}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set slice indices
    slice_indices = SLICE_INDICES

    visualize_comparison(IMAGE_PATH, MASK_PATH, SEGMENTED_OUTPUT_PATH, slice_indices, THRESHOLD)

if __name__ == "__main__":
    main()