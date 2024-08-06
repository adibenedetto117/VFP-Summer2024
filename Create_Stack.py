import os
import numpy as np
from PIL import Image
import tifffile

# User-defined variables
input_dir = "/Users/anthonydibenedetto/Desktop/VFP_MEETING/Tomo88/output_2"
output_dir = "/Users/anthonydibenedetto/Desktop/VFP_MEETING/Tomo88"
start_slice = 0
end_slice = 2100 

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of TIFF files
tiff_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".tiff")])

# Ensure start and end slices are within bounds
start_slice = max(0, min(start_slice, len(tiff_files) - 1))
end_slice = max(start_slice, min(end_slice, len(tiff_files)))

# Create output filename based on slice range
output_file = os.path.join(output_dir, f"tomographic_image_{start_slice}-{end_slice}.tiff")

total_files = end_slice - start_slice
images = []

for i, filename in enumerate(tiff_files[start_slice:end_slice], start=1):
    filepath = os.path.join(input_dir, filename)
    img = np.array(Image.open(filepath))
    images.append(img)

    if i % (total_files // 10 + 1) == 0 or i == total_files:
        print(f"Processed {i}/{total_files} images ({(i / total_files) * 100:.0f}%)")
        print(f"Image shape: {img.shape}")

# Convert list of images to 3D numpy array
stack = np.array(images)

# Save the 3D stack as a TIFF file
try:
    tifffile.imwrite(output_file, stack)
    print(f"Successfully saved 3D stack: {output_file}")
    print(f"Stack shape: {stack.shape}")
except Exception as e:
    print(f"Error saving file: {e}")

print("3D image stack creation and saving complete.")

# Verification step
try:
    loaded_stack = tifffile.imread(output_file)
    print(f"Verification - Loaded stack shape: {loaded_stack.shape}")
except Exception as e:
    print(f"Error loading file for verification: {e}")