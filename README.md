# VFP-Summer2024: Application of Data Analytics and Deep Learning for Improving and Accelerating Large-Scale Ptychographic Imaging Problems

## Overview
This repository contains code and resources for the project "Application of Data Analytics and Deep Learning for Improving and Accelerating Large-Scale Ptychographic Imaging Problems." The project focuses on 3D image segmentation using deep learning techniques, specifically applied to tomographic images of borosilicate glass spheres encased in a polypropylene matrix.

## Table of Contents
- [Introduction](#introduction)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Segmentation Methods](#segmentation-methods)
  - [3D U-Net](#3d-u-net)
  - [K-means Clustering](#k-means-clustering)
  - [Depth Maps](#depth-maps)
- [Training Process](#training-process)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Introduction
This project aims to enhance 3D image segmentation, a critical technique in fields such as medical imaging, materials science, and industrial quality control. We developed a modified 3D U-Net architecture to accurately segment complex 3D structures in volumetric data.

## Dataset and Preprocessing
### Sphere Dataset
- Samples with varied volume fractions of borosilicate glass spheres in a polypropylene matrix.
- Scanned using X-ray tomography at distances of 25 mm and 60 mm.
- 2000 projections per scan.

### Data Preprocessing
- Normalization to [0, 1] range.
- Initial mask generation using adaptive histogram equalization and Otsu thresholding.
- Data chunking to handle large 3D volumes efficiently.

## Segmentation Methods
### 3D U-Net
A modified 3D U-Net architecture was developed, focusing on maintaining spatial context along the z-axis while performing convolutions and pooling operations in the x and y dimensions.

### K-means Clustering
K-means clustering was used to segment voxel intensities along with their spatial coordinates, applying StandardScaler and thresholding before clustering.

### Depth Maps
Depth maps were created by calculating the gradient along the z-axis, smoothing with a Gaussian filter, and applying thresholding for segmentation.

## Training Process
- Data processed in chunks of 10 slices along the z-axis.
- Model trained using the Adam optimizer with an initial learning rate of 1e-4, a maximum of 100 epochs, and a batch size of 4.
- Multi-GPU setup for accelerated training and larger batch sizes.

## Results
### Segmentation Performance
- The 3D U-Net model achieved a Dice coefficient of 0.934, indicating excellent overlap between predicted and ground truth segmentations.

### K-means Clustering and Depth Map Segmentation
- K-means clustering and depth maps provided valuable insights and alternative approaches for 3D image segmentation.

### Validation on Additional Datasets
- The model was validated on the Tomography Round-Robin and Lorentz datasets, achieving consistent performance with high Dice coefficients.

## Conclusion
This study demonstrates the effectiveness of a 3D U-Net for segmenting borosilicate glass spheres in a polypropylene matrix, achieving high accuracy across different volume fractions and scanning conditions. K-means clustering and depth maps offered alternative segmentation approaches. Future work could explore advanced deep learning architectures and self-supervised learning techniques to further enhance segmentation accuracy.

## Acknowledgments
I express my sincere gratitude to the VFP program for their support and resources. Special thanks to Dr. Abuomar and Dr. Bicer for their invaluable guidance and support throughout this study.

For more details, please refer to the research paper.
