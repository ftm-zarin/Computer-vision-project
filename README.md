# Computer Vision Project: CNN Implementation and Neural Style Transfer

## Overview
This project, developed for an Introduction to Machine Learning course, consists of two main parts:
1.  **Implementing a basic Convolutional Neural Network (CNN) layer** from scratch using NumPy to understand the mechanics of forward and backward passes.
2.  **Implementing Neural Style Transfer (NST)** using PyTorch and a pre-trained VGG19 model to combine the content of one image with the style of another.

## Setup and Dependencies
To run the project, the following libraries are required:
- torch
- torchvision
- numpy
- matplotlib
- Pillow
- torchsummary (used for VGG architecture analysis)
- requests (used to fetch ImageNet class labels)

Install dependencies using:
%pip install torch torchvision numpy matplotlib Pillow torchsummary requests

## Section 1: CNN Implementation from Scratch

### Objective
To manually implement the fundamental components of a 2D Convolutional Layer: the forward pass (convolution operation) and the backward pass (gradient calculation).

### Implementation Details
- **Class `MyConv`**: Contains the logic for the forward and backward passes.
- **Forward Pass**: Implements the convolution operation, including padding and stride, to compute the output feature map. The output size calculation is based on input size, kernel size, stride, and padding.
- **Backward Pass**: Calculates the gradients $\mathbf{dx}$ (with respect to input), $\mathbf{dw}$ (with respect to weights/kernel), and $\mathbf{db}$ (with respect to bias).
- **Verification**: The implementation is verified against expected numerical outputs using the `rel_error` (relative error) function and numerical gradient checking (`eval_numerical_gradient_array`).

### Visualization
- The implemented `MyConv` layer is used with classic kernels (Grayscale, Sobel X, Sobel Y, Gaussian Blur) to demonstrate basic image processing effects on sample images.

## Section 2: Neural Style Transfer (NST)

### Objective
To generate a new image that combines the content structure of one image (Content Image) with the artistic style of another (Style Image).

### Core Components

1.  **Image Preprocessing**:
    - Images (content and style) are loaded, resized to a consistent dimension (e.g., 512x512 or 256x256), and converted into normalized PyTorch tensors.

2.  **Feature Extractor (VGG19)**:
    - A **pre-trained VGG19 model** is used to extract meaningful features from the images.
    - The model's weights are **frozen** (non-trainable) as it acts purely as a feature representation network.
    - The architecture is briefly analyzed, distinguishing between the `features` (feature extraction layers) and the `classifier` (classification layers).

3.  **Loss Functions**: The transfer is achieved by minimizing a total loss function composed of two parts:
    - **Content Loss (`ContentLoss`)**: Calculated as the Mean Squared Error (MSE) between the feature maps of the generated image and the content image. This preserves the high-level spatial arrangement and structure of the content. (Used at: `conv_4` layer).
    - **Style Loss (`StyleLoss`)**: Calculated as the MSE between the **Gram Matrices** of the feature maps of the generated image and the style image. The Gram Matrix captures the texture and style correlations. (Used at: `conv_1`, `conv_2`, `conv_3`, `conv_4`, `conv_5` layers).

4.  **Optimization**:
    - **Optimizer**: The **LBFGS** optimizer is used. Unlike standard gradient descent, LBFGS is a quasi-Newton method that is effective for optimizing the pixel values of the image itself.
    - **Target**: The pixel values of an input image (initialized as the content image clone) are adjusted to minimize the total loss.
    - **Total Loss**: $\text{Loss}_{\text{Total}} = \alpha \cdot \text{Loss}_{\text{Content}} + \beta \cdot \text{Loss}_{\text{Style}}$. The weights ($\alpha$, $\beta$) are hyperparameters to tune the content-style balance.
    - **Post-processing**: The image pixel values are clipped to the valid range [0, 1] after each step.

## Results
The notebook concludes with:
- The final **styled image** after optimization.
- Plots showing the convergence of the **Total Loss**, **Style Loss**, and **Content Loss** over the optimization iterations.
