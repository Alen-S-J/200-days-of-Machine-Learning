# Concepts and Theoretical Features

### 1. Convolutional Neural Networks (CNNs)
**Concept:** CNNs are specialized neural networks designed for processing grid-like data, such as images.  
**Theoretical Features:**
- **Local Connectivity:** Neurons in each layer are connected to only a small, local region of the input volume.
- **Weight Sharing:** Shared weights reduce parameters, enabling feature reuse.
- **Hierarchical Structure:** Layers progressively extract higher-level features from raw pixels.
- **Mathematical Elements:**
  - **Convolution Operation:** The formula involves element-wise multiplication and summing:
    ```
    S(i,j) = (I∗K)(i,j) = ∑ ∑ I(m,n)⋅K(i−m,j−n)
    ```
    Where S is the resulting feature map, I is the input image, K is the kernel/filter.
  - **Hierarchical Feature Learning:** Layers extract features using filters for higher-level feature extraction.

### 2. Convolutional Layers
**Concept:** Perform feature extraction using filters applied to input images.  
**Theoretical Features:**
- **Filter Operation:** Dot product of the filter and input image region.
- **Strides and Padding:** Impact output size - O = (W−K+2P) / S + 1.
- **Feature Map Generation:** Output represents learned features.
- **Mathematical Explanation:**
  - **Output Size Calculation:** O = (W−K+2P) / S + 1, where O is the output size, W is the input size, K is the kernel size, P is padding, and S is the stride.
  - **Weight Sharing Concept:** Shared weights across inputs capture similar features.

### 3. Activation Functions
**Concept:** Introduce non-linearities to CNNs.  
**Theoretical Features:**
- **ReLU:** Solves vanishing gradient problem effectively.
  - **Function Definition:** f(x) = max(0, x).
- **Sigmoid, Tanh:** Other activation functions and their properties.
- **Mathematical Explanation:**
  - **ReLU Derivative:** f'(x) = 1 if x > 0, 0 otherwise.
  - **Sigmoid:** f(x) = 1 / (1 + e^(-x)), Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)).

### 4. Pooling Layers
**Concept:** Downsample feature maps, reducing complexity.  
**Theoretical Features:**
- **Max Pooling, Average Pooling:** Techniques summarizing localized information.
- **Downsampling:** Reducing spatial dimensions while retaining essential information.
- **Mathematical Explanation:**
  - **Max Pooling:** Selects the maximum value from a pool window.
  - **Average Pooling:** Calculates the average value within a pool window.
