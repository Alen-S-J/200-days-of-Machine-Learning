# **Convolutional Layers in CNNs**

Convolutional layers are fundamental building blocks in Convolutional Neural Networks (CNNs) designed to extract and learn features from input images. These layers apply convolution operations to input data using learnable filters or kernels.

**Convolution Operation**
The convolution operation between an input image I and a filter K at a specific location can be represented as:

```
S(i,j)=(I∗K)*(i,j)=∑m∑nI(m,n)⋅K(i−m,j−n)
```

- S(i,j) denotes the output value at position
(i,j) in the feature map.

- I(m,n) represents the pixel value of the input image at location (m,n).

- K(i−m,j−n) corresponds to the filter kernel value at its relative position to the input image.

**Stride and Padding**

- Stride: Determines how much the filter shifts (slides) at each step during convolution.

- Padding: Addition of extra border pixels to the input image to control the output dimensions.

The formula for calculating the output size of a convolution operation given input size

```
O=W-F+2P/S+1
```

**Multiple Convolutional Layers**

Multiple convolutional layers stacked sequentially create deeper architectures. Each layer learns different filters and hierarchical features:

- Deeper layers capture complex patterns by combining low-level features from earlier layers.
- Filter sizes, number of filters, and stacking order impact the network's ability to learn intricate features.

### Mathematical Representation

Let \( I \) be the input image, \( K_1, K_2, ..., K_n \) be the filters in a convolutional layer, and \( S_1, S_2, ..., S_n \) represent the output feature maps obtained by applying these filters:

\[ S_i(i,j) = (I * K_i)(i,j) \]

Where \( i \) and \( j \) iterate over the spatial dimensions of the feature map \( S_i \).

This mathematical representation demonstrates how each filter interacts with the input to produce feature maps capturing specific patterns.

### Advanced Activation Functions

- **Leaky ReLU (LReLU)**: Extends ReLU by allowing a small gradient for negative inputs. The function is defined as:
  \[ f(x) = \begin{cases} x, & \text{if } x > 0 \\ ax, & \text{otherwise} \end{cases} \]
  Here, \( a \) is a small constant, typically around 0.01.

- **ELU (Exponential Linear Unit)**: Handles vanishing gradients better than ReLU. The function is defined as:
  \[ f(x) = \begin{cases} x, & \text{if } x > 0 \\ a(e^x - 1), & \text{otherwise} \end{cases} \]
  Here, \( a \) is a positive constant that smoothens the negative range.

- **PReLU (Parametric ReLU)**: Similar to LReLU but allows the slope to be learned during training, avoiding fixed negative slopes.

### Pooling Variations

- **Average Pooling**: Replaces each patch in the input with the average value of that patch. The mathematical representation for a \(2 \times 2\) average pooling is:
  \[ S(i, j) = \frac{1}{4} \sum_{m=0}^{1}\sum_{n=0}^{1} I(2i+m, 2j+n) \]

- **Global Pooling**: Performs pooling across the entire feature map, reducing spatial dimensions to 1x1.

### Feature Visualization and Interpretability

- **Activation Maximization**: Involves optimizing the input image to maximize the activation of specific neurons, revealing patterns the network is sensitive to.
  
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Generates heatmaps highlighting important regions in an image by examining gradient information flowing into the final convolutional layer.
