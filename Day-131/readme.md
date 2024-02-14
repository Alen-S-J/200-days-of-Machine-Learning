

# Computer Vision Concepts and Code Implementation

## Introduction:
Welcome to the Computer Vision Concepts repository! This repository aims to provide a comprehensive overview of fundamental computer vision concepts along with code implementations in Python using popular libraries such as OpenCV and TensorFlow.

## Table of Contents:
1. Image Processing Basics
2. Deep Learning Basics
3. Application: Object Detection
4. Application: Image Segmentation

---

## 1. Image Processing Basics:
In this section, we cover essential image processing techniques such as filtering, edge detection, and morphological operations.

### Code Implementation:
```python
import cv2
import numpy as np

# Read an image
image = cv2.imread('image.jpg')

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred_image, 50, 150)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 2. Deep Learning Basics:
This section covers basic deep learning concepts relevant to computer vision, including neural networks and convolutional layers.

### Code Implementation:
```python
import tensorflow as tf

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
```

---

## 3. Application: Object Detection:
In this section, we explore the application of object detection using pre-trained models and frameworks like YOLO (You Only Look Once) and SSD (Single Shot MultiBox Detector).

### Code Implementation:
```python
# Use a pre-trained YOLO model for object detection
import cv2

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

# Load class names
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Load image
image = cv2.imread("image.jpg")
height, width, _ = image.shape

# Detect objects
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

# Show results
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), 2)

# Display the results
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. Application: Image Segmentation:
In this section, we explore image segmentation techniques such as semantic segmentation and instance segmentation.

### Code Implementation:
```python
# Use a pre-trained Mask R-CNN model for instance segmentation
import cv2
import numpy as np

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow("mask_rcnn_inception_v2_coco.pb", "mask_rcnn_inception_v2_coco.pbtxt")

# Load the image
image = cv2.imread("image.jpg")
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)

# Set the input to the network
net.setInput(blob)

# Run forward pass
boxes, masks = net.forward(["detection_out_final", "detection_masks"])

# Display the results
for i in range(boxes.shape[2]):
    confidence = boxes[0, 0, i, 2]
    if confidence > 0.5:
        class_id = int(boxes[0, 0, i, 1])
        mask = masks[i, class_id]
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = (mask > 0.5)
        roi = image[mask]
        image[mask] = (0.0, 255.0, 0.0)

# Show the result
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

Certainly! Let's dive deeper into image processing by exploring some advanced techniques and their implementations in code:

---

## 1. Histogram Equalization:
Histogram equalization is a technique used to enhance the contrast of an image by redistributing the intensity values. This method is particularly useful for images with low contrast.

### Code Implementation:
```python
import cv2

# Read an image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 2. Image Denoising:
Image denoising techniques aim to remove noise from an image while preserving its important features. One popular method is using a Gaussian filter.

### Code Implementation:
```python
import cv2

# Read an image
image = cv2.imread('noisy_image.jpg')

# Apply Gaussian blur for denoising
denoised_image = cv2.GaussianBlur(image, (5, 5), 0)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 3. Image Sharpening:
Image sharpening enhances the edges and details in an image, making it appear clearer and more defined. One common approach is to use the Laplacian filter.

### Code Implementation:
```python
import cv2
import numpy as np

# Read an image
image = cv2.imread('image.jpg')

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Laplacian filter for sharpening
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
sharpened_image = np.uint8(np.clip(gray_image - 0.5*laplacian, 0, 255))

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. Edge Detection:
Edge detection techniques are used to identify the boundaries of objects within an image. The Canny edge detector is a popular choice due to its effectiveness.

### Code Implementation:
```python
import cv2

# Read an image
image = cv2.imread('image.jpg')

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray_image, 100, 200)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

