# YOLO Object Detection Program

This repository contains a Python script for performing object detection using the YOLO (You Only Look Once) algorithm. The script takes an input image and detects objects of different classes present in the image.

**Dependencies:**
- Python 3.x
- OpenCV (cv2)
- NumPy


**Usage:**
1. Clone the repository to your local machine:


2. Make sure you have the required dependencies installed. You can install them using pip:

```bash
pip install opencv-python numpy
```

3. Run the YOLO object detection script, providing the necessary arguments:
   - `-i` or `--image`: Path to the input image.
   - `-c` or `--config`: Path to the YOLO config file (.cfg).
   - `-w` or `--weights`: Path to the pre-trained YOLO weights file (.weights).
   - `-cl` or `--classes`: Path to the text file containing class names.

Example usage:

```bash
python yolo_object_detection.py -i input_image.jpg -c yolov3.cfg -w yolov3.weights -cl coco.names
```

**Files:**
- `yolo_object_detection.py`: The main Python script for performing object detection using YOLO.
- `yolov3.cfg`: YOLO configuration file defining the architecture of the neural network.
- `yolov3.weights`: Pre-trained weights for the YOLO model.
- `coco.names`: Text file containing class names used in the COCO dataset.

**Functionality:**
- The script loads the input image and the necessary YOLO configuration and weights files.
- It performs object detection on the image using YOLO.
- Detected objects are drawn with bounding boxes and labels on the image.
- The annotated image is displayed and saved as "object-detection.jpg".

**Note:**
- This implementation is based on YOLOv3. Make sure to use the corresponding configuration and weights files for other versions of YOLO.
- Adjust the confidence and NMS (Non-Maximum Suppression) thresholds as needed for your specific use case.

