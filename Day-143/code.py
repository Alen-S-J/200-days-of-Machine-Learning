import numpy as np
import os
import tensorflow as tf
import cv2

# Load the pre-trained SSD model
model_name = 'ssd_mobilenet_v2_coco'
model_dir = tf.keras.utils.get_file(
  fname=model_name, 
  origin="http://download.tensorflow.org/models/object_detection/tf2/20200711/{}".format(model_name),
  untar=True)

model_path = os.path.join(model_dir, 'saved_model')

# Load the saved model
detect_fn = tf.saved_model.load(model_path)

# Define the labels used in the pre-trained model
label_map_path = os.path.join(model_dir, 'mscoco_label_map.pbtxt')
category_index = {}
with open(label_map_path, 'r') as f:
    for line in f.readlines():
        if 'id:' in line:
            id_num = int(line.strip().split(':')[-1])
        elif 'display_name:' in line:
            display_name = line.strip().split(':')[-1].replace("'", "").strip()
            category_index[id_num] = {'name': display_name}

# Function to perform object detection on an image
def detect_objects(image_path):
    image_np = cv2.imread(image_path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Convert detection classes to int and filter out detections with confidence less than 0.5
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    detections = {key: value[detections['detection_scores'] > 0.5] for key, value in detections.items()}
    detections['num_detections'] = detections['detection_boxes'].shape[0]

    # Draw bounding boxes on the image
    for i in range(detections['num_detections']):
        class_id = int(detections['detection_classes'][i])
        display_name = category_index[class_id]['name']
        box = detections['detection_boxes'][i]
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * image_np.shape[1])
        ymin = int(ymin * image_np.shape[0])
        xmax = int(xmax * image_np.shape[1])
        ymax = int(ymax * image_np.shape[0])
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image_np, display_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image to perform object detection on
image_path = 'example.jpg'

# Perform object detection
detect_objects(image_path)
