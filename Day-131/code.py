
#Use a pre-trained YOLO model for object detection
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
