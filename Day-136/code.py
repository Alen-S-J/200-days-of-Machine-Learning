import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image  # Import the Image module from PIL

# Define your Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define your image preprocessing function
def preprocess_image(image):
    # Convert image to tensor
    image_tensor = F.to_tensor(image)
    # Normalize image
    image_normalized = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Add batch dimension
    image_normalized = image_normalized.unsqueeze(0)
    return image_normalized

# Load your image
image = Image.open("Day-136/OIF.jpeg")  # Provide the path to your image

# Preprocess the image
input_image = preprocess_image(image)

# Make predictions
with torch.no_grad():
    prediction = model(input_image)

# Visualize the predictions
# You can write code here to draw bounding boxes on the image based on the predictions
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to draw bounding boxes on the image
def draw_boxes(image, prediction):
    # Get predicted boxes, labels, and scores
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Draw bounding boxes
    for box, label, score in zip(boxes, labels, scores):
        # Get box coordinates
        x1, y1, x2, y2 = box

        # Calculate box width and height
        width = x2 - x1
        height = y2 - y1

        # Define box color based on label
        color = 'r' if label == 1 else 'g'  # Assuming label 1 is for 'person' and label 2 is for 'car'

        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor=color, facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add label and score as text
        ax.text(x1, y1, f'{label}: {score:.2f}', bbox=dict(facecolor='white', alpha=0.5))

    # Show the plot
    plt.show()

# Call the function to draw bounding boxes on the image
draw_boxes(image, prediction)

