import os
import torch
from PIL import Image
from ultralytics import YOLO
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.db import models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gdown
import json
import uuid
import requests
import json
import os
import uuid
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import os
import requests
from PIL import Image
import tempfile



def download_image(image_url, current_directory):
    # Ensure the target folder exists
    target_folder = os.path.join(current_directory, "images/")
    os.makedirs(target_folder, exist_ok=True)
    
    try:
        # Extracting image name from URL
        # print("1")
        image_name = image_url.split("/")[-1]
        # Creating full path for image
        image_path = os.path.join(target_folder, image_name)
        # print("2")
        # Downloading the image
        response = requests.get(image_url)
        response.raise_for_status()
        # print("3")
        # Saving the image to the specified directory
        with open(image_path, 'wb') as f:
            f.write(response.content)
            
        # print(f"Image downloaded to {image_path}")
        return image_path  # Return the path to the downloaded image

    except requests.RequestException as e:
        print(f"Error downloading image: {e}")  # Print error if occurred
        return None


def download_model(file_id, model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)

def get_or_download_model(model_folder, model_name):
    model_path = os.path.join(model_folder, model_name)
    # print("get_or_download_model")
    if not os.path.exists(model_path):
        if model_name == "yolov8m.pt":
            # print("download_model yolov8m")
            download_model("1HxMa9AYU6LRQYm7-To_Aav0AO7IbU0Q3", model_path)
        elif model_name == "yolov8n.pt":
            # print("download_model yolov8n")
            download_model("1KwP_Uzsb5E5IrS6eIAlMIdMwsA96T2NU", model_path)
    return YOLO(model_path)

def analyze_image(image_path, session_id=None):
    # Load models
    # print("Load models")
    # Get the directory of the current file
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the models folder
    model_folder = os.path.join(current_directory, "models/")

    model_m = get_or_download_model(model_folder, "yolov8m.pt")
    model_m.conf = 0.4  # Set the confidence threshold for the YOLOv8m model to 0.4
    # print("Load models yolov8m")

    model_n = get_or_download_model(model_folder, "yolov8n.pt")
    model_n.conf = 0.4  # Set the confidence threshold for the YOLOv8n model to 0.4
    # print("Load models yolov8n")

    # Load and convert image
    try:
        # print("Step 1: Download the image")
        image_path = download_image(image_path, current_directory)  # This should return the path to the downloaded image
        # print("Downloaded image path:", image_path)
        
        # print("Step 2: Read the image data as bytes")
        with open(image_path, 'rb') as file:
            image_data = file.read()  # Read image data in bytes
        
        # print("Step 3: Create a temporary file and write the image data to it")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image_data)  # Write the byte data to the file
            tmp_filename = tmp.name

        # Step 4: Load and convert the image from the temporary file
        img = Image.open(tmp_filename).convert('RGB')
    except Exception as e:
        print(f"Failed to analyze image at step: {e}")
        # Additional debugging information
        print(f"Data type of image_data: {type(image_data)}")
        return {"error": f"Failed to analyze image: {e}"}

    # Analyze image with both models
    try:
        # print("Analyzing image with YOLOv8m")
        results_m = model_m([img])
        # print("Analyzing image with YOLOv8n")
        results_n = model_n([img])
    except Exception as e:
        print(f"Failed to analyze image with models: {e}")
        return {"error": f"Failed to analyze image with models: {e}"}

    # Extract bounding boxes and labels
    # print("Extract bounding boxes and labels")
    boxes_m = results_m[0].boxes.xyxy
    boxes_n = results_n[0].boxes.xyxy

    labels_m = torch.tensor([0, 1] * len(boxes_m))
    labels_n = torch.tensor([2, 3] * len(boxes_n))

    # Combine results from both models
    # print("Extract bounding boxes and labels")
    all_boxes = torch.cat((boxes_m, boxes_n), dim=0)
    all_labels = torch.cat((labels_m, labels_n), dim=0)

    # Print combined boxes and labels for debugging
    # print("Combined Boxes:", all_boxes)
    # print("Combined Labels:", all_labels)

    # Initialize lists for unique boxes and labels
    unique_boxes = []
    unique_labels = []

    # Filter out non-unique boxes based on IOU
    for i, (box1, label1) in enumerate(zip(all_boxes, all_labels)):
        unique = True
        for box2, label2 in zip(all_boxes[i+1:], all_labels[i+1:]):
            # Calculate IOU
            intersection_area = (torch.min(box1[2], box2[2]) - torch.max(box1[0], box2[0])).clamp(0) * (torch.min(box1[3], box2[3]) - torch.max(box1[1], box2[1])).clamp(0)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            iou = intersection_area / (box1_area + box2_area - intersection_area)
            
            # Check for high IOU
            if iou > 0.8:
                unique = False
                break
        
        # Append unique boxes and labels
        if unique:
            unique_boxes.append(box1)
            unique_labels.append(label1)

    # Print unique boxes and labels for debugging
    # print("Unique Boxes:", unique_boxes)
    # print("Unique Labels:", unique_labels)

    # Create and draw on figure
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    # Draw bounding boxes on the image
    for box, label in zip(unique_boxes, unique_labels):
        x_min, y_min, x_max, y_max = box[:4]
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min, f'Class {label}', color='red')

    # Save the processed image to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='jpeg')
    plt.close(fig)
    buffer.seek(0)

    # Upload the image to ImgBB and handle response
    imgbb_api_key = '3c20e134a78ba35b50dfe856ad04d7e1'  # Replace with actual ImgBB API key
    files = {'image': buffer}
    data = {'key': imgbb_api_key}
    imgbb_response = requests.post('https://api.imgbb.com/1/upload', files=files, data=data)

    if imgbb_response.status_code == 200:
        imgbb_data = imgbb_response.json()
        if imgbb_data.get('success'):
            image_url = imgbb_data['data']['url']
            # print("Uploaded Image URL:", image_url)  # Print the URL for debugging
        else:
            raise Exception('Image upload to ImgBB failed: ' + imgbb_data.get('error', {}).get('message', ''))
    else:
        raise Exception('Network response was not ok during image upload.')

    buffer.close()

    # Compile information about the image and detection results
    image_info = {'width': img.width, 'height': img.height}
    boxes_info = [{'coordinates': box.tolist(), 'label': label.item(), 'class': f'Class {label.item()}'} for box, label in zip(unique_boxes, unique_labels)]

    response_data = {
        'image_info': image_info,
        'boxes_info': boxes_info,
        'total_boxes': len(unique_boxes),
        'yolo_output': json.dumps(boxes_info, indent=4),
        'image_url': image_url  # Include the image URL in the response
    }

    # Print final response data for debugging
    # print("Response Data:", response_data)

    return response_data
