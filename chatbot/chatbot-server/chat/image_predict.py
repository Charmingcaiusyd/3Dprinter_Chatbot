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
from .models import ChatSession
import uuid

def download_model(file_id, model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)

def get_or_download_model(model_folder, model_name):
    model_path = os.path.join(model_folder, model_name)
    if not os.path.exists(model_path):
        if model_name == "yolov8m.pt":
            download_model("1HxMa9AYU6LRQYm7-To_Aav0AO7IbU0Q3", model_path)
        elif model_name == "yolov8n.pt":
            download_model("1KwP_Uzsb5E5IrS6eIAlMIdMwsA96T2NU", model_path)
    return YOLO(model_path)

def analyze_image(image, session_id):
    model_folder = "models/"
    model_m = get_or_download_model(model_folder, "yolov8m.pt")
    model_n = get_or_download_model(model_folder, "yolov8n.pt")

    img = Image.open(image).convert('RGB')

    results_m = model_m([img])
    results_n = model_n([img])

    results_m_obj = results_m[0]
    results_n_obj = results_n[0]

    boxes_m = results_m_obj.boxes.xyxy
    boxes_n = results_n_obj.boxes.xyxy

    labels_m = torch.tensor([0, 1] * len(boxes_m))
    labels_n = torch.tensor([2, 3] * len(boxes_n))

    all_boxes = torch.cat((boxes_m, boxes_n), dim=0)
    all_labels = torch.cat((labels_m, labels_n), dim=0)

    unique_boxes = []
    unique_labels = []

    for i, (box1, label1) in enumerate(zip(all_boxes, all_labels)):
        unique = True
        for box2, label2 in zip(all_boxes[i+1:], all_labels[i+1:]):
            intersection_area = (torch.min(box1[2], box2[2]) - torch.max(box1[0], box2[0])).clamp(0) * (torch.min(box1[3], box2[3]) - torch.max(box1[1], box2[1])).clamp(0)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            iou = intersection_area / (box1_area + box2_area - intersection_area)
            if iou > 0.8:
                unique = False
                break
        if unique:
            unique_boxes.append(box1)
            unique_labels.append(label1)

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    for box, label in zip(unique_boxes, unique_labels):
        x_min, y_min, x_max, y_max = box[:4]
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min, f'Class {label}', color='red')

    result_image_name = f'result_image_{uuid.uuid4()}.jpg'
    result_image_path = os.path.join('uploads', result_image_name)
    plt.savefig(result_image_path)
    plt.close(fig)

    image_info = {
        'width': img.width,
        'height': img.height
    }
    boxes_info = [{
        'coordinates': box.tolist(),
        'label': label.item(),
        'class': f'Class {label.item()}'
    } for box, label in zip(unique_boxes, unique_labels)]

    response_data = {
        'image_info': image_info,
        'boxes_info': boxes_info,
        'total_boxes': len(unique_boxes),
        'yolo_output': json.dumps(boxes_info, indent=4)
    }

    chat_session = ChatSession.objects.get(id=session_id)
    chat_session.yolo_output = json.dumps(response_data['boxes_info'])
    chat_session.save()

    return response_data, result_image_path

@csrf_exempt
def upload_image(request, session_id):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image = request.FILES['image']
            file_path = default_storage.save('uploads/' + image.name, image)
            file_url = default_storage.url(file_path)

            yolo_result, result_image_path = analyze_image(image, session_id)

            response_data = {
                'url': file_url,
                'result_image_url': default_storage.url(result_image_path),
                'yolo_output': yolo_result['yolo_output']
            }

            chat_session = ChatSession.objects.get(id=session_id)
            message_content = "I've analyzed the image. Here are the results:\n" + yolo_result['yolo_output']
            chat_session.messages.create(content=message_content, sender='bot', image=result_image_path)
            chat_session.save()

            return JsonResponse(response_data, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)
