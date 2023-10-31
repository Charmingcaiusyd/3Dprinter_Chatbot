import os
import torch
from PIL import Image
from ultralytics import YOLO
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import urllib.request
import gdown
import json



def download_model(file_id, model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)


@csrf_exempt
def upload_image(request):
    response_data = {}
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # 上传图片
            image = request.FILES['image']
            file_path = default_storage.save('uploads/' + image.name, image)
            file_url = default_storage.url(file_path)

            model_folder = "models/"
            model_name_m = "yolov8m.pt"
            model_name_n = "yolov8n.pt"
            model_path_m = os.path.join(model_folder, model_name_m)
            model_path_n = os.path.join(model_folder, model_name_n)

            # 检查模型是否存在并下载（如果需要）
            if not os.path.exists(model_path_m):
                download_model("1HxMa9AYU6LRQYm7-To_Aav0AO7IbU0Q3", model_path_m)
            if not os.path.exists(model_path_n):
                download_model("1KwP_Uzsb5E5IrS6eIAlMIdMwsA96T2NU", model_path_n)

            # 加载模型
            model_m = YOLO(model_path_m)
            model_n = YOLO(model_path_n)

            # 从上传的图像文件中创建一个PIL图像对象
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

            # 用于存储非重复的边界框
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
                if label in [0, 1, 2, 3, 4]:
                    rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    plt.text(x_min, y_min, f'Class {label}', color='red')

            result_image_path = 'result_image.jpg'
            plt.savefig(result_image_path)
            plt.close(fig)


            # Add the following lines to collect information and create a response
            image_info = {
                'width': img.width,
                'height': img.height
            }
            boxes_info = [{
                'coordinates': box.tolist(),
                'label': label.item(),
                'class': 'Class {}'.format(label.item())
            } for box, label in zip(unique_boxes, unique_labels)]

            response_data.update({
                'url': file_url,
                'result_image_url': default_storage.url(result_image_path),
                'image_info': image_info,
                'boxes_info': boxes_info,
                'total_boxes': len(unique_boxes)
            })

            # Convert boxes_info to string as well
            response_data['boxes_info_str'] = json.dumps(boxes_info, indent=4)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse(response_data, status=200)

    return JsonResponse({'error': 'Invalid request'}, status=400)