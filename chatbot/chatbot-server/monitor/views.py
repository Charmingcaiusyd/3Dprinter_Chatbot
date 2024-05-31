# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

COUNTRY_CODES = {
    'AU': '61',
    'US': '1',
    'CN': '86',
}

SELECTED_COUNTRY_CODE = 'AU'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.utils.timezone import now
import logging
from .models import Detection, MonitorEvent, MonitorSetting, EmergencyContact, Box, SystemResourceUsage
from django.shortcuts import render
from ultralytics import YOLO
from django.http import JsonResponse
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import os
from PIL import Image
import tempfile
import requests
from collections import Counter
from datetime import datetime, timedelta
from .models import MonitorEvent, EmergencyContact
import time
from apscheduler.schedulers.background import BackgroundScheduler
from django.db import transaction, OperationalError
from django.conf import settings
from twilio.rest import Client
from django.utils import timezone
import psutil
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import redirect
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.utils.html import strip_tags
from django.core.mail import send_mail
from django.core.mail import EmailMessage
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from dkim import DKIM, sign
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# Set up basic configuration for logging
logger = logging.getLogger(__name__)

# System resource consumption LOG
def log_system_resource_usage(user):
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent

    SystemResourceUsage.objects.create(
        user=user,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage
    )

def download_model(file_id, model_path):
    url = f'https://drive.google.com/uc?id={file_id}&export=download'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download model with error code {response.status_code}")


def get_or_download_model(model_folder, model_name):
    model_path = os.path.join(model_folder, model_name)

    if not os.path.exists(model_path):
        if model_name == "yolov8m.pt":
            download_model("1HxMa9AYU6LRQYm7-To_Aav0AO7IbU0Q3", model_path)
        elif model_name == "yolov8n.pt":
            download_model("1KwP_Uzsb5E5IrS6eIAlMIdMwsA96T2NU", model_path)
    return YOLO(model_path)


# Get real-time data view
@login_required
@require_http_methods(["GET"])
def get_realtime_data(request):
    try:
        # Fetch the last detection for the user
        latest_detection = Detection.objects.filter(user=request.user).last()
        
        # Fetch the last detection for processing time
        processing_time = latest_detection.processing_time

        # Fetch the current the system usage
        log_system_resource_usage(request.user)

        # Initialize the variables
        object_count = 0
        total_area = 0
        total_confidence = 0 
        confidence_count = 0 
        class_counts = Counter()

        # Calculate time 10 minutes ago
        ten_minutes_ago = timezone.now() - timedelta(minutes=10)

        # Fetch all detections in the last 10 minutes for the user
        recent_detections = Detection.objects.filter(user=request.user, created_at__gte=ten_minutes_ago)

        # Calculate the total area of all boxes and class counts from recent detections
        for detection in recent_detections:
            boxes = Box.objects.filter(detection=detection)
            for box in boxes:
                # Calculate the area of the box
                box_area = (box.x_max - box.x_min) * (box.y_max - box.y_min)
                total_area += box_area

                # Normalize class name to ensure consistency (convert all to int)
                try:
                    class_number = int(float(box.class_name.split(' ')[-1]))
                    normalized_class_name = f'Class {class_number}'
                except ValueError:
                    normalized_class_name = box.class_name  # In case of non-numeric class name

                # Count the class occurrences
                class_counts[normalized_class_name] += 1

                # Accumulate confidence and count
                if box.confidence is not None:
                    total_confidence += box.confidence
                    confidence_count += 1

        # Calculate the average confidence level
        average_confidence = total_confidence / confidence_count if confidence_count > 0 else None

        # Prepare the data for the most recent detection
        if latest_detection:
            box_data = json.loads(latest_detection.yolo_output)
            object_count = len(box_data)

        # Fetch the most recent system resource usage for the user
        latest_resource_usage = SystemResourceUsage.objects.filter(user=request.user).order_by('-timestamp').first()

        if latest_resource_usage:
            resource_data = {
                'cpu_usage': latest_resource_usage.cpu_usage,
                'memory_usage': latest_resource_usage.memory_usage,
                'timestamp': latest_resource_usage.timestamp,
            }
        else:
            resource_data = {
                'cpu_usage': None,
                'memory_usage': None,
                'timestamp': None,
            }

        # Prepare the response data
        data = {
            'time': now().strftime('%H:%M:%S'),
            'objectCount': object_count,
            'totalArea': total_area,
            'classCounts': dict(class_counts),
            'system_resources': resource_data,
            'processing_time': processing_time,
            'average_confidence': average_confidence,  # Adding average confidence to response data
        }
        # print(data)
        return JsonResponse(data)

    except Exception as e:
        # logger.error('Error fetching real-time data: %s', e)
        return JsonResponse({'error': 'Error retrieving real-time data'}, status=500)


def download_image(image_url, current_directory):
    # Ensure the target folder exists
    target_folder = os.path.join(current_directory, "images/")
    os.makedirs(target_folder, exist_ok=True)
    
    try:
        # Extracting image name from URL
        image_name = image_url.split("/")[-1]
        # Creating full path for image
        image_path = os.path.join(target_folder, image_name)
        # Downloading the image
        response = requests.get(image_url)
        response.raise_for_status()
        # Saving the image to the specified directory
        with open(image_path, 'wb') as f:
            f.write(response.content)
            
        return image_path  # Return the path to the downloaded image

    except requests.RequestException as e:
        # print(f"Error downloading image: {e}")  # Print error if occurred
        return None



# 处理视频帧的视图
@login_required
@require_http_methods(["POST"])
def process_frame(request):
    try:
        start_time = time.time()  # Record start time

        image_url = request.POST.get('image_url')
        if not image_url:
            # print("No image URL provided")
            return JsonResponse({'status': 'error', 'message': 'No image URL provided'}, status=400)

        # Attempt to download the image from imgbb
        try:
            current_directory = os.path.dirname(os.path.abspath(__file__))

            # print("Step 1: Download the image")
            image_path = download_image(image_url, current_directory)  # This should return the path to the downloaded image
            
            # print("Step 2:  the image data as bytes")
            with open(image_path, 'rb') as file:
                image_data = file.read()  # Read image data in bytes
            
            # print("Step 3: Create a temporary file and write the image data to it")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(image_data)  # Write the byte data to the file
                tmp_filename = tmp.name

            # Step 4: Load and convert the image from the temporary file
            img = Image.open(tmp_filename).convert('RGB')

            # print("Image downloaded successfully from imgbb.")

            # Step 5: Retrieve the image_size from the user's settings
            image_size_setting = MonitorSetting.objects.filter(user=request.user, key='image_size').first()
            max_dimension = int(image_size_setting.value) if image_size_setting else 640  # Default to 640 if not set

            # Calculate the scale factor, maintaining the aspect ratio
            original_width, original_height = img.size
            scale_factor = min(max_dimension / original_width, max_dimension / original_height)

            # Calculate the new size, maintaining the aspect ratio
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # print("Image resized successfully while maintaining aspect ratio.")

        except requests.RequestException as e:
            # print(f"Error downloading image: {e}")  # Print error if occurred
            return JsonResponse({'status': 'error', 'message': 'Error downloading image'}, status=500)

        try:
            # Get YOLO parameter settings
            user = request.user
            confidence_setting = MonitorSetting.objects.filter(user=user, key='confidence_threshold').first()
            iou_setting = MonitorSetting.objects.filter(user=user, key='iou_threshold').first()
            class_prob_threshold = MonitorSetting.objects.filter(user=user, key='iou_threshold').first()
            max_detections = float(MonitorSetting.objects.filter(user=user, key='max_detections').first().value)
            min_layer_size = float(MonitorSetting.objects.filter(user=user, key='min_layer_size').first().value)
            nms_threshold = MonitorSetting.objects.filter(user=user, key='nms_threshold').first().value or 0.4  # Default value if not set
            image_size = MonitorSetting.objects.filter(user=user, key='image_size').first().value or 640  # Default value if not set

            # If the settings exist, apply them, otherwise use the defaults
            confidence_threshold = float(confidence_setting.value) if confidence_setting else 0.5
            iou_threshold = float(iou_setting.value) if iou_setting else 0.5

            # print("Get YOLO parameter settings.")
        except Exception as e:
            # print('Error during Get YOLO parameter settings: %s', e)
            return JsonResponse({'status': 'error', 'message': 'Error during Get YOLO parameter settings'}, status=500)

        try:

            # Construct the path to the models folder
            model_folder = os.path.join(current_directory, "models/")

            model_m = get_or_download_model(model_folder, "yolov8m.pt")
            # print("Load models yolov8m")
            model_n = get_or_download_model(model_folder, "yolov8n.pt")
            # print("Load models yolov8n")

            # print("Process the image with both models.")

        except Exception as e:
            # print('Error during Process the image with both models: %s', e)
            return JsonResponse({'status': 'error', 'message': 'Error during Process the image with both models'}, status=500)

        # Analyze image with both models
        try:
            # print("Analyzing image with YOLOv8m")
            results_m = model_m([img])
            # print("Analyzing image with YOLOv8n")
            results_n = model_n([img])
        except Exception as e:
            # print(f"Failed to analyze image with models: {e}")
            return {"error": f"Failed to analyze image with models: {e}"}

        try:

            # 提取边界框和置信度
            boxes_m = results_m[0].boxes.xyxy
            boxes_n = results_n[0].boxes.xyxy
            scores_m = results_m[0].boxes.conf  # Use the conf attribute to get a confidence level
            scores_n = results_n[0].boxes.conf  # Use the conf attribute to get a confidence level
            
            labels_m = torch.tensor([0, 1] * len(boxes_m))
            labels_n = torch.tensor([2, 3] * len(boxes_n))


            # Combine results from both models
            # print("Extract bounding boxes and labels")
            all_boxes = torch.cat((boxes_m, boxes_n), dim=0)
            all_labels = torch.cat((labels_m, labels_n), dim=0)
            all_scores = torch.cat((scores_m, scores_n), dim=0)

            # Calculate the average confidence level
            average_confidence = all_scores.mean().item() if len(all_scores) > 0 else 0

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
                    if iou > iou_threshold:
                        unique = False
                        break
                
                # Append unique boxes and labels
                if unique:
                    unique_boxes.append(box1)
                    unique_labels.append(label1)

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
                'image_url': image_url,  # Include the image URL in the response
                'average_confidence': average_confidence,  # Adding Average Confidence to Response Data
            }

            # print('Boxes the image.')
            # print(response_data)

        except Exception as e:
            # print('Error during Boxes the image: %s', e)
            return JsonResponse({'status': 'error', 'message': 'Error during Boxes the image'}, status=500)

        now_ = datetime.now()

        end_time = time.time()  # Record end time
        calculated_processing_time = end_time - start_time  # Calculate processing time

        # Creating a Detection Instance
        detection_instance = Detection.objects.create(
            id=int(now_.strftime('%Y%m%d%H%M%S')),
            user=request.user,
            image_width=image_info['width'],
            image_height=image_info['height'],
            total_boxes=len(unique_boxes),
            yolo_output=json.dumps(boxes_info),
            image_url=image_url,
            processing_time=calculated_processing_time,  
        )

        # Create Box instances for each unique_box
        for box_info in boxes_info:
            coordinates = box_info['coordinates']
            Box.objects.create(
                detection=detection_instance,
                label=box_info['label'],
                class_name=box_info['class'],
                x_min=coordinates[0],
                y_min=coordinates[1],
                x_max=coordinates[2],
                y_max=coordinates[3],
                confidence=average_confidence,  # Include the confidence value on average
            )

        current_layer_sizes = [(box[2] - box[0]) * (box[3] - box[1]) for box in unique_boxes]
        num_detections = len(unique_boxes)

        # Initialize a list to collect labels of boxes that meet the condition
        labels_for_monitoring = []

        # Sequential comparison of each size in current_layer_sizes with min_layer_size
        for size, box_info in zip(current_layer_sizes, boxes_info):
            if (size > min_layer_size) or (num_detections > max_detections):
                labels_for_monitoring.append(box_info['label'])

        # Count occurrences of each label using Counter
        label_counts = Counter(labels_for_monitoring)

        # Create MonitorEvent instances for each label
        for label, count in label_counts.items():
            MonitorEvent.objects.create(
                user=request.user,
                label=label,
                timestamp=datetime.now(),  # Assuming the event's time is the current time
                count=count,
                image_url=image_url  # Constructing a URL to the detection image
            )
        return JsonResponse(response_data)

    except Exception as e:
        # print('An error occurred: %s', e)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)



@login_required
def yolo_params(request):
    if request.method == 'POST':
        try:
            # Extract all parameters from POST request
            confidence_threshold = request.POST.get('confidenceThreshold')
            iou_threshold = request.POST.get('iouThreshold')
            max_detections = request.POST.get('maxDetections')
            min_layer_size = request.POST.get('minLayerSize')
            class_prob_threshold = request.POST.get('classProbThreshold')
            nms_threshold = request.POST.get('nmsThreshold')
            image_size = request.POST.get('imageSize')


            # Store or update YOLO parameters in MonitorSetting
            MonitorSetting.objects.update_or_create(
                user=request.user, key='confidence_threshold',
                defaults={'value': confidence_threshold}
            )
            MonitorSetting.objects.update_or_create(
                user=request.user, key='iou_threshold',
                defaults={'value': iou_threshold}
            )
            MonitorSetting.objects.update_or_create(
                user=request.user, key='max_detections',
                defaults={'value': max_detections}
            )
            MonitorSetting.objects.update_or_create(
                user=request.user, key='min_layer_size',
                defaults={'value': min_layer_size}
            )
            MonitorSetting.objects.update_or_create(
                user=request.user, key='class_prob_threshold',
                defaults={'value': class_prob_threshold}
            )
            MonitorSetting.objects.update_or_create(
                user=request.user, key='nms_threshold',
                defaults={'value': nms_threshold}
            )
            MonitorSetting.objects.update_or_create(
                user=request.user, key='image_size',
                defaults={'value': image_size}
            )
            # Return the new settings to the frontend for confirmation
            response_data = {
                'confidenceThreshold': confidence_threshold,
                'iouThreshold': iou_threshold,
                'maxDetections': max_detections,
                'minLayerSize': min_layer_size,
                'classProbThreshold':class_prob_threshold,
                'nmsThreshold': nms_threshold,
                'imageSize': image_size,
            }

            return JsonResponse({'status': 'success', 'message': 'YOLO parameters updated successfully.', 'data': response_data})

        except Exception as e:
            # Return error message on exception
            return JsonResponse({'status': 'error', 'message': 'Failed to update YOLO parameters.', 'error': str(e)}, status=500)

    else:
        # If the request method is not POST, return a method not allowed error
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


@login_required
def alert_settings(request):
    if request.method == 'POST':
        try:
            # Extracting emergency contact information from POST requests
            contact_name = request.POST.get('name')
            contact_email = request.POST.get('email')
            contact_phone = request.POST.get('phone')

            # Store or update emergency contact information to EmergencyContact
            emergency_contact, created = EmergencyContact.objects.get_or_create(
                associated_user=request.user,
                defaults={
                    'contact_name': contact_name,
                    'contact_phone': contact_phone,
                    'contact_email': contact_email
                }
            )

            if not created:
                # If the record already exists, update the information
                emergency_contact.contact_name = contact_name
                emergency_contact.contact_phone = contact_phone
                emergency_contact.contact_email = contact_email
                emergency_contact.save()

            # Return saved contact information to the front-end to confirm the update
            response_data = {
                'contact_name': emergency_contact.contact_name,
                'contact_email': emergency_contact.contact_email,
                'contact_phone': emergency_contact.contact_phone
            }

            return JsonResponse({'status': 'success', 'message': 'Emergency contact updated successfully.', 'data': response_data})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': 'Failed to update emergency contact.', 'error': str(e)}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)



@login_required
def initial_values(request):
    try:
        # Fetch YOLO parameters for the logged-in user
        yolo_params = MonitorSetting.objects.filter(user=request.user)
        yolo_params_dict = {param.key: param.value for param in yolo_params}

        # Fetch emergency contact for the logged-in user
        emergency_contact = EmergencyContact.objects.filter(associated_user=request.user).first()
        
        # If an emergency contact exists, prepare its data; otherwise, use defaults
        if emergency_contact:
            contact_info = {
                'contact_name': emergency_contact.contact_name,
                'contact_email': emergency_contact.contact_email,
                'contact_phone': emergency_contact.contact_phone
            }
        else:
            contact_info = {
                'contact_name': '',
                'contact_email': '',
                'contact_phone': ''
            }

        # Compile response data
        response_data = {
            'yolo_params': yolo_params_dict,
            'emergency_contact': contact_info,
        }

        # print(response_data)
        return JsonResponse(response_data)

    except Exception as e:
        # Log the error and return an appropriate error response
        logger.error('Error fetching initial values: %s', e)
        return JsonResponse({'error': str(e)}, status=500)



def get_or_download_private_key(base_dir):
    private_key_path = os.path.join(base_dir, "private.key")
    # print(f"Checking for private key at: {private_key_path}")
    if not os.path.exists(private_key_path):
        # print(f"Private key not found. Downloading from Google Drive...")
        download_file("1AvpmhRV0Rw1B8xRN-VPSoB-nMTYQiUbL", private_key_path)
        # print(f"Private key downloaded successfully to: {private_key_path}")
    # else:
        # print(f"Private key already exists at: {private_key_path}")
    return private_key_path



def send_email(subject, message_body, to_email, image_url, contact_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = get_or_download_private_key(base_dir)
    html_message = render_to_string('email.html', {
        'contact_name': contact_name,
        'message_body': message_body,
        'image_url': image_url
    })
    
    
    try:
        # print(f"Trying to send email...")
        # print(f"Private key loaded from {key_path}")
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Private key file not found at {key_path}")
        with open(key_path, 'r') as f:
            private_key = f.read()
        # print(f"Private key content:")
        # print(private_key)
        # print(f"Type of private_key before encoding: {type(private_key)}")

        private_key = private_key.encode('utf-8')  # Convert private_key to bytes
        # print(f"Type of private_key after encoding: {type(private_key)}")

        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = 'office@3dprinter.quest'
        msg['To'] = to_email
        msg['Reply-To'] = 'office@3dprinter.quest'
        msg.attach(MIMEText(html_message, 'html'))
        msg_string = msg.as_string()

        signature = sign(
            message=msg_string.encode('utf-8'),
            selector=b'default',
            domain=b'3dprinter.quest',
            privkey=private_key,
            include_headers=[b'From', b'To', b'Subject']
        )
        # print(f"Email signed with DKIM: {signature}")
        
        # print(f"Type of signature: {type(signature)}")
    
        msg_signed = signature.decode('utf-8').replace('\r\n', '') # Replace newline characters with spaces in the signature

        # print(f"Signed message: {msg_signed}")
        
        # Create EmailMessage Object
        email = EmailMessage()
        email.subject = subject  # Set the subject using the subject attribute
        email.from_email = 'office@3dprinter.quest'  # Set the from email using the from_email attribute
        email.to = [to_email]  # Set the to email using the to attribute (as a list)
        email.reply_to = ['office@3dprinter.quest']  # Set the reply-to using the reply_to attribute (as a list)
                # Set the email body and attach the HTML content
        
        # Add the DKIM signature to the email headers
        email.extra_headers = {'DKIM-Signature': msg_signed}

        # Set the email body and attach the HTML content
        email.content_subtype = 'html'
        email.body = html_message

        # print(f"Email object created: {email}")
        email.send()
        # print(f"Email sent successfully to {to_email}")
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")

def format_phone_number(phone_number):
    # Remove any non-digit characters
    phone_number = ''.join(filter(str.isdigit, phone_number))

    # Get the selected country code and corresponding area code
    country_code = COUNTRY_CODES[SELECTED_COUNTRY_CODE]
    area_code = f"+{country_code}"

    # Check the length of the phone number
    if len(phone_number) == len(country_code) + 7:
        # Local number without area code and leading "0"
        formatted_number = f"{area_code} 0{phone_number[-9:]}"
    elif len(phone_number) == len(country_code) + 8:
        if phone_number.startswith("0"):
            # Local number with leading "0"
            formatted_number = f"{area_code} {phone_number[-10:]}"
        else:
            # Local number without area code
            formatted_number = f"{area_code} 0{phone_number[-9:]}"
    elif len(phone_number) == len(country_code) + 9:
        if phone_number.startswith(country_code):
            # International number with country code
            formatted_number = f"+{phone_number}"
        else:
            # Invalid phone number format
            raise ValueError(f"Invalid phone number format: {phone_number}")
    elif len(phone_number) == len(country_code) + 10 and phone_number.startswith(country_code):
        # International number with country code and leading "0"
        formatted_number = f"+{phone_number[:len(country_code)]} 0{phone_number[len(country_code):]}"
    else:
        # Invalid phone number format
        raise ValueError(f"Invalid phone number format: {phone_number}")

    return formatted_number



def send_sms(twilio_client, message_body, to_phone, contact_name, image_url):
    try:

        # Format the phone number
        to_phone = format_phone_number(to_phone)
        # print(f"to_phone: {to_phone}")
        # Construct a more informative and professional SMS message
        sms_message = f"Dear {contact_name},\n\n"
        sms_message += "We have detected an important event that requires your attention:\n\n"
        sms_message += f"{message_body}\n\n"
        sms_message += f"Please check the following image for more details: {image_url}\n\n"
        sms_message += "If you have any questions or concerns, please don't hesitate to contact us.\n\n"
        sms_message += "Best regards,\nThe 3D Printer Quest Team"

        message = twilio_client.messages.create(
            body=sms_message,
            from_='+18044804795',  # Your Twilio number
            to=to_phone
        )
        # print(f"SMS sent to {to_phone} with SID {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS to {to_phone}: {e}")

def check_and_notify():
    max_retries = 5  # Set a max number of retries
    retry_sleep = 1  # Time to sleep between retries in seconds

    class_mapping = {
        0: "spaghetti",
        1: "stringing", 
        2: "zits",
        3: "Under-Extrusion",
        4: "warping"
    }
    
    for attempt in range(max_retries):
        try:
            with transaction.atomic():
                # Your existing setup for Twilio client
                twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

                # Determine the time 10 minutes ago from the current time
                ten_minutes_ago = timezone.now() - timedelta(minutes=10)

                # Fetch recent events
                recent_events = MonitorEvent.objects.filter(timestamp__gte=ten_minutes_ago)

                # If there are no recent events, log and exit function
                if not recent_events.exists():
                    # print("No recent events found. Exiting.")
                    return

                # Log the number of events found
                # print(f"Found {len(recent_events)} events in the last 10 minutes.")

                # Process each event
                for event in recent_events:
                    try:
                        emergency_contact = EmergencyContact.objects.filter(associated_user=event.user).first()
                        if emergency_contact:

                            # Get the label name from class_mapping
                            label_name = class_mapping.get(event.label, "Unknown")

                            # Construct a more informative and specific message body
                            message_body = f"Event Details:\n"
                            message_body += f"- Event Type: {label_name}\n"  # Use label name here
                            message_body += f"- Number of Occurrences: {event.count}\n\n"

                            # Provide specific guidance based on the event type
                            if label_name == "spaghetti":
                                message_body += "Possible Causes:\n"
                                message_body += "- Poor bed adhesion\n"
                                message_body += "- Incorrect first layer height\n\n"
                                message_body += "Recommended Actions:\n"
                                message_body += "- Level and clean the print bed\n"
                                message_body += "- Adjust first layer settings in the slicer\n"
                                message_body += "- Use a brim or raft for better adhesion\n"
                            elif label_name == "stringing":
                                message_body += "Possible Causes:\n"
                                message_body += "- Insufficient retraction settings\n"
                                message_body += "- Print temperature too high\n\n"
                                message_body += "Recommended Actions:\n"
                                message_body += "- Increase retraction distance and speed\n"
                                message_body += "- Reduce print temperature slightly\n"
                                message_body += "- Enable combing mode in the slicer\n"
                            elif label_name == "zits":
                                message_body += "Possible Causes:\n"
                                message_body += "- Over-extrusion\n"
                                message_body += "- Inconsistent filament diameter\n"
                                message_body += "- Moisture in the filament\n\n"
                                message_body += "Recommended Actions:\n"
                                message_body += "- Calibrate the extruder steps/mm\n"
                                message_body += "- Check for and remove any filament debris in the nozzle\n"
                                message_body += "- Dry the filament if it has absorbed moisture\n"
                            elif label_name == "under-extrusion":
                                message_body += "Possible Causes:\n"
                                message_body += "- Incorrect filament diameter setting\n"
                                message_body += "- Clogged nozzle\n"
                                message_body += "- Extruder tension too low\n\n"
                                message_body += "Recommended Actions:\n"
                                message_body += "- Check and adjust the filament diameter in the slicer settings\n"
                                message_body += "- Clean the nozzle thoroughly\n"
                                message_body += "- Increase the extruder tension if adjustable\n"
                            elif label_name == "warping":
                                message_body += "Possible Causes:\n"
                                message_body += "- Insufficient bed adhesion\n"
                                message_body += "- Printing environment too cold\n"
                                message_body += "- High temperature difference between the bed and the environment\n\n"
                                message_body += "Recommended Actions:\n"
                                message_body += "- Use a heated bed and/or adhesive (e.g., glue stick, hair spray)\n"
                                message_body += "- Enclose the printer to maintain a consistent temperature\n"
                                message_body += "- Reduce the bed temperature and/or increase the ambient temperature\n"
                            else:
                                message_body += "This is an unknown event type. Please check the provided image for more information.\n"


                            # Log the notification info
                            # print(f"Sending notification to {emergency_contact.contact_name} ({emergency_contact.contact_email}, {emergency_contact.contact_phone})")

                            # Send notifications as configured
                            if emergency_contact.contact_phone:
                                send_sms(twilio_client, message_body, emergency_contact.contact_phone, emergency_contact.contact_name, event.image_url)
                            if emergency_contact.contact_email:
                                send_email("New Event Detected!", message_body, emergency_contact.contact_email, event.image_url, emergency_contact.contact_name)

                    except Exception as e:
                        print(f"Failed to process event {event.id}: {e}")

                # Once all events are processed, delete them
                MonitorEvent.objects.filter(timestamp__gte=ten_minutes_ago).delete()
                # print(f"All recent events have been processed and deleted.")
                break  # Break out of the loop if successful

        except OperationalError as e:
            if 'database is locked' in str(e) and attempt < max_retries - 1:
                # print(f"Database is locked, retrying {attempt + 1}/{max_retries}")
                time.sleep(retry_sleep)  # Sleep and retry
            else:
                raise  # Re-raise exception if not a known recoverable error or max retries reached

scheduler = BackgroundScheduler()
scheduler.add_job(check_and_notify, 'interval', minutes=1)  # Schedule the job every 5 minutes
scheduler.start()


def login_view(request):
    # Redirect already logged in users
    if request.user.is_authenticated:
        return redirect('/api/monitor/')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('/api/monitor/')
        else:
            # Return an 'invalid login' error message.
            return HttpResponse("Invalid username or password.", status=401)
    # For a GET request, just display the login form.
    return render(request, 'login.html')

def logout_view(request):
    # Log out the user.
    logout(request)
    # Redirect to the login page or home page.
    return redirect('https://3dprinter.quest/api/monitor/login/')
    
def check_user_authenticated(user):
    # Return True if the user is authenticated, otherwise False
    return user.is_authenticated

@login_required
@require_http_methods(["POST"])
def clean_data(request):
    try:
        # Get all settings for the user
        user_settings = MonitorSetting.objects.filter(user=request.user)

        # Iterate over each setting and set its value to the midpoint
        for setting in user_settings:
            if setting.key == 'confidence_threshold':
                setting.value = '0.5'  # Midpoint value for confidence threshold
            elif setting.key == 'iou_threshold':
                setting.value = '0.5'  # Midpoint value for IOU threshold
            elif setting.key == 'max_detections':
                setting.value = '50'  # Midpoint value for max detections, assuming range 1-100
            elif setting.key == 'min_layer_size':
                setting.value = '500'  # Midpoint value for min layer size, assuming range 1-1000
            elif setting.key == 'class_prob_threshold':
                setting.value = '0.5'  # Midpoint value for class probability threshold
            elif setting.key == 'nms_threshold':
                setting.value = '0.5'  # Midpoint value for NMS threshold
            elif setting.key == 'image_size':
                setting.value = '640'  # Midpoint value for image size, assuming range 256-1024
            setting.save()  # Save the modified setting

        # Clear all detection data for the user
        Detection.objects.filter(user=request.user).delete()
        Box.objects.filter(detection__user=request.user).delete()

        # Clear all event data for the user
        MonitorEvent.objects.filter(user=request.user).delete()

        # Clear all resource usage data for the user
        SystemResourceUsage.objects.filter(user=request.user).delete()

        return JsonResponse({'status': 'success', 'message': 'Data cleaned successfully'})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': 'Failed to clean data', 'error': str(e)}, status=500)

@login_required(login_url='https://3dprinter.quest/api/monitor/login/')
def index(request):
    # print("Request object:", request)
    # print("User:", request.user)
    # print("Is authenticated:", request.user.is_authenticated)
    return render(request, 'index.html')
