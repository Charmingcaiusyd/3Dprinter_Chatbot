from django.db import models
from django.contrib.auth.models import User
from .validators import validate_image
from django.core.validators import MinValueValidator, MaxValueValidator


# Emergency contact information
class EmergencyContact(models.Model):
    # Foreign key association with the User model
    associated_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='emergency_contacts')
    contact_name = models.CharField(max_length=255)  # Contact person's name
    contact_email = models.EmailField()  # Contact person's email
    contact_phone = models.CharField(max_length=20)  # Contact person's phone number
    created_at = models.DateTimeField(auto_now_add=True)  # Creation time automatically set to current time
    updated_at = models.DateTimeField(auto_now=True)  # Update time automatically set to current time

    def __str__(self):
        # Content displayed when the object is converted to a string
        return f"{self.contact_name} ({self.contact_email}, {self.contact_phone})"


# Monitoring event information
class MonitorEvent(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='monitor_events')
    label = models.CharField(max_length=255)  # Event label
    timestamp = models.DateTimeField()  # Timestamp of the event occurrence
    count = models.PositiveIntegerField(default=0)  # Count of the label at the specified time
    image_url = models.URLField(null=True, blank=True)  # URL address of the monitoring image
    created_at = models.DateTimeField(auto_now_add=True)  # Creation time automatically set to current time
    updated_at = models.DateTimeField(auto_now=True)  # Update time automatically set to current time

    def __str__(self):
        # Content displayed when the object is converted to a string
        return f"Event {self.label} at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"


# Detection information
class Detection(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='detections')
    image_width = models.IntegerField()  # Image width
    image_height = models.IntegerField()  # Image height
    total_boxes = models.IntegerField()  # Total number of bounding boxes
    yolo_output = models.JSONField()  # JSON data output by YOLO
    processing_time = models.FloatField(null=True, default=0.0)  # Add default processing time
    image_url = models.URLField(null=True, blank=True)  # Replace the original image_file field
    created_at = models.DateTimeField(auto_now_add=True)  # Creation time automatically set to current time
    updated_at = models.DateTimeField(auto_now=True)  # Update time automatically set to current time


class Box(models.Model):
    detection = models.ForeignKey(Detection, on_delete=models.CASCADE, related_name='boxes')
    label = models.CharField(max_length=50)  # Label name
    class_name = models.CharField(max_length=50)  # Class name
    x_min = models.FloatField()  # Minimum X coordinate of the bounding box
    y_min = models.FloatField()  # Minimum Y coordinate of the bounding box
    x_max = models.FloatField()  # Maximum X coordinate of the bounding box
    y_max = models.FloatField()  # Maximum Y coordinate of the bounding box
    confidence = models.FloatField(null=True, blank=True)  # Allow confidence to be null
    created_at = models.DateTimeField(auto_now_add=True)  # Creation time automatically set to current time
    updated_at = models.DateTimeField(auto_now=True)  # Update time automatically set to current time

    
# Monitoring settings information
class MonitorSetting(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='monitor_settings')
    key = models.CharField(max_length=255)  # Key of the setting item
    value = models.TextField()  # Value of the setting item
    description = models.TextField(blank=True)  # Description of the setting item
    created_at = models.DateTimeField(auto_now_add=True)  # Creation time automatically set to current time
    updated_at = models.DateTimeField(auto_now=True)  # Update time automatically set to current time

    def __str__(self):
        # Content displayed when the object is converted to a string
        return f"Setting {self.key}"


class SystemResourceUsage(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    cpu_usage = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        null=True,  # Allow null values, indicating that data may not always be available
        blank=True
    )
    memory_usage = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        null=True,  # Allow null values, indicating that data may not always be available
        blank=True
    )
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='resource_usages'
    )

    def __str__(self):
        return f"{self.user.username} - CPU: {self.cpu_usage}%, Memory: {self.memory_usage}%"

    class Meta:
        verbose_name = "System Resource Usage"
        verbose_name_plural = "System Resource Usages"