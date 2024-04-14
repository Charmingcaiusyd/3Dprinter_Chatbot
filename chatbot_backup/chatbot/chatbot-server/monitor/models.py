from django.db import models
from django.contrib.auth.models import User
from .validators import validate_image
from django.core.validators import MinValueValidator, MaxValueValidator

# 紧急联系人信息
class EmergencyContact(models.Model):
    # 与用户模型的外键关联
    associated_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='emergency_contacts')
    contact_name = models.CharField(max_length=255)  # 联系人姓名
    contact_email = models.EmailField()  # 联系人电子邮件
    contact_phone = models.CharField(max_length=20)  # 联系人电话号码
    created_at = models.DateTimeField(auto_now_add=True)  # 创建时间自动设置为当前时间
    updated_at = models.DateTimeField(auto_now=True)  # 更新时间自动设置为当前时间

    def __str__(self):
        # 对象转换为字符串时显示的内容
        return f"{self.contact_name} ({self.contact_email}, {self.contact_phone})"

# 监控事件信息
class MonitorEvent(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='monitor_events')
    label = models.CharField(max_length=255)  # 事件标签
    timestamp = models.DateTimeField()  # 事件发生的时间戳
    count = models.PositiveIntegerField(default=0)  # 该标签在指定时间的计数
    image_url = models.URLField(null=True, blank=True)  # 监控图片的URL地址
    created_at = models.DateTimeField(auto_now_add=True)  # 创建时间自动设置为当前时间
    updated_at = models.DateTimeField(auto_now=True)  # 更新时间自动设置为当前时间

    def __str__(self):
        # 对象转换为字符串时显示的内容
        return f"Event {self.label} at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

# 检测信息
class Detection(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='detections')
    image_width = models.IntegerField()  # 图像宽度
    image_height = models.IntegerField()  # 图像高度
    total_boxes = models.IntegerField()  # 总边界框数量
    yolo_output = models.JSONField()  # YOLO输出的JSON数据
    processing_time = models.FloatField(null=True, default=0.0)  # 添加默认处理时间
    image_url = models.URLField(null=True, blank=True)  # 替换原来的 image_file 字段
    created_at = models.DateTimeField(auto_now_add=True)  # 创建时间自动设置为当前时间
    updated_at = models.DateTimeField(auto_now=True)  # 更新时间自动设置为当前时间


class Box(models.Model):
    detection = models.ForeignKey(Detection, on_delete=models.CASCADE, related_name='boxes')
    label = models.CharField(max_length=50)  # 标签名称
    class_name = models.CharField(max_length=50)  # 类别名称
    x_min = models.FloatField()  # 边界框最小X坐标
    y_min = models.FloatField()  # 边界框最小Y坐标
    x_max = models.FloatField()  # 边界框最大X坐标
    y_max = models.FloatField()  # 边界框最大Y坐标
    confidence = models.FloatField(null=True, blank=True)  # 允许置信度为空
    created_at = models.DateTimeField(auto_now_add=True)  # 创建时间自动设置为当前时间
    updated_at = models.DateTimeField(auto_now=True)  # 更新时间自动设置为当前时间
    
# 监控设置信息
class MonitorSetting(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='monitor_settings')
    key = models.CharField(max_length=255)  # 设置项的键
    value = models.TextField()  # 设置项的值
    description = models.TextField(blank=True)  # 设置项的描述
    created_at = models.DateTimeField(auto_now_add=True)  # 创建时间自动设置为当前时间
    updated_at = models.DateTimeField(auto_now=True)  # 更新时间自动设置为当前时间

    def __str__(self):
        # 对象转换为字符串时显示的内容
        return f"Setting {self.key}"


class SystemResourceUsage(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    cpu_usage = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        null=True,  # 允许为空，表示数据可能不总是可用的
        blank=True
    )
    memory_usage = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        null=True,  # 允许为空，表示数据可能不总是可用的
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