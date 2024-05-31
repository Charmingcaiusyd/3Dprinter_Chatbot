from django.contrib import admin
from .models import EmergencyContact, MonitorEvent, Detection, Box, MonitorSetting, SystemResourceUsage
from django.utils.html import format_html

# Register your models here.
@admin.register(EmergencyContact)
class EmergencyContactAdmin(admin.ModelAdmin):
    list_display = ('contact_name', 'contact_email', 'contact_phone', 'created_at', 'updated_at')
    search_fields = ('contact_name', 'contact_email')
    list_filter = ('created_at', 'updated_at')

@admin.register(MonitorEvent)
class MonitorEventAdmin(admin.ModelAdmin):
    list_display = ('user', 'label', 'timestamp', 'count', 'image_tag', 'created_at', 'updated_at')
    search_fields = ('label',)
    list_filter = ('timestamp',)
    
    def image_tag(self, obj):
        if obj.image_url:
            return format_html('<img src="{}" style="width: 45px; height:45px;" />', obj.image_url)
        return "-"
    image_tag.short_description = 'Image'

    
@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('user', 'image_width', 'image_height', 'image_tag', 'total_boxes', 'processing_time', 'created_at', 'updated_at')  
    search_fields = ('user__username') 
    list_filter = ('created_at', 'updated_at') 

    def image_tag(self, obj):
        if obj.image_url:
            return format_html('<img src="{}" style="width: 45px; height:45px;" />', obj.image_url)
        return "-"
    image_tag.short_description = 'Image'


@admin.register(Box)
class BoxAdmin(admin.ModelAdmin):
    list_display = ('detection', 'label', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'created_at', 'updated_at')  
    search_fields = ('label', 'class_name')  
    list_filter = ('created_at', 'updated_at')

@admin.register(MonitorSetting)
class MonitorSettingAdmin(admin.ModelAdmin):
    list_display = ('user', 'key', 'value', 'description', 'created_at', 'updated_at')
    search_fields = ('key', 'value')
    list_filter = ('created_at', 'updated_at')


@admin.register(SystemResourceUsage)
class SystemResourceUsageAdmin(admin.ModelAdmin):
    list_display = ('user', 'cpu_usage', 'memory_usage', 'timestamp')
    search_fields = ('user__username', 'cpu_usage', 'memory_usage')
    list_filter = ('timestamp',)

    def has_change_permission(self, request, obj=None):
        # You can adjust the permission settings as needed.
        return super().has_change_permission(request, obj=obj)

    def has_delete_permission(self, request, obj=None):
        # You can adjust the permission settings as needed.
        return super().has_delete_permission(request, obj=obj)