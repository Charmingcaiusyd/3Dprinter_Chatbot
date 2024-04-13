from django.urls import path
from .views import yolo_params, alert_settings, process_frame, get_realtime_data, index, initial_values, login_view, logout_view

app_name = 'monitor'

urlpatterns = [
    path('', index, name='monitor-index'),  # 将/monitor路径映射到index视图
    path('login/', login_view, name='login_view'),
    path('logout/', logout_view, name='logout_view'),
    path('yolo_params/', yolo_params, name='yolo_params'),
    path('alert_settings/', alert_settings, name='alert_settings'),
    path('process_frame/', process_frame, name='process_frame'),
    path('initial_values/', initial_values, name='initial_values'),
    path('get_realtime_data/', get_realtime_data, name='get_realtime_data'),
]
