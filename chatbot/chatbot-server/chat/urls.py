from django.urls import include, path
from rest_framework import routers
from .views import ConversationViewSet, MessageViewSet, PromptViewSet, EmbeddingDocumentViewSet, SettingViewSet
from .image_predict import upload_image

router = routers.SimpleRouter()
router.register(r'conversations', ConversationViewSet, basename='conversationModel')
router.register(r'messages', MessageViewSet, basename='messageModel')
router.register(r'prompts', PromptViewSet, basename='promptModel')
router.register(r'embedding_document', EmbeddingDocumentViewSet, basename='embeddingDocumentModel')
router.register(r'settings', SettingViewSet, basename='settingModel')

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('api/upload_image/', upload_image, name='upload_image'),
    response_data['result_image_url'] = default_storage.url(result_image_path),

]