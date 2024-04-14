from django.apps import AppConfig
from utils.chroma_db import DocumentQuery
from django.conf import settings


class ChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chat"

    def ready(self):

        global document_query
        document_query = DocumentQuery()
        document_query.initialize()
        document_query.query("init")
