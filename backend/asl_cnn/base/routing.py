# base/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/predict/$', consumers.PredictConsumer.as_asgi()),
]