from django.urls import path
from .views import detection_list, upload, upload_time

urlpatterns = [
    path('', detection_list, name='detection_list'),
    path('upload', upload, name='upload'),
    path('upload_time', upload_time, name='upload_time'),
]