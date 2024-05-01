from django.urls import path
from .views import detection_list, upload, upload_time

# url 부분
urlpatterns = [
    # 기본 pybo로 되어있고 뒤에 어떤게 붙냐에따라 url지정
    path('', detection_list, name='detection_list'),
    path('upload', upload, name='upload'),
    path('upload_time', upload_time, name='upload_time'),
]