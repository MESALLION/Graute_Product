from django.urls import path
from . import views

app_name = 'pybo'

# url 부분
urlpatterns = [
    # 기본 pybo로 되어있고 뒤에 어떤게 붙냐에따라 url지정
    path('', views.main, name='main'),
    # 드론 감지 정보를 담는 URL
    path('detect', views.detection_list, name='detection_list'),
    # 드론 이미지, 최초의 발견 시간
    path('upload', views.upload, name='upload'),
    # 드론이 머문시간
    path('upload_time', views.upload_time, name='upload_time'),
    # 감지된 드론 하나만 보는 URL
    path('<int:detection_id>/', views.detection_detail, name="detection_detail")
]