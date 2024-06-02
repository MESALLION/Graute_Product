from django.urls import path
from . import views

app_name = 'pybo'

# url 부분
urlpatterns = [
    # 기본 pybo로 되어있고 뒤에 어떤게 붙냐에따라 url지정
    path('', views.main, name='main'),
    # 게시판 형태로 보여주기
    path('notice', views.detection_notice, name='detection_notice'),
    # 드론 감지 정보를 담는 URL
    path('detect', views.detection_list, name='detection_list'),
    # 드론 이미지, 최초의 발견 시간
    path('upload', views.upload, name='upload'),
    # 드론이 머문시간
    path('upload_time', views.upload_time, name='upload_time'),
    # 감지된 드론 하나만 보는 URL
    path('<int:detection_id>/', views.detection_detail, name="detection_detail"),
    # 달력처럼 보이는 URL
    path('calendar', views.detection_calendar, name="detection_calendar"),
    # 달력처럼 보이는 URL
    path('calendar/<int:year>/<int:month>', views.detection_calendar, name="detection_calendar"),
    # 달력에서 누르면 보이는 URL
    path('detections/<int:year>/<int:month>/<int:day>/', views.detection_day_detail, name='detection_day_detail'),
    # 엑셀 다운로드 URL
    path('export/<str:date>/', views.export_to_excel, name='export_to_excel')
]