from django.urls import path

from . import views

urlpatterns = [
    path('', views.index),
    path('receive_detection/', views.receive_detection, name='receive_detection'),
]