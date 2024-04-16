from django.db import models


class Detection(models.Model):
    image = models.ImageField(upload_to='images/')
    detection_time = models.DateTimeField()
    elapsed_time = models.FloatField(null=True, blank=True)  # 드론이 머문 시간 (초)
