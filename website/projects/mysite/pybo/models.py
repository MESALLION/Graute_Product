from django.db import models


# 넘어오는 모델
class Detection(models.Model):
    # 이미지는 장고의 모델에 이미지 필드로 들어오는 데 업로드 들어오는 곳은 세팅의 media의 images로 추가.
    image = models.ImageField(upload_to='images/')
    # 최초의 발견 시간 장고 모델의 데이트 타임 필드로 추가가 된다.
    detection_time = models.DateTimeField()
    # 드론이 머문시간을 장고 모델의 초단위로 들어온다.
    elapsed_time = models.FloatField(null=True, blank=True)