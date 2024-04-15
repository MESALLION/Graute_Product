from django.db import models

# Create your models here.

class Detection(models.Model):
    image = models.ImageField(upload_to='detection/')
    detected_at = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True, null=True)