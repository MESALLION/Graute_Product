from django import template
from django.utils import timezone

register = template.Library()

@register.filter
def date_filter(value):
    local_time = timezone.localtime(value)  # UTC to local time
    return local_time.strftime("%Y년 %m월 %d일 %H:%M %p")