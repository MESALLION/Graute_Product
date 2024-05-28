from django import template
from django.utils import timezone

register = template.Library()

@register.filter
def date_filter(value):
    local_time = timezone.localtime(value)  # UTC to local time
    return local_time.strftime("%Y년 %m월 %d일 %H:%M %p")


@register.filter
def elapsed_filter(value):
    if value is None:
        return "측정불가"
    
    try:
        value = round(float(value), 1)  # 소수점 한 자리까지 반올림
        return f"{value:.1f}초"
    except (ValueError, TypeError):
        return "측정불가"