# 날짜 통일을 위한 필터
from django import template

register = template.Library()

@register.filter
def date_filter(value):
    return value.strftime("%Y년 %m월 %d일 %H:%M %p")