from django import template

register = template.Library()

@register.filter
def make_range(value):
    return range(1, value + 1)  # 1부터 value까지의 범위
