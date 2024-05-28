from django import template
import re

register = template.Library()

@register.filter
def format_image_name(value):
    file_name = value.name.split('/')[-1]  # '현재_위치_1.jpg'
    # 정규 표현식으로 날짜와 시간 부분을 추출 및 포맷팅
    match = re.match(r"(.+)_(\d+)", file_name)
    if match:
        # 그룹 추출
        location = match.group(1).replace('_', ' ')
        number = match.group(2)
        # 새로운 형식으로 문자열 조합
        formatted_name = f"{location} {number}"
        return formatted_name
    return file_name  # 정규 표현식 매치 실패 시 원래 파일 이름 반환