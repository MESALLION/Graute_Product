from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Detection
from datetime import datetime
# import 꼬이는 문제 해결
import datetime as dat
import calendar
from django.core.paginator import Paginator
import pandas as pd
import io


# 메인 페이지
def main(request):
    return render(request, 'detection/main.html')


# 게시판 뷰
def detection_notice(request):
    # 페이지
    page = request.GET.get('page', '1')
    # 모델의 디텍을 다 가져오는 변수
    detections = Detection.objects.order_by('-detection_time', '-id')
    # 페이지당 10개씩 보여주기
    paginator = Paginator(detections, 4)
    page_obj = paginator.get_page(page)
    detections = {'detections': page_obj}
    return render(request, 'detection/detection_notice.html',  detections)


# 저장된 모든 감지 정보를 보여주는 뷰
def detection_list(request):
    # 페이지
    page = request.GET.get('page', '1')
    # 모델의 디텍 오브젝트를 다 가져오는 변수
    detections = Detection.objects.order_by('-detection_time', '-id')
    paginator = Paginator(detections, 6)
    page_obj = paginator.get_page(page)
    # 위에서 가져온 변수를 html문서에 변수로 다시 전달해서 html에서 참조가 가능하게 리턴
    return render(request, 'detection/detection_all.html',  {'page_obj': page_obj})


# 저장된 감지 정보 중 하나만 보여주는 뷰
def detection_detail(request, detection_id):
    detection = get_object_or_404(Detection, pk=detection_id)
    detection_one = {'detection': detection}
    return render(request, 'detection/detection_detail.html', detection_one)


# 달력처럼 보여주는 뷰
def detection_calendar(request):
    today = dat.date.today()
    year = int(request.GET.get('year', today.year))
    month = int(request.GET.get('month', today.month))

    cal = calendar.Calendar(firstweekday=7)
    month_days = list(cal.itermonthdays4(year, month))  # 해당 월의 모든 날짜 가져오기

    # 각 날짜에 해당하는 감지 이벤트 수 집계
    detections_per_day = {}
    for day in month_days:
        if day[1] == month:  # 해당 월의 날짜인 경우만 처리
            date = dat.date(day[0], day[1], day[2])
            detections_count = Detection.objects.filter(detection_time__date=date).count()
            detections_per_day[date] = detections_count

    # 주차별로 분리
    weeks = []
    week = []
    for day in month_days:
        date = dat.date(day[0], day[1], day[2])
        if day[1] == month or day[2] != 0:  # 해당 월이거나, 빈 날짜가 아닐 경우
            week.append((date, detections_per_day.get(date, 0)))
        if len(week) == 7:
            weeks.append(week)
            week = []
    if week:  # 마지막 주 처리
        weeks.append(week)

    context = {
        'year': year,
        'month': month,
        'weeks': weeks
    }
    return render(request, 'detection/detection_calendar.html', context)


# 달력에서 누르면 보이는 뷰
def detection_day_detail(request, year, month, day):
    date = dat.date(year, month, day)
    detections = Detection.objects.filter(detection_time__date=date).order_by('-detection_time', '-id')
    page = request.GET.get('page', '1')
    paginator = Paginator(detections, 4)
    page_obj = paginator.get_page(page)
    
    context = {
        'detections': page_obj,
        'date': date,
    }
    
    return render(request, 'detection/detection_day_detail.html', context)


# 드론 감지 이미지와 시간을 받아 저장하는 뷰
@csrf_exempt  # CSRF 검증 비활성화
def upload(request):
    # 만약 POST방식으로 왔다면
    if request.method == 'POST':
        # 이미지는 request받은 파일 중에 이미지를 저장하는 변수
        image = request.FILES.get('image')
        # 마찬가지로 request받은 파일 중에 최초발견 시간을 저장하는 변수
        detection_time = datetime.strptime(request.POST.get('time'), '%Y-%m-%d %H:%M:%S')
        # 위의 변수들을 모델을 참조해서 통합 변수
        detection = Detection(image=image, detection_time=detection_time)
        # 위의 변수를 세이브
        detection.save()
        # 리턴
        return JsonResponse({'status': 'success', 'message': 'Detection data saved.'})
    # 리턴
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)


# 드론이 사라진 후 경과 시간을 받아 마지막 감지 레코드를 업데이트하는 뷰
@csrf_exempt
def upload_time(request):
    # 만약 POST방식으로 왔다면
    if request.method == 'POST':
        # 머문 시간
        elapsed_time = request.POST.get('elapsed_time')
        # 머문 시간추가할 곳 잡기
        last_detection = Detection.objects.latest('id')
        # 머문 시간까지 추가
        last_detection.elapsed_time = float(elapsed_time)
        # 머문 시간 세이브
        last_detection.save()
        # 리턴
        return JsonResponse({'status': 'success', 'message': 'Elapsed time updated.'})
    # 리턴
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)



def export_to_excel(request, date):
    # 문자열로 전달된 날짜를 datetime 객체로 변환
    try:
        current_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        return HttpResponse("Invalid date format. Please use 'YYYY-MM-DD' format.")

    # 해당 날짜에 해당하는 Detection 객체를 가져옵니다.
    detections = Detection.objects.filter(detection_time__date=current_date)

    # 데이터 프레임 변환을 위한 리스트 초기화
    data = []
    for detection in detections:
        # 이미지 이름 포맷팅
        image_name = detection.image.name.split('/')[-1].split('_')[0] + " " + detection.image.name.split('/')[-1].split('_')[1]
        # 머문 시간이 없는 경우 '측정불가'로 대체
        elapsed_time = detection.elapsed_time if detection.elapsed_time is not None else "측정불가"
        data.append({
            '번호': detection.id,
            '위치': image_name,
            '감지된 시간': current_date,
            '머문 시간': elapsed_time
        })

    # 데이터프레임 생성
    df = pd.DataFrame(data)

    # 데이터를 엑셀 파일로 변환
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Detections')

        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename=detections_{current_date}.xlsx'
        return response