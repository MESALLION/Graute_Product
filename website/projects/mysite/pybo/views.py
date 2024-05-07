from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Detection
from datetime import datetime
from django.core.paginator import Paginator  


# 메인 페이지
def main(request):
    # 페이지
    page = request.GET.get('page', '1')
    # 모델의 디텍을 다 가져오는 변수
    detections = Detection.objects.order_by('-detection_time')
    # 페이지당 10개씩 보여주기
    paginator = Paginator(detections, 10)
    page_obj = paginator.get_page(page)
    detections = {'detections': page_obj}
    return render(request, 'detection/main.html',  detections)

# 저장된 모든 감지 정보를 보여주는 뷰
def detection_list(request):
    # 모델의 디텍 오브젝트를 다 가져오는 변수
    detections = Detection.objects.order_by('-detection_time')
    # 위에서 가져온 변수를 html문서에 변수로 다시 전달해서 html에서 참조가 가능하게 리턴
    return render(request, 'detection/list.html', {'detections': detections})

# 저장된 감지 정보 중 하나만 보여주는 뷰
def detection_detail(request, detection_id):
    detection = get_object_or_404(Detection, pk=detection_id)
    detection_one = {'detection': detection}
    return render(request, 'detection/detection_detail.html', detection_one)

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
        # 머문 시간까지 추가
        last_detection = Detection.objects.latest('id')
        # 머문 시간까지 추가
        last_detection.elapsed_time = float(elapsed_time)
        # 머문 시간 세이브
        last_detection.save()
        # 리턴
        return JsonResponse({'status': 'success', 'message': 'Elapsed time updated.'})
    # 리턴
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)