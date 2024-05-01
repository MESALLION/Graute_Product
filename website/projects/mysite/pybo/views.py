from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Detection
from datetime import datetime

def detection_list(request):
    """저장된 모든 감지 정보를 보여주는 뷰"""
    detections = Detection.objects.all()
    return render(request, 'detection/list.html', {'detections': detections})

@csrf_exempt  # CSRF 검증 비활성화
def upload(request):
    """드론 감지 이미지와 시간을 받아 저장하는 뷰"""
    if request.method == 'POST':
        image = request.FILES.get('image')
        detection_time = datetime.strptime(request.POST.get('time'), '%Y-%m-%d %H:%M:%S')
        detection = Detection(image=image, detection_time=detection_time)
        detection.save()
        return JsonResponse({'status': 'success', 'message': 'Detection data saved.'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

@csrf_exempt
def upload_time(request):
    """드론이 사라진 후 경과 시간을 받아 마지막 감지 레코드를 업데이트하는 뷰"""
    if request.method == 'POST':
        elapsed_time = request.POST.get('elapsed_time')
        last_detection = Detection.objects.latest('id')
        last_detection.elapsed_time = float(elapsed_time)
        last_detection.save()
        return JsonResponse({'status': 'success', 'message': 'Elapsed time updated.'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)