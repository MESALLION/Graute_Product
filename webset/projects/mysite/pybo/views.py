from django.http import HttpResponse
from django.http import JsonResponse
from .models import Detection
from django.views.decorators.csrf import csrf_exempt
import datetime


def index(request):
    return HttpResponse("안녕하세요 pybo에 오신것을 환영합니다.")

@csrf_exempt
def receive_detection(request):
    if request.method == "POST":
        # 파일과 데이터를 받습니다.
        image = request.FILES.get('image')
        description = request.POST.get('description', '')

        # 모델에 저장합니다.
        detection = Detection.objects.create(
            image=image,
            description=description
        )

        return JsonResponse({"success": True, "detected_at": detection.detected_at}, status=201)
    return JsonResponse({"success": False, "error": "Invalid request"}, status=400)