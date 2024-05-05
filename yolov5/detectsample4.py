# 의존성 import
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch

# posixpath때문에
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 결과값을 웹브라우저 넘기기 위한 import
import requests
from datetime import datetime, timedelta

# 추가한 코드(딜레이용)
import time


# 파일 경로 설정
FILE = Path(__file__).resolve()
# YOLOv5 루트 디렉토리 설정
ROOT = FILE.parents[0]
# ROOT가 sys.path에 포함되어 있는지 확인하고, 포함되어 있지 않으면 추가합니다.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# 상대 경로로 ROOT 설정
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 욜로 코드를 import
# 욜로 코드의 세부 모듈들을 import
# ultralytics.utils.plotting 모듈에서 Annotator, colors, save_one_box 가져오기
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# utils.general 모듈에서 필요한 함수와 클래스들 가져오기
from utils.general import (
    LOGGER,                    # 로그 출력을 위한 LOGGER 객체
    Profile,                   # 성능 프로파일링을 위한 Profile 클래스
    check_file,                # 파일 존재 여부를 확인하는 함수
    check_img_size,            # 이미지 사이즈를 확인하는 함수
    check_imshow,              # 이미지 뷰어를 확인하는 함수
    check_requirements,        # 필요한 패키지가 설치되어 있는지 확인하는 함수
    colorstr,                  # 색상을 문자열에 적용하는 함수
    cv2,
    increment_path,            # 중복되지 않는 경로를 생성하는 함수
    non_max_suppression,       # 비최대 억제를 수행하는 함수
    print_args,                # 인자를 출력하는 함수
    scale_boxes,               # 박스 좌표를 이미지 크기에 맞게 스케일링하는 함수
    strip_optimizer,           # 옵티마이저 정보를 제거하는 함수
    xyxy2xywh,                 # (x1, y1, x2, y2) 형태의 박스 좌표를 (x_center, y_center, width, height) 형태로 변환하는 함수
)
# utils.torch_utils 모듈에서 select_device, smart_inference_mode 가져오기
from utils.torch_utils import select_device, smart_inference_mode

# 추가한 코드
# 전역 변수
DRONE_DETECTED = False
LAST_DETECTED_TIME = None

# 추가한 코드
def process_frame(frame, model, device):
    # 이미지를 모델 입력 크기로 조정
    # 모델 추론을 수행
    # 결과를 통해 프레임에 주석 달기
    frame = cv2.resize(frame, (640, 640))  # 예시 크기
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).to(device).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        preds = model(tensor)
    # 예시: 결과를 프레임에 그리기
    # 이 부분은 탐지된 객체에 따라 주석을 달기 위해 사용자가 적절히 구현해야 합니다.
    return frame  # 처리된 프레임 반환

# 추가한 코드
def save_and_send(image, path, save_dir):
    print("Saving and sending image...")
    save_path = str(save_dir / path.name)
    cv2.imwrite(save_path, image)  # Ensure this is the image with bounding boxes
    print(f"Image saved to {save_path}")
    send_detection(save_path, datetime.now())

# 추가한 코드
# 감지된 드론의 이미지와 시간을 웹 서버로 전송
def send_detection(image_path, detection_time):
    files = {'image': open(image_path, 'rb')}
    data = {'time': detection_time.strftime('%Y-%m-%d %H:%M:%S')}
    response = requests.post('http://127.0.0.1:8000/pybo/upload', files=files, data=data)
    print('Detection sent. Status code:', response.status_code)

# 추가한 코드
# 드론이 사라진 후 경과 시간 계산 및 전송
def calculate_and_send_elapsed_time():
    global LAST_DETECTED_TIME, DRONE_DETECTED
    if DRONE_DETECTED:
        elapsed_time = datetime.now() - LAST_DETECTED_TIME
        data = {'elapsed_time': elapsed_time.total_seconds()}
        response = requests.post('http://127.0.0.1:8000/pybo/upload_time', data=data)
        print('Elapsed time sent. Status code:', response.status_code)
        DRONE_DETECTED = False


# 추론을 실행하는 함수
@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # 모델 경로 또는 triton URL
    source=ROOT / "data/images",  # 파일/디렉토리/URL/글로브/스크린/0(웹캠)
    data=ROOT / "data/coco128.yaml",  # 데이터셋 yaml 경로
    imgsz=(640, 640),  # 추론 크기 (높이, 너비)
    conf_thres=0.25,  # 신뢰 임계값
    iou_thres=0.45,  # NMS IOU 임계값
    max_det=1000,  # 이미지당 최대 탐지 개수
    device="",  # cuda 장치, 예: 0 또는 0,1,2,3 또는 cpu
    view_img=False,  # 결과 표시
    save_txt=False,  # *.txt로 결과 저장
    save_csv=False,  # CSV 형식으로 결과 저장
    save_conf=False,  # --save-txt 레이블에 신뢰도 저장
    save_crop=False,  # 잘린 예측 상자 저장
    nosave=False,  # 이미지/비디오 저장 안 함
    classes=None,  # 클래스로 필터링: --class 0 또는 --class 0 2 3
    agnostic_nms=False,  # 클래스에 대한 NMS
    augment=False,  # 증강된 추론
    visualize=False,  # 특징 시각화
    update=False,  # 모든 모델 업데이트
    project=ROOT / "runs/detect",  # 결과를 저장할 프로젝트/이름
    name="exp",  # 결과를 저장할 프로젝트/이름
    exist_ok=False,  # 기존 프로젝트/이름 ok, 증가 안 함
    line_thickness=3,  # 바운딩 박스 두께 (픽셀)
    hide_labels=False,  # 레이블 숨기기
    hide_conf=False,  # 신뢰도 숨기기
    half=False,  # FP16 반정밀도 추론 사용
    dnn=False,  # ONNX 추론을 위해 OpenCV DNN 사용
    vid_stride=1,  # 비디오 프레임 속도 간격
):
    source = str(source)   # 입력된 source 경로를 문자열로 변환합니다.
    save_img = not nosave and not source.endswith(".txt")  # 이미지를 저장할지 여부를 결정합니다. , # nosave가 False이고 source가 .txt로 끝나지 않으면 이미지를 저장합니다.
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # source가 파일인지 여부를 확인합니다.
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))  # source가 URL인지 여부를 확인합니다.
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)  # source가 웹캠 번호, 스트림 파일 또는 URL이면 webcam을 True로 설정합니다.
    screenshot = source.lower().startswith("screen")  # source가 화면 캡처를 나타내는지 확인합니다.
    # 만약 URL이고 파일인 경우, 파일을 다운로드합니다.
    if is_url and is_file:
        source = check_file(source)  # 다운로드

    # 결과를 저장할 디렉토리를 설정합니다.
    # 저장할 디렉토리를 증가시킵니다.
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    # 결과를 저장할 디렉토리를 생성합니다. (save_txt가 True이면 'labels' 디렉토리도 생성합니다.)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 모델을 로드합니다.
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # 이미지 크기를 확인합니다.
    imgsz = check_img_size(imgsz, s=stride)

    # 데이터 로더를 설정합니다.
    bs = 1  # 배치 크기
    # 만약 웹캠이면
    if webcam:
        # 이미지 보기를 확인하고, 경고 메시지를 표시합니다.
        view_img = check_imshow(warn=True)
        # LoadStreams를 사용하여 데이터셋을 로드합니다. 이미지 크기, 스트라이드, 오토 설정, 비디오 스트라이드를 설정합니다.
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 데이터셋의 길이를 배치 크기로 설정합니다.
        bs = len(dataset)
    # 만약 스크린샷이면
    elif screenshot:
        # LoadScreenshots를 사용하여 데이터셋을 로드합니다. 이미지 크기, 스트라이드, 오토 설정을 설정합니다.
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    # 그렇지 않으면
    else:
        # LoadImages를 사용하여 데이터셋을 로드합니다. 이미지 크기, 스트라이드, 오토 설정, 비디오 스트라이드를 설정합니다.
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # 비디오 경로와 비디오 라이터를 배치 크기만큼 None으로 초기화합니다.
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 추론 실행
    # 모델을 워밍업합니다. (이미지 크기에 따라 조정)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    # 추론에 사용되는 변수 초기화
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    # 데이터셋에서 이미지를 가져와서 추론을 실행합니다.
    
    # 추가한 변수
    capture_interval = 5  # 드론 감지 후 프레임 캡처 간격 (초)
    last_capture_time = None  # 마지막 캡처 시간 초기화
    

    for i, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        if Path(path).suffix in VID_FORMATS:
            if vid_cap:
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                codec = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = save_dir / f"{Path(path).stem}_processed.mp4"
                out = cv2.VideoWriter(str(output_path), codec, fps, (w, h))

                while True:
                    ret, frame = vid_cap.read()
                    if not ret:
                        break
                    processed_frame = process_frame(frame, model, device)  # 모델과 디바이스는 추론에 사용됩니다.
                    out.write(processed_frame)
                out.release()
                
        with dt[0]:
            # 이미지를 토치 텐서로 변환하고 모델 디바이스로 전송합니다.
            im = torch.from_numpy(im).to(model.device)
            # 모델이 fp16이면 이미지를 half-precision으로 변환하고 그렇지 않으면 float으로 변환합니다.
            im = im.half() if model.fp16 else im.float()
            # 이미지를 0에서 1로 정규화합니다.
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 이미지가 3차원이면 배치 차원을 확장합니다.
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # 모델이 xml 파일이고 이미지 개수가 1보다 크면 이미지를 청크로 나눕니다.
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # 추론
        with dt[1]:
            # 시각화할 경우 경로를 설정합니다.
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 모델이 xml 파일이고 이미지 개수가 1보다 크면 이미지를 순회하면서 추론합니다.
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                # 이미지를 추론합니다.
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            # 비최대 억제를 실행하여 중복된 예측을 제거합니다.
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 두 번째 스테이지 분류기 (옵션)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # CSV 파일을 위한 경로 정의
        csv_path = save_dir / "predictions.csv"

        # CSV 파일을 생성하거나 추가합니다.
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            # 이미지의 예측 데이터를 CSV 파일에 작성하며, 파일이 이미 존재하면 추가합니다.
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # 예측 처리 - 추가
        for i, det in enumerate(pred):  # 이미지별
            global DRONE_DETECTED, LAST_DETECTED_TIME  # 전역 변수 사용 선언 추가
            seen += 1
            if webcam:  # 배치 크기 >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            path_obj = Path(p)
            save_path = str(save_dir / path_obj.name)  # 이미지 저장 경로

            # 추가코드
            now = datetime.now()

            # 드론 감지 및 전송 로직 추가 - 시작
            if len(det) and 'drone' in [names[int(cls)] for *xyxy, conf, cls in det]:
                if not DRONE_DETECTED:
                    DRONE_DETECTED = True
                    LAST_DETECTED_TIME = now
                    last_capture_time = now - timedelta(seconds=capture_interval)  # 즉시 첫 캡처를 트리거하기 위해 초기화

                # 추가한 코드
                # 드론 감지 이후 일정 간격으로 프레임 저장
                if (now - last_capture_time).total_seconds() >= capture_interval:
                    # 비디오 파일명에서 확장자를 제거하고 이미지 확장자를 추가합니다.
                    img_file_name = os.path.splitext(path_obj.name)[0] + '.jpg'
                    img_save_path = os.path.join(save_dir, img_file_name)
                    cv2.imwrite(img_save_path, im0)  # 이미지 저장
                    #send_detection(img_save_path, LAST_DETECTED_TIME)  # 웹 서버로 전송
                    last_capture_time = now
            else:
                if DRONE_DETECTED:
                    calculate_and_send_elapsed_time()  # 경과 시간 전송
                    RONE_DETECTED = False  # 감지 상태 초기화
            # 드론 감지 및 전송 로직 추가 - 끝

            p = Path(p)  # Path로 변환
            save_path = str(save_dir / p.name)  # im.jpg # 이미지 저장 경로
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt # 텍스트 저장 경로
            s += "%gx%g " % im.shape[2:]  # print string # 출력 문자열
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh # 정규화 게인 whwh
            imc = im0.copy() if save_crop else im0  # save_crop를 위해
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            drone_detected_in_this_frame = False
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 결과 출력
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 클래스별 탐지 개수
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 문자열에 추가

                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    if names[int(cls)] == 'drone':
                        drone_detected_in_this_frame = True
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

                        if names[int(cls)] == 'drone':  # 드론이 감지된 경우
                            label = f'{names[int(cls)]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(int(cls), True))
                annotated_image = annotator.result()  # Annotated image 생성
                path_obj = Path(p)  # 파일 경로 객체 생성

                if drone_detected_in_this_frame:
                    # 드론이 감지된 경우, 이미지를 저장하고 웹으로 전송
                    save_and_send(annotated_image, path_obj, save_dir)

                # 드론이 감지되지 않았지만 이전에 감지된 적이 있다면 경과 시간 전송
                elif DRONE_DETECTED:
                    calculate_and_send_elapsed_time()
                    DRONE_DETECTED = False

                if vid_writer[i]:
                    vid_writer[i].write(annotated_image)
                else:
                    # 이미지 저장 및 전송 로직
                    save_and_send(annotated_image, Path(path), save_dir)
            else:
                if vid_writer[i]:
                    vid_writer[i].write(im0s)  # 변화가 없는 원본 프레임 쓰기

    LOGGER.info(f"{s}{'' if len(pred[0]) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    # 모든 비디오 작업을 완료한 후, 비디오 라이터를 정리
    for writer in vid_writer:
        if writer:
            writer.release()

    # 결과 출력
    t = tuple(x.t / seen * 1e3 for x in dt)  # 이미지당 속도
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 모델 업데이트 (SourceChangeWarning 수정)

def parse_opt():
    #YOLOv5 탐지를 위한 명령행 인자를 구문 분석하여 추론 옵션과 모델 구성을 설정합니다.
    # argparse 모듈을 사용하여 명령행 인자를 구문 분석합니다.
    parser = argparse.ArgumentParser()
    # 모델 경로 또는 Triton URL을 설정하는 인자입니다.
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    # 입력 소스(파일/디렉토리/URL/글로브/스크린/0(웹캠))를 설정하는 인자입니다.
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    # 데이터셋 YAML 파일 경로를 설정하는 인자입니다.
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    # 추론 크기를 설정하는 인자입니다. (높이, 너비)
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    # 신뢰도 임계값을 설정하는 인자입니다.
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    # NMS IoU 임계값을 설정하는 인자입니다.
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    # 이미지 당 최대 탐지 수를 설정하는 인자입니다.
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    # CUDA 장치 또는 CPU를 설정하는 인자입니다.
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 결과를 표시하는지 여부를 설정하는 인자입니다.
    parser.add_argument("--view-img", action="store_true", help="show results")
    # 결과를 *.txt 파일에 저장하는지 여부를 설정하는 인자입니다.
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    # 결과를 CSV 형식으로 저장하는지 여부를 설정하는 인자입니다.
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    # --save-txt 레이블에 신뢰도를 저장하는지 여부를 설정하는 인자입니다.
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    # 예측된 상자를 저장하는지 여부를 설정하는 인자입니다.
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    # 이미지/동영상을 저장하지 않는지 설정하는 인자입니다.
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    # 클래스로 필터링하는 인자입니다.
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    # 클래스에 대한 NMS를 설정하는 인자입니다.
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    # 증강된 추론을 설정하는 인자입니다.
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    # 특성을 시각화하는지 여부를 설정하는 인자입니다.
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    # 모델을 업데이트하는지 여부를 설정하는 인자입니다.
    parser.add_argument("--update", action="store_true", help="update all models")
    # 결과를 저장할 프로젝트 경로를 설정하는 인자입니다.
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    # 결과를 저장할 이름을 설정하는 인자입니다.
    parser.add_argument("--name", default="exp", help="save results to project/name")
    # 기존 프로젝트/이름이 있는지 여부를 설정하는 인자입니다.
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 바운딩 박스 두께를 설정하는 인자입니다. (픽셀)
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    # 레이블을 숨기는지 여부를 설정하는 인자입니다.
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    # 신뢰도를 숨기는지 여부를 설정하는 인자입니다.
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    # FP16 반 정밀도 추론을 사용하는지 여부를 설정하는 인자입니다.
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # ONNX 추론에 OpenCV DNN을 사용하는지 여부를 설정하는 인자입니다.
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    # 비디오 프레임 속도 간격을 설정하는 인자입니다.
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    # 파싱된 옵션을 가져옵니다.
    opt = parser.parse_args()
    # 추론 크기를 확장합니다.
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 파싱된 옵션을 출력합니다.
    print_args(vars(opt))
    # 파싱된 옵션을 반환합니다.
    return opt


# 메인 함수
def main(opt):
    # 주어진 옵션으로 YOLOv5 모델 추론을 실행하고, 모델을 실행하기 전에 요구 사항을 확인합니다.
    # 모델을 실행하기 전에 요구 사항을 확인합니다.
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    # YOLOv5 모델 추론을 실행합니다.
    run(**vars(opt))


# 메인 스크립트
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)