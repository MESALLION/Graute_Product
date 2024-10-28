import subprocess
import threading

def start_camera_stream():
    # 카메라 스트리밍 명령어 실행
    process = subprocess.Popen(
        ["bash", "camera.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    process.communicate()

def start_gps():
    # GPS 전송 파이썬 스크립트 실행
    process = subprocess.Popen(
        ["python3", "send_gps.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    process.communicate()

if __name__ == "__main__":
    # 두 스레드 각각 시작
    camera_thread = threading.Thread(target=start_camera_stream, daemon=True)
    gps_thread = threading.Thread(target=start_gps, daemon=True)

    camera_thread.start()
    gps_thread.start()

    # 메인 프로그램 종료 방지
    try:
        camera_thread.join()
        gps_thread.join()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")