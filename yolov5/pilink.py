import subprocess
import threading
import time
import queue

def enqueue_output(out, queue):
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()

def run_camera_stream():
    # detectdrone.py 실행
    process = subprocess.Popen(
        ['python', 'detectdrone.py', '--weights', 'finalbest.pt', '--source', '<IOT의 주소:8082>', '--imgsz', '640', '--conf-thres', '0.7'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1
    )

    q = queue.Queue()
    thread = threading.Thread(target=enqueue_output, args=(process.stdout, q))
    thread.daemon = True  # 데몬 스레드로 설정하여 메인 프로그램 종료 시 자동 종료
    thread.start()

    try:
        while process.poll() is None:
            try:
                line = q.get(timeout=0.1)  # 출력 스트림에서 데이터를 가져옴 (타임아웃 설정)
                print(f"[Camera Stream] {line}", end='')
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nCamera stream interrupted by user.")
    finally:
        process.terminate()

def run_gps_receiver():
    # rev_gps.py 실행
    process = subprocess.Popen(
        ['python', r'GPS\rev_gps.py'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1
    )

    q = queue.Queue()
    thread = threading.Thread(target=enqueue_output, args=(process.stdout, q))
    thread.daemon = True
    thread.start()

    try:
        while process.poll() is None:
            try:
                q.get(timeout=0.1)  # 데이터를 가져오지만 출력하지 않음
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nGPS receiver interrupted by user.")
    finally:
        process.terminate()

if __name__ == '__main__':
    # 카메라와 GPS 스레드 각각 실행
    camera_thread = threading.Thread(target=run_camera_stream, daemon=True)
    gps_thread = threading.Thread(target=run_gps_receiver, daemon=True)

    camera_thread.start()
    gps_thread.start()

    try:
        # 메인 루프 실행
        while True:
            user_input = input("Enter 'q' to quit: ")
            if user_input.lower() == 'q':
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

    # 스레드 종료
    camera_thread.join()
    gps_thread.join()