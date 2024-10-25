import cv2

# 스트리밍 URL을 사용합니다.
stream_url = '<IOT의 주소:8082>' 
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Cannot open stream")
else:
    print("Stream opened successfully")
    while True:
        ret, frame = cap.read()
        if ret:
            # 프레임을 화면에 표시합니다.
            cv2.imshow('Stream', frame)

            # 'q' 키를 누르면 스트리밍을 종료합니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Cannot read frame")
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()