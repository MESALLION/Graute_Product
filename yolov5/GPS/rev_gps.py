import socket

# sock통신
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('<연결하고자하는 PC의 주소>', 5005))

# 출력 및 결과값
while True:
    data, _ = sock.recvfrom(1024)
    print(f"Received GPS data: {data.decode('utf-8')}")