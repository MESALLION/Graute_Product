import socket

# 소켓 주소는 사용하고자하는 인터넷 IP주소
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('인터넷 IP주소', 5005))

# 출력 및 결과값
while True:
    data, _ = sock.recvfrom(1024)
    print(f"Received GPS data: {data.decode('utf-8')}")