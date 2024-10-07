import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('192.168.219.168', 5005))

while True:
    data, _ = sock.recvfrom(1024)
    print(f"Received GPS data: {data.decode('utf-8')}")