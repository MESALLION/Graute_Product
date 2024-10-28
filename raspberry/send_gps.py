import socket
import serial

# GPS 모듈 설정
ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)

# 소켓 설정 - 목표 주소값 설정(보내고자 하는 IP주소)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('<보내고자 하는 인터넷 IP주소>', 5005)

while True:
    data = ser.readline().decode('utf-8').strip()
    if data:
        sock.sendto(data.encode('utf-8'), server_address)