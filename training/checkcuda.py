# 제대로 GPU를 사용하는지 체크
import torch
# GPU 이름이 제대로 나와야함
print(torch.cuda.get_device_name(0))
# GPU 이름이 제대로 나와야함
print(torch.version.cuda)
# True 반환을 해야함
print(torch.cuda.is_available())