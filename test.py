import torch

if torch.cuda.is_available():
    print("CUDA 可用")
    print("CUDA 设备数量:", torch.cuda.device_count())
    print("当前 CUDA 设备:", torch.cuda.current_device())
    print("CUDA 设备名称:", torch.cuda.get_device_name(0))
else:
    print("CUDA 不可用")