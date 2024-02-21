import  torch
print(torch.__version__)
print('gpu:', torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())
print(torch.device)