import torch
# conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

device_count = torch.cuda.device_count() # 可用的GPU数量
current_device = torch.cuda.current_device() #返回当前所选设备的索引
device_name = torch.cuda.get_device_name(current_device) #返回设备名， 默认值为当前设备
device_capability = torch.cuda.get_device_capability(current_device) # 设备的最大和最小的cuda容量, 默认值为当前设备
device_properties = torch.cuda.get_device_properties(current_device)
# device_properties = torch.cuda.get_device_properties(0)
is_available = torch.cuda.is_available() # 当前CUDA是否可用
device_cuda = torch.device("cuda") # GPU设备





print('device_count: {device_count}'.format(device_count=device_count))

print('current_device: {current_device}'.format(current_device=current_device))

print('device_name: {device_name}'.format(device_name=device_name))
print('device_capability: {device_capability}'.format(device_capability=device_capability))
print('device_properties: {device_properties}'.format(device_properties=device_properties))
print('is_available: {is_available}'.format(is_available=is_available))
print('device_cuda: {device_cuda}'.format(device_cuda=device_cuda))

