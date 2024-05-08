
import torch # 如果pytorch安装成功即可导入
print(torch.__version__)
print(torch.cuda.is_available()) # 查看CUDA是否可用，如果True表示可以使用
print(torch.cuda.device_count()) # 查看可用的CUDA数量，0表示有一个
print(torch.version.cuda) # 查看CUDA的版本号
