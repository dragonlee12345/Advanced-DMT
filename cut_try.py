import math
import torch

h = w = 4
# 创建一个大小为 (1, 1, 4, 4) 的三维张量
tensor = torch.linspace(1, h ** 2 * 2, h ** 2 * 2).reshape(1, 2, h, w)

# 打印原始张量
print("Original Tensor:")
print(tensor)
print("\nShape:", tensor.shape)

b, c, h, w = tensor.shape
kernel_size = math.ceil(h / 3)
if h % 3 == 2:  # 需要对 3n+2 特殊处理
    kernel_size += 1
stride = int((h - kernel_size) / 2)
padding = 0
print(kernel_size, stride)
unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
print(unfold(tensor))
unfolded_tensor = unfold(tensor).permute(0, 2, 1).reshape(b, 3, 3, c, kernel_size, kernel_size)
unfolded_tensor = unfolded_tensor.permute(0, 3, 1, 2, 4, 5)
# 打印滑动窗口后的张量
print("\nUnfolded Tensor:")
print(unfolded_tensor)
print("\nShape:", unfolded_tensor.shape)


