import torch
import numpy as np

# 输出版本号
print("PyTorch版本:", torch.__version__)
print("NumPy版本:", np.__version__)

# 测试autograd自动微分
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print("x的梯度:\n", x.grad)
