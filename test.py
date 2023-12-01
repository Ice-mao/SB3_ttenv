import numpy as np
import os
import yaml
from skimage.transform import rotate

from ttenv.maps.map_utils import GridMap
import ttenv.util as util
from ttenv.maps.dynamic_map import DynamicMap

# if __name__ == '__main__':
#     print("Test DynamicMap")
#     d = DynamicMap(map_dir_path='ttenv/maps', map_name='dynamic_map', )
#     for _ in range(5):
#         d.generate_map()
#         import matplotlib.pyplot as plt
#         plt.imshow(d.map, cmap='gray_r')
#         plt.show()
#     plt.close()

import torch
import torch.nn as nn

n_input_channels = 5
input_shape = (1, n_input_channels, 28, 28)  # Batch size of 1

# 创建一个随机输入张量
input_tensor = torch.randn(input_shape)

# 定义网络
cnn = nn.Sequential(
    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
    nn.ReLU(),
    nn.Flatten(),
)

# 前向传播
output = cnn(input_tensor)
print("输出形状:", output.shape)
