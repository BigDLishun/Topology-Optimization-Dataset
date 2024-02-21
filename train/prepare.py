import numpy as np
import os
input_data = np.zeros((10000, 64, 128))
target_data = np.zeros((10000, 64, 128))
for i in range(10000):
    print(i+150000)
    npz_path = f"./simply-supported-beam_64_128/{i+150000}.npz"  # 假设您的数据文件名是 "data_0.npz" 至 "data_999.npz"
    npz = np.load(npz_path)
    data = npz['arr_0']  # 每个.npz文件包含一个名为 'arr_0' 的numpy数组，形状为 (40, 32, 64)
    input_data[i] = data[0]
    target_data[i] = data[-1]
# Save input and target data as .npz files
np.savez('./data/simply-supported-beam_64_128_input_data_10000.npz', input=input_data)
np.savez('./data/simply-supported-beam_64_128_target_data_10000.npz', target=target_data)
print("数据加载完毕")