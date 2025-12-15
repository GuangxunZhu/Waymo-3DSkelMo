import numpy as np
def read_npy_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data

path = "/dataset/guangxun/test/generated_1/joints_10fps/validation_5_75_15.npy"
data = read_npy_file(path)
print(data.shape)