import pickle
import os

# 加载第一个 pkl 文件
with open('/home/guangxun/dataset/waymo/SMPL-data/training/training.pkl', 'rb') as f1:
    data1 = pickle.load(f1)

# 加载第二个 pkl 文件
with open('/home/guangxun/dataset/waymo/SMPL-data/validation/validation.pkl', 'rb') as f2:
    data2 = pickle.load(f2)

# 合并数据（假设数据是列表）
merged_data = data1 + data2

# 保存合并后的数据到新的 pkl 文件
with open('/home/guangxun/dataset/waymo/SMPL-data/merged.pkl', 'wb') as f_merged:
    pickle.dump(merged_data, f_merged)

# 加载第二个 pkl 文件
with open('/home/guangxun/dataset/waymo/SMPL-data/merged.pkl', 'rb') as f:
    data = pickle.load(f)
    print(len(data))
