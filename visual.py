import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import torch
import h5py
from modeling.MobileCount_multidia import MobileCount  # 请替换为你的模型文件路径

# 创建可视化保存路径
save_path = r'C:\Thesis\Honours\Visual\SHTechB'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 图像归一化
def imnormalize(img,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return (img - mean) / std

# 显示密度图（热力图）
def show_map(input):
    input[input < 0] = 0  # 将负值置为 0
    input = input[0][0]  # 提取单通道数据
    fidt_map = input / np.max(input) * 255  # 归一化到 0-255
    fidt_map = fidt_map.astype(np.uint8)
    fidt_map = cv2.applyColorMap(fidt_map, cv2.COLORMAP_JET)  # 转换为彩色热力图
    return fidt_map

# 加载模型
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model = MobileCount()
checkpoint_best = torch.load(
    r"C:/Thesis/Honours/weights/mobilecount-multy-coonv-epoch500-BATCH32-no-bn/mobilecount-multy-coonv-epoch500-BATCH32-no-bn_model_best.pth"
)
model.load_state_dict(checkpoint_best['state_dict'])
model = model.cuda()
model.eval()

# 加载测试数据
part_B_test = os.path.join('C:/Thesis/Honours/datasets/part_B_final/test_data', 'images')
val_list = []
for img_path in glob.glob(os.path.join(part_B_test, '*.jpg')):
    val_list.append(img_path)

# 遍历测试集图像并生成可视化
for img_path in val_list:
    # 读取原始图像
    img = cv2.imread(img_path)
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')

    # 加载真实密度图
    with h5py.File(gt_path, 'r') as h5:
        ground_truth_density = np.asarray(h5['density'])

    # 归一化图像并转换为张量
    img = imnormalize(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)
    img = img.permute(0, 3, 1, 2)
    img = torch.tensor(img, dtype=torch.float32).cuda()

    # 模型预测
    with torch.no_grad():
        output = model(img)
    predicted_density = output.cpu().data.numpy()

    # 可视化
    gt_heatmap = show_map(ground_truth_density[np.newaxis, np.newaxis, ...])  # GT 热力图
    pred_heatmap = show_map(predicted_density)  # 预测热力图

    # 保存可视化结果
    fname = os.path.basename(img_path).replace('.jpg', '_visual.png')
    save_file = os.path.join(save_path, fname)
    plt.figure(figsize=(20, 10))

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')

    # 真实密度图
    plt.subplot(1, 3, 2)
    plt.title(f"Ground Truth (Sum: {ground_truth_density.sum():.2f})")
    plt.imshow(gt_heatmap)
    plt.axis('off')

    # 预测密度图
    plt.subplot(1, 3, 3)
    plt.title(f"Predicted (Sum: {predicted_density.sum():.2f})")
    plt.imshow(pred_heatmap)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

print(f"Visualizations saved to {save_path}")
