import os
import glob
import cv2
import numpy as np
import torch
import h5py
import matplotlib.cm as CM
from tqdm import tqdm
from matplotlib import pyplot as plt
from modeling.MobileCount_multidia import MobileCount


# 配置参数集中管理
class Config:
    save_path = r'C:/Honours/Thesis/Visual/SHTechA'
    model_weights = r"C:/Honours/Thesis/weights/mobilecount-multy-coonv-epoch500-BATCH32-no-bn/mobilecount-multy-coonv-epoch500-BATCH32-no-bn_model_best.pth"
    test_data_path = 'C:/Honours/Thesis/datasets/part_A_final/test/images'
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    heatmap_cmap = CM.jet  # 使用更明显的颜色映射
    dpi = 150  # 平衡清晰度和文件大小
    figsize = (24, 8)  # 调整画布比例
    density_cmap = CM.viridis  # 改用viridis色图，更适合科学可视化
    use_grayscale = False


def safe_mkdir(path):
    """创建目录并处理异常"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"Failed to create directory: {e}")
            raise


def enhance_density_map(density_map):
    """优化密度图生成逻辑"""
    density_map = np.maximum(density_map, 0)

    # 自适应归一化
    vmax = np.percentile(density_map, 99)
    vmin = np.min(density_map)
    normalized = np.clip((density_map - vmin) / (vmax - vmin + 1e-7), 0, 1)

    # 应用平滑
    normalized = cv2.GaussianBlur(normalized, (5, 5), 0)

    # 根据配置选择颜色模式
    if Config.use_grayscale:
        colored = plt.cm.gray(normalized)[..., :3]
    else:
        colored = plt.get_cmap(Config.density_cmap)(normalized)[..., :3]

    return (colored * 255).astype(np.uint8)


def load_model():
    """封装模型加载逻辑"""
    model = MobileCount().cuda()
    checkpoint = torch.load(Config.model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def process_single_image(model, img_path):
    """单张图像处理流水线"""
    # 数据加载与预处理
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {img_path}")

    # 转换颜色空间
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 密度图加载
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    with h5py.File(gt_path, 'r') as h5:
        gt_density = np.asarray(h5['density'])

    # 模型推理预处理
    input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
    input_tensor = torch.from_numpy(input_tensor).float()  # 转换为float张量
    input_tensor = (input_tensor - torch.tensor(Config.mean)) / torch.tensor(Config.std)  # 归一化
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).cuda()  # 调整维度

    # 模型推理
    with torch.no_grad():
        pred_density = model(input_tensor).cpu().numpy()[0, 0]

    return original_img, gt_density, pred_density


def visualize_results(original, gt, pred, save_path):
    """生成可视化结果并保存"""
    plt.figure(figsize=Config.figsize, dpi=Config.dpi)

    # 原图显示
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image", fontsize=12)
    plt.axis('off')

    # 真实密度图（保持原始数值）
    plt.subplot(1, 3, 2)
    plt.imshow(enhance_density_map(gt))
    plt.title(f"Ground Truth\nSum: {gt.sum():.2f}", fontsize=12)
    plt.axis('off')

    # 预测密度图（数值除以100）
    plt.subplot(1, 3, 3)
    plt.imshow(enhance_density_map(pred))
    adjusted_sum = pred.sum() / 100  # 应用数值调整
    plt.title(f"Predicted\nSum: {adjusted_sum:.2f}", fontsize=12)
    plt.axis('off')

    # 紧凑布局并保存
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    # 初始化环境
    safe_mkdir(Config.save_path)
    model = load_model()

    # 获取测试图像列表
    img_list = glob.glob(os.path.join(Config.test_data_path, "*.jpg"))
    if not img_list:
        raise ValueError("No test image found")

    # 使用进度条
    pbar = tqdm(img_list, desc="Processing", unit="img")
    for img_path in pbar:
        try:
            original, gt, pred = process_single_image(model, img_path)
            save_name = os.path.basename(img_path).replace('.jpg', '_visual.png')
            save_file = os.path.join(Config.save_path, save_name)
            visualize_results(original, gt, pred, save_file)
        except Exception as e:
            print(f"处理 {img_path} 失败: {e}")
            continue


if __name__ == "__main__":
    main()
    print(f"The visualization results have been saved to: {Config.save_path}")
