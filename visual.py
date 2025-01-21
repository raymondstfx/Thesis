import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import torch
import h5py
from modeling.MobileCount_multidia import MobileCount

save_path = r'C:\Thesis\Honours\Visual\SHTechB'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Image normalization
def imnormalize(img,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return (img - mean) / std

# Display density map (heat map)）
def show_map(input):
    input[input < 0] = 0  # Set negative values ​​to 0
    input = input[0][0]  # Extract single channel data
    fidt_map = input / np.max(input) * 255  # Normalize to 0-255
    fidt_map = fidt_map.astype(np.uint8)
    fidt_map = cv2.applyColorMap(fidt_map, cv2.COLORMAP_JET)  # Convert to color heat map
    return fidt_map

# Loading model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model = MobileCount()
checkpoint_best = torch.load(
    r"C:/Thesis/Honours/weights/mobilecount-multy-coonv-epoch500-BATCH32-no-bn/mobilecount-multy-coonv-epoch500-BATCH32-no-bn_model_best.pth"
)
model.load_state_dict(checkpoint_best['state_dict'])
model = model.cuda()
model.eval()

# Loading test data
part_B_test = os.path.join('C:/Thesis/Honours/datasets/part_B_final/test_data', 'images')
val_list = []
for img_path in glob.glob(os.path.join(part_B_test, '*.jpg')):
    val_list.append(img_path)

# Traverse the test set images and generate visualizations
for img_path in val_list:
    # Read the original image
    img = cv2.imread(img_path)
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')

    # Load the real density map
    with h5py.File(gt_path, 'r') as h5:
        ground_truth_density = np.asarray(h5['density'])

    # Normalize the image and convert it to a tensor
    img = imnormalize(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)
    img = img.permute(0, 3, 1, 2)
    img = torch.tensor(img, dtype=torch.float32).cuda()

    # Model prediction
    with torch.no_grad():
        output = model(img)
    predicted_density = output.cpu().data.numpy()

    # Visualization
    gt_heatmap = show_map(ground_truth_density[np.newaxis, np.newaxis, ...])  # GT heat map
    pred_heatmap = show_map(predicted_density)  # Prediction heat map

    # Save the visualization results
    fname = os.path.basename(img_path).replace('.jpg', '_visual.png')
    save_file = os.path.join(save_path, fname)
    plt.figure(figsize=(20, 10))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')

    # Real density map
    plt.subplot(1, 3, 2)
    plt.title(f"Ground Truth (Sum: {ground_truth_density.sum():.2f})")
    plt.imshow(gt_heatmap)
    plt.axis('off')

    # Prediction Density Plot
    plt.subplot(1, 3, 3)
    plt.title(f"Predicted (Sum: {predicted_density.sum():.2f})")
    plt.imshow(pred_heatmap)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

print(f"Visualizations saved to {save_path}")
