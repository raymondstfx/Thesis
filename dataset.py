import random
from PIL import Image
import cv2



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # ## 数据生成器


# In[25]:


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            xmin = w - bbx[:, 3]
            xmax = w - bbx[:, 1]
            bbx[:, 1] = xmin
            bbx[:, 3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        if bbx is None:
            return img, mask
        return img, mask, bbx


# In[26]:


def adap_net(img, den, scale=32):
    if img.shape[0] % scale:  # 如果高不能被8整除
        # 记录需要填充的高度
        pad_height = scale - img.shape[0] % scale
        left, right = 0, 0
        top, bottom = int(pad_height / 2), pad_height - int(pad_height / 2)
        img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])  # 从图像边界向上,下,左,右扩的像素数目
        den = cv2.copyMakeBorder(den, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])  # 从图像边界向上,下,左,右扩的像素数目

    if img.shape[1] % scale:  # 如果宽不能被8整除
        # 记录需要填充的高度
        pad_width = scale - img.shape[1] % scale
        left, right = int(pad_width / 2), pad_width - int(pad_width / 2)
        top, bottom = 0, 0
        img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])  # 从图像边界向上,下,左,右扩的像素数目
        den = cv2.copyMakeBorder(den, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])  # 从图像边界向上,下,左,右扩的像素数目

    return img, den


# In[27]:


def norm_img_size(img, den, dst_size):
    if img.shape[1] < dst_size[0]:
        width_scale = dst_size[0] / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1] * width_scale), int(img.shape[0] * width_scale)),
                         interpolation=cv2.INTER_CUBIC)
        den = cv2.resize(den, (int(den.shape[1] * width_scale), int(den.shape[0] * width_scale)),
                         interpolation=cv2.INTER_CUBIC) / width_scale / width_scale
    if img.shape[0] < dst_size[1]:
        height_scale = dst_size[1] / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * height_scale), int(img.shape[0] * height_scale)),
                         interpolation=cv2.INTER_CUBIC)
        den = cv2.resize(den, (int(den.shape[1] * height_scale), int(den.shape[0] * height_scale)),
                         interpolation=cv2.INTER_CUBIC) / height_scale / height_scale

    img, den = adap_net(img, den)

    return img, den


# In[28]:


def random_crop(img, den, dst_size):
    # dst_size：w,h
    # 若宽高不够，则先将其等比放大至最小大小
    img, den = norm_img_size(img, den, dst_size)
    ts_hd, ts_wd, _ = img.shape

    x1 = random.randint(0, ts_wd - dst_size[0])
    y1 = random.randint(0, ts_hd - dst_size[1])
    x2 = x1 + dst_size[0]
    y2 = y1 + dst_size[1]

    label_x1 = x1
    label_y1 = y1
    label_x2 = x2
    label_y2 = y2

    return img[y1:y2, x1:x2, :], den[label_y1:label_y2, label_x1:label_x2]





