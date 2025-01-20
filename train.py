import glob
import json
import math
import os

import cv2
from torch.utils.data import Dataset
import h5py
import numpy as np

from dataset import random_crop, adap_net
from modeling.MobileCount_multidia import MobileCount

import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import argparse
import time
import random
cfg = dict()
cfg['gpu'] = ''  # GPU id to u se
cfg[
    'task'] = 'mobilecount-multy-coonv-epoch500-BATCH32-no-bn'  # task is to use
cfg['pre'] = "weights/" + cfg['task'] + "/" + cfg['task'] + '_checkpoint.pth'  # path to the pretrained model

# 训练相关参数
cfg['start_epoch'] = 0  # 起始epoch（影响学习率）
cfg['epochs'] = 1000  # 总轮次vihicle
cfg['best_prec1'] = 1e6  # 最优精度？
cfg['best_rmse'] = 1e6
cfg['best_psnr'] = 0
cfg['best_ssim'] = 0
cfg['original_lr'] = 1e-4  # 初始学习率
cfg['lr'] = 1e-4  # 学习率（可变？）
cfg['batch_size'] = 32  # batch_size仅为1？
cfg['momentum'] = 0.95  # SGD动量
cfg['decay'] = 1e-4  # 学习率衰减
cfg['steps'] = [-1, 1, 100, 150]  # ？
cfg['scales'] = [1, 1, 1, 1]  # ？
cfg['workers'] = 4  # 线程数？
cfg['seed'] = time.time()  # 随机种子？
cfg['stand_by'] = 10
cfg['print_freq'] = 10  # 打印队列？
cfg['crop_size'] = [256, 256]
cfg['SHHB.MEAN_STD'] = (
     [0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])

# cfg['SHHB.MEAN_STD'] = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])  # SHA

# cfg['SHHB.MEAN_STD'] = (
#     [0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])  # QNRF
# cfg['SHHB.MEAN_STD'] = (    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RSOC

cfg['LABEL_FACTOR'] = 100.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.manual_seed(cfg['seed'])
part_B_train = os.path.join('C:/Thesis/Honours/datasets/part_B_final/test_data', 'images')
part_B_test = os.path.join('C:/Thesis/Honours/datasets/part_B_final/train_data', 'images')
cur_lr_list = []
class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4):
        #         if train:
        #             root = root * 4
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]
        #         print(img_path)

        img, target = self.load_data(img_path, self.train)

        # img = 255.0 * F.to_tensor(img)

        # img[0,:,:]=img[0,:,:]-92.8207477031
        # img[1,:,:]=img[1,:,:]-95.2757037428
        # img[2,:,:]=img[2,:,:]-104.877445883

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def load_data(self, img_path, train=True):
        gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        #     print(gt_path)
        img = Image.open(img_path).convert('RGB')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
        if train:
            if random.random() > 0.5 or 1:  # random() 方法返回随机生成的一个实数，它在[0,1)范围内。
                target = np.fliplr(target).copy()  # np.fliplr作用是将数组在左右方向上翻转。
                img = np.fliplr(img).copy()

            if cfg['crop_size']:
                img, target = random_crop(np.array(img), target, cfg['crop_size'])
        else:
            img = np.array(img)
            img, target = adap_net(img, target)

        img = np.array(img)
        img, target = adap_net(img, target)
        target = cv2.resize(target, (int(target.shape[1] ), int(target.shape[0] )),
                            interpolation=cv2.INTER_CUBIC)
        target = target * cfg['LABEL_FACTOR']
        return img, target

def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')

def main():
    parser = argparse.ArgumentParser(description='PyTorch ASPDNet')
    parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                        help='path to the pretrained model')
    global args, best_prec1
    temp = 'test_images'
    path_sets_1 = [part_B_train]
    path_sets_2 = [part_B_test]

    train_list = []
    for path in path_sets_1:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            train_list.append(str(img_path))
    val_list = []
    for path in path_sets_2:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            val_list.append(str(img_path))

    torch.cuda.manual_seed(cfg['seed'])
    model = MobileCount()
    model = model.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), cfg['lr'],
                                 weight_decay=cfg['decay'])
    if cfg['pre']:
        if os.path.isfile(cfg['pre']):
            print("=> loading checkpoint '{}'".format(cfg['pre']))
            checkpoint = torch.load(cfg['pre'])
            cfg['start_epoch'] = checkpoint['epoch']

            checkpoint_best = torch.load("weights/" + cfg['task'] + "/" + cfg['task'] + '_model_best.pth')

            best_prec1 = checkpoint_best['best_prec1']
            model.load_state_dict(checkpoint_best['state_dict'])
            optimizer.load_state_dict(checkpoint_best['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg['pre'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg['pre']))

    prec1 = cfg['best_prec1']

    # In[37]:

    if os.path.isfile(cfg['task'] + '.json'):
        with open(cfg['task'] + '.json', 'r') as f:
            history = json.load(f)
        print('load history file ' + cfg['task'] + '.json')
    else:
        history = {}
        history['epoch'] = []
        history['loss'] = []
        history['val_loss'] = []
        history['lr'] = []
    for epoch in range(cfg['start_epoch'], cfg['epochs']):
        loss = train(train_list, model, criterion, optimizer, epoch)
        mae, rmse = validate(val_list, model, criterion)
        prec1 = mae
        is_best = mae < cfg['best_prec1']
        cfg['best_prec1'] = min(mae, cfg['best_prec1'])
        cfg['best_rmse'] = min(rmse, cfg['best_rmse'])
        print(' * best MAE {mae:.3f} '
              .format(mae=cfg['best_prec1']))
        print(' * best RMSE {mae:.3f} '
              .format(mae=cfg['best_rmse']))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': cfg['pre'],
            'state_dict': model.state_dict(),
            'best_prec1': cfg['best_prec1'],
            #         'best_psnr':cfg['best_psnr'],
            #         'best_ssim':cfg['best_ssim'],
            'optimizer': optimizer.state_dict(),
        }, is_best, cfg['task'])
        history['epoch'].append(int(epoch))
        history['loss'].append(float(loss))
        history['val_loss'].append(float(prec1))
        history['lr'].append(float(cfg['lr']))
        with open(cfg['task'], 'w+') as f:
            json.dump(history, f)


def train(train_list, model, criterion, optimizer, epoch):
    # 记录损失、训练时间、数据加载时间
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # 读取训练数据（含数据增强）
    train_loader = torch.utils.data.DataLoader(
        listDataset(train_list,
                    shuffle=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize(mean=cfg['SHHB.MEAN_STD'][0],
                                                                    std=cfg['SHHB.MEAN_STD'][1]),
                    ]),
                    train=True,
                    #                             seen=model.seen,
                    batch_size=cfg['batch_size'],
                    num_workers=cfg['workers']),
        batch_size=cfg['batch_size'])
    print('epoch %d, processed %d samples, lr %.10f' %
          (epoch, epoch * len(train_loader.dataset), cfg['lr']))

    model.train()
    end = time.time()
    # print('cur_lr:', scheduler.get_lr()[0])
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)

        # print(output.shape)
        # exit()
        # -------------------------# -------------------------
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)
        # print(target.shape)
        loss_output = criterion(output, target)
        # loss = loss_output + d * lc
        loss = loss_output
        losses.update(loss.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()

        # -------------------------# -------------------------
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    # scheduler.step()
    # cfg['lr'] = optimizer.param_groups[-1]['lr']
    # cfg['lr'] = scheduler.get_lr()[0]

    return losses.avg


def validate(val_list, model, criterion):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        listDataset(val_list,
                    shuffle=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize(mean=cfg['SHHB.MEAN_STD'][0],
                                                                    std=cfg['SHHB.MEAN_STD'][1]),
                    ]), train=False),
        batch_size=1)

    model.eval()

    mae = 0
    rmse = 0
    #     psnr = 0
    #     ssim = 0

    for i, (img, target) in enumerate(test_loader):

        #         print(val_list)

        img = img.cuda()
        img = Variable(img)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        output = model(img)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        mae += abs(output.data.sum() -
                   target.sum().type(torch.FloatTensor).cuda())

        rmse += (output.data.sum() -
                 target.sum().type(torch.FloatTensor).cuda()) ** 2

    mae = mae / len(test_loader)
    rmse = math.sqrt(rmse / len(test_loader))

    print(' * MAE {mae:.3f} '
          .format(mae=mae))
    print(' * RMSE {rmse:.3f} '
          .format(rmse=rmse))
    return mae, rmse


def adjust_learning_rate(optimizer, epoch, prec1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    cfg['lr'] = cfg['original_lr'] * 0.995 ** epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg['lr']


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


def save_checkpoint(state, is_best, task_id, filename='_checkpoint.pth'):
    if not os.path.isdir("weights/" + task_id):
        os.makedirs("weights/" + task_id)
    torch.save(state, "weights/" + task_id + "/" + task_id + filename)
    if is_best:
        shutil.copyfile("weights/" + task_id + "/" + task_id + filename,
                        "weights/" + task_id + "/" + task_id + '_model_best.pth')


if __name__ == '__main__':
    main()
