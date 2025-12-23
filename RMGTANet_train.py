import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime
from tqdm import tqdm

from model.RMGTANet_models import MCCNet_VGG
from data import get_loader
from utils import clip_gradient, adjust_lr
from APBLoss import AdaptivePBLoss
from PBLoss import PBLoss

import pytorch_iou
import pytorch_fm

from thop import profile
from thop import clever_format

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./MCC-logs/PV/My_model')
from evaluation import metric as M
from data import test_dataset

torch.cuda.set_device(-1)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = MCCNet_VGG()

num_params = 0
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

for p in model.parameters():
    num_params += p.numel()
print("The number of parameters: {}".format(num_params))
# -------------------------------------------------------------
# 计算 FLOPs 和 参数量
# -------------------------------------------------------------
from thop import profile, clever_format

# 构造一个假的输入样本（符合模型输入尺寸）
dummy_input = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()

# 使用 thop.profile 计算 FLOPs 和参数
flops, params = profile(model, inputs=(dummy_input,), verbose=False)

# 格式化输出
macs, params = clever_format([flops, params], "%.3f")
print(f"Model FLOPs: {macs}")
# -------------------------------------------------------------

# 改
image_root = './dataset/PV08/train/image/'
gt_root = './dataset/PV08/train/gt/'

# image_root = './dataset/EORSSD/train/image/'
# gt_root = './dataset/EORSSD/train/gt/'
# edge_root = './dataset/EORSSD/train/edge/'

# image_root = './dataset/ors-4199/train/image/'
# gt_root = './dataset/ors-4199/train/gt/'
# edge_root = './dataset/ors-4199/train/edge/'

# image_root = './dataset/RSISOD/train/image/'
# gt_root = './dataset/RSISOD/train/gt/'
# edge_root = './dataset/RSISOD/train/edge/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)


def binaryDiceLoss(pred, target, eps=1e-5):
    if torch.max(pred) > 1:
        pred = pred.contiguous() / 255
    else:
        pred = pred.contiguous()

    if torch.max(target) > 1:
        target = target.contiguous() / 255
    else:
        target = target.contiguous()

    """
    # This is incorrect. (1-(ab/a+b) + 1-(cd/c+d)) is not same with 1*2(ab+cd/a+b+c+d) 
    inter = torch.dot(pred.view(-1), target.view(-1))
    union = torch.sum(pred) + torch.sum(target)

    loss = 1*batch_num - (2 * inter + smooth) / (union + smooth) # 1*2(ab+cd/a+b+c+d) 
    """
    if len(pred.size()) == 4 and len(target.size()) == 4:  # case of batch (Batchsize, C==1, H, W)
        intersection = (pred * target).sum(dim=2).sum(dim=2)  # sum of H,W axis
        loss = (1 - ((2. * intersection + eps) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + eps)))
        # loss shape : (batch_size, 1)
    elif len(pred.size()) == 3 and len(target.size()) == 3:  # case of image shape (C==1,H,W)
        intersection = (pred * target).sum(dim=1).sum(dim=1)
        coeff = (1 - (2. * intersection) / (pred.sum(dim=1).sum(dim=1) + target.sum(dim=1).sum(dim=1) + eps))
    return loss.mean()  # (1-(ab/a+b) + 1-(cd/c+d)) / batch_size


CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
PB = PBLoss()
APB = AdaptivePBLoss()


def train(train_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0

    train_loader_tqdm = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch}")
    for i, pack in train_loader_tqdm:
        optimizer.zero_grad()
        images, gts = pack
        images = images.cuda()
        gts = gts.cuda()

        s1, s2 = model(images)
        s1_sig = torch.sigmoid(s1)
        s2_sig = torch.sigmoid(s2)

        # loss
        loss1 = CE(s1, gts) + IOU(s1_sig, gts) + binaryDiceLoss(s1_sig, gts)
        loss2 = CE(s2, gts) + IOU(s2_sig, gts) + binaryDiceLoss(s2_sig, gts)

        loss = loss1 + loss2

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        total_loss += loss.item()
        train_loader_tqdm.set_postfix({'Loss': f"{total_loss / i:.4f}"})

    writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)
    print(f'第{epoch}个epoch的loss: {total_loss / len(train_loader):.4f}')

    # EORSSD
    # writer.add_scalar('Loss/train', total_loss / 1400, epoch)
    # print('第{}个epoch的loss: {:.4f}'.format(epoch, total_loss/1400))

    # ors-4199
    # writer.add_scalar('Loss/train', total_loss / 2000, epoch)
    # print('第{}个epoch的loss: {:.4f}'.format(epoch, total_loss / 2000))

    save_path = 'save/RPV8/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + 'MCCNet_VGG.pth' + '.%d' % epoch)
    print("在第{}个epoch测试和评估模型".format(epoch))
    # 将模型切换到评估模式
    model.eval()

    # 初始化评估指标
    Sm_fun = M.Smeasure()
    Em_fun = M.Emeasure()
    FM_fun = M.Fmeasure_and_FNR()
    MAE_fun = M.MAE()

    # ✅ 新增：IoU 评估
    iou_list = []

    # 测试数据集配置
    test_image_root = './dataset/PV08/test/image/'
    test_gt_root = './dataset/PV08/test/gt/'

    test_loader = test_dataset(test_image_root, test_gt_root, opt.testsize)

    # 遍历测试数据集
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        image = image.cuda()

        with torch.no_grad():
            _, res = model(image)

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # 更新评估指标
        Sm_fun.step(pred=res, gt=gt)
        FM_fun.step(pred=res, gt=gt)
        Em_fun.step(pred=res, gt=gt)
        MAE_fun.step(pred=res, gt=gt)

        # ✅ 计算 IoU
        res_bin = (res >= 0.5).astype(np.float32)
        gt_bin = (gt >= 0.5).astype(np.float32)
        intersection = np.sum(res_bin * gt_bin)
        union = np.sum(res_bin) + np.sum(gt_bin) - intersection + 1e-8
        iou = intersection / union
        iou_list.append(iou)

    # 计算并输出评估结果
    sm = Sm_fun.get_results()['sm']
    fm = FM_fun.get_results()[0]['fm']['curve'].max()
    em = Em_fun.get_results()['em']['curve'].max()
    mae = MAE_fun.get_results()['mae']
    iou_mean = np.mean(iou_list)  # ✅ 平均 IoU

    print('第{}个epoch的S-measure: {:.4f}, F-measure: {:.4f}, E-measure: {:.4f}, MAE: {:.4f}, IoU: {:.4f}'.format(
        epoch, sm, fm, em, mae, iou_mean))

    # ✅ 将IoU也写入TensorBoard
    writer.add_scalars('Metrics/test', {
        'Sm': sm,
        'Fm': fm,
        'Em': em,
        'MAE': mae,
        'IoU': iou_mean
    }, epoch)

print("Let's go!")

if __name__ == '__main__':
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
