import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import imageio

from model.RMGTANet_models import MCCNet_VGG
from data import test_dataset
import pytorch_iou  # 用于计算 IOU

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--iou_path', type=str, default='output/', help='IOU 保存路径')
parser.add_argument('--iou_file', type=str, default='OPV.txt', help='IOU 文件名')
opt = parser.parse_args()

# 数据集路径
dataset_path = './dataset/PV08/test/'

# 模型
model = MCCNet_VGG()
model.load_state_dict(
    {k: v for k, v in torch.load('save/RPV8/MCCNet_VGG.pth.49', map_location='cpu').items()
     if not any(s in k for s in ['total_ops', 'total_params'])},
    strict=False
)
model.cuda()
model.eval()

# 测试
test_datasets = ['PV']

IOU_metric = pytorch_iou.IOU(size_average=True)

for dataset in test_datasets:
    save_path = './results/VGG/' + dataset + '/'
    os.makedirs(save_path, exist_ok=True)

    image_root = dataset_path + '/image/'
    gt_root = dataset_path + '/gt/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    # IOU 保存路径
    os.makedirs(opt.iou_path, exist_ok=True)
    iou_file_path = os.path.join(opt.iou_path, opt.iou_file)

    time_sum = 0
    with open(iou_file_path, 'w') as f_iou:
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)  # 归一化

            image = image.cuda()
            time_start = time.time()
            _, res = model(image)
            time_end = time.time()
            time_sum += (time_end - time_start)

            # 调整到 GT 尺寸
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res_sig = res.sigmoid().data.cpu().numpy().squeeze()
            res_sig = (res_sig - res_sig.min()) / (res_sig.max() - res_sig.min() + 1e-8)

            # 二值化保存
            threshold = 0.5
            binary_res = (res_sig > threshold).astype(np.uint8) * 255
            imageio.imsave(save_path + name, binary_res)

            # 计算 IOU 并写入文件
            pred_tensor = torch.from_numpy(res_sig).unsqueeze(0).unsqueeze(0).cuda()
            gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).cuda()
            iou_val = IOU_metric(pred_tensor, gt_tensor).item()
            f_iou.write(f"{name} {iou_val:.6f}\n")

            if (i + 1) % 20 == 0 or i == test_loader.size - 1:
                print(f"[{i+1}/{test_loader.size}] {name} IOU: {iou_val:.4f}")

        print('Average running time: {:.5f}s, FPS: {:.4f}'.format(time_sum / test_loader.size, test_loader.size / time_sum))
        print(f"IOU results saved to {iou_file_path}")
