import os
from time import time

import numpy as np
import matplotlib.pyplot as plt

import torch

# My libraries
import datasets.data_loader as data_loader
import utils.data_utils as data_utils
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU
from models.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
import torch.nn as nn
import argparse
from tqdm import tqdm
import json
from tensorboardX import SummaryWriter
import datetime
from models.loss import BCEWithLogitsLossWeighted

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--save_rate', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--model_suffix', type=str, default='')
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--load_pretrained', action='store_true')

args = parser.parse_args()

args.model_suffix += args.config_name
with open(os.path.join('configs', args.config_name + '.json')) as f:
    data_loading_params = json.load(f)

if 'OCID' in data_loading_params['dataset']: 
    dl = data_loader.get_OCID_train_dataloader(data_loading_params['dataset'], data_loading_params, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
else:
    dl = data_loader.get_TOD_train_dataloader(data_loading_params['dataset'], data_loading_params, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
loss_func = BCEWithLogitsLossWeighted(weighted=True) # torch.nn.BCEWithLogitsLoss(reduction='none')
net = TwoStreamAttentionLangFusionLat().cuda()
params_exclude_clip = net.params_exclude_clip
net = torch.nn.DataParallel(net)
optimizer = torch.optim.Adam(params_exclude_clip, lr=args.lr)
total_losses = data_utils.AverageMeter()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3, 5, 7], gamma=0.5)
total_iter = 0
logdir = 'tensorboard/{}_{}'.format(args.model_suffix, datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
mkdir_if_missing(logdir)
print("logdir:", logdir)

tb_logger = SummaryWriter(logdir=logdir)
mkdir_if_missing('output')

# train
epoch_num = args.epoch
for n in range(epoch_num):
    for data in tqdm(dl):
        foreground_labels = data['target_labels'] # Shape: [N x H x W]
        N, H, W = foreground_labels.shape
        x = torch.cat((data['rgb'], data['xyz']), dim=1)
        x = torch.nn.functional.interpolate(x, (240, 320))
        fg_logits  = net(x.cuda(), data['tokenize_prompt'].cuda())
        ### Foreground Loss ###

        foreground_labels = torch.nn.functional.interpolate(foreground_labels[:,None], (240, 320)) 
        fg_masks = foreground_labels.clamp(0,2).cuda() 
        fg_loss = loss_func(fg_logits, fg_masks).mean()
        optimizer.zero_grad()
        loss = fg_loss
        loss.backward()
        optimizer.step()
        total_losses.update(loss.item())
        print('epoch: {} iter: {} loss: {}'.format(n, total_iter, total_losses.avg))
        total_iter += 1
        
        if total_iter % 10 == 0:
            tb_logger.add_scalar('bce_loss', total_losses.avg, total_iter)
            # also write image
        if total_iter % 100 == 0:
            tb_logger.add_image("gt_image", data_utils.torch_to_numpy(data['rgb'], is_standardized_image=True)[0], total_iter, dataformats='HWC')            
            tb_logger.add_image("gt_depth", data_utils.torch_to_numpy(data['xyz'])[0], total_iter, dataformats='HWC')
            tb_logger.add_image("gt_mask", fg_masks[0], total_iter, dataformats='CHW')
            tb_logger.add_image("pred_mask", torch.sigmoid(fg_logits[0]), total_iter, dataformats='CHW')

        if total_iter % args.save_rate == 0:
            filename = "output/model_{}.pth".format(args.model_suffix)
            checkpoint = {'model' : net.state_dict()}
            torch.save(checkpoint, filename)
    scheduler.step()
    print("epoch: {}".format(n))




# test
