import os
from time import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import json
# My libraries
import datasets.data_loader as data_loader
import utils.data_utils as data_utils
from models.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
import IPython
from tqdm import trange
import argparse
from models.loss import BCEWithLogitsLossWeighted


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--model_suffix', type=str, required=True)
parser.add_argument('--test_num', type=int, default=1200)
parser.add_argument('--mask_thre', type=float, default=0.5)
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--num_workers', type=int, default=12)

args = parser.parse_args()

args.model_suffix += args.config_name
with open(os.path.join('configs', args.config_name + '.json')) as f:
    data_loading_params = json.load(f)

# dataset
if 'OCID' in data_loading_params['dataset']:
    dl_test = data_loader.get_OCID_test_dataloader(data_loading_params['dataset'],
                                                   data_loading_params, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True)
    dl_train = data_loader.get_OCID_train_dataloader(data_loading_params['dataset'],
                                                     data_loading_params,
                                                     batch_size=args.batch_size,
                                                     num_workers=args.num_workers, shuffle=True)
else:
    dl_test = data_loader.get_TOD_test_dataloader(data_loading_params['dataset'], data_loading_params,
                                                  batch_size=args.batch_size,
                                                  num_workers=8, shuffle=True)
    dl_train = data_loader.get_TOD_train_dataloader(data_loading_params['dataset'], data_loading_params,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers, shuffle=True)

ckpt_path = "output/model_{}.pth".format(args.model_suffix)
checkpoint = torch.load(ckpt_path)  # , map_location=torch.device('cpu')
net = TwoStreamAttentionLangFusionLat()
net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
net = torch.nn.DataParallel(net).cuda()
loss_func = BCEWithLogitsLossWeighted(weighted=True)
net.eval()
eval_log_dir = 'output/eval_{}'.format(args.model_suffix)
mkdir_if_missing(eval_log_dir)


def benchmark(dataloader, log_dir, split):
    dl_iter = dataloader.__iter__()
    iou_meter = data_utils.AverageMeter()
    loss_meter = data_utils.AverageMeter()

    fig_index = 0
    for idx in trange(args.test_num // args.batch_size):
        data = next(dl_iter)
        rgb_imgs = data_utils.torch_to_numpy(data['rgb'], is_standardized_image=True)  # Shape: [N x H x W x 3]
        xyz_imgs = data_utils.torch_to_numpy(data['xyz'])  # Shape: [N x H x W x 3]
        foreground_labels = data['foreground_labels']  # Shape: [N x H x W]
        data['target_labels'] = torch.nn.functional.interpolate(data['target_labels'][:, None].cuda(), (240, 320))[:, 0]
        target_labels = data_utils.torch_to_numpy(data['target_labels'])  # Shape: [N x H x W]

        # center_offset_labels = util_.torch_to_numpy(batch['center_offset_labels']) # Shape: [N x 2 x H x W]
        N, H, W = foreground_labels.shape[:3]

        ### Compute segmentation masks ###
        x = torch.cat((data['rgb'], data['xyz']), dim=1)
        x = torch.nn.functional.interpolate(x, (240, 320))

        with torch.no_grad():
            fg_logits = net(x.cuda(), data['tokenize_prompt'].cuda())
            foreground_labels = torch.nn.functional.interpolate(foreground_labels[:, None], (240, 320))
            fg_masks = foreground_labels.clamp(0, 2).cuda()
            fg_loss = loss_func(fg_logits, fg_masks)
            mask_pred = (fg_logits > args.mask_thre)[:, 0]
            iou = torch.sum((mask_pred & (data['target_labels'] > 0)), (1, 2)) / torch.sum(
                (mask_pred | (data['target_labels'] > 0)), (1, 2))
            if torch.isnan(iou).sum() > 0:
                iou = torch.zeros_like(iou)
                continue
                
        prompts = data['prompt']

        for i in range(N):
            print(iou[i].item())
            iou_meter.update(iou[i].item())
            loss_meter.update(fg_loss[i].item())

            if idx % 5 == 0:
                fig_index += 1
                fig = plt.figure(fig_index)
                fig.suptitle('prompt: {}'.format(prompts[i]), fontsize=30)
                fig.set_size_inches(20, 5)

                # Plot image
                plt.subplot(1, 4, 1)
                plt.imshow(rgb_imgs[i, ...].astype(np.uint8))
                plt.title('Image {0}'.format(i + 1))

                # Plot Depth
                plt.subplot(1, 4, 2)
                plt.imshow(xyz_imgs[i, ..., 2])
                plt.title('Depth')

                # Plot prediction
                plt.subplot(1, 4, 3)
                # plt.imshow(fg_logits[i,...][0] > 0.9)
                resized_imgs = data_utils.torch_to_numpy(x[:, :3], is_standardized_image=True)
                plt.imshow(
                    (resized_imgs * mask_pred[..., None] + resized_imgs * (1 - mask_pred[..., None]) * 0.2)[i].astype(
                        np.uint8))  # 0.9

                plt.title("Predicted Masks: IOU: {:.6f}".format(iou[i].item()))

                # Plot Initial Masks
                plt.subplot(1, 4, 4)
                plt.imshow(data_utils.get_color_mask(target_labels[i, ...]))
                plt.title(f"Target Mask ")
                plt.tight_layout()

                plt.savefig(os.path.join(log_dir, '{}_{}.png'.format(split, fig_index)))
                plt.close()

    stats = dict(
        iou=iou_meter.avg,
        loss=loss_meter.avg,
        mask_threshold=args.mask_thre
    )

    with open(os.path.join(eval_log_dir, '{}.json'.format(split)), 'a+') as outfile:
        json.dump(stats, outfile)


benchmark(dl_train, eval_log_dir, 'train')
benchmark(dl_test, eval_log_dir, 'test')
