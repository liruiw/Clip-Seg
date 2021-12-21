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
import argparse
from models.clip import tokenize
from datasets import data_augmentation
import cv2

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--model_suffix', type=str, default='')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--mask_thre', type=float, default=0.1)
parser.add_argument('--config_name', type=str, default='')
parser.add_argument('--input_file', type=str, default='')

args = parser.parse_args()
with open(os.path.join('configs', args.config_name + '.json')) as f:
    data_loading_params = json.load(f)

filename = "output/model_{}.pth".format(args.model_suffix + args.config_name)
print("testing model:", filename)

checkpoint = torch.load(filename) # , map_location=torch.device('cpu')
net = TwoStreamAttentionLangFusionLat()
net.load_state_dict({k.replace('module.', ''): v for k,v in checkpoint['model'].items()})
net = torch.nn.DataParallel(net).cuda()

net.eval()
mkdir_if_missing('output/output_figures_{}'.format(args.model_suffix))

if len(args.config_name) > 0:
    args.model_suffix = args.config_name
    with open(os.path.join('configs', args.config_name + '.json')) as f:
        data_loading_params = json.load(f)
        

# dataset
if 'OCID' in data_loading_params['dataset']:
    dl = data_loader.get_OCID_test_dataloader(data_loading_params['dataset'],
                                               data_loading_params,
                                               batch_size=1,
                                               num_workers=0, shuffle=True)
else:
    dl = data_loader.get_TOD_test_dataloader(data_loading_params['dataset'], data_loading_params,
                                              batch_size=1,
                                              num_workers=0, shuffle=True)

dl_iter = dl.__iter__()
for idx in range(args.test_num):
    batch = next(dl_iter)

    if len(args.input_file) > 0:
        print("load image: {}!!".format(args.input_file))
        ycb_intrinsics_path = '/home/liruiw/Projects/BlenderProc/object_models/ycbv/camera_uw.json'
        # import json
        # mat = json.load(ycb_intrinsics_path)[]
        intrinsics = np.array([1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]).reshape([3, 3])
        color_path = os.path.join('test_input', 'color_{}.png'.format(args.input_file))
        depth_path = os.path.join('test_input', 'depth_{}.png'.format(args.input_file))
        
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 5000. # This reads a 16-bit single-channel image. Shape: [H x W]
        print("depth mean:", depth_img.mean())
        color_img = cv2.imread(color_path)        
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        xyz_img = data_utils.backproject(depth_img, intrinsics, 1.)  

        rgb_img = dl.dataset.process_rgb(color_img)
        batch['rgb'] = data_augmentation.array_to_tensor(rgb_img)[None]
        batch['xyz'] = data_augmentation.array_to_tensor(xyz_img)[None]
    
    rgb_imgs = data_utils.torch_to_numpy(batch['rgb'], is_standardized_image=True) # Shape: [N x H x W x 3]
    xyz_imgs = data_utils.torch_to_numpy(batch['xyz']) # Shape: [N x H x W x 3]
    foreground_labels = data_utils.torch_to_numpy(batch['foreground_labels']) # Shape: [N x H x W]
    
    for _ in range(5):
        # center_offset_labels = util_.torch_to_numpy(batch['center_offset_labels']) # Shape: [N x 2 x H x W]
        N, H, W = foreground_labels.shape[:3]
        print("Number of images: {0}".format(N))
         
        ### Compute segmentation masks ###
        data = batch
        st_time = time()

        # Plot image
        fig = plt.figure(1);
        fig.set_size_inches(5,5)
        plt.subplot(1,1,1)
        plt.imshow(rgb_imgs[0,...].astype(np.uint8))
        plt.title('Image')
        plt.show()

        # print("original prompts:", batch['prompt'])
        prompt = input("please input prompt: ")
        if prompt == 'q':
            break
        data['tokenize_prompt'][0] = tokenize([prompt])
        batch['prompt'][0] = prompt

        x = torch.cat((batch['rgb'], batch['xyz']), dim=1)
        x = torch.nn.functional.interpolate(x, (240, 320))

        with torch.no_grad():
            fg_logits  = net(x.cuda(), data['tokenize_prompt'].cuda())
        fg_logits = torch.sigmoid(fg_logits).detach().cpu().numpy()

        # Get segmentation masks in numpy
        fig_index = 1
        prompts = batch['prompt']

        for i in range(N):
            fig = plt.figure(fig_index); fig_index += 1
            mask_pred = (fg_logits > args.mask_thre)[:, 0]
            # print(np.sum((mask_pred & (target_labels > 0)).astype(np.float32)), np.sum((mask_pred | (target_labels > 0)).astype(np.float32)))
            fig.suptitle('prompt: {}'.format(prompts[i]), fontsize=30)
            fig.set_size_inches(20,5)

            # Plot image
            plt.subplot(1,3,1)
            plt.imshow(rgb_imgs[i,...].astype(np.uint8))
            plt.title('Image {0}'.format(i+1))

            # Plot Depth
            plt.subplot(1,3,2)
            plt.imshow(xyz_imgs[i,...,2])
            plt.title('Depth')

            # Plot prediction
            plt.subplot(1, 3, 3)
            resized_imgs = data_utils.torch_to_numpy(x[:, :3], is_standardized_image=True)
            plt.imshow((resized_imgs * mask_pred[..., None] + resized_imgs * (1 - mask_pred[..., None]) * 0.2)[i].astype(np.uint8)) # 0.9
            plt.title("Predicted Masks")

            plt.tight_layout()
            plt.show()
     