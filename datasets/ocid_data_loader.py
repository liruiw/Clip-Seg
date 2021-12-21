# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.utils.data as data
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# import pcl
from pathlib import Path
import utils.data_utils as util_
import IPython
from . import data_augmentation
from models.clip import tokenize
import json


def chromatic_transform(im, label=None, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
    if d_l is None:
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)

    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]
    return new_im


def add_noise(image, level=0.1):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.9:
        row, col, ch = image.shape
        mean = 0
        noise_level = random.uniform(0, level)
        sigma = np.random.rand(1) * noise_level * 256
        gauss = sigma * np.random.randn(row, col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy.astype('uint8')


class OCIDObject(data.Dataset):
    def __init__(self, image_set, config, ocid_object_path=None):
        self.mode = image_set
        self.config = config
        self._name = 'ocid_object_' + image_set
        self._image_set = image_set
        self._ocid_object_path = self._get_default_path() if ocid_object_path is None \
            else ocid_object_path

        self._pixel_mean = torch.tensor(np.array([[[102.9801, 115.9465, 122.7717]]]) / 255.0).float()
        self._width = 640
        self._height = 480

        print("load language...")
        self.language_prompts = json.load(
            open('{}_expressions.json'.format(os.path.join(self._ocid_object_path, image_set))))

        self.test_scene = ['ARID20/table/bottom/seq07', 'ARID20/table/top', 'ARID20/table/bottom/seq09',
                           'ARID10/table/top/fruits', 'YCB10/floor/curved', 'YCB10/floor/top/mixed/seq08']

        self.setup_language_image_path()
        print("image num: {} seq num: {} prompt num: {}".format(len(self.img_set), len(self.seq_set),
                                                                len(self.target_instance_idx)))

        # get all the prompt corresponding paths and

        # print('%d images for dataset %s' % (len(self.image_paths), self._name))
        self._size = len(self.language_prompt_list)
        assert os.path.exists(self._ocid_object_path), \
            'ocid_object path does not exist: {}'.format(self._ocid_object_path)

    def setup_language_image_path(self):
        self.scene_paths = []
        self.references_list = []
        self.target_instance_idx = []
        self.target_class_idx = []
        self.target_class_name = []
        self.language_prompt_list = []
        self.img_set = set()
        self.seq_set = set()

        for key, val in self.language_prompts.items():
            num_of_the = sum(['the' in s for s in val['tokens']])
            # print("number of the appears: ", num_of_the)
            if num_of_the < 2:
                test_scene_cnt = np.sum([prefix in val['sequence_path'] for prefix in self.test_scene])
                # print(val['sequence_path'], test_scene_cnt)
                # if (self.mode == 'train' and test_scene_cnt == 0) or (self.mode == 'test' and test_scene_cnt > 0):
                if (self.mode == 'train') or (self.mode == 'test'):
                    self.references_list.append(val['sentence'])
                    self.scene_paths.append(os.path.join(self._ocid_object_path, val['scene_path']))
                    self.target_instance_idx.append(val['scene_instance_id'])
                    self.target_class_name.append(val)
                    self.language_prompt_list.append(val)
                    self.img_set.add(val['scene_path'])
                    self.seq_set.add(val['sequence_path'])

                    # IPython.embed()

            # filter relations
            # sentence_list = val['sentence'].lower().split(' ')
            # print(val.keys())
            # IPython.embed()
            # print(val['tokens'])
            # IPython.embed()
            # self.target_class_idx.append(val['class_instance'])

        # {
        #   "[sentence_id]": {
        #     "seq_id": A sequence ID that the current scene(image) belongs to in our pre-defined seq_list.txt. ,
        #     "scene_id": A scene ID representing an image in our pre-defined scene_list.txt. ,
        #     "take_id": A take ID representing the order of the image being taken in the sequence. ,
        #     "scene_path": A relative image path (usage: ~/OCID-Dataset/[scene_path]). ,
        #     "sequence_path": A relative sequence path rooted from the (usage: ~/OCID-Dataset/[sequence_path]). ,
        #     "sub_dataset": A metadata marked for the image split from the original OCID dataset. ([ARID10/ARID20/YCB10]) ,
        #     "instance_id": An instance ID representing an object instance over the whole dataset. ,
        #     "scene_instance": An instance ID representing an object instance over current scene(image). ,
        #     "class": A class name for the object. ,
        #     "class_instance": An instance name related to class of the object in current scene. ,
        #     "new_class": A class name for the object. ,
        #     "sentence": A referring expression corresponded to the object. ,
        #     "tokens": An array of tokens split from sentence. ,
        #     "bbox": A bounding box for the object in format of (x, y, w, h). ,
        #     "sentence_type": The type of sentence generation method used.
        #   },

    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels

    def process_rgb(self, rgb_img):
        """ Process RGB image
                - random color warping
        """
        rgb_img = rgb_img.astype(np.float32)

        if self.config['use_data_augmentation']:
            # rgb_img = data_augmentation.random_color_warp(rgb_img)
            # pass
            # if self.mode == 'train' and np.random.rand(1) > 0.1:
            #     rgb_img = chromatic_transform(rgb_img)
            if self.mode == 'train' and np.random.rand(1) > 0.1:
                rgb_img = add_noise(rgb_img)

        rgb_img = data_augmentation.standardize_image(rgb_img)

        return rgb_img

    def process_depth(self, depth_img):
        """ Process depth channel
                TODO: CHANGE THIS
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """
        # IPython.embed()
        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # # add random noise to depth

        if self.config['use_data_augmentation'] and self.mode == 'train':
            depth_img = data_augmentation.add_noise_to_depth(depth_img, self.config)
            depth_img = data_augmentation.dropout_random_ellipses(depth_img, self.config)

        # Compute xyz ordered point cloud
        xyz_img = util_.compute_xyz(depth_img, self.config)
        if self.config['use_data_augmentation'] and self.mode == 'train':
            xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth_img, self.config)

        return xyz_img

    def __len__(self):
        return self._size

    def __getitem__(self, idx):

        # BGR image
        filename = str(self.scene_paths[idx])

        rgb_img_np = cv2.imread(filename)
        # if self.mode == 'train' and np.random.rand(1) > 0.1:
        #     rgb_img_np = chromatic_transform(rgb_img_np)
        # if self.mode == 'train' and np.random.rand(1) > 0.1:
        #     rgb_img_np = add_noise(rgb_img_np)

        rgb_img_np = cv2.cvtColor(rgb_img_np, cv2.COLOR_BGR2RGB)

        # Label
        labels_filename = filename.replace('rgb', 'label')
        foreground_labels = util_.imread_indexed(labels_filename)

        # mask table as background
        foreground_labels[foreground_labels == 1] = 0
        if 'table' in labels_filename:
            foreground_labels[foreground_labels == 2] = 0
        foreground_labels_np = self.process_label(foreground_labels)
        label_blob = torch.from_numpy(foreground_labels).unsqueeze(0)
        index = filename.find('OCID')

        # Depth image
        depth_filename = filename.replace('rgb', 'depth')
        depth_img = cv2.imread(depth_filename,
                               cv2.IMREAD_ANYDEPTH)  # This reads a 16-bit single-channel image. Shape: [H x W]

        # TODO: filter depth imag == 1
        xyz_img = self.process_depth(depth_img)
        rgb_img = self.process_rgb(rgb_img_np)
        rgb_img = data_augmentation.array_to_tensor(rgb_img)  # Shape: [3 x H x W]
        xyz_img = data_augmentation.array_to_tensor(xyz_img)  # Shape: [3 x H x W]
        if (xyz_img[0] == 1.).sum() > 100000:  # bad depth
            return self[np.random.randint(self._size)]
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels_np)  # Shape: [H x W]
        target_object_idx = self.target_instance_idx[idx]
        # plus 2 if table and plus 1 if floor
        base_idx = 1 if 'floor' in filename else 2
        target_label_mask = (foreground_labels == target_object_idx - base_idx).float()
        prompt = self.references_list[idx].replace('_', ' ').replace('-', ' ')

        # add a picture of
        # prompt = 'A picture of ' + prompt.replace('The', 'a')
        VIS = False

        if VIS:
            fig = plt.figure(figsize=(12.8 * 4, 4.8 * 3))
            fig.suptitle('prompt: {}'.format(prompt), fontsize=30)
            ax = fig.add_subplot(1, 4, 1)
            plt.imshow((data_augmentation.unstandardize_image(rgb_img.T.numpy())).transpose(1, 0, 2))
            ax = fig.add_subplot(1, 4, 2)
            plt.imshow(xyz_img.T.numpy().transpose(1, 0, 2))
            ax = fig.add_subplot(1, 4, 3)
            plt.imshow(target_label_mask)
            ax = fig.add_subplot(1, 4, 4)
            plt.imshow(foreground_labels_np)
            # plt.show()
            plt.savefig('output/dataset/image_{}.png'.format(idx))

        tokenize_prompt = tokenize([prompt]).detach().numpy()[0]

        return {'rgb': rgb_img,
                'xyz': xyz_img,
                'foreground_labels': foreground_labels,
                'target_labels': target_label_mask,
                'prompt': prompt,
                'tokenize_prompt': tokenize_prompt,
                }

    def _get_default_path(self):
        """
        Return the default path where ocid_object is expected to be installed.
        """
        return os.path.join('test_data', 'OCID', 'OCID-dataset')