import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import cv2
import json

import utils.data_utils as util_
from . import data_augmentation
from models.clip import tokenize
import matplotlib.pyplot as plt
import IPython
import random
from datasets.ocid_data_loader import OCIDObject

NUM_VIEWS_PER_SCENE = 5 # 7

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2


###### Some utilities #####

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


############# Synthetic Tabletop Object Dataset #############

class Tabletop_Object_Dataset(Dataset):
    """ Data loader for Tabletop Object Dataset
    """
    def __init__(self, base_dir, train_or_test, config):
        self.base_dir = base_dir
        self.config = config
        self.train_or_test = train_or_test
        self.use_object_descriptions = config['use_object_descriptions']
        #  False if 'TOD' in self.base_dir else True #
        split_suffix = '{}_set'.format(train_or_test)

        # Get a list of all scenes
        if train_or_test == 'train':
            split_suffix = 'training_set'
        self.scene_dirs = sorted(glob.glob(os.path.join(self.base_dir, split_suffix) + '/*'))
        self.len = len(self.scene_dirs) * NUM_VIEWS_PER_SCENE
        
        self.name = 'TableTop'
        class_id = '03797390'
        # class_description_file = os.path.join(base_dir, 'shapenet_class_descriptions.txt'.format(class_id))
        class_description_file = os.path.join(base_dir, 'shapenet_class_{}.txt'.format(class_id))

        with open(class_description_file, 'r+') as f:
            self.object_description_list = f.readlines()
            
        self.shapenet_taxonomy = json.load(open(os.path.join(base_dir, 'shapenet_taxonomy.json')))

        if 'v6' in os.path.join(self.base_dir, split_suffix):
            global OBJECTS_LABEL
            OBJECTS_LABEL = 4

    def __len__(self):
        return self.len

    def process_rgb(self, rgb_img):
        """ Process RGB image
                - random color warping
        """
        rgb_img = rgb_img.astype(np.float32)

        if self.config['use_data_augmentation']:
            # rgb_img = data_augmentation.random_color_warp(rgb_img)
            pass
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

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # add random noise to depth
        if self.config['use_data_augmentation']:
            depth_img = data_augmentation.add_noise_to_depth(depth_img, self.config)
            # depth_img = data_augmentation.dropout_random_ellipses(depth_img, self.config)

        # Compute xyz ordered point cloud
        xyz_img = util_.compute_xyz(depth_img, self.config)
        if self.config['use_data_augmentation']:
            xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth_img, self.config)

        return xyz_img

    def sample_random_description(self, des):
        return random.sample(des.split(','), 1)[0]

    def get_scene_object_descriptions(self, object_descriptions):
        object_prompts = []
        valid_indexes = []
        filter_words = ['anonymous', 'challenge', '@']
        filter_word = lambda x: np.sum([w in x.lower() for w in filter_words]) == 0
        threshold_length = [3, 6]

        for idx in range(len(object_descriptions)):
            target_object_file = object_descriptions[idx]['mesh_filename'] 
            class_id, object_id = target_object_file.split('/')[3], target_object_file.split('/')[4]

            related_obj_desription = [l.split(',')[-2] for l in self.object_description_list if object_id in l]
            related_class_desription = [self.sample_random_description(info["name"])  for info in self.shapenet_taxonomy if class_id in info['synsetId']]
            # print("class description:", related_class_desription)
            # print("object description:", related_obj_desription)
            
            if not self.use_object_descriptions:
                if len(related_class_desription) > 0:
                    obj_prompt = related_class_desription[-1]
                    object_prompts.append(obj_prompt)
                    valid_indexes.append(idx)                    
            else:
                if len(related_obj_desription) > 0:
                    # filter 
                    obj_prompt = related_obj_desription[-1]
                    prompt_len = len(obj_prompt.split(' '))
                    if filter_word(obj_prompt) and prompt_len >= threshold_length[0] and prompt_len < threshold_length[1]:
                        object_prompts.append(obj_prompt)
                        valid_indexes.append(idx)
        
        # print(object_prompts)
        return object_prompts, valid_indexes

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        # Get scene directory
        scene_idx = idx // NUM_VIEWS_PER_SCENE
        scene_dir = self.scene_dirs[scene_idx]

        # Get view number
        view_num = idx % NUM_VIEWS_PER_SCENE

        # RGB image
        rgb_img_filename = os.path.join(scene_dir, f"rgb_{view_num:05d}.jpeg")
        # print(rgb_img_filename)
        rgb_img = cv2.imread(rgb_img_filename)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = self.process_rgb(rgb_img)

        # Depth image
        depth_img_filename = os.path.join(scene_dir, f"depth_{view_num:05d}.png")
        depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
        xyz_img = self.process_depth(depth_img)

        # Labels
        foreground_labels_filename = os.path.join(scene_dir, f"segmentation_{view_num:05d}.png")
        foreground_labels = util_.imread_indexed(foreground_labels_filename)
        scene_description_filename = os.path.join(scene_dir, "scene_description.txt")
        scene_description = json.load(open(scene_description_filename))
        scene_description['view_num'] = view_num
        object_descriptions = scene_description['object_descriptions'] 
        label_abs_path = '/'.join(foreground_labels_filename.split('/')[-2:]) # Used for evaluation

        # Turn these all into torch tensors
        rgb_img = data_augmentation.array_to_tensor(rgb_img) # Shape: [3 x H x W]
        xyz_img = data_augmentation.array_to_tensor(xyz_img) # Shape: [3 x H x W]
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels) # Shape: [H x W]
        
        ############### language prompt and mask labels #####################
        # get shapenet object descriptions from shapenet org
        target_object_classname = ''
        target_object_label = ''
        object_num = len(object_descriptions)
        base_idx = 2
        obj_prompts, valid_indices = self.get_scene_object_descriptions(object_descriptions)
        assert len(obj_prompts) == len(valid_indices)
        c = np.random.randint(len(valid_indices))
        target_object_idx = valid_indices[c]
        target_name = obj_prompts[c]

        target_label_mask = foreground_labels == target_object_idx + base_idx
        for i in range(len(valid_indices)):
            if obj_prompts[i] == target_name:
                target_label_mask |= foreground_labels == valid_indices[i] + base_idx

        target_label_mask = target_label_mask.float()
        prompt = 'a {} on a table in a simulation engine'.format(target_name)
        # print("prompt:", prompt)

        if not target_label_mask.sum() > 100 * 100: # hardcoded pixel numer
            # regenerate sample
            return self[np.random.randint(self.len)]
        VIS = False

        if VIS:        
            fig = plt.figure(figsize=(12.8 * 3, 4.8 * 3))
            fig.suptitle('scene: {} prompt: {}'.format(scene_idx, prompt), fontsize=30)
            ax = fig.add_subplot(1, 3, 1)
            plt.imshow((data_augmentation.unstandardize_image(rgb_img.T.numpy())).transpose(1,0,2))
            ax = fig.add_subplot(1, 3, 2)
            plt.imshow(xyz_img.T.numpy().transpose(1,0,2))
            ax = fig.add_subplot(1, 3, 3)
            mask = target_label_mask.numpy()
            plt.imshow(mask)
            plt.show()    
        tokenize_prompt = tokenize([prompt]).detach().numpy()[0]

        return {'rgb' : rgb_img,
                'xyz' : xyz_img,
                'foreground_labels' : foreground_labels,
                'target_labels': target_label_mask,
                'scene_dir' : scene_dir,
                'prompt': prompt,
                'view_num' : view_num,
                'label_abs_path' : label_abs_path,
                'tokenize_prompt': tokenize_prompt,                
                }


def get_OCID_train_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = OCIDObject('train', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)

def get_OCID_test_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = OCIDObject('test', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)

def get_TOD_train_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = Tabletop_Object_Dataset(base_dir, 'train', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)


def get_TOD_test_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=False):

    config = config.copy()
    dataset = Tabletop_Object_Dataset(base_dir, 'test', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)




############# RGB Images Dataset (Google Open Images) #############

class RGB_Objects_Dataset(Dataset):
    """ Data loader for Tabletop Object Dataset
    """


    def __init__(self, base_dir, start_list_file, train_or_test, config):
        self.base_dir = base_dir
        self.config = config
        self.train_or_test = train_or_test

        # Get a list of all instance labels
        f = open(base_dir + start_list_file)
        lines = [x.strip() for x in f.readlines()]
        self.starts = lines
        self.len = len(self.starts)

        self.name = 'RGB_Objects'

    def __len__(self):
        return self.len

    def pad_crop_resize(self, img, morphed_label, label):
        """ Crop the image around the label mask, then resize to 224x224
        """

        H, W, _ = img.shape

        # Get tight box around label/morphed label
        x_min, y_min, x_max, y_max = util_.mask_to_tight_box(label)
        _xmin, _ymin, _xmax, _ymax = util_.mask_to_tight_box(morphed_label)
        x_min = min(x_min, _xmin); y_min = min(y_min, _ymin); x_max = max(x_max, _xmax); y_max = max(y_max, _ymax)

        # Make bbox square
        x_delta = x_max - x_min
        y_delta = y_max - y_min
        if x_delta > y_delta:
            y_max = y_min + x_delta
        else:
            x_max = x_min + y_delta

        sidelength = x_max - x_min
        padding_percentage = np.random.beta(self.config['padding_alpha'], self.config['padding_beta'])
        padding_percentage = max(padding_percentage, self.config['min_padding_percentage'])
        padding = int(round(sidelength * padding_percentage))
        if padding == 0:
            print(f'Whoa, padding is 0... sidelength: {sidelength}, %: {padding_percentage}')
            padding = 25 # just make it 25 pixels

        # Pad and be careful of boundaries
        x_min = max(x_min - padding, 0)
        x_max = min(x_max + padding, W-1)
        y_min = max(y_min - padding, 0)
        y_max = min(y_max + padding, H-1)

        # Crop
        if (y_min == y_max) or (x_min == x_max):
            print('Fuck... something is wrong:', x_min, y_min, x_max, y_max)
            print(morphed_label)
            print(label)
        img_crop = img[y_min:y_max+1, x_min:x_max+1]
        morphed_label_crop = morphed_label[y_min:y_max+1, x_min:x_max+1]
        label_crop = label[y_min:y_max+1, x_min:x_max+1]

        # Resize
        img_crop = cv2.resize(img_crop, (224,224))
        morphed_label_crop = cv2.resize(morphed_label_crop, (224,224))
        label_crop = cv2.resize(label_crop, (224,224))

        return img_crop, morphed_label_crop, label_crop

    def transform(self, img, label):
        """ Process RGB image
                - standardize_image
                - random color warping
                - random horizontal flipping
        """

        img = img.astype(np.float32)

        # Data augmentation for mask
        morphed_label = label.copy()
        if self.config['use_data_augmentation']:
            if np.random.rand() < self.config['rate_of_morphological_transform']:
                morphed_label = data_augmentation.random_morphological_transform(morphed_label, self.config)
            if np.random.rand() < self.config['rate_of_translation']:
                morphed_label = data_augmentation.random_translation(morphed_label, self.config)
            if np.random.rand() < self.config['rate_of_rotation']:
                morphed_label = data_augmentation.random_rotation(morphed_label, self.config)

            sample = np.random.rand()
            if sample < self.config['rate_of_label_adding']:
                morphed_label = data_augmentation.random_add(morphed_label, self.config)
            elif sample < self.config['rate_of_label_adding'] + self.config['rate_of_label_cutting']:
                morphed_label = data_augmentation.random_cut(morphed_label, self.config)
                
            if np.random.rand() < self.config['rate_of_ellipses']:
                morphed_label = data_augmentation.random_ellipses(morphed_label, self.config)

        # Next, crop the mask with some padding, and resize to 224x224. Make sure to preserve the aspect ratio
        img_crop, morphed_label_crop, label_crop = self.pad_crop_resize(img, morphed_label, label)

        # Data augmentation for RGB
        # if self.config['use_data_augmentation']:
        #     img_crop = data_augmentation.random_color_warp(img_crop)
        img_crop = data_augmentation.standardize_image(img_crop)

        # Turn into torch tensors
        img_crop = data_augmentation.array_to_tensor(img_crop) # Shape: [3 x H x W]
        morphed_label_crop = data_augmentation.array_to_tensor(morphed_label_crop) # Shape: [H x W]
        label_crop = data_augmentation.array_to_tensor(label_crop) # Shape: [H x W]

        return img_crop, morphed_label_crop, label_crop

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        # Get label filename
        label_filename = self.starts[idx]

        label = cv2.imread(str(os.path.join(self.base_dir, 'Labels', label_filename))) # Shape: [H x W x 3]
        label = label[..., 0] == 255 # Turn it into a {0,1} binary mask with shape: [H x W]
        label = label.astype(np.uint8)

        # find corresponding image file
        img_file = label_filename.split('_')[0] + '.jpg'
        img = cv2.imread(str(os.path.join(self.base_dir, 'Images', img_file)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # These might not be the same size. resize them to the smaller one
        if label.shape[0] < img.shape[0]:
            new_size = label.shape[::-1] # (W, H)
        else:
            new_size = img.shape[:2][::-1]
        label = cv2.resize(label, new_size)
        img = cv2.resize(img, new_size)

        img_crop, morphed_label_crop, label_crop = self.transform(img, label)

        return {
            'rgb' : img_crop,
            'initial_masks' : morphed_label_crop,
            'labels' : label_crop
        }

def get_RGBO_train_dataloader(base_dir, config, batch_size=8, num_workers=4, shuffle=True):

    dataset = RGB_Objects_Dataset(base_dir, config['starts_file'], 'train', config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)


if __name__ == '__main__':
    test_dataloader = get_TOD_train_dataloader('test_data', config, batch_size=1, num_workers=0)
