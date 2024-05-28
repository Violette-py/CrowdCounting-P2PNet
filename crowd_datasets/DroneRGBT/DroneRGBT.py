import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

class DroneRGBT(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None, train=False, patch=False, flip=False):
        # self.root_path = data_root
        
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        
        self.img_map = {}
        self.img_list = []
        
        img_paths = [filename for filename in os.listdir(img_dir) if filename.endswith('.jpg')]
        for filename in img_paths:
            # TODO: RGB & TIR
            # self.img_map[os.path.join(img_dir, filename)] = os.path.join(gt_dir, f"{filename.split('.')[0]}.txt") # tir
            self.img_map[os.path.join(img_dir, filename)] = os.path.join(gt_dir, f"{filename.split('.')[0]}R.txt")  # rgb
        self.img_list = sorted(list(self.img_map.keys()))
        
        """
        # 每次随机划分训练集和验证集
        
        self.img_dir_list = img_dirs.split('|')
        self.gt_dir_list = gt_dirs.split('|')
        
        if len(self.img_dir_list) != len(self.gt_dir_list):
            raise Exception('img_dirs和gt_dirs长度不相等')
        
        self.img_map = {}
        self.img_list = []
        for i, img_dir in enumerate(self.img_dir_list):
            img_paths = [filename for filename in os.listdir(
                img_dir) if filename.endswith('.jpg')]
            for filename in img_paths:
                self.img_map[os.path.join(img_dir, filename)] = os.path.join(self.gt_dir_list[i], f"{filename.split('.')[0]}R.txt")
        self.img_list = sorted(list(self.img_map.keys()))
        """
        
        """
        # baseline
        # TODO: 生成list文件并修改此处路径
        # TODO: 采用交叉验证，每次以不同方式划分数据集，动态生成两个list文件
        # self.train_lists = "shanghai_tech_part_a_train.list"
        # self.eval_list = "shanghai_tech_part_a_test.list"
        self.train_lists ="DroneRGBT_train.txt"
        self.eval_list = "DroneRGBT_test.txt"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}  # img_map[img_path] = gt_path
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:  # list文件放在 DATA_ROOT 下
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))  # 将字典键值排序后，存储在列表中
        
        """
        
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip
        
        self.count = 0  # NOTE

    def __len__(self):
        return self.nSamples
    
    def remove_n_points(self, points, n):
        if len(points) <= n:  # 如果列表为空或n不大于0，直接返回原列表
            raise Exception('points被删没了')
        # 随机打乱嵌套列表
        random.shuffle(points)
        # 移除前n个子列表
        points = points[n:]
        return points

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]  # 绝对路径
        gt_path = self.img_map[img_path] 
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        
        # NOTE: 随机mask
        # point_num = len(point)
        # if point_num > 30 and point_num < 60:
            # point = self.remove_n_points(point, int(point_num * 0.1))  
            # print(f"图片 {img_path} 点数被降为 {len(point)}")
        
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        # random crop augumentaiton
        if self.train and self.patch:
            # print(f"==== processing {img_path.split(os.sep)[-1]} ====")
            img, point = random_crop(img, point)
            self.count += 1  # NOTE
            # print(f"processed: {self.count}")
            
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            # NOTE
            # 将图片的绝对路径，按照'/'分割（Windows和Linux文件分隔符不同），取列表最后一个元素，为文件名
            # 将文件名按照'.'分割，取第一个元素，即去除了文件扩展名
            # 将纯文件名按照'_'分割，取最后一个元素，为文件id
            # TODO: RGB & TIR
            # image_id = int(img_path.split(os.sep)[-1].split('.')[0].split('_')[-1][:-1]) # tir
            image_id = int(img_path.split(os.sep)[-1].split('.')[0].split('_')[-1])  # rgb
            # image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])

    return img, np.array(points)

import matplotlib.pyplot as plt

# random crop augumentation
def random_crop(img, den, num_patch=4):
    # print(f"img.shape = {img.shape}")
    # print(f"len(den) = {len(den)}")
    if len(den) <= 0:
        print("empty den")
        img_array = np.array(img)  #  (3, 543, 679)
        # 调整图像数据的形状，将通道维度放到最后
        img_array = np.transpose(img_array, (2, 1, 0))
        # img_array = np.transpose(img_array, (1, 2, 0))
        plt.imshow(img_array)
        plt.axis('off')  # 可选择是否显示坐标轴
        plt.show()
    # else:
        # print(f"len(den[0]) = {len(den[0])}")
        
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        # BUG: 若den为空则会有问题
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den