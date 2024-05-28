import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

class DroneRGBTDual(Dataset):
    # def __init__(self, img_dir, gt_dir, transform=None, train=False, patch=False, flip=False):
    def __init__(self, rgb_img_dir, tir_img_dir, gt_dir, rgb_transform=None, tir_transform=None, train=False, patch=False, flip=False):
        
        # self.img_dir = img_dir
        self.rgb_img_dir = rgb_img_dir
        self.tir_img_dir = tir_img_dir
        self.gt_dir = gt_dir
        
        self.img_map = {}
        self.img_list = []
        
        # 创建可见光图像映射
        img_paths = [filename for filename in os.listdir(rgb_img_dir) if filename.endswith('.jpg')]
        for filename in img_paths:
            # 假设可见光和红外光图像文件名相同，只是目录不同
            rgb_img_path = os.path.join(rgb_img_dir, filename)
            tir_img_path = os.path.join(tir_img_dir, filename.replace('.jpg', 'R.jpg'))
            gt_path = os.path.join(gt_dir, f"{filename.split('.')[0]}R.txt")
            
            # 确保可见光和红外光图像文件都存在
            if os.path.isfile(rgb_img_path) and os.path.isfile(tir_img_path):
                # self.img_map['D:/1.jpg'] = ('D:/1R.jpg', 'D:/1.txt')
                self.img_map[rgb_img_path] = (tir_img_path, gt_path)  # NOTE
                self.img_list.append(rgb_img_path)
        self.img_list = sorted(list(self.img_map.keys()))
        
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.rgb_transform = rgb_transform
        self.tir_transform = tir_transform
        # self.transform = transform
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
        rgb_img, rgb_target = self.__getsingleitem__(index, 'rgb')
        tir_img, tir_target = self.__getsingleitem__(index, 'tir')

        # 封装最终的输出，包括可见光图像、红外光图像和标注
        return rgb_img, tir_img, tir_target
        

    def __getsingleitem__(self, index, img_type):
        assert index <= len(self), 'index range error'

        rgb_img_path = self.img_list[index]
        tir_img_path, gt_path = self.img_map[rgb_img_path] 
        
        # load image and ground truth
        # TODO: 红外光图像以什么形式读入
        # NOTE: 读取图片和点坐标
        if img_type == 'rgb':
            img, point = load_data((rgb_img_path, gt_path), self.train)
        elif img_type == 'tir':
            img, point = load_data((tir_img_path, gt_path), self.train)
        else:
            raise Exception('无法处理这种图像：', img_type)
        
        # NOTE: 随机mask
        # point_num = len(point)
        # if point_num > 30 and point_num < 60:
            # point = self.remove_n_points(point, int(point_num * 0.1))  
            # print(f"图片 {img_path} 点数被降为 {len(point)}")
        
        # applu augumentation
        # if self.transform is not None:
        #     img = self.transform(img)
        
        # NOTE: transform
        if img_type == 'rgb' and self.rgb_transform is not None:
            img = self.rgb_transform(img)
        elif img_type == 'tir' and self.tir_transform is not None:
            img = self.tir_transform(img)

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
            image_id = int(rgb_img_path.split(os.sep)[-1].split('.')[0].split('_')[-1]) 
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target

# NOTE: 红外光会默认转换成彩色三通道
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