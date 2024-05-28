import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    
    # NOTE: 输入文件夹
    # TODO: RGB & TIR
    # parser.add_argument('--input_dir', default='/root/notebook/violette/dataset/test/tir',  # tir
    # parser.add_argument('--input_dir', default='/root/notebook/violette/dataset/test/rgb',  # rgb
    parser.add_argument('--input_dir', default='D:/Desktop/AIA/DroneRGBT/Final',  # rgb
                        help='path where to read picture and predict')
    # NOTE: 输出文件夹  
    # parser.add_argument('--output_dir', default='/root/notebook/violette/CrowdCounting-P2PNet/output',  
    parser.add_argument('--output_dir', default='D:/Desktop/AIA/CrowdCounting-P2PNet/output',  
                        help='path where to save')
    # NOTE: 训练好的模型权重
    # parser.add_argument('--weight_path', default='/root/notebook/violette/CrowdCounting-P2PNet/weights/best_mae.pth', 
    # parser.add_argument('--weight_path', default='/root/notebook/violette/CrowdCounting-P2PNet/weights/SHTechA.pth', 
    parser.add_argument('--weight_path', default='D:/Desktop/AIA/CrowdCounting-P2PNet/weights/best_mae.pth', 
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    # os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    # print("======= parameter ======")
    # for name, param in model.named_parameters():
    #     print(name, param.size())
    # print("======= parameter ======")
    
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    rgb_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    
    tir_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.492, 0.168, 0.430],
            std=[0.317, 0.174, 0.191]),
    ])

    rgb_paths = [f"{args.input_dir}/rgb/{i}.jpg" for i in range(1, 1001)]  
    tir_paths = [f"{args.input_dir}/tir/{i}R.jpg" for i in range(1, 1001)]  
    
    count_list = []
    
    for i in range(len(rgb_paths)):
        
        rgb_path = rgb_paths[i]
        tir_path = tir_paths[i]
    
        # load the images
        # img_raw = Image.open(img_path)  # tir
        rgb_img_raw = Image.open(rgb_path).convert('RGB')  # rgb
        tir_img_raw = Image.open(tir_path).convert('RGB')  
        # round the size
        width, height = rgb_img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        
        #img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        # 新版本的Pillow
        rgb_img_raw = rgb_img_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)
        tir_img_raw = tir_img_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # pre-proccessing
        rgb_img = rgb_transform(rgb_img_raw)
        tir_img = tir_transform(tir_img_raw)

        rgb_samples = torch.Tensor(rgb_img).unsqueeze(0)
        tir_samples = torch.Tensor(tir_img).unsqueeze(0)
        rgb_samples = rgb_samples.to(device)
        tir_samples = tir_samples.to(device)
        # run inference
        outputs = model(rgb_samples, tir_samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        # NOTE: 预测的点
        outputs_points = outputs['pred_points'][0]
        # print(f"outputs_points = {outputs_points}")
        # print(f"outputs_points.shape = {outputs_points.shape}")  # 20480 * 2

        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # print(f"after filter = {points}")
        # print(f"predict count = {predict_cnt}")
        
        # NOTE: 重定向到文件
        count_list.append(predict_cnt)
        # print(f"{i+1},{predict_cnt}")

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]
        # print(f"outputs_scores = {outputs_scores}")
        # print(f"outputs_points = {outputs_points}")
        # print(f"outputs_points.shape = {outputs_points.shape}") 
        
        # draw the predictions
        size = 2
        img_to_draw = cv2.cvtColor(np.array(rgb_img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        # save the visualized image
        # cv2.imwrite(f"./output/{img_path}", img_to_draw)
        # cv2.imwrite(os.path.join(args.output_dir, img_path), img_to_draw)
        cv2.imwrite(os.path.join(args.output_dir, f"{i+1}-{predict_cnt}.jpg"), img_to_draw)
        # cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)
    
    with open('ans.txt', 'w') as file:
        for i in range(len(rgb_paths)):
            file.write(f"{i+1},{count_list[i]}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)