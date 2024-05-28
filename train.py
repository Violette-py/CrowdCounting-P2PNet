import argparse
import datetime
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler, random_split

from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    # TODO: 这两个lr有什么不同
    parser.add_argument('--lr', default=1e-4, type=float)  # NOTE
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # NOTE
    parser.add_argument('--batch_size', default=8, type=int)  
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)  # NOTE
    # parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    # TODO: 重写 数据集路径
    parser.add_argument('--dataset_file', default='DroneRGBT')
    # parser.add_argument('--dataset_file', default='SHHA')
    # NOTE: 用于loading_data，最好设置成环境变量的绝对路径
    # parser.add_argument('--data_root', default='D:\Desktop\AIA\DroneRGBT',  
    #                     help='path where the dataset is')
    
    # TODO: RGB & TIR
    parser.add_argument('--train_img_dir', default='/root/notebook/violette/DroneRGBT/Train/RGB',  
                        help='path where the train image in')
    parser.add_argument('--train_gt_dir', default='/root/notebook/violette/DroneRGBT/Train/GT_Point',  
                        help='path where the train ground truth in')
    parser.add_argument('--test_img_dir', default='/root/notebook/violette/DroneRGBT/Random_Val/RGB',  
                        help='path where the test image in')
    parser.add_argument('--test_gt_dir', default='/root/notebook/violette/DroneRGBT/Random_Val/GT_Point',  
                        help='path where the test ground truth in')  
    
    # parser.add_argument('--img_dirs', default='D:/Desktop/AIA/DroneRGBT/Train/RGB|D:/Desktop/AIA/DroneRGBT/Test/RGB',  
    #                     help='path where the train image in')
    # parser.add_argument('--gt_dirs', default='D:/Desktop/AIA/DroneRGBT/Train/GT_Point|D:/Desktop/AIA/DroneRGBT/Test/GT_Point',  
    #                     help='path where the train ground truth in') 
    
    parser.add_argument('--output_dir', default='./logs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./weights',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./logs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # NOTE: 从指定的checkpoint恢复训练
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)  # NOTE
    parser.add_argument('--eval_freq', default=1, type=int,  # NOTE
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # create the logging file
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the P2PNet model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    # 允许优化器对模型的不同部分使用不同的学习率或其他优化参数
    param_dicts = [
        # 不需要梯度的参数
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        # 需要梯度的参数，设置lr
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    # TODO: 换用其他的调度器试一下
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)  # NOTE: 这个的lr_scheduler.step()不需要传参
    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    # TODO: 重写 导入数据集
    # BUG: 原始数据集的测试集，没有132、181、296、314
    # TODO: 验证集目前用的是原始数据集的前452 - 1张，可以每次从里面随机选一些
    print("==== preprocessing dataset ====")
    # dataset = loading_data(args.img_dirs, args.gt_dirs)
    # # 按照 8:2 划分训练集和验证集
    # train_size = int(0.8 * len(dataset))  
    # val_size = len(dataset) - train_size
    # train_set, val_set = random_split(dataset, [train_size, val_size])
    train_set, val_set = loading_data(args.train_img_dir, args.train_gt_dir, args.test_img_dir, args.test_gt_dir)
    # print(f"len(train_set) = {len(train_set)}")
    # print(f"train_set.img_list = {train_set.img_list}")
    # print(f"train_set.img_map = {train_set.img_map}")
    
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)  # 随机采样
    sampler_val = torch.utils.data.SequentialSampler(val_set)  # 顺序采样

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    # TODO: 重写 DataLoader？
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=True, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)
                                    # drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume the weights and training state if exists
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)
    
    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
                log_file.write("loss/loss_ce@{}: {}\n".format(epoch, stat['loss_ce']))
                
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]\n' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        # change lr according to the scheduler
        # BUG: 换成了 ReduceLROnPlateau，这里的 step 需要一个metric参数
        lr_scheduler.step(stat['loss'])
        # lr_scheduler.step()
        # save latest weights every epoch
        # 最新和最佳的都保存下来了
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
        }, checkpoint_latest_path)
        # run evaluation
        if epoch % args.eval_freq == 0:
        # if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            # TODO: 可以将可视化结果保存下来
            result = evaluate_crowd_no_overlap(model, data_loader_val, device)  
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])
            # print the evaluation results
            print('=======================================test=======================================')
            print("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", np.min(mae), )
            with open(run_log_name, "a") as log_file:
                log_file.write("mae:{}, mse:{}, time:{}, best mae:{}\n".format(result[0], 
                                result[1], t2 - t1, np.min(mae)))
            print('=======================================test=======================================')
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, result[0]))
                    log_file.write("metric/mse@{}: {}".format(step, result[1]))
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

            # save the best model since begining
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_mae.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_best_path)
    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)