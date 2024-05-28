import torchvision.transforms as standard_transforms
from .DroneRGBT_Dual import DroneRGBTDual

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def loading_data(train_rgb_dir, train_tir_dir, train_gt_dir, test_rgb_dir, test_tir_dir, test_gt_dir):
    
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
            
    # create the training dataset
    # NOTE: 划分patch
    # dataset = DroneRGBT(img_dirs, gt_dirs, train=True, transform=transform, patch=True, flip=True)  
    train_set = DroneRGBTDual(train_rgb_dir, train_tir_dir, train_gt_dir, rgb_transform=rgb_transform, tir_transform=tir_transform, train=True, patch=True, flip=True)  
    # train_set = DroneRGBT(train_img_dir, train_gt_dir, train=True, transform=transform, patch=True, flip=True)  
    # create the validation dataset
    val_set = DroneRGBTDual(test_rgb_dir, test_tir_dir, test_gt_dir, rgb_transform=rgb_transform, tir_transform=tir_transform, train=True, patch=True, flip=True)  

    # return dataset
    return train_set, val_set