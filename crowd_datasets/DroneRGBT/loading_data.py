import torchvision.transforms as standard_transforms
from .DroneRGBT import DroneRGBT

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def loading_data(train_img_dir, train_gt_dir, test_img_dir, test_gt_dir,):
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    
    # create the training dataset
    # NOTE: 划分patch
    # dataset = DroneRGBT(img_dirs, gt_dirs, train=True, transform=transform, patch=True, flip=True)  
    train_set = DroneRGBT(train_img_dir, train_gt_dir, train=True, transform=transform, patch=True, flip=True)  
    # train_set = DroneRGBT(train_img_dir, train_gt_dir, train=True, transform=transform, patch=True, flip=True)  
    # create the validation dataset
    val_set = DroneRGBT(test_img_dir, test_gt_dir, train=False, transform=transform)

    # return dataset
    return train_set, val_set