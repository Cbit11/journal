import numpy as np 
from torch.utils.data import Dataset
from basicsr.utils import img2tensor
import os
import cv2 
from basicsr.data.transforms import augment, paired_random_crop
class dataset(Dataset):
    def __init__(self, hr_file_pth, lr_file_pth, train_type, scale , gt_size):
        super(dataset, self).__init__()
        self.hr_files = sorted(os.listdir(hr_file_pth))
        self.lr_files = sorted(os.listdir(lr_file_pth))
        self.hr_path = hr_file_pth
        self.lr_path = lr_file_pth
        self.train_type = train_type
        self.scale= scale
        self.gt_size= gt_size
        assert len(self.hr_files)== len(self.lr_files), "Mismatch in number of HR and LR images!"
    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, index):
        img_gt_pth = os.path.join(self.hr_path,self.hr_files[index])
        img_lr_pth = os.path.join(self.lr_path,self.lr_files[index])
        img_gt= cv2.imread(img_gt_pth, cv2.IMREAD_COLOR).astype(np.float32)
        img_lq= cv2.imread(img_lr_pth, cv2.IMREAD_COLOR).astype(np.float32)
        
        if self.train_type == 'train':
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, self.scale, img_gt_pth)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], hflip=True, rotation=True)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        return {'lq': img_lq/255. , 'gt': img_gt/255. , 'lq_path': img_lr_pth, 'gt_path': img_gt_pth}
    
class test_dataset(Dataset):
    def __init__(self, file_pth_hr, file_pth_lr):
        super(test_dataset, self).__init__()
        self.hr_pth = file_pth_hr
        self.lr_pth = file_pth_lr
        self.hr_files = sorted([x for x in os.listdir(file_pth_hr) if x.endswith(('.png', '.jpg'))])
        self.lr_files = sorted([x for x in os.listdir(file_pth_lr) if x.endswith(('.png', '.jpg'))])
        # Safety Check
        assert len(self.hr_files) == len(self.lr_files), "Mismatch in number of HR and LR images!"
    def __len__(self):
        return len(self.hr_files)
    def __getitem__(self, index):
        hr_img_pth= os.path.join(self.hr_pth, self.hr_files[index])
        lr_img_pth= os.path.join(self.lr_pth, self.lr_files[index])
        img_gt =cv2.imread(hr_img_pth, cv2.IMREAD_COLOR).astype(np.float32)
        img_lq= cv2.imread(lr_img_pth, cv2.IMREAD_COLOR).astype(np.float32)
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        
        # Normalize to [0, 1]
        img_gt = img_gt / 255.
        img_lq = img_lq / 255.
        
        # Return raw images. Pad in the loop!
        # pass the filename too, it's useful for saving results
        return {
            "gt": img_gt, 
            "lq": img_lq, 
            "lq_path": lr_img_pth,
            "gt_path": hr_img_pth
        }