import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T

from segment_anything.utils.transforms import ResizeLongestSide

##########################################################################################################################################

class cus_Dataset(Dataset):

    def __init__(self, CFG, model, data):

        self.df = data
        self.image_dir = CFG['Image_Dir']
        self.mask_dir = CFG['Mask_Dir']
        self.role = CFG['Role']
        self.resize = T.Resize((CFG['Resize'], CFG['Resize']))

        self.tranform = ResizeLongestSide(1024)
        self.preprocess = model.preprocess
        self.img_size = model.image_encoder.img_size


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        image, image_name = self.load_image(self, idx)
        mask = self.load_mask(self, idx)

        if self.role == 'Test':
            return image, image_name, mask
        else:
            return image, mask
    

    def load_image(self, idx):

        img_name = self.df['image_name'][idx]
        img_path = f"{self.image_dir}/{img_name}"

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.tranform.apply_image(img)
        img = torch.as_tensor(img)
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0)
        
        img = self.preprocess(img)

        return img, img_name


    def load_mask(self, idx):

        mask_path = f"{self.mask_dir}/{self.df['mask_name'][idx]}"

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = self.tranform.apply_image(mask)
        mask = mask.unsqueeze(0)

        h, w = mask.shape[-2:]

        padh = self.img_size - h 
        padw = self.img_size - w

        mask = F.pad(mask, (0, padw, 0, padh))

        mask = self.resize(mask).squeeze(0)
        mask = (mask != 0) * 1

        return mask