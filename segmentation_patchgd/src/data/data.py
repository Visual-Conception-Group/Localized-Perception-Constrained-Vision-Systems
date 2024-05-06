
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob 
import json
from torch.utils.data import DataLoader
import albumentations as A
# from torchvision import transforms

import sys
lib_path = "./src"
# print(os.listdir((lib_path)))
sys.path.append(lib_path)
from utils import generate_small_patches

# ALBUMENTATION TRANSFORM
transformations = A.Compose([
    # A.RandomCrop(width=400, height=400),
    # A.RandomBrightnessContrast(p=0.2),
    # A.RGBShift(),
    # A.ElasticTransform(),
    # A.MaskDropout((10,15), p=1),
    # A.CoarseDropout(p=1)
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(),
    A.Blur(),
    A.GaussNoise(),
])



class SegmentDataset(Dataset):
    def __init__(self, images_path, masks_path, img_size, transforms=None, n_patch_split=1):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_patch_split = n_patch_split
        self.total_patches = self.n_patch_split*self.n_patch_split
        self.n_samples = len(images_path)*self.total_patches

        if type(img_size)==int:
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size
            
        self.prev_img = None
        self.prev_mask = None
        self.transforms = transforms

        self.ctr = 0
        self.image_mask = []
        # self.load_image_mask()
        self.patch_dict = {}
        self.populate_image_dict()
        # print(self.patch_dict)
        self.buff_idx=None
        
    def populate_image_dict(self):
        for i in range(self.n_samples):
            j = i//(self.total_patches)
            patch_no = i%self.total_patches
            # print(self.masks_path[j], patch_no)
            # self.patch_dict[i]=[self.images_path[j], self.masks_path[j], patch_no]
            self.patch_dict[i]=[j, patch_no]
        pass

    def load_image_mask(self):
        for i in range(len(self.images_path)):
            image, mask = self.get_image(i)
            image = image.unsqueeze(dim=0)
            mask = mask.unsqueeze(dim=0)
            # print(image.shape, mask.shape)
            new_x, new_y = generate_small_patches(image, mask, self.n_patch_split)
            
            for img, mask in zip(new_x, new_y):
                print(img.shape, mask.shape)
                self.image_mask.append([img, mask])
        print(f"Generated: {len(self.image_mask)} Images")

    def load_image_mask_one(self, i):
        self.image_mask = []
        image, mask = self.get_image(i)
        # print(image.shape)
        # exit()
        image = image.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)
        # print(image.shape, mask.shape)
        new_x, new_y = generate_small_patches(image, mask, self.n_patch_split)
        
        for img, mask in zip(new_x, new_y):
            # print(img.shape, mask.shape)
            self.image_mask.append([img, mask])
        # print(f"Generated: {len(self.image_mask)} Images")

    def preprocess_img(self, image):
        """
        Reading image
        Imp: keeping BGR as image format
        """
        image = cv2.resize(image, self.img_size)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        return image

    def preprocess_mask(self, mask):
        """ Reading mask """
        mask = cv2.resize(mask, self.img_size)
        mask = mask/255.0   ## (512, 512)
        mask = mask>0.5
        # mask = (mask-1)*-1 # invert mask
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        # print(image.shape, mask.shape)
        return mask

    def get_image(self, index):
        try:
            image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

            if self.transforms is not None:
                # apply the transformations to both image and its mask
                transformed = self.transforms(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            image = self.preprocess_img(image)
            mask = self.preprocess_mask(mask)

        except Exception as e:
            print(f"error at: {self.images_path[index]} - {self.masks_path[index]} [{e}]")
            image = self.prev_img
            mask = self.prev_mask
            # raise e
        return image, mask

    def load_more(self, ):
        self.ctr


    def __getitem__(self, index):
        i, n_patch = self.patch_dict[index]

        # CACHING TO MAKE THE DATALOADING FASTER
        if not(i==self.buff_idx):
            # print("loading image",i, n_patch, index, self.buff_idx)
            self.buff_idx=i
            self.load_image_mask_one(i)

        return self.image_mask[n_patch]

    def __len__(self):
        return self.n_samples
    

def load_datasets(
        train_path, 
        valid_path,
        batch_size=1,
        img_size=(512, 512), 
        n_samples=None,
        n_samples_train=None,
        n_samples_valid=None,
        dtype="None",
        transforms=None,
        n_patch_split=1
    ):

    """ Seeding """
    print("train:",train_path)
    print("valid:",valid_path)

    """ Load dataset from JSON """
    dist = 0.2
    if dtype=="json":
        json_data = json.load(open(train_path))
        train_x, train_y = [], []
        valid_x, valid_y = [], []
        for key in json_data.keys():
            data_pair = json_data[key]
            img_path = data_pair['image']
            msk_path = data_pair['mask']

            if np.random.rand()>dist:
                train_x.append(img_path)
                train_y.append(msk_path)
            else:
                valid_x.append(img_path)
                valid_y.append(msk_path)
    else:
        """ Load dataset """
        train_x = sorted(glob(f"{train_path}/image/*"))
        train_y = sorted(glob(f"{train_path}/mask/*"))
        valid_x = sorted(glob(f"{valid_path}/image/*"))
        valid_y = sorted(glob(f"{valid_path}/mask/*"))

    print(len(train_x)), len(valid_x)

    # REDUCE SAMPLES FOR TESTING
    if n_samples is not None:
        n_samples_train, n_samples_valid= n_samples, n_samples
    if n_samples_train is None: n_samples_train = len(train_x)
    if n_samples_valid is None: n_samples_valid = len(valid_x)
    print(n_samples_train, n_samples_valid)
    train_x = train_x[:n_samples_train]
    train_y = train_y[:n_samples_train]
    valid_x = valid_x[:n_samples_valid]
    valid_y = valid_y[:n_samples_valid]
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)
    # return train_x, train_y, valid_x, valid_y

    """ Dataset and loader """
    train_dataset = SegmentDataset(train_x, train_y, img_size, transforms=transforms, n_patch_split=n_patch_split)
    valid_dataset = SegmentDataset(valid_x, valid_y, img_size, n_patch_split=n_patch_split)


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Dataset Size: {len(train_dataset)}, {len(valid_dataset)} | ")
    print(f"Dataloader Size: {len(train_loader)} | Batch Size: {batch_size} | Samples: {len(train_loader)*batch_size} ")

    return train_loader, valid_loader


if __name__=="__main__":
    # DATALOADER 
    train_data_root = "data/retina"
    train_path = f"{train_data_root}/train"
    valid_path = f"{train_data_root}/test"

    img_size = (512, 512)

    train_loader, valid_loader = load_datasets(
        train_path,
        valid_path,
        batch_size=4,
        img_size=img_size,
        n_samples=None,
        n_patch_split=1
    )

    print("Main Dataloader", len(train_loader), len(valid_loader))
    for x, y in train_loader:
        print(x.shape, y.shape)