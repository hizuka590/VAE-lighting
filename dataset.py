import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import glob
import random
from PIL import Image
from torchvision import transforms
import torchvision

#######################################################
#               Define Transforms
#######################################################
transform_img = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((416,416)),
                                #transforms.PILToTensor()])
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                )
transform_label = transforms.Compose([
                                transforms.Resize((416,416)),
                                transforms.ToTensor()]) # range [0, 255] -> [0.0,1.0]

#######################################################
#               Define train,val,test sets path
#######################################################
def get_alldata_path(train_image_path, type='train'):
    list_image_paths = []  # to store image paths in list
    # 1.
    # get all the paths from train_data_path and append image paths and class to to respective lists
    # eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
    # eg. class -> 26.Pont_du_Gard
    for data_path in glob.glob(train_image_path + '/*'):
        # print(data_path)
        list_image_paths.append(glob.glob(data_path + '/*'))
        # print('appended: ',glob.glob(data_path + '/*'))
        # break

    list_image_paths = [item for sublist in list_image_paths for item in sublist] # flatten needed for shuffle because ori is a 2d list
    # print('ori:',train_image_paths)
    random.shuffle(list_image_paths)
    # print('after:', train_image_paths)

    print('{}_image_path example number:{},random sample:{} '.format(type, len(list_image_paths),random.choice(list_image_paths)))
    return list_image_paths

#######################################################
#               Define exposure fundaion
#######################################################
def get_dataset(transform_on=True):
    train_image_path = '/opt/sdb/polyu/VSD_dataset/train/images'
    valid_image_path = '/opt/sdb/polyu/VSD_dataset/test/images'
    if transform_on == True:
        print('transformed dataset loaded')
        train_dataset = VSD_DataSet(get_alldata_path(train_image_path), transform_img, transform_label)
        valid_dataset = VSD_DataSet(get_alldata_path(valid_image_path, type='val'),
                                    transform_img,transform_label)  # test transforms are applied

    else:
        print('original dataset loaded')
        train_dataset = VSD_DataSet(get_alldata_path(train_image_path))
        valid_dataset = VSD_DataSet(get_alldata_path(valid_image_path, type='val'))  # test transforms are applied

    return train_dataset, valid_dataset

class VSD_DataSet(Dataset):
    def __init__(self, image_paths, transform_img=False,transform_label=False):
        self.image_paths = image_paths
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = np.array(Image.open(image_filepath).convert('RGB'))
        labbel_filepath = image_filepath.replace('images','labels').replace(".jpg",".png")
        label = (Image.open(labbel_filepath).convert('L'))
        if self.transform_img is not None:
            image = self.transform_img(image)
        if self.transform_label is not None:
            label = self.transform_label(label)
        return image, label

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  CelebA Dataset  =========================
    
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

class VSDDataset(LightningDataModule):

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = ''
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_image_path = '/opt/sdb/polyu/VSD_dataset/train/images'
        valid_image_path = '/opt/sdb/polyu/VSD_dataset/test/images'
        train_dataset, valid_dataset = get_dataset()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.valid_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


