import torch.utils.data as data
import os
import numpy as np
import glob

from PIL import Image
from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode

dataset_folder_path = 'dataset/training_folder'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

from transforms import build_transforms

transform_train, transform_test = build_transforms(
    224,
    224,
    transforms=['random_flip', 'random_erase'],
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225]
)


class Dataset_normal(data.Dataset):
    def __init__(self):
        target_folder = dataset_folder_path + '/train/normal'
        file_list = glob.glob(target_folder + '/**/*.JPG', recursive=True)
        file_list2 = glob.glob(target_folder + '/**/*.jpg', recursive=True)
        file_list.extend(file_list2)
        print(target_folder)
        self.list = file_list
        # self.transform = transforms.Compose([
        #     # transforms.ToPILImage(),
        #     # transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
        #     transforms.Resize((112, 112)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        #     # transforms.Resize((256,256)),         
        #     # transforms.CenterCrop(224),            
        # ])

        self.transform = transform_train
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        label = 0
        im = Image.open(self.list[index]).convert('RGB')  
        img = self.transform(im)
        return img, label



class Dataset_abnormal(data.Dataset):
    def __init__(self):
        target_folder = dataset_folder_path + '/train/abnormal'
        file_list = glob.glob(target_folder + '/**/*.JPG', recursive=True)
        file_list2 = glob.glob(target_folder + '/**/*.jpg', recursive=True)
        file_list.extend(file_list2)
        self.list = file_list
        # self.transform = transforms.Compose([
        #     # transforms.ToPILImage(),
        #     # transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
        #     transforms.Resize((112, 112)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        #     # transforms.Resize((256,256)),         
        #     # transforms.CenterCrop(224),            
        # ])

        self.transform = transform_train
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        label = 1        
        im = Image.open(self.list[index]).convert('RGB')
        img = self.transform(im)
        return img, label


class Dataset_test(data.Dataset):
    def __init__(self):
        target_folder = dataset_folder_path + '/test'
        file_list = glob.glob(target_folder + '/**/*.JPG', recursive=True)
        file_list2 = glob.glob(target_folder + '/**/*.jpg', recursive=True)
        file_list.extend(file_list2)
        self.list = file_list
        # self.transform = transforms.Compose([
        #     # transforms.ToPILImage(),
        #     # transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
        #     transforms.Resize((112, 112)),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        #     # transforms.Resize((256,256)),         
        #     # transforms.CenterCrop(224),            
        # ])

        self.transform = transform_test
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        image_path = self.list[index]
        check_str = '/test/abnormal/'
        if check_str in image_path:
            label = 1
        else:
            label = 0
        im = Image.open(self.list[index]).convert('RGB')  
        img = self.transform(im)
        return img, label



     

if __name__ == '__main__':
    from torch.utils.data import DataLoader    
    ntrain_dataset = Dataset_normal()
    temp11 = iter(ntrain_dataset)
    get_ntrain = next(temp11)

    ntrain_loader = DataLoader(
        ntrain_dataset,
        batch_size=16, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)
    
    temp11 = iter(ntrain_loader)
    get_ntrain = next(temp11)

    abtrain_dataset = Dataset_abnormal()
    temp12 = iter(abtrain_dataset)
    get_abtrain = next(temp12)
    abtrain_loader = DataLoader(
        abtrain_dataset,
        batch_size=16, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)

    temp12 = iter(abtrain_loader)
    get_abtrain = next(temp12)
    
    test_dataset = Dataset_test()
    temp12 = iter(test_dataset)
    get_abtrain = next(temp12)
    all_loader = DataLoader(
        test_dataset,
        batch_size=16, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False)

    temp12 = iter(all_loader)
    get_abtrain = next(temp12)

    print('end')
