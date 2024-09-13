import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.transforms as transforms

import os

from random_erasing import RandomErasing


class TrainingDataset(Dataset):
    def __init__(self, data_folder, person_ids, transform=None, num_sample_persons=4, num_sample_imgs=8, random_mask=False):
        assert os.path.isdir(data_folder)
        self.data_folder = data_folder
        self.person_ids = person_ids
        self.unread_ids = set(self.person_ids)
        assert len(self.person_ids) > 0
        self.person_img_path_dict = {}
        self.random_mask = random_mask
        count = 0
        for pid in self.person_ids:
            pfolder = os.path.join(self.data_folder, pid)
            assert os.path.isdir(pfolder)
            img_paths = [os.path.join(pfolder, x) for x in os.listdir(pfolder)]
            num_img_paths = len(img_paths)
            miss_num = num_sample_imgs - num_img_paths
            miss_img_paths = []
            for miss_i in range(miss_num):
                miss_img_paths.append(random.choice(img_paths))
            img_paths += miss_img_paths
            self.person_img_path_dict[pid] = img_paths
            count += len(self.person_img_path_dict[pid])
        self.num_sample_persons = num_sample_persons
        self.num_sample_imgs = num_sample_imgs
        self.transform = transform
        #self.length = math.ceil(1.*len(person_ids)/num_sample_persons)
        self.length = int(1.*len(person_ids)/num_sample_persons)
        if self.random_mask:
            self.random_mask_obj = RandomErasing(random_fill=True)

    def __getitem__(self, index):
        person_samples = random.sample(self.unread_ids, self.num_sample_persons)
        self.unread_ids = self.unread_ids - set(person_samples)

        if len(self.unread_ids) < self.num_sample_persons:
            self.unread_ids = set(self.person_ids)

        imgs_mini_batch_list = []
        for pid in person_samples:
            img_samples = [Image.open(x).convert('RGB')
                           for x in random.sample(self.person_img_path_dict[pid], self.num_sample_imgs)]
            if self.transform is not None:
                img_samples = [self.transform(x) for x in img_samples]
            else:
                img_samples = [transforms.ToTensor()(x) for x in img_samples]
            if self.random_mask:
                for pimg_id in range(len(img_samples)):
                    img_samples[pimg_id] = self.random_mask_obj(img_samples[pimg_id])

            imgs_mini_batch_list += img_samples
        return torch.stack(imgs_mini_batch_list)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    
    dataset_path = 'dataset/MARS/bbox_train/bbox_train'
    assert os.path.isdir(dataset_path)
    train_person_ids = os.listdir(dataset_path)
    # train_dataset = TrainingDataset(dataset_path, train_person_ids, random_mask=True)
    # temp11 = iter(train_dataset)
    # temp12 = next(temp11)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_image_size = [int(x * 1.125) for x in [224,224]]
    train_transform = transforms.Compose([
        transforms.Resize(scale_image_size),
        transforms.RandomCrop([224,224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,])
    

    train_dataset = TrainingDataset(
        data_folder=dataset_path,
        person_ids=train_person_ids,
        num_sample_persons=2,
        num_sample_imgs=3,
        transform=train_transform,
        random_mask=True)

    
    temp11 = iter(train_dataset)
    temp12 = next(temp11)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        num_workers=0, pin_memory=True)

    temp21 = iter(train_loader)
    temp22 = next(temp21)
    print('end')

