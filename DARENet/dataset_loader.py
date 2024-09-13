import os
import torch
import torchvision.transforms as transforms
from datasets import TrainingDataset




def get_train_loader():
    dataset_path = 'dataset/MARS/bbox_train/bbox_train'
    assert os.path.isdir(dataset_path)
    train_person_ids = os.listdir(dataset_path)
    

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

    
    


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        num_workers=0, pin_memory=True)

    return train_loader

if __name__ == '__main__':
    train_loader = get_train_loader()
    print('End')