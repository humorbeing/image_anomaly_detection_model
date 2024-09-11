import torch.utils.data as data
import glob

from PIL import Image


dataset_folder_path = 'dataset/training_folder/test'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


from torchvision.transforms import Resize, Compose, ToTensor, Normalize

normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
transform_test = Compose([
    Resize((224, 224)),
    ToTensor(),
    normalize,
])

class Dataset_test(data.Dataset):
    def __init__(self):
        target_folder = dataset_folder_path
        file_list = glob.glob(target_folder + '/**/*.JPG', recursive=True)
        file_list2 = glob.glob(target_folder + '/**/*.jpg', recursive=True)
        file_list.extend(file_list2)
        self.list = file_list        

        self.transform = transform_test
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        image_path = self.list[index]
        check_str = '/abnormal/'
        if check_str in image_path:
            label = 1
        else:
            label = 0
        im = Image.open(self.list[index]).convert('RGB')  
        img = self.transform(im)
        return img, label, image_path

 

if __name__ == '__main__':
    from torch.utils.data import DataLoader   
    
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
