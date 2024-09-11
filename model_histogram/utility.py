

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage

normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


transform_crop_check = Compose([
    ToPILImage(),
    Resize((224, 224)),
    ToTensor(),
    normalize,
])




transform_disease_detection = Compose([
    ToPILImage(),
    Resize((224, 224)),
    ToTensor(),
    normalize,
])
