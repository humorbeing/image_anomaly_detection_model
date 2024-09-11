from model import get_disease_detection_model
import torch
from utility import transform_disease_detection
import cv2
from torch.nn.functional import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_disease_detection_model().to(device).eval()


def detect(img_path):
    bgr_frame = cv2.imread(img_path)
    im = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    images = transform_disease_detection(im)
    images = images[None, :]
    images = images.to(device)
    with torch.no_grad():
        temp12 = model.eval_forward(images)
        
        sf = softmax(temp12, dim=1)        
        prediction = sf[:,1].item()

    return prediction

if __name__ == '__main__':        
    
    img_path = 'dataset/training_folder/train/abnormal/83627_objt_rs_2020-12-05_10-07-18-66_003.JPG'
    score = detect(img_path)
    print(score)
