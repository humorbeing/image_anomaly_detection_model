from model import get_disease_detection_model
import torch
from utility import transform_disease_detection
import cv2
from torch.nn.functional import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_disease_detection_model().to(device).eval()
print('end')

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
    print('End')
    img_path = 'dataset/training_folder/test/abnormal/144459_objt_rs_2020-12-17_10-33-34-33_002.JPG'
    score = detect(img_path)
    print()
