import torch
import tqdm
import csv
import numpy as np
from torch.nn.functional import softmax
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

import os
current_path = os.path.dirname(__file__)



csvfile = open(current_path + '/test_results/detection_result.csv', 'w', newline='') 
writer = csv.writer(csvfile)    
results = [
    ['Image Path', 'Prediction', 'Ground Truth']
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset_loader import Dataset_test
test_dataset = Dataset_test()
from detection_model import get_model

model = get_model()
model = model.to(device=device)
model.eval()

pred_RB_list = torch.zeros(0).to(device)
labels_RB_list = []

pred_list = torch.zeros(0).to(device)
labels_list = []

for (images, labels, image_path) in tqdm.tqdm(test_dataset):
    images = images[None, :]
    images = images.to(device)
    with torch.no_grad():
        temp12 = model.eval_forward(images)
        
        sf = softmax(temp12, dim=1)        
        prediction = sf[:,1]

        labels_list.append(labels)        
        pred_list = torch.cat((pred_list, prediction))
        results.append([image_path, prediction.item(), int(labels)])

        labels_RB_list.append(labels)        
        pred_RB_list = torch.cat((pred_RB_list, prediction))
    




writer.writerows(results)
csvfile.close()

def get_AUCROC(pred_list, labels_list):
    pred_np = pred_list.cpu().detach().numpy()
    gt_np = np.array(labels_list)

    fpr, tpr, threshold = roc_curve(gt_np, pred_np)
    auc_roc = auc(fpr, tpr)
    return auc_roc

total_AUCROC = get_AUCROC(pred_list, labels_list)
RB_AUCROC = get_AUCROC(pred_RB_list, labels_RB_list)
# RS_AUCROC = get_AUCROC(pred_RS_list, labels_RS_list)


with open(current_path + '/test_results/auc_roc.txt', 'w') as f:
    f.writelines(f'Total AUC-ROC: {total_AUCROC}\n')
    f.writelines(f'Rock Bream AUC-ROC: {RB_AUCROC}\n')
    # f.writelines(f'Red Seabream AUC-ROC: {RS_AUCROC}\n')


print('Test end.')

