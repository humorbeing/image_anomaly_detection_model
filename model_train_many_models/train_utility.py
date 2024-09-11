
import os
import torch
from transforms import build_transforms

transform_train, transform_test = build_transforms(
    224,
    224,
    transforms=['random_flip', 'random_erase'],
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225]
)


import os.path as osp
from PIL import Image

def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img






# from torchreid import metrics
from torch.nn.functional import softmax
from sklearn.metrics import auc as auc_rename_avoid_duplication
from sklearn.metrics import roc_curve



def train(
    model,
    ntrain_loader,
    abtrain_loader,
    test_loader,
    optimizer,
    num_steps,
    loss_fn,
    NUM_ACCUMULATION_STEPS = 5
    ):

    
    device = next(model.parameters()).device
    STEP_TRACKER = 0   

    best_AUCROC = 0
    for step in range(num_steps):    

        if (step) % len(ntrain_loader) == 0:
            ntrain_iter = iter(ntrain_loader)

        if (step) % len(abtrain_loader) == 0:
            abtrain_iter = iter(abtrain_loader)
        model.train()
        ninput, nlabel = next(ntrain_iter)
        ainput, alabel = next(abtrain_iter)
        input_features = torch.cat((ninput, ainput), 0).to(device)
        labels = torch.cat((nlabel, alabel), 0).to(device)

        outputs = model(input_features)

        

        
        loss = loss_fn(outputs, labels)
        
        
        loss = loss / NUM_ACCUMULATION_STEPS
        loss.backward()
        STEP_TRACKER = STEP_TRACKER + 1
        if (STEP_TRACKER % NUM_ACCUMULATION_STEPS == 0):
            optimizer.step()
            optimizer.zero_grad()
        

        if step % 20 == 0:
            model.eval()
            pred_list1 = torch.zeros(0).to(device)
            pred_list2 = torch.zeros(0).to(device)
            labels_list = torch.zeros(0)
            for (images, labels) in test_loader:
                images = images.to(device)
                with torch.no_grad():
                    temp12 = model.eval_forward(images)
                    
                    sf = softmax(temp12, dim=1)
                    prediction1 = sf[:,0]
                    prediction2 = sf[:,1]

                    labels_list = torch.cat((labels_list, labels))            
                    pred_list1 = torch.cat((pred_list1, prediction1))
                    pred_list2 = torch.cat((pred_list2, prediction2))
                
            

            pred_np1 = pred_list1.cpu().detach().numpy()
            pred_np2 = pred_list2.cpu().detach().numpy()
            gt_np = labels_list.cpu().detach().numpy()

            

            

            fpr, tpr, threshold = roc_curve(gt_np, pred_np1)
            auc_roc1 = auc_rename_avoid_duplication(fpr, tpr)

            fpr, tpr, threshold = roc_curve(gt_np, pred_np2)
            auc_roc2 = auc_rename_avoid_duplication(fpr, tpr)

            print(f'1: {auc_roc1}, 2: {auc_roc2}')

            this_auc_roc = auc_roc2
            if this_auc_roc > best_AUCROC:
                best_AUCROC = this_auc_roc
                print(f'saving best: {this_auc_roc}')
                save_folder = os.path.dirname(__file__) + '/model_weights'
                os.makedirs(save_folder, exist_ok=True)
                torch.save(model.state_dict(), save_folder+'/this_model.pkl')        
    



print('')