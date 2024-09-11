import torch
from dataset_utility import get_dataset_loader

loaders = get_dataset_loader(2)
ntrain_loader = loaders['normal loader']
abtrain_loader = loaders['abnormal loader']
test_loader = loaders['test loader']


from models.osnet import osnet_x1_0

model = osnet_x1_0(num_classes=2, pretrained=False)

from losses.cross_entropy_loss import CrossEntropyLoss


criterion_x = CrossEntropyLoss(
    num_classes=2,
    use_gpu=True,
    label_smooth=True
)


import models
num_classes = 2
pretrained = True  # one successful run
for model_name in models.__model_factory:
    print(f'model: {model_name} <<<-------------------')
    model = models.build_model(model_name, num_classes=num_classes, pretrained=pretrained)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0015,
        weight_decay=0.0005,
        betas=(0.9, 0.999),
        amsgrad=True,
    )
    optimizer.zero_grad()



    from train_utility import train

    train(
        model,
        ntrain_loader,
        abtrain_loader,
        test_loader,
        optimizer,
        num_steps=50,
        loss_fn=criterion_x,
        NUM_ACCUMULATION_STEPS = 5
        )