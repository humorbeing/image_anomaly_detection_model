

def get_fish_detection_model():
    import super_gradients
    checkpoint_path = 'models/fish_detection_model/fish_detection_yolonas.pth'
    classes = ['fish']
    fish_detection_yolo_nas_l_model = super_gradients.training.models.get("yolo_nas_l", num_classes=len(classes),
        checkpoint_path=checkpoint_path)
    
    return fish_detection_yolo_nas_l_model


def get_disease_detection_model():
    from models.osnet import osnet_x1_0
    import torch
    fish_disease_detection_model = osnet_x1_0(num_classes=2, pretrained=False, loss='triplet')

    model_weight_path = 'histogram_model/this_model_round2.pkl'
    # model_path = os.path.abspath(os.path.join(inspect.getfile(fish_disease_detection_model), "..", model_weight_path))
    fish_disease_detection_model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    return fish_disease_detection_model


def get_crop_check_model():
    import models.crop_check_model.resnet as resnet
    import torch
    crop_check_model = resnet.resnet152(    
        num_classes=2,
        loss='softmax',
        pretrained=False,    
    )


    model_weight_path = 'models/crop_check_model/fish_crop_check.pkl'

    crop_check_model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    return crop_check_model

if __name__ == "__main__":
    # model = get_fish_detection_model()
    # model = get_disease_detection_model()
    model = get_crop_check_model()
    print()