from models.osnet import osnet_x1_0
import torch
import os
current_path = os.path.dirname(__file__)
def get_model():
    fish_disease_detection_model = osnet_x1_0(num_classes=2, pretrained=False, loss='triplet')

    model_weight_path = current_path + '/model_weights/this_model.pkl'
    # model_path = os.path.abspath(os.path.join(inspect.getfile(fish_disease_detection_model), "..", model_weight_path))
    fish_disease_detection_model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    return fish_disease_detection_model


