

def get_disease_detection_model():
    from models.osnet import osnet_x1_0
    import torch
    fish_disease_detection_model = osnet_x1_0(num_classes=2, pretrained=False, loss='triplet')

    model_weight_path = 'histogram_model/model_weights/this_model.pkl'
    # model_path = os.path.abspath(os.path.join(inspect.getfile(fish_disease_detection_model), "..", model_weight_path))
    fish_disease_detection_model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    return fish_disease_detection_model



if __name__ == "__main__":    
    model = get_disease_detection_model()    
    print()