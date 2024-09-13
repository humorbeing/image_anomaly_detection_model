from models import dare_resnet
from models import dare_densenet

def dare_R(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7),gen_stage_features = False, **kwargs):
    print('model: resnet')
    return dare_resnet.resnet50(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
                                global_pooling_size=gap_size, drop_rate=0, gen_stage_features = gen_stage_features)



def dare_D1(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    print('model: densenet201')
    return dare_densenet.densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
                                     global_pooling_size=gap_size)
    

def dare_D2(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    print('model: densenet169')
    return dare_densenet.densenet169(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
                                     global_pooling_size=gap_size)
    

def dare_D3(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    print('model: densenet161')
    return dare_densenet.densenet161(pretrained=False, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
                                     global_pooling_size=gap_size)  # No pretrain
    

def dare_D4(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    print('model: densenet121')
    return dare_densenet.densenet121(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
                                     global_pooling_size=gap_size)

model_list = [dare_R, dare_D1, dare_D2, dare_D3, dare_D4]