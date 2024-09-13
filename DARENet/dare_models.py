from models import dare_resnet
from models import dare_densenet

def dare_R(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7),gen_stage_features = False, **kwargs):
    return dare_resnet.resnet50(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
                                global_pooling_size=gap_size, drop_rate=0, gen_stage_features = gen_stage_features)


 
def dare_D(pretrained=True, fc_layer1=1024, fc_layer2=128, gap_size=(7,7), **kwargs):
    # return dare_densenet.densenet201(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
    #                                  global_pooling_size=gap_size)
    # return dare_densenet.densenet169(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
    #                                  global_pooling_size=gap_size)
    # return dare_densenet.densenet161(pretrained=False, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
    #                                  global_pooling_size=gap_size)  # No pretrain
    return dare_densenet.densenet121(pretrained=pretrained, fc_layer1=fc_layer1, fc_layer2=fc_layer2,
                                     global_pooling_size=gap_size)
