import torch
import torchvision
from models.resnet import *
from models.vit import *
def get_encoder(args):
    model=None
    if args.backbone=="res50":
        pretrained=args.pretrained
        return resnet50(pretrained=pretrained)
        # print(model)
    elif args.backbone=="res101":
        pretrained = args.pretrained
        return resnet101(pretrained=pretrained)
    return model

def get_feature_extraction(args):
    model=vit()
    return model
