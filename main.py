import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='配置网络参数')
parser.add_argument('--patchs', default=4, type=int, help="将最后网络输出变成多少个patch")
parser.add_argument('--batch_size', default=8, type=int, help="每次传入网络的batch")
parser.add_argument('--encoder', default="res101", type=str, help="backbone")
parser.add_argument('--root', default=r'data',type=str, help='dataset root')
parser.add_argument('--lr', default=2e-4,type=int, help='学习率')
parser.add_argument('--cuda', default=True,type=bool, help='是否使用gpu')
parser.add_argument('--pretrained', default=True,type=bool, help='是否使用预训练模型')
parser.add_argument('--image_size', default=224, type=int, help="图像的大小")
parser.add_argument('--is_embed', default=False, type=bool, help="配置VIT的参数，是否使用patch_embed")
args=parser.parse_args()
from models.backbone import get_encoder,Feature2Patch,get_feature_extraction
from utils.dataset import SSLData
def build_model(args):
    return get_encoder(args)

def main(args):
    ssldata=SSLData(args)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    encoder=get_encoder(args).to(device)
    dataloader=DataLoader(ssldata,batch_size=2,shuffle=True)
    feature2patch=Feature2Patch(args).to(device)
    feextraction=get_feature_extraction(args).to(device)
    for images in dataloader:
        images.to(device)
        features=encoder(images)  #使用resnet提取特征
        patch_feature=feature2patch(features,images) #将特征打成patch同时增加细节纹理信息
        exfeature=feextraction(patch_feature)







# if __name__ == '__main__':
#     tens = torch.randn((10, 3, 256, 256))
#     main(tens,args)
#     # print(model(tens).reshape(-1,2048,7,7).shape)
