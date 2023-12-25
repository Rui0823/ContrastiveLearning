import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser(description='配置网络参数')
parser.add_argument('--patch_size', default=4, type=int, help="将最后网络输出变成多少个patch")
parser.add_argument('--batch_size', default=8, type=int, help="每次传入网络的batch")
parser.add_argument('--backbone', default="res101", type=str, help="backbone")
parser.add_argument('--root', default=r'/data',type=str, help='dataset root')
parser.add_argument('--lr', default=2e-4,type=int, help='学习率')
parser.add_argument('--cuda', default=True,type=bool, help='是否使用gpu')
parser.add_argument('--pretrained', default=True,type=bool, help='是否使用预训练模型')
args=parser.parse_args()
from models.backbone import get_encoder
def build_model(args):
    return get_encoder(args)

def main(image,args):
    print(args)
    # model = build_model(args)
    # feature=model(image)
    feature=torch.randn([10,32,1,1])
    B,C,H,W=feature.shape
    print(feature)
    # print(model)
    feature_lis=feature.view(B,args.patch_size*args.patch_size,C//(args.patch_size*args.patch_size),H,W).contiguous()
    print("***************************")
    print(feature_lis.shape)

if __name__ == '__main__':
    tens = torch.randn((10, 3, 256, 256))
    main(tens,args)
    # print(model(tens).reshape(-1,2048,7,7).shape)
