from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from main import build_model
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.vit import VisionTransformer

parser = argparse.ArgumentParser(description='配置网络参数')
parser.add_argument('--patch_size', default=16, type=int, help="将最后网络输出变成多少个patch")
parser.add_argument('--batch_size', default=8, type=int, help="每次传入网络的batch")
parser.add_argument('--backbone', default="res101", type=str, help="backbone")
parser.add_argument('--root', default=r'/data',type=str, help='dataset root')
parser.add_argument('--lr', default=2e-4,type=int, help='学习率')
parser.add_argument('--cuda', default=True,type=bool, help='是否使用gpu')
parser.add_argument('--pretrained', default=True,type=bool, help='是否使用预训练模型')
args=parser.parse_args()


def readImage(path='a.jpg', size=256):#这里可以替换成自己的图片
    image = Image.open(path)
    transform1 = transforms.Compose([
        # transforms.Scale(size),
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    tens = transform1(image)
    # print(tens.shape)
    return tens


def showTorchImage(image):
    mode = transforms.ToPILImage()(image)
    # print(type(mode))
    # plt.imshow(mode[0])
    # plt.imshow(mode[1])
    # plt.imshow(mode[2])
    plt.imshow(mode)
    plt.show()


if __name__ == '__main__':
    image = readImage(size=256)
    image=image.unsqueeze(0)
    i_b,i_c,i_h,i_w=image.shape
    model = build_model(args)
    feature=model(image)
    # showTorchImage(mode)
    image=image.view(i_b, 3, args.patch_size, 256 // args.patch_size, args.patch_size, 256 // args.patch_size).permute(0, 2, 4,
                                                                                                              1, 3,
                                                                                                              5).contiguous().view(
        i_b, args.patch_size * args.patch_size, 3, 256 //args.patch_size, 256 // args.patch_size)
    print(image.shape, feature.shape)
    image = image.mean(dim=2).unsqueeze(2)
    B, C, H, W = feature.shape
    feature=feature.view(B, args.patch_size * args.patch_size, C // (args.patch_size * args.patch_size), H,W).contiguous()
    ps = torch.nn.PixelShuffle(32 // args.patch_size if args.patch_size < 32 else args.patch_size // 32)
    feature=ps(feature)
    print(image.shape,feature.shape)
    final = torch.cat([image, feature], dim=2).view(B,args.patch_size*args.patch_size*3,i_h//args.patch_size,i_w//args.patch_size)
    conv1=nn.Conv2d(args.patch_size*args.patch_size*3,args.patch_size*args.patch_size*3*((i_w//args.patch_size)**2),kernel_size=16,stride=16,padding=0,groups=args.patch_size*args.patch_size)
    final=conv1(final).flatten(2).view(i_b,args.patch_size*args.patch_size,args.patch_size*args.patch_size*3).contiguous()
    vit=VisionTransformer(img_size=i_h,patch_size=args.patch_size,embed=False)
    print(vit(final).shape)
    # mode = mode.view(1,3,args.patch_size,256//args.patch_size,args.patch_size,256//args.patch_size).permute(0,2,4,1,3,5).contiguous().view(1,args.patch_size*args.patch_size,3,256//int(args.patch_size),256//int(args.patch_size))
    # B, C, H, W = feature.shape
    #
    #
    # mode=mode.mean(dim=2).unsqueeze(2)
    # print(mode.shape)
    # # print(model)
    # feature_lis = feature.view(B, args.patch_size * args.patch_size, C // (args.patch_size * args.patch_size), H,
    #                            W).contiguous()
    #
    # ps=torch.nn.PixelShuffle(32//args.patch_size if args.patch_size<32 else args.patch_size//32)
    # output=ps(feature_lis)
    # print(output.shape)
    # final=torch.cat([mode, output], dim=2)
    # # conv=nn.Conv2d(in_channels=3,out_channels=args.patch_size*args.patch_size,kernel_size=args.patch_size,stride=args.patch_size,padding=0)
    #
    #
    # print(final.shape)
    # for i in range(args.patch_size*args.patch_size):
    #     showTorchImage(final.squeeze()[i])


