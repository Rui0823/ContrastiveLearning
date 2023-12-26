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
from models.backbone import Feature2Patch

parser = argparse.ArgumentParser(description='配置网络参数')
parser.add_argument('--patchs', default=16, type=int, help="将最后网络输出变成多少个patch")
parser.add_argument('--batch_size', default=8, type=int, help="每次传入网络的batch")
parser.add_argument('--encoder', default="res101", type=str, help="backbone")
parser.add_argument('--root', default=r'/data',type=str, help='dataset root')
parser.add_argument('--lr', default=2e-4,type=int, help='学习率')
parser.add_argument('--cuda', default=True,type=bool, help='是否使用gpu')
parser.add_argument('--pretrained', default=True,type=bool, help='是否使用预训练模型')
parser.add_argument('--image_size', default=224, type=int, help="图像的大小")
parser.add_argument('--is_embed', default=False, type=bool, help="配置VIT的参数，是否使用patch_embed")
args=parser.parse_args()


def readImage(path='data/a.png', size=256):#这里可以替换成自己的图片
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
    image = readImage(size=224)
    image=image.unsqueeze(0).cuda()
    i_b,i_c,i_h,i_w=image.shape
    model = build_model(args).cuda()
    features=model(image)
    fea_patch=Feature2Patch(args).cuda()
    output=fea_patch(features,image)
    print(output.shape)








    # showTorchImage(mode)

    # image=image.view(i_b, i_c, args.patchs, i_w // args.patchs, args.patchs, i_w // args.patchs).permute(0, 2, 4,
    #                                                                                                           1, 3,
    #                                                                                                           5).contiguous().view(
    #     i_b, args.patchs * args.patchs, i_c, i_w //args.patchs, i_w // args.patchs)
    # print(image.shape, feature.shape)
    # image = image.mean(dim=2).unsqueeze(2)
    # B, C, H, W = feature.shape
    # feature=feature.view(B, args.patchs * args.patchs, C // (args.patchs * args.patchs), H,W).contiguous()
    # ps = torch.nn.PixelShuffle(32 // args.patchs if args.patchs < 32 else args.patchs // 32)
    # feature=ps(feature)
    # print(image.shape,feature.shape)
    # final = torch.cat([image, feature], dim=2).view(B,args.patchs*args.patchs*3,i_h//args.patchs,i_w//args.patchs).cuda() #将图像与特征进行cat，并且以每三张为一组
    # conv1=nn.Conv2d(args.patchs*args.patchs*3,args.patchs*args.patchs*3*((i_w//args.patchs)**2),kernel_size=i_w//args.patchs,stride=i_w//args.patchs,padding=0,groups=args.patchs*args.patchs).cuda()
    # final=conv1(final).flatten(2) #以三个为一组进行分组卷积，然后每组再产生 3*(i_w//args.patchs)*(i_w//args.patchs)个卷积，这个地方其实就是vit的patch_embed,然后flatten开，再分成patcha_size*patchs组
    # print(final.shape)
    # final=final.view(i_b,args.patchs*args.patchs,i_w//args.patchs*(i_w//args.patchs)*3).contiguous()
    # vit=VisionTransformer(img_size=i_h,patch_size=i_w//args.patchs,embed=False,embed_dim=i_w//args.patchs*(i_w//args.patchs)*3)
    # vit_feature=vit(final)
    # print(vit_feature.shape)

    # mode = mode.view(1,3,args.patchs,256//args.patchs,args.patchs,256//args.patchs).permute(0,2,4,1,3,5).contiguous().view(1,args.patchs*args.patchs,3,256//int(args.patchs),256//int(args.patchs))
    # B, C, H, W = feature.shape
    #
    #
    # mode=mode.mean(dim=2).unsqueeze(2)
    # print(mode.shape)
    # # print(model)
    # feature_lis = feature.view(B, args.patchs * args.patchs, C // (args.patchs * args.patchs), H,
    #                            W).contiguous()
    #
    # ps=torch.nn.PixelShuffle(32//args.patchs if args.patchs<32 else args.patchs//32)
    # output=ps(feature_lis)
    # print(output.shape)
    # final=torch.cat([mode, output], dim=2)
    # # conv=nn.Conv2d(in_channels=3,out_channels=args.patchs*args.patchs,kernel_size=args.patchs,stride=args.patchs,padding=0)
    #
    #
    # print(final.shape)
    # for i in range(args.patchs*args.patchs):
    #     showTorchImage(final.squeeze()[i])


