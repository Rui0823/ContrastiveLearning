import torch
import torchvision
from models.res_encoder import *
from models.res_decoder import res_de101,res_de50,res_de152
from models.vit import *
import torch.nn as nn
def get_encoder(args):
    model=None
    if args.encoder=="res50":
        pretrained=args.pretrained
        return res_en50(pretrained=pretrained)
        # print(model)
    elif args.encoder=="res101":
        pretrained = args.pretrained
        return res_en101(pretrained=pretrained)
    return model
def get_decoder(args):
    model = None
    if args.encoder == "res50":
        pretrained = args.pretrained
        return res_de50()
        # print(model)
    elif args.encoder == "res101":
        pretrained = args.pretrained
        return res_de101()
    return model


def get_feature_extraction(args):
    model=VisionTransformer(img_size=args.image_size,patch_size=args.image_size//args.patchs,embed_dim=args.embed_dim)
    return model

class Feature2Patch(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.embed_dim=args.embed_dim
        self.patchs = args.patchs
        self.conv1 = nn.Conv2d(args.patchs * args.patchs * 3, args.patchs * args.patchs * self.embed_dim,
                          kernel_size=args.image_size // args.patchs, stride=args.image_size // args.patchs, padding=0,
                          groups=args.patchs * args.patchs)
        self.scale=args.scale_size // args.patchs if args.patchs < args.scale_size else args.patchs // args.scale_size
        self.pixelshuffle=torch.nn.PixelShuffle(self.scale)
        self.conv1x1=nn.Conv2d(3,1,1,1,0) #将RGB图像的三个通道转为一个通道
    def forward(self,features,image): # B 2048 7 7 ->B 256*  7 7   B 3 14 14
        image=self.conv1x1(image)
        i_b, i_c, i_h, i_w = image.shape
        image = image.view(i_b, i_c, self.patchs, i_w // self.patchs, self.patchs, i_w // self.patchs).permute(0, 2, 4,
                                                                                                               1, 3,
                                                                                                               5).contiguous().view(
            i_b,self.patchs * self.patchs, i_c, i_w // self.patchs, i_w //self.patchs)
        B, C, H, W = features.shape
        features = features.view(B, self.patchs * self.patchs, C // (self.patchs * self.patchs), H, W).contiguous()
        features=self.pixelshuffle(features)
        output=torch.cat([image, features], dim=2).view(i_b, self.patchs * self.patchs * 3, i_h // self.patchs,
                                                i_w // self.patchs)
        output=self.conv1(output)
        output=output.flatten(2).view(i_b,self.patchs*self.patchs,self.embed_dim).contiguous()
        return output


class Patch2Feature(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.embed_dim=args.embed_dim
        self.patch_size=args.image_size//args.patchs if args.patchs>args.scale_size else args.image_size//args.scale_size
        self.proj=nn.Linear(self.embed_dim,self.patch_size**2)
        self.conv1=nn.Sequential(
            nn.Conv2d(args.patchs ** 2, args.encoder_dim, 1, 1, 0),
            nn.LayerNorm([args.encoder_dim,self.patch_size,self.patch_size]),
            nn.PReLU()
        )   #从patchs**2个通道重新映射成经过resnet后的通道数

    def forward(self,patch):
        proj_patch=self.proj(patch)
        B,C,_=proj_patch.size()
        proj_patch=proj_patch.view(B,C,self.patch_size,self.patch_size)
        feature=self.conv1(proj_patch)
        return feature







class MCL(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder=get_encoder(args)
