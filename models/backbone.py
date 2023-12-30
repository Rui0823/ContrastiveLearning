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


class RandomMask(nn.Module):
    #由于对patch采用了映射，是在映射前进行随机掩码还是映射后进行随机掩码这个地方需要进行实验
    def __init__(self,args):
        super().__init__()
        self.mask_index=torch.randint(2, (args.patchs*args.patchs, 1))
        self.embed_dim=args.embed_dim
        self.patchs=args.patchs
        self.args=args

    def forward(self,x):
        device = x.device
        mask=Variable(self.mask_index,requires_grad=False).to(device)
        index=torch.arange(0,self.patchs*self.patchs).unsqueeze(1).to(device)
        mask_index=(index*mask).flatten().to(device)
        out=x.index_fill(1,mask_index,0).to(device)
        return out,mask

class PatchClass(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=args.image_size // args.patchs, stride=args.image_size // args.patchs),
            nn.Sigmoid()
        )
    def forward(self,x):
        B=x.size()[0]
        out=self.conv(x)
        return out.view(B,-1)


class PatchMask(nn.Module):
    #由随机掩码对应每个patch，生成二维的patch
    def __init__(self,args):
        super().__init__()
        self.patch_size=args.image_size//args.patchs
        self.patchs=args.patchs
    def forward(self,x):
        x=x.repeat(1,1,self.patch_size*self.patch_size)
        mask=x.view(1,self.patchs,self.patchs,self.patch_size,self.patch_size)\
            .contiguous().view(1,self.patchs,self.patchs*self.patch_size,self.patch_size)\
            .permute(0,2,1,3).reshape(1,self.patch_size*self.patchs,self.patch_size*self.patchs)
        return mask


