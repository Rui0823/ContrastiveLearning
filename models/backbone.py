import torch
import torchvision
from models.resnet import *
from models.vit import *
import torch.nn as nn
def get_encoder(args):
    model=None
    if args.encoder=="res50":
        pretrained=args.pretrained
        return resnet50(pretrained=pretrained)
        # print(model)
    elif args.encoder=="res101":
        pretrained = args.pretrained
        return resnet101(pretrained=pretrained)
    return model

def get_feature_extraction(args):
    model=VisionTransformer(img_size=args.image_szie,embed=args.embed,embed_dim=args.image_szie//args.patchs*(args.image_szie//args.patchs)*3)
    return model

class Feature2Patch(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.conv1 = nn.Conv2d(args.patchs * args.patchs * 3, args.patchs * args.patchs * 3 * ((args.image_size // args.patchs) ** 2),
                          kernel_size=args.image_size // args.patchs, stride=args.image_size // args.patchs, padding=0,
                          groups=args.patchs * args.patchs)
        self.pixelshuffle=torch.nn.PixelShuffle(32 // args.patchs if args.patchs < 32 else args.patchs // 32)
        self.patchs=args.patchs
        self.conv1x1=nn.Conv2d(3,1,1,1,0) #将图像的三个通道转为
    def forward(self,features,image):
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
        output=output.flatten(2).view(i_b,self.patchs*self.patchs,i_w//self.patchs*(i_w//self.patchs)*3).contiguous()
        return output



class MCL(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder=get_encoder(args)
