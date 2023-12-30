import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='配置网络参数')
parser.add_argument('--patchs', default=16, type=int, help="将最后网络输出变成多少个patch")
parser.add_argument('--epoch', default=100, type=int, help="训练多少个周期")
parser.add_argument('--batch_size', default=8, type=int, help="每次传入网络的batch")
parser.add_argument('--encoder', default="res50", type=str, help="encoder for image")
parser.add_argument('--decoder', default="res50", type=str, help="decoder for feature")
parser.add_argument('--root', default=r'data',type=str, help='dataset root')
parser.add_argument('--lr', default=2e-4,type=int, help='学习率')
parser.add_argument('--cuda', default=True,type=bool, help='是否使用gpu')
parser.add_argument('--pretrained', default=True,type=bool, help='是否使用预训练模型，主要是encoder的预训练模型')
parser.add_argument('--image_size', default=256, type=int, help="图像的大小")
parser.add_argument('--is_patch_embed', default=False, type=bool, help="配置VIT的参数，是否使用patch_embed")
parser.add_argument('--is_proj', default=True, type=bool, help="是否进行映射,默认为True")
parser.add_argument('--embed_dim', default=1024, type=int, help="默认对转成patch的向量进行同一映射,类比vit中的embed_dim")
parser.add_argument('--scale_size', default=32, type=int, help="在使用编码器时，编码后的特征缩小了多少倍，需手动更改,resnet101默认等于32")
parser.add_argument('--encoder_dim', default=2048, type=int, help="在使用编码器时，编码后的特征维度是多少，需手动更改,resnet101默认等于2048")
args=parser.parse_args()
from models.MyNet import Net
from utils.dataset import SSLData
from torch.optim import AdamW
from utils.Loss import TotalLoss

def main(args):

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    net = Net(args).to(device)
    optimizer = AdamW(params=net.parameters(),lr=args.lr,betas=(0.9,0.999),weight_decay=0.5e-4)
    ssldata=SSLData(args)
    dataloader=DataLoader(ssldata,batch_size=2,shuffle=True)
    lossfunc=TotalLoss(args).to(device)
    for i in range(args.epoch):
        for index,images in enumerate(dataloader):
            images=images.to(device)
            output,mask_index=net(images)
            loss=lossfunc(output,images,mask_index)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


main(args)




# if __name__ == '__main__':
#     tens = torch.randn((10, 3, 256, 256))
#     main(tens,args)
#     # print(model(tens).reshape(-1,2048,7,7).shape)
