import torch
import torch.nn as nn
from models.backbone import Feature2Patch,Patch2Feature,RandomMask,PatchClass,PatchMask
from models.res_encoder import res_en50,res_en101
from models.res_decoder import res_de101,res_de50
from models.vit import VisionTransformer
class Net(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.encoder=self.get_encoder()
        self.decoder=self.get_decoder()
        self.f2p=Feature2Patch(args)
        self.fex=self.get_feature_extraction()  #中间特征提取模块
        self.p2f=Patch2Feature(args)
        self.randommask=RandomMask(args)

    def forward(self,x):
        en_features=self.encoder(x)
        patchs=self.f2p(en_features,x)
        fex=self.fex(patchs)
        maskfeature,mask_index=self.randommask(fex)
        features=self.p2f(maskfeature)
        output=self.decoder(features)
        return output,mask_index


    def get_encoder(self):
        model = None
        if self.args.encoder == "res50":
            pretrained = self.args.pretrained
            return res_en50(pretrained=pretrained)
            # print(model)
        elif self.args.encoder == "res101":
            pretrained = self.args.pretrained
            return res_en101(pretrained=pretrained)
        return model

    def get_decoder(self):
        model = None
        if self.args.encoder == "res50":
            pretrained = self.args.pretrained
            return res_de50()
            # print(model)
        elif self.args.encoder == "res101":
            pretrained = self.args.pretrained
            return res_de101()
        return model

    def get_feature_extraction(self,blocks=6):
        model = nn.ModuleList()
        for i in range(blocks):
            model.append(VisionTransformer(img_size=self.args.image_size, patch_size=self.args.image_size // self.args.patchs,
                                  embed_dim=self.args.embed_dim))

        return nn.Sequential(*model)