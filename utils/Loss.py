import torch
import torch.nn as nn
from models.backbone import PatchClass,PatchMask
class TotalLoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.patchclass=PatchClass(args)
        self.patchmask=PatchMask(args)
        self.bceloss=nn.BCELoss()
        self.l2loss=nn.L1Loss()
    def forward(self,out,label,mask_index):
        pc=self.patchclass(label)
        pm=self.patchmask(mask_index)
        bceloss = self.bceloss(pc.float(), mask_index.flatten().repeat(pc.size()[0],1).float())
        out=out*pm
        label=label*pm
        l2loss=self.l2loss(out,label)
        return bceloss+l2loss
