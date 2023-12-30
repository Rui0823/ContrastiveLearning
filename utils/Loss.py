import torch
import torch.nn as nn
from models.backbone import PatchClass,PatchMask
class TotalLoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.patchclass=PatchClass(args)
        self.patchmask=PatchMask(args)
        self.bceloss=nn.BCEWithLogitsLoss()
        self.l2loss=nn.MSELoss()
    def forward(self,out,label,mask_index):
        pc=self.patchclass(label)
        bceloss=self.bceloss(pc,mask_index)
        pm=self.patchmask(mask_index)
        out=out*pm
        label=label*pm
        l2loss=self.l2loss(out,label)
        return bceloss+l2loss
