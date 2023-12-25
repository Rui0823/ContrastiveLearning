from torch.utils import data
import os
class SSLData(data.dataset):
    def __init__(self,root,args,transform):
        self.root=root
        self.train=args.train

    def __len__(self):
        pass