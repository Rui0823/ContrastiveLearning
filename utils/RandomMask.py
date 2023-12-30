import torch
import torch.nn as nn

class PatchMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask=torch.randint(2,(1,16)).repeat(1,16,1)

# lis=torch.zeros((1,16,5))
# index=torch.LongTensor([0,2])
# b=a.index_fill(1,index,0)
# print(b)
a=torch.randint(2, (16, 1)).repeat(1,1,49)
print(a)
a=a.view(1,4,4,7,7).contiguous()
print(a)
a=a.view(1,4,28,7).permute(0,2,1,3)
print(a)
a=a.reshape(1,28,28)
print(a)
from torchvision.transforms import ToPILImage
show = ToPILImage() # 可以把Tensor转成Image，方便可视化
show(a.float()).show()
# b=torch.arange(0,16).unsqueeze(1)
# out=lis.index_fill(1,(a*b).flatten(),1)
# print(out)