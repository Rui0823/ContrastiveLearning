from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms
from PIL import Image
class SSLData(Dataset):
    def __init__(self,args,transform=None):
        if transform==None:
            self.transform=transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
        else:
            self.transform=transform
        self.root_dir=args.root
        # self.train=args.train
        self.images=os.listdir(self.root_dir)
    def __getitem__(self, index):
        image_name=self.images[index]
        image_path=os.path.join(self.root_dir,image_name)
        image=Image.open(image_path)
        return self.transform(image)

    def __len__(self):
        return len(self.images)