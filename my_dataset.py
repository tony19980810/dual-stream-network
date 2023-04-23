import os

from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as scio

class MyDataSet(Dataset):
    def __init__(self,txt_path, transform=None):
        images_path=[]
        images_class=[]
        au_gcn=[]
        f=open(txt_path)
        readline=f.readline()
        while readline:
            tmp=readline.split(' ')
            images_path.append(tmp[0])
            images_class.append(int(tmp[1]))
            tmp1=tmp[0].split('\\')
            au_gcn.append(os.path.join('.','data','MEIP_au_O_A_30_flatten',tmp1[-1]+'.mat'))
            readline=f.readline()
        f.close
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.gcn_path=au_gcn
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        imgs=[]
        for f in range(1, 31):
            img = Image.open(os.path.join(self.images_path[item], str(f) + '.jpg'))
            # print(os.path.join(self.images_path[item], str(f) + '.jpg'))
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        label = self.images_class[item]
        gcn = scio.loadmat(self.gcn_path[item])
        gcn = gcn['data']
        gcn = torch.tensor(gcn)
        gcn = gcn.to(torch.float32)
        gcn = torch.squeeze(gcn)
        gcn = torch.unsqueeze(gcn, dim=1)
        imgs=torch.stack(imgs,dim=0)
        return imgs, gcn,label

    @staticmethod
    def collate_fn(batch):
        images, gcns,labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        gcns = torch.stack(gcns, dim=0)
        labels = torch.as_tensor(labels)
        return images, gcns,labels
