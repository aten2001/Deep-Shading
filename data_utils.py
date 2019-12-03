from torch.utils.data import Dataset, DataLoader
from data_process import *
import os


class AODataset(Dataset):
    def __init__(self, indexpath, datapath, split):
        pwd = os.path.abspath(os.path.dirname(__file__))
        self.datapath = os.path.join(pwd, datapath)
        self.positions = []
        self.normals = []
        self.gts = [] # groundtruths
        buffers = ["position", "normal", "groundtruth"]
        for buffer_type in buffers:
            filepath = indexpath + split + "_" + buffer_type + ".txt"
            with open(filepath) as f:
                for line in f.readlines():
                    path = line.split()[0]
                    p = os.path.join(self.datapath, path)
                    if not os.path.isfile(p):
                        continue
                    if buffer_type == "position":
                        self.positions.append(path)
                    elif buffer_type == "normal":
                        self.normals.append(path)
                    else:
                        self.gts.append(path)
            
    def __len__(self):
        return len(self.positions)
        # n = 0
        # for path in self.positions:
        #     p = os.path.join(self.datapath, path)
        #     if os.path.isfile(p):
        #         #print(p)
        #         n += 1
        # return n

    def __getitem__(self, idx):
        n = idx
        # for path in self.positions:
        #     if os.path.isfile(path):
        #         n += 1
        #     if idx == n:
        #         break

        pos = np.array(exr_loader(os.path.join(self.datapath, self.positions[n]), ndim=3))
        normal = np.array(exr_loader(os.path.join(self.datapath, self.normals[n]), ndim=3))
        gt = np.array(exr_loader(os.path.join(self.datapath, self.gts[n]), ndim=1))

        return pos, normal, gt