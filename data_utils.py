from torch.utils.data import Dataset, DataLoader
from data_process import *
import os
import torch


class GIDataset(Dataset):
    def __init__(self, indexpath, datapath, split="training", savePt=False):
        pwd = os.path.abspath(os.path.dirname(__file__))
        self.datapath = os.path.join(pwd, datapath)
        self.positions = []
        self.normals = []
        self.lights = []
        self.gts = [] # groundtruths
        buffers = ["position", "light", "normal", "ground_truth"]
        n = 0
        for buffer_type in buffers:
            filepath = indexpath + split + "_" + buffer_type + ".txt"
            with open(filepath) as f:
                n = 0
                for line in f.readlines():
                    path = line.split()[0]
                    p = os.path.join(self.datapath, path)
                    if not os.path.isfile(p):
                        continue
                    if not savePt:
                        words = path.split('/')
                        words[1] = 'Position'
                        p1 = os.path.join(self.datapath, "/".join(words) + ".pt")
                        words[1] = 'Normals'
                        p2 = os.path.join(self.datapath, "/".join(words) + ".pt")
                        words[1] = 'GroundTruth'
                        p3 = os.path.join(self.datapath, "/".join(words) + ".pt")
                        words[1] = 'Light'
                        p4 = os.path.join(self.datapath, "/".join(words) + ".pt")
                        if os.path.isfile(p1) and  os.path.isfile(p2) and  os.path.isfile(p3) and  os.path.isfile(p4):
                            if buffer_type == "position":
                                self.positions.append(p + ".pt")
                            elif buffer_type == "normal":
                                self.normals.append(p + ".pt")
                            elif buffer_type == "light":
                                self.lights.append(p + ".pt")
                            else:
                                self.gts.append(p + ".pt")
                        continue
                    if buffer_type == "position":
                        if savePt:
                            arr = np.array(exr_loader(os.path.join(self.datapath, path), ndim=3))
                            ts = torch.tensor(arr).permute(2, 0, 1)
                            torch.save(ts, p + ".pt")
                        self.positions.append(p + ".pt")
                    elif buffer_type == "normal":
                        if savePt:
                            arr = np.array(exr_loader(os.path.join(self.datapath, path), ndim=3))
                            ts = torch.tensor(arr).permute(2, 0, 1)
                            torch.save(ts, p + ".pt")
                        self.normals.append(p + ".pt")
                    elif buffer_type == "light":
                        if savePt:
                            arr = np.array(exr_loader(os.path.join(self.datapath, path), ndim=3))
                            ts = torch.tensor(arr).permute(2, 0, 1)
                            torch.save(ts, p + ".pt")
                        self.lights.append(p + ".pt")
                    else:
                        if savePt:
                            arr = np.array(exr_loader(os.path.join(self.datapath, path), ndim=3))
                            ts = torch.tensor(arr).permute(2, 0, 1)
                            torch.save(ts, p + ".pt")
                        self.gts.append(p + ".pt")
                    n += 1
                    if n%500 == 0:
                        print(buffer_type, n)
                    if n > 2000:
                        break
                    
    def __len__(self):
        return len(self.positions) * 3

    def __getitem__(self, idx):
        list_idx = idx // 3
        rgb_idx = idx % 3
        pos = torch.load(self.positions[list_idx])
        normal = torch.load(self.normals[list_idx])
        light = torch.load(self.lights[list_idx])[rgb_idx,:,:].unsqueeze(0)
        gt = torch.load(self.gts[list_idx])[rgb_idx,:,:].unsqueeze(0)

        return pos, normal, light, gt

class AODataset(Dataset):
    def __init__(self, indexpath, datapath, split="training", savePt=False):
        pwd = os.path.abspath(os.path.dirname(__file__))
        self.datapath = os.path.join(pwd, datapath)
        self.positions = []
        self.normals = []
        self.gts = [] # groundtruths
        buffers = ["position", "normal", "groundtruth"]
        n = 0
        for buffer_type in buffers:
            filepath = indexpath + split + "_" + buffer_type + ".txt"
            with open(filepath) as f:
                for line in f.readlines():
                    path = line.split()[0]
                    p = os.path.join(self.datapath, path)
                    if not os.path.isfile(p):
                        continue
                    if not savePt:
                        words = path.split('/')
                        words[1] = 'Position'
                        p1 = os.path.join(self.datapath, "/".join(words) + ".pt")
                        words[1] = 'Normals'
                        p2 = os.path.join(self.datapath, "/".join(words) + ".pt")
                        words[1] = 'GroundTruth'
                        p3 = os.path.join(self.datapath, "/".join(words) + ".pt")
                        if os.path.isfile(p1) and  os.path.isfile(p2) and  os.path.isfile(p3):
                            if buffer_type == "position":
                                self.positions.append(p + ".pt")
                            elif buffer_type == "normal":
                                self.normals.append(p + ".pt")
                            else:
                                self.gts.append(p + ".pt")
                        continue
                    if buffer_type == "position":
                        if savePt:
                            arr = np.array(exr_loader(os.path.join(self.datapath, path), ndim=3))
                            ts = torch.tensor(arr).permute(2, 0, 1)
                            torch.save(ts, p + ".pt")
                        self.positions.append(p + ".pt")
                    elif buffer_type == "normal":
                        if savePt:
                            arr = np.array(exr_loader(os.path.join(self.datapath, path), ndim=3))
                            ts = torch.tensor(arr).permute(2, 0, 1)
                            torch.save(ts, p + ".pt")
                        self.normals.append(p + ".pt")
                    else:
                        if savePt:
                            arr = np.array(exr_loader(os.path.join(self.datapath, path), ndim=1))
                            ts = torch.tensor(arr).unsqueeze(0)
                            torch.save(ts, p + ".pt")
                        self.gts.append(p + ".pt")
                    n += 1
                    if n%1000 == 0:
                        print(buffer_type, n)
                    
                        
            
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        pos = torch.load(self.positions[idx])
        normal = torch.load(self.normals[idx])
        gt = torch.load(self.gts[idx])

        return pos, normal, gt