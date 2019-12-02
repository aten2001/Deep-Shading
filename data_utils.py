from torch.utils.data import Dataset, DataLoader
from data_process import *

class AODataset(Dataset):
    def __init__(self, root="./dataset/", split="train"):
        if split == "train":
            data = load_train("AO", root=root)
        else:
            data = load_test("AO", root=root)
        self.position_data = data["position"]
        self.normal_data = data["normal"]
        self.groundtruth_data = data["groundtruth"]

    def __len__(self):
        return self.groundtruth_data.shape[0]

    def __getitem__(self, idx):
        return self.position_data[idx], self.normal_data[idx], self.groundtruth_data[idx]