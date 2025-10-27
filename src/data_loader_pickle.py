import pickle
from torch_geometric.data import Dataset

class PickleEventGraphDataset(Dataset):
    def __init__(self, pickle_path, **kwargs):
        super(PickleEventGraphDataset, self).__init__()
        with open(pickle_path, "rb") as f:
            self.graph = pickle.load(f)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx >= 1:
            raise IndexError("This dataset contains only one graph.")
        return self.graph
