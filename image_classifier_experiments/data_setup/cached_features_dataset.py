import torch
from torch.utils.data import Dataset


class CatchedFeaturesDataset(Dataset):
    def __init__(self, features_path: str):
        cached_feature_data = torch.load(features_path)
        self.features = cached_feature_data["features"]
        self.labels = cached_feature_data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]
