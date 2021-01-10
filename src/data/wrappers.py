from torch.utils import data


class TorchvisionDatasetWrapper(data.Dataset):
    def __init__(self, torchvision_dataset: data.Dataset):
        self.dataset = torchvision_dataset

    def __getitem__(self, item):
        features, targets = self.dataset[item]
        return {
            "features": features,
            "targets": targets,
        }
