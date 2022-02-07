from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self, inputs1, inputs2, scores) -> None:
        super().__init__()
        self.inputs1 = inputs1
        self.inputs2 = inputs2
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx: int):
        # get inputs
        return self.inputs1[idx], self.inputs2[idx], self.scores[idx]


class AugmentationDataset(Dataset):
    def __init__(self, dataset: SentenceDataset, augmentation_indices, scores) -> None:
        super().__init__()
        self.inputs1 = dataset.inputs1
        self.inputs2 = dataset.inputs2
        self.augmentation_indices = augmentation_indices
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx: int):
        # extract augmentation index for input 1 & 2
        idx1, idx2 = self.augmentation_indices[idx]
        # get inputs
        return self.inputs1[idx1], self.inputs2[idx2], self.scores[idx]
