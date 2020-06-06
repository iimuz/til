import torch


class MinMaxScaler:
    def __init__(self, min_val, max_val):
        self.min_val = torch.from_numpy(min_val)
        self.scale = torch.from_numpy(1.0 / (max_val - min_val))

    def __call__(self, tensor):
        result = (tensor - self.min_val) * self.scale
        result = result.float()
        return result
