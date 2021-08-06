import torch
import torchvision
import sklearn.preprocessing as pp
from torch.utils.data import Dataset

class MNIST(Dataset):
    def __init__(
        self, train=True, noise=None,
    ):

        self.noise = noise

        # MNIST with full range of data mapped to (0,1)

        train_data = torchvision.datasets.MNIST("data/", train=True, download=True).data
        train_labels = torchvision.datasets.MNIST(
            "data/", train=True, download=True
        ).targets
        train_data = train_data.reshape(train_data.shape[0], 28 * 28)
        train_data = pp.minmax_scale(train_data)
        train_data = torch.Tensor(train_data)

        test_data = torchvision.datasets.MNIST("data/", train=False, download=True).data
        test_labels = torchvision.datasets.MNIST(
            "data/", train=False, download=True
        ).targets
        test_data = test_data.reshape(test_data.shape[0], 28 * 28)
        test_data = pp.minmax_scale(test_data)
        test_data = torch.Tensor(test_data)

        if train:
            self.data = train_data
            self.labels = train_labels

            theta = torch.zeros(self.data.shape[0], 2, 3)
            for i in range(self.data.shape[0]):
                As = torch.randn(2) * 0.5
                Ts = torch.randn(2) * 3.0

                theta[i, 0, 0] = 1
                theta[i, 0, 1] = As[0]
                theta[i, 0, 2] = Ts[0] * 2 / 28 + 1 + As[0] - 1
                theta[i, 1, 0] = As[1]
                theta[i, 1, 1] = 1
                theta[i, 1, 2] = Ts[1] * 2 / 28 + 1 + As[1] - 1

            grid = torch.nn.functional.affine_grid(
                theta, [self.data.shape[0], 1, 28, 28]
            )

            transform = torch.nn.functional.grid_sample(
                self.data.data.reshape(self.data.shape[0], 1, 28, 28), grid
            )

            self.data.data = transform.reshape(self.data.shape[0], 28 * 28)

        else:
            self.data = test_data
            self.labels = test_labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.noise is None:
            return self.data[idx]
        else:
            return self.data[idx] + self.noise.sample(self.data[idx].shape)
