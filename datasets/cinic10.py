"""
The CINIC-10 dataset.
For more information about CINIC-10, refer to
https://github.com/BayesWatch/cinic-10
"""

import os
import logging
from torchvision import datasets, transforms

from config import Config
from datasets import base


class Dataset(base.Dataset):
    """The CINIC-10 dataset."""
    def __init__(self, path):
        super().__init__(path)
        if not os.path.exists(path):
            os.makedirs(path)

        logging.info(
            "Downloading the CINIC-10 dataset. This may take a while.")
        url = Config().data.download_url
        if not os.path.exists(path + url.split('/')[-1]):
            Dataset.download(url, path)

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                 [0.24205776, 0.23828046, 0.25874835])
        ])
        self.trainset = datasets.ImageFolder(root=self.cinic_path + '/train',
                                             transform=_transform)
        self.testset = datasets.ImageFolder(root=self.cinic_path + '/test',
                                            transform=_transform)

    @staticmethod
    def num_train_examples():
        return 90000

    @staticmethod
    def num_test_examples():
        return 90000

    @staticmethod
    def num_classes():
        return 10
