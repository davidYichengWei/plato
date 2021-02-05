"""
The COCO dataset.

For more information about COCO 128, which contains the first 128 images of the
COCO 2017 dataset, refer to https://www.kaggle.com/ultralytics/coco128.

For more information about the COCO 2017 dataset, refer to http://cocodataset.org.
"""

import os
import logging

from config import Config
from datasets import base
from utils.yolov5.datasets import LoadImagesAndLabels
from utils.yolov5.general import check_img_size


class Dataset(base.Dataset):
    """The COCO dataset."""
    def __init__(self, path):
        super().__init__(path)

        if not os.path.exists(path):
            os.makedirs(path)

        logging.info("Downloading the COCO dataset. This may take a while.")

        urls = Config().data.download_urls
        for url in urls:
            if not os.path.exists(path + url.split('/')[-1]):
                Dataset.download(url, path)

        assert 'grid_size' in Config().params

        self.grid_size = Config().params['grid_size']
        self.image_size = check_img_size(Config().data.image_size,
                                         self.grid_size)

        print(self.grid_size)
        print(self.image_size)
        self.train_set = None
        self.test_set = None

    @staticmethod
    def num_train_examples():
        return Config().data.num_train_examples

    @staticmethod
    def num_test_examples():
        return Config().data.num_test_examples

    @staticmethod
    def num_classes():
        return Config().data.num_classes

    def classes(self):
        """Obtains a list of class names in the dataset."""
        return Config().data.classes

    def get_train_set(self):
        single_class = (Config().data.num_classes == 1)

        if self.train_set is None:
            self.train_set = LoadImagesAndLabels(
                Config().data.train_path,
                self.image_size,
                Config().trainer.batch_size,
                augment=False,  # augment images
                hyp=None,  # augmentation hyperparameters
                rect=False,  # rectangular training
                cache_images=False,
                single_cls=single_class,
                stride=int(self.grid_size),
                pad=0.0,
                image_weights=False,
                prefix='')

        return self.train_set

    def get_test_set(self):
        single_class = (Config().data.num_classes == 1)

        if self.test_set is None:
            self.test_set = LoadImagesAndLabels(
                Config().data.test_path,
                self.image_size,
                Config().trainer.batch_size,
                augment=False,  # augment images
                hyp=None,  # augmentation hyperparameters
                rect=False,  # rectangular training
                cache_images=False,
                single_cls=single_class,
                stride=int(self.grid_size),
                pad=0.0,
                image_weights=False,
                prefix='')

        return self.test_set
