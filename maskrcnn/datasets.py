# @Time : 2020-04-30 12:50 
# @Author : Ben 
# @Version：V 0.1
# @File : datasets.py
# @desc :数据集载入类

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
# from torchvision import transforms
from maskrcnn.utils import transforms


class Datasets(Dataset):
    def __init__(self, root,transforms, isTrain=True):
        self.root = root
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[item])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[item])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # print(obj_ids)
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([item])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def get_transform(train):
    trans = []
    # converts the image, a PIL image, into a PyTorch Tensor
    trans.append(transforms.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        trans.append(transforms.RandomHorizontalFlip(0.5))

    return transforms.Compose(trans)


if __name__ == '__main__':
    data_train = Datasets('data', None)
    print(data_train[0])
    print(data_train[1])
