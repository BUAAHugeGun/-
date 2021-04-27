import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets
from torch.nn import functional as F
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
import os
import numpy as np
import dataset.cifar10 as cifar10
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

ImageFile.LOAD_TRUNCATED_IMAGES = True


class cifar10_dataset(Dataset):
    def __init__(self, path, train=True, **kwargs):
        super(cifar10_dataset, self).__init__()
        data_all = cifar10.load_CIFAR10(path)
        if train:
            self.set = [data_all[0], data_all[1]]
        else:
            self.set = [data_all[2], data_all[3]]
        labels = kwargs.get('labels')
        data_sum = kwargs.get('data_sum')
        if labels is not None:
            self.set[0] = [self.set[0][i] for i in range(len(self.set[0])) if self.set[1][i] in labels]
            self.set[1] = [self.set[1][i] for i in range(len(self.set[1])) if self.set[1][i] in labels]
        if data_sum is not None:
            self.set[0] = self.set[0][0:data_sum]
            self.set[1] = self.set[1][0:data_sum]
        self.transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

    def __len__(self):
        return len(self.set[0])

    def __getitem__(self, id):
        return [self.transform(Image.fromarray(self.set[0][id].astype(np.uint8))), self.set[1][id]]


class facades_dataset(Dataset):
    def __init__(self, path, train=True):
        super(facades_dataset, self).__init__()
        if train:
            path = os.path.join(path, "train")
        else:
            path = os.path.join(path, "test")
        self.a_path = os.path.join(path, "a")
        self.b_path = os.path.join(path, "b")
        self.image_filenames = [x for x in os.listdir(self.a_path)]

        self.transform = transforms.Compose(
            [  # transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, id):
        a = Image.open(os.path.join(self.a_path, self.image_filenames[id])).convert('RGB')
        b = Image.open(os.path.join(self.b_path, self.image_filenames[id])).convert('RGB')
        a = a.resize((286, 286), Image.BICUBIC)
        b = b.resize((286, 286), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))

        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]  # flip
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)
        return a, b


class coco_obj_dataset(Dataset):

    def __init__(self, path, train=True, **kwargs):
        super(coco_obj_dataset, self).__init__()
        if train:
            self.path = os.path.join(path, "train_cut")
            self.label_path = os.path.join(path, "train_label_cut")
            self.mask_path = os.path.join(path, "train_mask_cut")
        else:
            self.path = os.path.join(path, "val_cut")
            self.label_path = os.path.join(path, "val_label_cut")
            self.mask_path = os.path.join(path, "val_mask_cut")
        data_list_path = os.path.join(self.path, "data_list.txt")
        f = open(data_list_path)
        classes = kwargs.get('classes', None)
        if classes is None:
            assert 0
        classes_inv = {}
        for i in range(0, len(classes)):
            classes_inv[classes[i]] = i
        lines = f.readlines()

        # annotation_dir = os.path.join(path, "annotations", "instances_{}2017.json".format("train" if train else 'val'))
        # self.coco = COCO(annotation_dir)

        self.file_name_label_list = []
        self.image_size = kwargs.get('image_size', 128)
        for line in lines:
            line = line[0:-1]
            filename, class_num, class_name = line.split(' ', 2)
            id, obj_id = filename.split('.')[0].split('_')
            id, obj_id, class_num = int(id), int(obj_id), int(class_num)
            if class_num in classes:
                self.file_name_label_list.append(
                    [filename, obj_id, classes_inv[class_num]])  # class number start from 0

        self.transform = transforms.Compose(
            [transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
             transforms.ToTensor(),
             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

    def __len__(self):
        return len(self.file_name_label_list)

    def __getitem__(self, id):
        a = self.transform(Image.open(os.path.join(self.path, self.file_name_label_list[id][0])))
        b = self.transform(Image.open(os.path.join(self.label_path, self.file_name_label_list[id][0])))
        mask = transforms.ToTensor()(transforms.Resize((self.image_size, self.image_size), Image.NEAREST)(
            Image.open(os.path.join(self.mask_path, self.file_name_label_list[id][0]))))
        t1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        t2 = transforms.Normalize((0.5), (0.5))
        if a.shape[0] == 1:
            a = a.expand([3, -1, -1])
        a, b = t1(a), t2(b)
        return a, b, mask, torch.tensor(self.file_name_label_list[id][2])


def build_data(tag, path, batch_size, training, num_worker, **kwargs):
    if tag == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        mnist = tv_datasets.MNIST(root="../data/", train=training, transform=transform, download=False)
        dataloader = DataLoader(mnist, batch_size, shuffle=True, num_workers=num_worker)
        return dataloader
    elif tag == "cifar10":
        return DataLoader(cifar10_dataset(os.path.join(path, "cifar10"), **kwargs), batch_size, shuffle=True,
                          num_workers=num_worker)
    elif tag == "facades":
        return DataLoader(facades_dataset(os.path.join(path, "facades")), batch_size, shuffle=True,
                          num_workers=num_worker)
    elif tag == "coco_obj":
        return DataLoader(coco_obj_dataset(os.path.join(path, 'COCO'), **kwargs), batch_size, shuffle=True,
                          num_workers=num_worker)


if __name__ == "__main__":
    # data = build_data("cifar10", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"), 8, True, 0)
    # data = build_data("facades", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"), 8, True, 0)
    data = build_data('coco_obj', os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"), 16, True, 0,
                      classes=[1])
    print(len(data))
    for i, data in enumerate(data):
        print(i, data[0].shape, data[1].shape)
        exit(0)
