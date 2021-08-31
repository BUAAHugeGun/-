import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets
from torch.nn import functional as F
from torch import nn
from PIL import Image, ImageFilter
from PIL import ImageFile
from torchvision import transforms
import os
import numpy as np
import dataset.cifar10 as cifar10
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import math
from tqdm import tqdm
from torchvision.utils import save_image

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
        count = []
        Max = 0
        for x in classes:
            Max = x if Max < x else Max
        for i in range(0, Max + 1):
            count.append(0)

        self.file_name_label_list = []
        self.image_size = kwargs.get('image_size', 128)
        for line in lines:
            line = line[0:-1]
            filename, class_num, class_name = line.split(' ', 2)
            id, obj_id = filename.split('.')[0].split('_')
            id, obj_id, class_num = int(id), int(obj_id), int(class_num)
            if class_num not in classes:
                continue
            if len(classes) > 1 and count[class_num] > 10000:
                continue
            count[class_num] += 1
            self.file_name_label_list.append([filename, obj_id, classes_inv[class_num]])  # class number start from 0

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


class coco_synthesis_dataset(Dataset):
    def __init__(self, path, train, **kwargs):
        super(coco_synthesis_dataset, self).__init__()
        self.classes = kwargs.get('classes', None)
        if self.classes is None:
            assert 0
        self.obj_model = kwargs.get('obj_model', None)
        if self.obj_model is None:
            assert 0
        self.classes_inv = {}
        for i in range(0, len(self.classes)):
            self.classes_inv[self.classes[i]] = i

        self.bg_image_dir = path
        self.base_image_dir = path[0:-2]
        self.annotation_dir = os.path.join(path, "..", "annotations",
                                           "instances_{}2017.json".format("train" if train else 'val'))
        self.origin_image_dir = os.path.join(path, "..", "{}_image".format("train" if train else "val"))
        self.origin_label_dir = os.path.join(path, "..", "{}_label".format("train" if train else "val"))
        # self.obj_mask_dir = os.path.join(path, "..", "{}_mask_cut".format("train" if train else "val"))
        # self.obj_label_dir = os.path.join(path, "..", "{}_label_cut".format("train" if train else "val"))

        data_path = os.path.join(self.origin_image_dir, "..")
        class_num = len(self.classes)
        self.data_path = os.path.join(data_path, "synthesis_{}_{}".format("train" if train else "val", class_num))

        file_name_list_file = open(os.path.join(path, "file_name.txt"), "r")
        lines = file_name_list_file.readlines()
        self.image_id_to_file_name = []
        for line in lines:
            self.image_id_to_file_name.append(line.split('.')[0])

        file_name_list_file = open(os.path.join(self.base_image_dir, "file_name.txt"), "r")
        lines = file_name_list_file.readlines()
        self.file_name_to_base_image_name = {}
        for i in range(len(lines)):
            line = lines[i]
            self.file_name_to_base_image_name[line.split('.')[0]] = "img" + str(i).zfill(6) + ".png"

        self.image_size = kwargs.get('image_size', 64)
        self.transform = transforms.Compose(
            [transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
             transforms.ToTensor(),
             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        self.data = []
        if not self.check_data():
            self.valid_data=[]
            self.coco = COCO(self.annotation_dir)
            self.data = []
            print("preparing synthesis data")
            for i in tqdm(range(len(self.image_id_to_file_name))):
                origin_image_id = self.image_id_to_file_name[i]
                image_file_name = origin_image_id + ".png"
                image_file_path = os.path.join(self.data_path, image_file_name)
                #if os.path.exists(image_file_path):
                #    continue
                idata = self.prepare(i)
                if idata is None:
                    continue
                self.valid_data.append(i)
                image = (idata[0] / 2 + 0.5).clamp(0, 1)
                if not os.path.exists(self.data_path):
                    os.mkdir(self.data_path)
                save_image(image, image_file_path)

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, id):
        id=self.valid_data[id]
        t1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        origin_image_id = self.image_id_to_file_name[id]
        image_file_name = origin_image_id + ".png"
        origin_image_file_name = origin_image_id + ".jpg"
        image_file_path = os.path.join(self.data_path, image_file_name)
        origin_image_file_path = os.path.join(self.origin_image_dir, origin_image_file_name)

        image = t1(self.transform(Image.open(image_file_path)))
        ori = Image.open(origin_image_file_path)
        shape = torch.tensor(transforms.ToTensor()(ori).shape)
        ori = self.transform(ori)
        if ori.shape[0] == 1:
            ori = ori.expand([3, -1, -1])
        ori = t1(ori)
        return image, ori, shape

    def check_data(self):
        print("checking synthesis data")
        for id in tqdm(range(len(self.image_id_to_file_name))):
            origin_image_id = self.image_id_to_file_name[id]
            image_file_name = origin_image_id + ".png"
            image_file_path = os.path.join(self.data_path, image_file_name)
            if not os.path.exists(image_file_path):
                return False
        return True

    def prepare(self, id):
        upsample = nn.UpsamplingBilinear2d((self.image_size, self.image_size))
        t1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        t2 = transforms.Normalize((0.5), (0.5))

        bg_image_file_name = "img" + str(id).zfill(6) + ".png"
        origin_image_id = self.image_id_to_file_name[id]
        origin_image_file_name = origin_image_id + ".jpg"

        bg_image = transforms.ToTensor()(Image.open(os.path.join(self.bg_image_dir, bg_image_file_name))).cuda()
        # if origin_image_id in self.file_name_to_base_image_name.keys():
        #     base_image_file_name = self.file_name_to_base_image_name[origin_image_id]
        #     base_image = transforms.ToTensor()(Image.open(os.path.join(self.base_image_dir, base_image_file_name)))
        # else:
        #     base_image = None
        origin_image = transforms.ToTensor()(
            Image.open(os.path.join(self.origin_image_dir, origin_image_file_name))).cuda()
        if origin_image.shape[0] == 1:
            origin_image = origin_image.expand([3, -1, -1])
        origin_label = Image.open(os.path.join(self.origin_label_dir, origin_image_id + ".png"))

        bg_image = (nn.UpsamplingBilinear2d(size=origin_image.shape[1:])(bg_image.unsqueeze(0))).squeeze(0).cuda()
        # if base_image is not None:
        #     base_image = (nn.UpsamplingBilinear2d(size=origin_image.shape[1:])(base_image.unsqueeze(0))).squeeze(0)

        # print(origin_image_id)
        annIds = self.coco.getAnnIds(imgIds=int(origin_image_id), catIds=[], iscrowd=None)
        anns = self.coco.loadAnns(annIds)

        objs = []
        for ann in anns:
            if ann['category_id'] in self.classes:
                bbox = ann['bbox']
                for i in range(4):
                    bbox[i] = math.floor(bbox[i])
                if bbox[2] < 5 or bbox[3] < 5:
                    continue
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

                mask = self.coco.annToMask(ann) * 255
                mask = Image.fromarray(mask)
                obj_mask = mask.crop(bbox)
                obj_mask = transforms.ToTensor()(obj_mask)

                obj_label = origin_label.crop(bbox)

                W, H = origin_label.size
                w, h = obj_label.size
                if w < 63 or h < 63:
                    continue
                if bbox[0] < 6 or bbox[1] < 6 or bbox[2] >= W - 6 or bbox[3] >= H - 6:
                    continue

                obj_label = self.transform(obj_label).cuda()

                obj_input_catid = torch.tensor(self.classes_inv[ann['category_id']]).unsqueeze(0)

                obj_label = t2(obj_label)
                objs.append([obj_label, obj_mask, obj_input_catid, bbox, [h, w]])

                # print(origin_image.shape)
                # print(bbox, w, h)
                # transforms.ToPILImage()(origin_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]).show()
                # transforms.ToPILImage()(transforms.ToTensor()(origin_label)[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]).show()
                # transforms.ToPILImage()(obj_label).show()
                # transforms.ToPILImage()(obj_mask).show()
                # transforms.ToPILImage()(bg_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]).show()
                # transforms.ToPILImage()(bg_image).show()
                # if base_image is not None:
                #     transforms.ToPILImage()(base_image).show()
                #     transforms.ToPILImage()(origin_image).show()
                #     exit(0)

        if len(objs) == 0:
            return None#t1(upsample(bg_image.unsqueeze(0)).squeeze(0)), \
                   #t1(upsample(origin_image.unsqueeze(0)).squeeze(0)), torch.tensor(origin_image.shape)
        objs_label = torch.cat([objs[i][0].cuda().unsqueeze(0) for i in range(0, len(objs))], 0)
        objs_mask = [objs[i][1].cuda() for i in range(0, len(objs))]
        objs_catid = torch.cat([objs[i][2].cuda() for i in range(0, len(objs))], 0)
        objs_g = (self.obj_model.generate(objs_label, objs_catid) / 2 + 0.5).clamp(0, 1)
        # for i in range(objs_g.shape[0]):
        #    transforms.ToPILImage()(objs_g[i].squeeze(0)).show()
        synthesis_image = bg_image.clone().cuda()
        for i in range(objs_g.shape[0]):
            # obj_g = transforms.ToPILImage()()
            # obj_g = obj_g.filter(ImageFilter.GaussianBlur(1))
            # obj_g = transforms.ToTensor()(objs_g[i])
            obj_g = nn.UpsamplingBilinear2d(objs[i][4])(objs_g[i].unsqueeze(0)).cuda()
            bbox = objs[i][3]
            obj_mask = objs_mask[i]
            # print(synthesis_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]].shape, obj_g.shape)
            synthesis_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = synthesis_image[:, bbox[1]:bbox[3],
                                                                   bbox[0]:bbox[2]] * (1 - obj_mask) + obj_g * obj_mask
        # transforms.ToPILImage()(synthesis_image.squeeze(0)).show()
        shape = origin_image.shape
        origin_image = t1(upsample(origin_image.unsqueeze(0)).squeeze(0))
        # synthesis_image=transforms.ToPILImage()(synthesis_image).filter(ImageFilter.GaussianBlur(1))
        # synthesis_image=transforms.ToTensor()(synthesis_image)
        synthesis_image = t1(upsample(synthesis_image.unsqueeze(0)).squeeze(0))
        return synthesis_image, origin_image, torch.tensor(shape)


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
    elif tag == 'coco_synthesis':
        return DataLoader(coco_synthesis_dataset(path, train=training, **kwargs), batch_size, shuffle=True,
                          num_workers=num_worker)


if __name__ == "__main__":
    # data = build_data("cifar10", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"), 8, True, 0)
    # data = build_data("facades", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"), 8, True, 0)
    # data = build_data('coco_obj', os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"), 16, True, 0,
    #                 classes=[1])
    from tools.single_obj import SingleObj, open_config

    single_model = SingleObj(open_config('../experiments/pix2pix_person'),
                             '../experiments/pix2pix_person')
    data = build_data('coco_synthesis',
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/COCO/results_coco_val_1"), 1,
                      False, 0, classes=[1], obj_model=single_model)
    print(len(data))

    for i, d in enumerate(data):
        # x = data[0][0].squeeze(0) / 2 + 0.5
        # transforms.ToPILImage()(x).show()
        # print(x.min(), x.max())
        print(i, len(data))
        print(d[0].shape, d[1].shape, d[2].shape)
        exit(0)
        # save_image(d[0], "../data/COCO/synthesis_train/{}.png".format(d[2]))
