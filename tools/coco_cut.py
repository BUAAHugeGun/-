import os
import argparse
import warnings
from pycocotools.coco import COCO
import numpy as np
from PIL import Image, ImageOps
import skimage.io as io
from skimage import transform
from skimage.draw import polygon
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import sys

classes = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
           9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
           16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
           24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
           34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
           40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
           47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
           55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
           63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
           74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
           81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
           89: 'hair drier', 90: 'toothbrush'}

if __name__ == "__main__":
    assert len(sys.argv) == 2
    data_path = sys.argv[1]
    print("loading data from: {}".format(data_path))

    pre = "val"

    warnings.filterwarnings("ignore")
    annotation_dir = os.path.join(data_path, "annotations", "instances_{}2017.json".format(pre))
    coco = COCO(annotation_dir)
    input_dir = os.path.join(data_path, "{}_image".format(pre))
    output_dir = os.path.join(data_path, "{}_cut".format(pre))
    input_label_dir = os.path.join(data_path, "{}_label".format(pre))
    output_label_dir = os.path.join(data_path, "{}_label_cut".format(pre))
    output_mask_dir = os.path.join(data_path, "{}_mask_cut".format(pre))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_label_dir):
        os.mkdir(output_label_dir)
    data_list_file = open(os.path.join(output_dir, "data_list.txt"), "w")

    cats = coco.loadCats(coco.getCatIds())
    imgIds = coco.getImgIds(catIds=coco.getCatIds(cats))
    for ix in tqdm(range(len(imgIds))):
        id = imgIds[ix]
        img_dict = coco.loadImgs(id)[0]
        filename = img_dict["file_name"]
        image_path = os.path.join(input_dir, filename)
        filename = filename[0:-4] + '.png'
        label_path = os.path.join(input_label_dir, filename)
        img = Image.open(image_path)
        label = Image.open(label_path)
        annIds = coco.getAnnIds(imgIds=id, catIds=[], iscrowd=False)
        anns = coco.loadAnns(annIds)
        '''
        plt.axis('off')
        plt.clf()
        plt.imshow(img)
        coco.showAnns(anns)
        plt.show()
        '''
        for ann in anns:
            bbox = ann['bbox']
            for i in range(4):
                bbox[i] = math.floor(bbox[i])
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            mask = coco.annToMask(ann) * 255
            mask = Image.fromarray(mask)

            obj_mask = mask.crop(bbox)
            # obj = img.crop(bbox)
            obj_label = label.crop(bbox)

            W, H = label.size
            w, h = obj_label.size
            if w < 64 or h < 64:
                continue
            if bbox[0] < 5 or bbox[1] < 5 or bbox[2] >= W - 5 or bbox[3] >= H - 5:
                continue
            # obj = np.array(obj)
            # obj_label = np.array(obj_label)

            # obj_label = np.array([255]) - obj_label
            # obj_label = (obj_label) * obj_mask
            # if len(obj.shape) == 3:
            #    obj_mask = np.expand_dims(obj_mask, axis=2)
            # obj = obj * obj_mask
            # obj = ImageOps.invert(Image.fromarray(obj))
            # obj_label = Image.fromarray(obj_label)

            obj_name = str(id) + "_" + str(ann['id']) + ".png"

            # obj_path = os.path.join(output_dir, obj_name)
            # obj_label_path = os.path.join(output_label_dir, obj_name)
            obj_mask_path = os.path.join(output_mask_dir, obj_name)
            # obj.save(obj_path)
            # obj_label.save(obj_label_path)
            obj_mask.save(obj_mask_path)
            print(obj_name, ann['category_id'], classes[ann['category_id']], file=data_list_file)
