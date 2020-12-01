import os
import argparse
import warnings
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from skimage.draw import polygon
from tqdm import tqdm

warnings.filterwarnings("ignore")
annotation_dir = os.path.join("..", "data", "COCO", "annotations", "instances_val2017.json")
input_dir = os.path.join("..", "data", "COCO", "val2017")
ouotput_dir = os.path.join("..", "data", "COCO", "val_inst")

coco = COCO(annotation_dir)

if __name__ == "__main__":
    cats = coco.loadCats(coco.getCatIds())
    imgIds = coco.getImgIds(catIds=coco.getCatIds(cats))
    for ix in tqdm(range(len(imgIds))):
        id = imgIds[ix]
        img_dict = coco.loadImgs(id)[0]
        filename = img_dict["file_name"].replace("jpg", "png")
        label_name = os.path.join(input_dir, filename)
        inst_name = os.path.join(ouotput_dir, filename)
        img = io.imread(label_name, as_gray=True)

        annIds = coco.getAnnIds(imgIds=id, catIds=[], iscrowd=None)
        anns = coco.loadAnns(annIds)
        count = 0
        for ann in anns:
            if type(ann["segmentation"]) == list:
                if "segmentation" in ann:
                    for seg in ann["segmentation"]:
                        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                        rr, cc = polygon(poly[:, 1] - 1, poly[:, 0] - 1)
                        img[rr, cc] = count
                    count += 1

        io.imsave(inst_name, img)
