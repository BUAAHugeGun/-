import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from PIL import Image
from tqdm import tqdm
import shutil
from loss.PSNR_Loss import Loss as PNSR
from loss.SSIM_Loss import MSSSIM
import torch
import torch.nn as nn
from torchvision import transforms

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str, nargs=1)
    args = parser.parse_args()
    path = args.path[0]
    origin_path = os.path.join("../data/COCO/val_image")
    msssim = MSSSIM()
    psnr = PNSR()
    l1 = nn.L1Loss()
    p = 0.
    m = 0.
    l = 0.
    if os.path.exists(os.path.join(path, "file_name.txt")):
        file_name_file = open(os.path.join(path, "file_name.txt"))
        lines = file_name_file.readlines()
        names = []
        for line in lines:
            names.append(line.split('.')[0])
        data_sum = len(names)
        for i in tqdm(range(len(names))):
            origin_id = names[i]
            image_name = "img" + str(i).zfill(6) + ".png"
            image = Image.open(os.path.join(path, image_name))
            origin_image = Image.open(os.path.join(origin_path, origin_id + ".jpg")).resize(size=(64, 64))

            image = transforms.ToTensor()(image).unsqueeze(0)
            origin_image = transforms.ToTensor()(origin_image).unsqueeze(0)
            if origin_image.shape[1] == 1:
                origin_image = origin_image.expand([1, 3, -1, -1])

            p += psnr(origin_image, image)
            m += msssim(origin_image, image)
            l += l1(origin_image, image)

    else:
        for root, dirs, files in os.walk(path):
            data_sum = len(files)
            for i in tqdm(range(len(files))):
                file = files[i]
                origin_id = file.split('.')[0]

                image = Image.open(os.path.join(root, file))
                origin_image = Image.open(os.path.join(origin_path, origin_id + ".jpg")).resize(size=(64, 64))
                image = transforms.ToTensor()(image).unsqueeze(0)
                origin_image = transforms.ToTensor()(origin_image).unsqueeze(0)
                if origin_image.shape[1] == 1:
                    origin_image = origin_image.expand([1, 3, -1, -1])

                p += psnr(origin_image, image)
                m += msssim(origin_image, image)
                l += l1(origin_image, image)

    print(p / data_sum, m / data_sum, l / data_sum)
