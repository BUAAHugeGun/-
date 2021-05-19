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
    if os.path.exists("./compare"):
        shutil.rmtree("./compare")
    os.mkdir("./compare")
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str, nargs=2)
    args = parser.parse_args()
    path = args.path
    file_name_file = open(os.path.join(path[0], "file_name.txt"))
    lines = file_name_file.readlines()
    names = []
    for line in lines:
        names.append(line.split('.')[0])
    origin_path = os.path.join(path[0], "..", "val_image")
    exp_path = path[1]
    baseline_path = path[0]

    msssim = MSSSIM()
    psnr = PNSR()
    l1 = nn.L1Loss()
    data_sum = len(names)
    p = [0., 0.]
    m = [0., 0.]
    l = [0., 0.]
    for i in tqdm(range(len(names))):
        origin_id = names[i]
        if not os.path.exists(os.path.join(exp_path, origin_id + ".png")):
            data_sum -= 1
            continue
        baseline_image_name = "img" + str(i).zfill(6) + ".png"

        baseline_image = Image.open(os.path.join(baseline_path, baseline_image_name))
        exp_image = Image.open(os.path.join(exp_path, origin_id + ".png"))
        origin_image = Image.open(os.path.join(origin_path, origin_id + ".jpg")).resize(size=(64, 64))

        origin_id = os.path.join("./compare", origin_id)
        baseline_image.save(origin_id + "base.png")
        origin_image.save(origin_id + "ori.png")
        exp_image.save(origin_id + "exp.png")

        baseline_image = transforms.ToTensor()(baseline_image).unsqueeze(0)
        exp_image = transforms.ToTensor()(exp_image).unsqueeze(0)
        origin_image = transforms.ToTensor()(origin_image).unsqueeze(0)
        if origin_image.shape[1] == 1:
            origin_image = origin_image.expand([1, 3, -1, -1])

        p[0] += psnr(baseline_image, origin_image)
        p[1] += psnr(exp_image, origin_image)

        m[0] += msssim(baseline_image, origin_image)
        m[1] += msssim(exp_image, origin_image)

        #l[0] += l1(baseline_image, origin_image)
        #l[1] += l1(exp_image, origin_image)

    print(p[0] / data_sum, p[1] / data_sum, '\n', m[0] / data_sum, m[1] / data_sum)#, l[0] / data_sum, l[1] / data_sum)
