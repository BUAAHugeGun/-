import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from PIL import Image
from tqdm import tqdm
import shutil

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
    for i in tqdm(range(len(names))):
        origin_id = names[i]
        if not os.path.exists(os.path.join(exp_path, origin_id + ".png")):
            continue
        baseline_image_name = "img" + str(i).zfill(6) + ".png"
        baseline_image = Image.open(os.path.join(baseline_path, baseline_image_name))
        exp_image = Image.open(os.path.join(exp_path, origin_id + ".png"))
        origin_image = Image.open(os.path.join(origin_path, origin_id + ".jpg")).resize(size=(64, 64))
        origin_id = os.path.join("./compare", origin_id)
        baseline_image.save(origin_id + "base.jpg")
        origin_image.save(origin_id + "ori.jpg")
        exp_image.save(origin_id + "exp.jpg")
