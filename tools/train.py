from dataset.data_builder import build_data
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
from model.flow import GLOW
import math
import torch

import yaml
import os
import argparse

log_file = None


def to_log(s, output=True):
    global log_file
    if output:
        print(s)
    print(s, file=log_file)


def open_config(root):
    f = open(os.path.join(root, "config.yaml"))
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load(models, epoch, root):
    def _detect_latest():
        """
        detect the latest file in log_dir with format <prefix><epoch><suffix>
        :param prefix:
        :param suffix:
        :return: epoch, if here's no checkpoints, return a negative value
        """
        checkpoints = os.listdir(os.path.join(root, "logs"))
        checkpoints = [f for f in checkpoints if f.startswith("flow_model_epoch-") and f.endswith(".pth")]
        checkpoints = [int(f[len("flow_model_epoch-"):-len(".pth")]) for f in checkpoints]
        checkpoints = sorted(checkpoints)
        _epoch = checkpoints[-1] if len(checkpoints) > 0 else None
        return _epoch

    if epoch == -1:
        epoch = _detect_latest()
    if epoch == None:
        return -1
    for name, model in models.items():
        ckpt = torch.load(os.path.join(root, "logs/" + name + "_epoch-{}.pth".format(epoch)))
        ckpt = {k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        to_log("load model: {} from epoch: {}".format(name, epoch))
    # print("loaded from epoch: {}".format(epoch))
    return epoch


def calc_z_shapes(n_channel, input_size, n_levels):
    z_shapes = []

    for i in range(n_levels - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bits):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -n_bits * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (math.log(2) * n_pixel)).mean(),
        (log_p / (math.log(2) * n_pixel)).mean(),
        (logdet / (math.log(2) * n_pixel)).mean(),
    )


def train(args, root):
    global log_file
    if not os.path.exists(os.path.join(root, "logs")):
        os.mkdir(os.path.join(root, "logs"))
    if not os.path.exists(os.path.join(root, "logs/result/")):
        os.mkdir(os.path.join(root, "logs/result/"))
    if not os.path.exists(os.path.join(root, "logs/result/event")):
        os.mkdir(os.path.join(root, "logs/result/event"))
    log_file = open(os.path.join(root, "logs/log.txt"), "w")
    to_log(args)
    writer = SummaryWriter(os.path.join(root, "logs/result/event/"))

    dataloader = build_data(args['data_tag'], args['data_path'], args["bs"], True, num_worker=args["num_workers"])

    z_sample = []
    z_shapes = calc_z_shapes(3, args['img_size'], args['n_levels'])
    for z in z_shapes:
        z_new = torch.randn(args['n_sample'], *z) * args['temp']
        z_sample.append(z_new.cuda())

    flow_model = GLOW(3, args['n_flows'], args['n_levels']).cuda()
    flow_opt = torch.optim.Adam(flow_model.parameters(), lr=args['lr'])
    flow_sch = torch.optim.lr_scheduler.MultiStepLR(flow_opt, args["lr_milestone"], gamma=0.5)

    load_epoch = load({"flow_model": flow_model, "flow_opt": flow_opt, "flow_sch": flow_sch}, args["load_epoch"], root)

    tot_iter = (load_epoch + 1) * len(dataloader)
    for epoch in range(args['load_epoch'] + 1, args['epoch']):
        flow_opt.step()
        for i, (image, label) in enumerate(tqdm(dataloader)):
            tot_iter += 1
            image, label = image.cuda(), label.cuda()
            image -= 0.5
            log_p, ldj, _ = flow_model(image + torch.rand_like(image) / 255.)  # 256?

            ldj = ldj.mean()
            loss, log_p, ldj = calc_loss(log_p, ldj, args['img_size'], 8)
            flow_model.zero_grad()
            loss.backward()

            flow_opt.step()

            if tot_iter % args['show_interval'] == 0:
                to_log('epoch: {}, batch: {}, tot_batch: {}, loss_total: {:.5f}, lr: {:.5f}'.format(
                    epoch, i, len(dataloader), loss.item(), flow_sch.get_last_lr()[0]), output=False)
                writer.add_scalar("loss_total", loss.item(), tot_iter)

        if epoch % args['snapshot_interval'] == 0:
            torch.save(flow_model.state_dict(), os.path.join(root, "logs/flow_model_epoch-{}.pth".format(epoch)))
            torch.save(flow_opt.state_dict(), os.path.join(root, "logs/flow_opt_epoch-{}.pth".format(epoch)))
            torch.save(flow_sch.state_dict(), os.path.join(root, "logs/flow_sch_epoch-{}.pth".format(epoch)))
        if epoch % args['test_interval']:
            with torch.no_grad():
                save_image(
                    flow_model.reverse(z_sample).cpu().data + 0.5,
                    f"sample/{str(epoch + 1).zfill(3)}.png",
                    normalize=True,
                    nrow=10,
                )


"""
        if epoch % args["test_interval"] == 0:
            label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda()
            input = torch.randn([10, 2, 28, 28]).cuda()
            input[:, 1, :, :] = label.reshape(10, 1, 1).expand(10, 28, 28)
            G_out = G(input)
            save_image(G_out, os.path.join(root, "logs/result/output-{}.png".format(epoch)))
            # save_image(data.cpu(), os.path.join(root, "logs/result/gt-{}.png".format(epoch)))
"""


def test(args):
    print("testing\n", args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--test", default=False, action='store_true')
    args = parser.parse_args()
    if args.test == True:
        test(open_config(args.root), args.root)
    else:
        train(open_config(args.root), args.root)
