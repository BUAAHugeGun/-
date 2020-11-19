from dataset.dataset import build_data
from nets.generator import get_G
from nets.discriminator import get_D
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch

import yaml
import os
import argparse

log_file = None


def to_log(s):
    global log_file
    print(s)
    print(s, file=log_file)


def open_config(root):
    f = open(os.path.join(root, "config.yaml"))
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load(models, epoch, root):
    for name, model in models.items():
        ckpt = torch.load(os.path.join(root, "logs/" + name + "_epoch-{}.pth".format(epoch)))
        ckpt = {k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        to_log("load model: {} from epoch: {}".format(name, epoch))


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

    G = get_G("mnist").cuda()
    D = get_D("mnist").cuda()
    dataloader = build_data("mnist", args["bs"], True, num_worker=args["num_workers"])

    loss_function = nn.CrossEntropyLoss()

    g_opt = torch.optim.Adam(G.parameters(), lr=args["lr"])
    d_opt = torch.optim.Adam(D.parameters(), lr=args["lr"])
    g_sch = torch.optim.lr_scheduler.MultiStepLR(g_opt, args["lr_milestone"], gamma=0.5)
    d_sch = torch.optim.lr_scheduler.MultiStepLR(d_opt, args["lr_milestone"], gamma=0.5)

    if args["load_epoch"] != -1:
        load({"G": G, "D": D, "g_opt": g_opt, "d_opt": d_opt, "g_sch": g_sch, "d_sch": d_sch}, args["load_epoch"],
             root)

    tot = (args["load_epoch"] + 1) * len(dataloader)
    g_opt.step()
    d_opt.step()
    for epoch in range(args["load_epoch"] + 1, args["epoch"]):
        g_sch.step()
        d_sch.step()
        for i, (data, label) in enumerate(dataloader):
            tot = tot + 1
            data = data.cuda()
            label = label.cuda()
            real_label = label.cuda()
            fake_label = torch.tensor([10]).expand(data.shape[0]).cuda().long()

            D_out = D(data)
            D_loss_real = loss_function(D_out, real_label)

            input = torch.randn([data.shape[0], 2, 28, 28]).cuda()
            input[:, 1, :, :] = label.reshape(data.shape[0], 1, 1).expand(data.shape[0], 28, 28)
            G_out = G(input)
            GD_out = D(G_out)
            D_loss_fake = loss_function(GD_out, fake_label)

            d_loss = D_loss_fake + D_loss_real
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            input = torch.randn([data.shape[0], 2, 28, 28]).cuda()
            input[:, 1, :, :] = label.reshape(data.shape[0], 1, 1).expand(data.shape[0], 28, 28)
            G_out = G(input)
            GD_out = D(G_out)
            g_loss = loss_function(GD_out, real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if tot % args["show_interval"] == 0:
                to_log(
                    'epoch: {}, batch: {}, d_loss_total: {:.5f}, d_loss_real: {:.5f}, d_loss_fake: {:.5f}, g_loss: {:.5f}, lr: {:.5f}'.format(
                        epoch, i, d_loss.item(), D_loss_real.item(), D_loss_fake.item(), g_loss.item(),
                        g_sch.get_last_lr()[0]))
                writer.add_scalar("d_loss_total", d_loss.item(), tot)
                writer.add_scalar("d_loss_real", D_loss_real.item(), tot)
                writer.add_scalar("d_loss_fake", D_loss_fake.item(), tot)
                writer.add_scalar("g_loss", g_loss.item(), tot)

        if epoch % args["snapshot_interval"] == 0:
            torch.save(G.state_dict(), os.path.join(root, "logs/G_epoch-{}.pth".format(epoch)))
            torch.save(D.state_dict(), os.path.join(root, "logs/D_epoch-{}.pth".format(epoch)))
            torch.save(g_opt.state_dict(), os.path.join(root, "logs/g_opt_epoch-{}.pth".format(epoch)))
            torch.save(d_opt.state_dict(), os.path.join(root, "logs/d_opt_epoch-{}.pth".format(epoch)))
            torch.save(g_sch.state_dict(), os.path.join(root, "logs/g_sch_epoch-{}.pth".format(epoch)))
            torch.save(d_sch.state_dict(), os.path.join(root, "logs/d_sch_epoch-{}.pth".format(epoch)))
        if epoch % args["test_interval"] == 0:
            label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda()
            input = torch.randn([10, 2, 28, 28]).cuda()
            input[:, 1, :, :] = label.reshape(10, 1, 1).expand(10, 28, 28)
            G_out = G(input)
            save_image(G_out, os.path.join(root, "logs/result/output-{}.png".format(epoch)))
            # save_image(data.cpu(), os.path.join(root, "logs/result/gt-{}.png".format(epoch)))


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
