import os.path
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from utils import loadyaml, _get_logger, mk_path, BoxMaskGenerator
from model import build_model
from datasets import build_loader
from utils import build_lr_scheduler, build_optimizer, Dense_Loss
import math
from torchvision.utils import make_grid

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
import matplotlib.pyplot as plt


# define the utils
def show_image(image, title=''):
    plt.imshow(image,cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    return
  
def cutmix_img_show(batch_mix1, batch_mix2):
    batch_mix_im1 = batch_mix1.cpu().detach()
    batch_mix_im2 = batch_mix2.cpu().detach()
    print(batch_mix_im1.shape)
    print(batch_mix_im2.shape)
    
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [6, 12]
    for i in range(4):
        im1 = batch_mix_im1[i].numpy().squeeze()
        im2 = batch_mix_im2[i].numpy().squeeze()
        plt.subplot(4, 2, 1+2*i)
        show_image(im1, "cutmix1")

        plt.subplot(4, 2, 2+2*i)
        show_image(im2, "cutmix2")
    plt.show()
    fig.savefig("swinmae_add/cutmix.png")
    
def compute_contrastive_loss(args, criterion, p1, p2, z1, z2):
    return args.contr_weight * (-(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5)


def main():

    path = r"config/swinmae_30k_224x224_ISIC_pr.yaml"
    root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径
    args = loadyaml(os.path.join(root, path))  # 加载yaml

    if args.cuda:
        args.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    else:
        args.device = torch.device("cpu")

    torch.manual_seed(args.seed)  # 设置随机种子
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = False  # 单卡的不需要分布式
    torch.backends.cudnn.benchmark = True  # 寻找最佳 的训练路径

    root = os.path.dirname(os.path.realpath(__file__))  # 获取绝对路径
    args.save_path = os.path.join(root, args.save_path) # 获取文件保存位置
    mk_path(args.save_path)  # 创建文件保存位置
    # 创建 tensorboardX日志保存位置
    mk_path(os.path.join(args.save_path, "tensorboardX"))
    mk_path(os.path.join(args.save_path, "model"))  # 创建模型保存位置
    args.finetune_save_path = os.path.join(args.save_path, "model", "finetune_model.pth")
    args.pretrain_save_path = os.path.join(args.save_path, "model", "pretrain_model.pth")
    args.supervise_save_path = os.path.join(args.save_path, "model", "supervise_model.pth")  # 设置模型名称

    args.writer = SummaryWriter(os.path.join(args.save_path, "tensorboardX"))
    args.logger = _get_logger(os.path.join(args.save_path, "log.log"), "info")
    args.tqdm = os.path.join(args.save_path, "tqdm.log") # tqdm是Python进度条库

    # step 1: 构建数据集
    dataloader = build_loader(args)
    args.num_instances = len(dataloader.dataset)
    args.logger.info(f"length of training dataset: {args.num_instances}")
    # total_itrs: 30000 
    # len(train_loader)==1312/16=82 (注意这里配置文件中batch_size为16)
    # args.epochs=366
    args.epochs = args.total_itrs // len(dataloader)  # 30000//82=365
    args.logger.info("==============> epochs :{}".format(args.epochs))

    # step 2: 构建模型
    model = build_model(args=args).to(device=args.device)

    # step 3: 训练模型
    UnSupervise(model=model, dataloader=dataloader, args=args)


def UnSupervise(model: nn.Module, dataloader, args):
    # optimizer = build_optimizer(args=args, model=model)
    # lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args)) #创建了一个优化器对象，并将其分配给变量optimizer。
    lr_scheduler, epochs = create_scheduler(args, optimizer)
    args.logger.info("==============> epochs :{}".format(epochs))

    model.train()
    
    
    class config:
        cutmix_mask_prop_range = (0.15, 0.3)
        cutmix_boxmask_n_boxes = 3
        cutmix_boxmask_fixed_aspect_ratio = False
        cutmix_boxmask_by_size = False
        cutmix_boxmask_outside_bounds = False
        cutmix_boxmask_no_invert = False

    mask_generator = BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range,
                                      n_boxes=config.cutmix_boxmask_n_boxes,
                                      random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                      prop_by_area=not config.cutmix_boxmask_by_size,
                                      within_bounds=not config.cutmix_boxmask_outside_bounds,
                                      invert=not config.cutmix_boxmask_no_invert)
    
    cur_itrs = 0 # 目前的迭代次数，初始为0
    train_loss = 0.0
    best_dice = 0.0
    criterion = torch.nn.CosineSimilarity(dim=1).to(args.device)

    # 加载原模型
    # 检查是否存在预训练的模型和优化器状态文件，如果存在，就加载这些状态以恢复之前训练的模型和优化器的状态，从而可以继续训练或进行其他任务，而不是从头开始训练模型。
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=True)
        optimizer.load_state_dict(state_dict["optimizer"], strict=True)

    for epoch in range(args.epochs):
        for i, (img1, label_true1, img2, label_true2) in enumerate(tqdm(dataloader)):
            cur_itrs += 1
            img1 = img1.to(args.device).float()
            label_true1 = label_true1.to(args.device).long()
            img2 = img2.to(args.device).float()
            label_true2 = label_true2.to(args.device).long()
            
            
            cutmix_mask1 = mask_generator.generate_params(n_masks=img1.shape[0],
                                                mask_shape=(args.train_crop_size[0], args.train_crop_size[1]))
            cutmix_mask1 = torch.tensor(cutmix_mask1, dtype=torch.float).to(args.device)
            cutmix_mask2 = mask_generator.generate_params(n_masks=img1.shape[0],
                                                mask_shape=(args.train_crop_size[0], args.train_crop_size[1]))
            cutmix_mask2 = torch.tensor(cutmix_mask2, dtype=torch.float).to(args.device)

            batch_mix1 = img1 * (1.0 - cutmix_mask1) + img2 * cutmix_mask1
            batch_mix2 = img2 * (1.0 - cutmix_mask2) + img1 * cutmix_mask2

            # print(img1.shape)
            # print(img2.shape)
            # cutmix_img_show(batch_mix1, batch_mix2)
            
            
            swinmae_loss, _, _, _, _, p1, p2, z1, z2 = model(view1=batch_mix1, view2=batch_mix2)
            contr_loss = compute_contrastive_loss(args, criterion, p1, p2, z1, z2)
            loss = swinmae_loss + contr_loss
            
            # print("swinmae_loss:",swinmae_loss)
            # print("contr_loss:",contr_loss)
            # print("loss:",loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch=epoch)
            lr = optimizer.param_groups[0]["lr"]
            train_loss += loss.item()
            args.writer.add_scalar('supervise/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('supervise/lr', lr, cur_itrs)

            if cur_itrs % args.step_size == 0:
                model.eval()
                # y--模型预测的结果
                # mask--掩码，0为保留，1为掩码
                _, y, mask, _, _, _, _, _, _ = model(view1=batch_mix1, view2=batch_mix2)
                # y = model.unpatchify(y)
                # einsum操作将输入张量 y 的通道维度与高度和宽度维度进行了交换，实现了维度的重排。然后将结果分离并移动到CPU上。
                y = torch.einsum('nchw->nhwc', y).detach().cpu()

                mask = mask.detach()
                # mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size ** 2 * args.in_channels)  # (N, H*W, p*p*3)
                # mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
                mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

                x = torch.einsum('nchw->nhwc', batch_mix1).detach().cpu()
                # 获得掩码后的图像masked image
                im_masked = x * (1 - mask)
                # y = y * mask
                # MAE reconstruction pasted with visible patches
                im_paste = x * (1 - mask) + y * mask

                image = make_grid(tensor=batch_mix1, nrow=4, normalize=True, scale_each=True)
                im_masked = make_grid(tensor=im_masked.permute(0, 3, 1, 2), nrow=4, normalize=True, scale_each=True)
                im_paste = make_grid(tensor=im_paste.permute(0, 3, 1, 2), nrow=4, normalize=True, scale_each=True)

                # 用于将图像数据添加到TensorBoard日志中，以便用户可以使用TensorBoard工具来可视化和分析模型的训练进程中的图像数据。
                args.writer.add_image('SwinMae/image', image, cur_itrs)
                args.writer.add_image('SwinMae/im_masked', im_masked, cur_itrs)
                args.writer.add_image('SwinMae/im_paste', im_paste, cur_itrs)
                model.train()

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "lr_scheduler": lr_scheduler.state_dict(),
        }, args.supervise_save_path)

        if cur_itrs > args.total_itrs:
            return

        args.logger.info("Train [{}/{} ({:.0f}%)]\t loss: {:.5f}\t  ".format(cur_itrs, args.total_itrs,
                                                                             100. * cur_itrs / args.total_itrs,
                                                                             train_loss/len(dataloader)))
        train_loss = 0


if __name__ == "__main__":
    main()
