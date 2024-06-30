import torch
import torch.nn as nn
import argparse
import numpy as np
import os
from model.swinunet import get_swinunet  # 导入Swin-UNet模型的定义
from datasets import build_loader
from utils import loadyaml, _get_logger, mk_path
from utils import build_lr_scheduler, build_optimizer, Med_Sup_Loss
from tensorboardX import SummaryWriter
from tqdm import tqdm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from val import test_acdc, test_lidc


def get_finetue_model():
    # 创建Swin-MAE模型
    swinunet_model = get_swinunet(img_size=224, num_classes=args.num_classes, in_channels=args.in_channels)

    # 加载Swin-MAE的预训练权重
    chkpt_dir = '/media/ext_disk/xcm/paper/checkpoint/LIDC_checkpoint/LIDC_SwinMAE_30k_224x224_ms_pr/model/supervise_model.pth'
    swinmae_checkpoint = torch.load(chkpt_dir, map_location='cpu')
    swinmae_state_dict = swinmae_checkpoint['model']

    # for name, param in swinunet_model.named_parameters():
    #     print(name)
    # for name,values in swinmae_state_dict.items():
    #     print(name)
    
    #将Swin-MAE编码器部分的权重传递给Swin-UNet的编码器部分
    for name, param in swinunet_model.encoder.named_parameters():
        if name in swinmae_state_dict:
            print("编码器部分:",name)
            param.data.copy_(swinmae_state_dict[name])

    # 将Swin-MAE编码器中的Swin Transformer Blocks
    for name, param in swinunet_model.decoder.named_parameters():
        if name in swinmae_state_dict:
            if "attn" in name or "blocks_conv" in name:
                k = name[len('layers_up.')]
                k = str(2 - int(k))
                name = 'layers.' + k + name[len('layers_up.k'):]
                print("解码器部分:",name)
                param.data.copy_(swinmae_state_dict[name])
            else:
                # print("name:",name)
                # print("param:",param)
                if 'weight' in name:
                    # print(len(param.shape))
                    if len(param.shape) < 2:
                        nn.init.xavier_normal_(param.unsqueeze(0))
                    else:
                        nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)  # 初始化偏置为零
    return swinunet_model




def Supervise(model, train_loader, test_loader, args):
    optimizer = build_optimizer(args=args, model=model)
    lr_scheduler = build_lr_scheduler(args=args, optimizer=optimizer)
    max_epoch = args.total_itrs // len(train_loader) + 1
    args.logger.info("==============> max_epoch :{}".format(max_epoch))

    # criterion=BCEDiceLoss()
    criterion=Med_Sup_Loss(args.num_classes)

    model.train()
    cur_itrs = 0
    train_loss = 0.0
    best_dice = 0.0

    #  加载原模型
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        state_dict = torch.load(args.ckpt)
        cur_itrs = state_dict["cur_itrs"]
        model = state_dict["model"]
        optimizer = state_dict["optimizer"]
        lr_scheduler = state_dict["lr_scheduler"]
        best_dice = state_dict["best_score"]

    for epoch in range(max_epoch):
        for i, (img, label_true) in enumerate(tqdm(train_loader)):
            cur_itrs += 1
            img = img.to(args.device).float()
            label_true = label_true.to(args.device).long()
            label_pred = model(img)
            loss= criterion(label_pred, label_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            train_loss += loss.item()
            args.writer.add_scalar('supervise/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('supervise/lr', lr, cur_itrs)

            if cur_itrs % args.step_size == 0:
                dice, hd95, jaccard, asd = test_lidc(model=model, test_loader=test_loader, args=args, cur_itrs=cur_itrs)
                args.writer.add_scalar('supervise/{}_dice'.format(args.name), dice, cur_itrs)
                args.writer.add_scalar('supervise/{}_hd95'.format(args.name), hd95, cur_itrs)
                args.writer.add_scalar('supervise/{}_jaccard'.format(args.name), jaccard, cur_itrs)
                args.writer.add_scalar('supervise/{}_asd'.format(args.name), asd, cur_itrs)
                args.logger.info("epoch:{} \t dice:{:.5f} \t hd95:{:.5f} \t jaccard:{:.5f} \t asd:{:.5f}".format(epoch, dice, hd95, jaccard, asd))

                if dice > best_dice:
                    best_dice = dice
                    #  保存模型
                    torch.save({
                        "cur_itrs":cur_itrs,
                        "best_dice":best_dice,
                        "model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "lr_scheduler":lr_scheduler.state_dict(),
                    },os.path.join(args.save_path, "model", "model_{:.4f}.pth".format(best_dice)))

                model.train()

            if cur_itrs > args.total_itrs:
                return

        args.logger.info("Train [{}/{} ({:.0f}%)]\t loss: {:.5f}\t best_dice:{:.5f} ".format(cur_itrs, args.total_itrs,
                                                                          100. * cur_itrs / args.total_itrs,
                                                                          train_loss,best_dice
                                                                          ))
        train_loss = 0


if __name__ == '__main__':
    path = r"config/swinunet_30k_224x224_LIDC_dw.yaml"
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
    args.save_path = os.path.join(root, args.save_path)
    mk_path(args.save_path)  # 创建文件保存位置
    # 创建 tensorboardX日志保存位置
    mk_path(os.path.join(args.save_path, "tensorboardX"))
    mk_path(os.path.join(args.save_path, "model"))  # 创建模型保存位置
    args.supervise_save_path = os.path.join(args.save_path, "model", "supervise_model.pth")  # 设置模型名称
    args.writer = SummaryWriter(os.path.join(args.save_path, "tensorboardX"))
    args.logger = _get_logger(os.path.join(args.save_path, "log.log"), "info")
    args.tqdm = os.path.join(args.save_path, "tqdm.log")

    # step 1: 构建数据集
    train_loader, test_loader = build_loader(args)  # 构建数据集
    args.epochs = args.total_itrs // args.step_size  # 设置模型epoch
    args.logger.info("==========> train_loader length:{}".format(len(train_loader.dataset)))
    args.logger.info("==========> test_dataloader length:{}".format(len(test_loader.dataset)))
    args.logger.info("==========> epochs length:{}".format(args.epochs))
    
    # step 2: 构建模型
    model = get_finetue_model()
    model.to(device=args.device)

    # step 3: 训练模型
    Supervise(model=model, train_loader=train_loader, test_loader=test_loader, args=args)
