from .swinunet import get_swinunet, get_swinunet_plus
from .swin_mae import swin_mae, contr_mae_vit_base_patch16

def build_model(args):
    if args.model == 'swinunet':
        if isinstance(args.train_crop_size, list) or isinstance(args.train_crop_size, tuple):
            image_size = args.train_crop_size[0]
        else:
            image_size = args.train_crop_size
        model = get_swinunet(img_size=image_size, num_classes=args.num_classes, in_channels=args.in_channels)
    elif args.model == "swinunet_plus":
        if isinstance(args.train_crop_size, list) or isinstance(args.train_crop_size, tuple):
            image_size = args.train_crop_size[0]
        else:
            image_size = args.train_crop_size
        model = get_swinunet_plus(img_size=image_size, num_classes=args.num_classes, in_channels=args.in_channels)
    elif args.model=="swinmae":
        model=swin_mae(in_channels=args.in_channels,mask_ratio=args.mask_ratio)
    elif args.model=="swinmae_contr":
        model=contr_mae_vit_base_patch16(in_channels=args.in_channels,mask_ratio=args.mask_ratio)   
    else:
        raise NotImplementedError

    return model


def build_backbone(args):
    pass
