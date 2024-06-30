from .ACDC import get_acdc_loader, get_ssl_acdc_loader, get_part_acdc_loader
from .LIDC import get_lidc_loader, get_ssl_lidc_loader, get_part_lidc_loader
from .ISIC import get_isic_loader, get_ssl_isic_loader, get_part_isic_loader

def build_loader(args):
    if args.datasets == "acdc":
        data_loader = get_ssl_acdc_loader(
            root=args.data_path,
            train_crop_size=args.train_crop_size,
            batch_size=args.batch_size)
        return  data_loader
    elif args.datasets == "sup_acdc":
        train_loader, test_loader = get_acdc_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size)
        return train_loader, test_loader
    elif args.datasets == "sup_part_acdc":
        label_loader, test_loader = get_part_acdc_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size,
            label_num=args.label_num)
        return  label_loader, test_loader
    
    elif args.datasets == "lidc":
        data_loader = get_ssl_lidc_loader(
            root=args.data_path,
            train_crop_size=args.train_crop_size,
            batch_size=args.batch_size)
        return data_loader
    elif args.datasets == "sup_lidc":
        train_loader, test_loader = get_lidc_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size)
        return train_loader, test_loader
    elif args.datasets == "sup_part_lidc":
        label_loader, test_loader = get_part_lidc_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size,
            label_num=args.label_num)
        return  label_loader, test_loader
    
    elif args.datasets == "isic":
        data_loader = get_ssl_isic_loader(
            root=args.data_path,
            train_crop_size=args.train_crop_size,
            batch_size=args.batch_size)
        return data_loader
    elif args.datasets == "sup_isic":
        train_loader, test_loader = get_isic_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size)
        return train_loader, test_loader
    elif args.datasets == "sup_part_isic":
        label_loader, test_loader = get_part_isic_loader(
            root=args.data_path,
            batch_size=args.batch_size,
            train_crop_size=args.train_crop_size,
            label_num=args.label_num)
        return  label_loader, test_loader
    else:
        raise NotImplementedError
