import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from scipy import ndimage
from skimage import io
from albumentations.augmentations import transforms
from scipy.ndimage import zoom
import random


PALETTE = np.array([
    [0, 0, 0],
    [255, 255, 255],
])
    
class LIDC(Dataset):
    def __init__(self, root=r"C:\Users\xcm\Desktop\data\LIDC", split="train", transform=None, index=None):
        super(LIDC, self).__init__()
        self.PALETTE = PALETTE
        self.split = split
        self.root = root
        self.transform = transform
        self.img_dir = []
        self.ann_dir = []
        self.load_annotations()  # 加载文件路径
        print("total {} samples".format(len(self.img_dir)))

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        image = np.array(Image.open(self.img_dir[idx]).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(self.ann_dir[idx]).convert("L"), dtype=np.uint8)
        image = image.astype('float32') / 255
        mask[mask == 255] = 1

        if self.transform is not None:
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]

        return image, mask

    def label_to_img(self, label):
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.astype(np.uint8)
        label[label == 255] = 0
        img = self.PALETTE[label]
        if len(img.shape) == 4:
            img = torch.tensor(img).permute(0, 3, 1, 2)
            img = make_grid(tensor=img, nrow=8, scale_each=True)
            img = img.permute(1, 2, 0).numpy()

        return img.astype(np.uint8)

    def load_annotations(self):
        if self.split == "train":
            with open(self.root + "/train.txt", "r") as f:
                self.sample_list = f.readlines()
        elif self.split == "val":
            with open(self.root + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
        else:
            with open(self.root + "/test.txt", "r") as f:
                self.sample_list = f.readlines()

        self.sample_list = [item.replace("\n", "")
                            for item in self.sample_list]
        self.img_dir = [
            self.root + "/image_roi/{}.png".format(item) for item in self.sample_list]
        self.ann_dir = [self.root + "/mask_roi/LIDC_Mask_{}.png".format(
            item.split("_")[1]) for item in self.sample_list]
        self.img_dir = np.array(self.img_dir)
        self.ann_dir = np.array(self.ann_dir)

class CustomDataset(Dataset):
    def __init__(self, lidc_dataset, transform1=None, transform2=None):
        self.PALETTE = PALETTE
        self.lidc_dataset = lidc_dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.lidc_dataset)

    def __getitem__(self, idx):
        image, mask = self.lidc_dataset[idx]  # 获取图像和标签（这里不使用标签）
        # 应用第一个数据增强管道
        if self.transform1 is not None:
            result1 = self.transform1(image=image, mask=mask)
            image1 = result1["image"]
            mask1 = result1["mask"]
        # 应用第二个数据增强管道
        if self.transform2 is not None:
            result2 = self.transform2(image=image, mask=mask)
            image2 = result2["image"]
            mask2 = result2["mask"]
        # 返回两个增强后的图像
        return image1, mask1, image2, mask2
    
    def label_to_img(self, label):
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.astype(np.uint8)
        label[label == 255] = 0
        img = self.PALETTE[label]
        if len(img.shape) == 4:
            img = torch.tensor(img).permute(0, 3, 1, 2)
            img = make_grid(tensor=img, nrow=8, scale_each=True)
            img = img.permute(1, 2, 0).numpy()

        return img.astype(np.uint8)

def get_ssl_lidc_loader(root=r'C:\Users\xcm\Desktop\data\LIDC', batch_size=4, train_crop_size=(224, 224)):
    """
    :param root: 数据集路径
    :param batch_size: 有标注数据批次大小
    :param label: 有标签的数量
    :return:
    """
    train_transform = A.Compose([
        A.RandomResizedCrop(height=train_crop_size[0], width=train_crop_size[1], scale=(0.75, 1.5)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.4,0.4,0.4,p=0.5),
        ToTensorV2()
    ])
    lidc_dataset = LIDC(root=root, split="train")
    # 创建自定义数据集实例
    custom_dataset = CustomDataset(lidc_dataset, transform1=train_transform, transform2=train_transform)
    # 创建数据加载器
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
    return data_loader

    
def get_lidc_loader(root=r'C:\Users\xcm\Desktop\data\LIDC', batch_size=4, train_crop_size=(224, 224)):
    """
    :param root:
    :param batch_size: 批次大小
    :param label: 有标签的数量
    :return:
    """
    train_transform = A.Compose([
        A.RandomResizedCrop(height=train_crop_size[0], width=train_crop_size[1], scale=(0.75, 1.5)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.4,0.4,0.4,p=0.5),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(train_crop_size[0], train_crop_size[1]),
        ToTensorV2()
    ])

    train_dataset = LIDC(root=root, split="train", transform=train_transform)
    test_dataset = LIDC(root=root, split="test", transform=test_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_dataloader, test_dataloader


def get_part_lidc_loader(root=r'C:\Users\xcm\Desktop\data\LIDC',
                        batch_size=8,
                        train_crop_size=(224, 224),
                        label_num=0.2):
    """
    :param root: 数据集路径
    :param batch_size: 有标注数据批次大小
    :param label_num: 有标签的数量
    :return:
    """
    train_transform = A.Compose([
        A.RandomRotate90(),
        # A.ShiftScaleRotate(p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=0.2),
        A.OneOf([
            transforms.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                          always_apply=False),  # 调整色相饱和度
            transforms.RandomBrightnessContrast(),  # 随机亮度和对比度
        ], p=1),  # 按照归一化的概率选择执行哪一个
        A.Resize(train_crop_size[0],train_crop_size[1]),
        # A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(height=train_crop_size[0], width=train_crop_size[1]),
        ToTensorV2()
    ])

    train_dataset = LIDC(root=root, split="train", transform=train_transform)
    label_length = int(len(train_dataset) * label_num)
    train_label, _ = torch.utils.data.random_split(dataset=train_dataset,
                                                               lengths=[label_length, len(train_dataset) - label_length])

    test_dataset = LIDC(root=root, split="test", transform=test_transform)
    label_loader = DataLoader(train_label, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return label_loader, test_loader


def show(im, path="swinmae_ms/datasets/img/lidc_image.png"):
    im = im.permute(1, 2, 0).numpy()
    fig=plt.figure()
    plt.imshow(im, cmap="gray")
    plt.show()
    fig.savefig(path)

def show_label(mask, path="swinmae_ms/datasets/img/lidc_mask.jpg"):
    plt.figure()
    plt.imshow(mask)
    plt.show()
    Image.fromarray(mask).save(path)


if __name__ == '__main__':
    lidc_data_loader = get_ssl_lidc_loader()
    print(len(lidc_data_loader))# 328
    print(len(lidc_data_loader.dataset))# 1312
    for image1, mask1, image2, mask2 in lidc_data_loader:
        print(image1.shape)
        print(image2.shape)
        print(mask1.shape)
        print(mask2.shape)
        print(np.unique(mask1.numpy()))
        print(np.unique(mask2.numpy()))
        show(image1[0], path="swinmae_ms/datasets/img/lidc_image1.png")
        show(image2[0], path="swinmae_ms/datasets/img/lidc_image2.png")
        show_label(lidc_data_loader.dataset.label_to_img(mask1[0]), path="swinmae_ms/datasets/img/lidc_mask1.jpg")
        show_label(lidc_data_loader.dataset.label_to_img(mask2[0]), path="swinmae_ms/datasets/img/lidc_mask2.jpg")
        break
    
    train_dataloader, test_dataloader = get_lidc_loader()
    print(len(train_dataloader.dataset))
    print(len(test_dataloader.dataset))
    for image, label in train_dataloader:
        print(image.shape)
        print(label.shape)
        print(np.max(image.numpy()))
        print(np.min(image.numpy()))
        print(np.unique(label.numpy()))
        show(image[0])
        show_label(train_dataloader.dataset.label_to_img(label[0]))
        break

    for sample in test_dataloader:
        image, label = sample
        print(image.shape)
        print(label.shape)
        print(np.max(image.numpy()))
        print(np.min(image.numpy()))
        print(np.unique(label.numpy()))
        show(image[0])
        show_label(test_dataloader.dataset.label_to_img(label[0]))
        break
