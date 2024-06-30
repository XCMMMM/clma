import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader,RandomSampler
import albumentations as A
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import h5py
from torchvision.utils import make_grid
from .utils import RandomGenerator,TwoStreamBatchSampler,patients_to_slices
import torchvision.transforms as transforms

PALETTE = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
])


class ACDC(Dataset):
    def __init__(self, root=r"C:\Users\xcm\Desktop\data\ACDC", split="train", label_num=137,transform=None):
        super(ACDC, self).__init__()
        self.PALETTE = PALETTE
        self.split = split
        self.root = root
        self.transform = transform
        self.sample_list = []
        self.label_num = label_num
        self.load_annotations()  # 加载文件路径
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, "r")

        image = np.array(h5f["image"][:],dtype=np.float32)
        mask = np.array(h5f["label"][:],dtype=np.uint8)

        if self.transform is not None:
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]

        return image, mask
    
    def label_to_img(self, label):
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.astype(np.uint8)
        label[label == 255] = 0
        img = self.PALETTE[label]
        if len(img.shape) == 4:
            img = torch.tensor(img).permute(0, 3, 1, 2)
            img = make_grid(tensor=img, nrow=2, scale_each=True)
            img = img.permute(1, 2, 0).numpy()

        return img.astype(np.uint8)

    def load_annotations(self):
        if "train" in self.split:
            with open(self.root + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            self.sample_list = [self.root + "/data/slices/{}.h5".format(item) for item in self.sample_list]
            self.sample_list = np.array(self.sample_list)
            
            if self.split == "train_l":
                idxs = np.array(range(0, self.label_num))
                self.sample_list = self.sample_list[idxs]

            elif self.split == "train_u":
                idxs = np.array(range(self.label_num, len(self.sample_list)))
                self.sample_list = self.sample_list[idxs]

        elif self.split == "val":
            with open(self.root + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            self.sample_list = [self.root + "/data/{}.h5".format(item) for item in self.sample_list]
        else:
            with open(self.root + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            self.sample_list = [self.root + "/data/{}.h5".format(item) for item in self.sample_list]
            
        self.sample_list=np.array(self.sample_list)

class CustomDataset(Dataset):
    def __init__(self, acdc_dataset, transform1=None, transform2=None):
        self.PALETTE = PALETTE
        self.acdc_dataset = acdc_dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.acdc_dataset)

    def __getitem__(self, idx):
        image, mask = self.acdc_dataset[idx]  # 获取图像和标签（这里不使用标签）
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
            label = label.numpy()
        if not isinstance(label, np.ndarray):
            label = np.array(label)
        label = label.astype(np.uint8)
        label[label == 255] = 0
        img = self.PALETTE[label]
        if len(img.shape) == 4:
            img = torch.tensor(img).permute(0, 3, 1, 2)
            img = make_grid(tensor=img, nrow=2, scale_each=True)
            img = img.permute(1, 2, 0).numpy()

        return img.astype(np.uint8)
    

def get_ssl_acdc_loader(root=r'C:\Users\xcm\Desktop\data\ACDC', batch_size=4, train_crop_size=(224, 224)):
    """

    :param root:
    :param batch_size: 批次大小
    :param label: 有标签的数量
    :return:
    """
    train_transform=RandomGenerator(train_crop_size)
    acdc_dataset = ACDC(root=root, split="train")
    # 创建自定义数据集实例
    custom_dataset = CustomDataset(acdc_dataset, transform1=train_transform,transform2=train_transform)
    # 创建数据加载器
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return data_loader


def get_acdc_loader(root=r'C:\Users\xcm\Desktop\data\ACDC', batch_size=4, train_crop_size=(224, 224)):
    """
    :param root:
    :param batch_size: 批次大小
    :param label: 有标签的数量
    :return:
    """
    train_transform=RandomGenerator(train_crop_size)
    train_dataset = ACDC(root=root, split="train", transform=train_transform)
    test_dataset = ACDC(root=root, split="test")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    return train_dataloader, test_dataloader


def patients_to_slices(patiens_num):
    if patiens_num == 0.05:
        return 68
    elif patiens_num == 0.1:
        return 136
    elif patiens_num == 0.2:
        return 256
    elif patiens_num == 0.5:
        return 664
    else:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 824, "300": 1024}

        return ref_dict[str(patiens_num)]

def get_part_acdc_loader(root=r'C:\Users\xcm\Desktop\data\ACDC', batch_size=8, train_crop_size=(224, 224), label_num=0.2):
    """
    :param root: 数据集路径
    :param batch_size: 有标注数据批次大小
    :param unlabel_batch_size: 无标注数据的batch大小
    :param label_num: 有标签的数量
    :return:
    """

    label_num= patients_to_slices(label_num)
    train_transform=RandomGenerator(train_crop_size)
    label_dataset = ACDC(root=root, split="train_l", label_num=label_num, transform=train_transform)
    test_dataset = ACDC(root=root, split="test")
    label_loader = DataLoader(label_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    return label_loader, test_loader


def show(im, path="swinmae_add/datasets/img/image_acdc.png"):
    im = im.numpy().squeeze()
    fig=plt.figure()
    plt.imshow(im, cmap="gray")
    plt.show()
    fig.savefig(path)


def show_label(mask, path="swinmae_add/datasets/img/mask_acdc.jpg"):
    plt.figure()
    plt.imshow(mask)
    plt.show()
    Image.fromarray(mask).save(path)

if __name__ == '__main__':
    acdc_data_loader = get_ssl_acdc_loader()
    print(len(acdc_data_loader))# 328
    print(len(acdc_data_loader.dataset))# 1312
    for image1, mask1, image2, mask2 in acdc_data_loader:
        print(image1.shape)
        print(image2.shape)
        print(mask1.shape)
        print(mask2.shape)
        print(np.unique(mask1.numpy()))
        print(np.unique(mask2.numpy()))
        show(image1[0], path="swinmae_add/datasets/img/image1_acdc.png")
        show(image2[0], path="swinmae_add/datasets/img/image2_acdc.png")
        show_label(acdc_data_loader.dataset.label_to_img(mask1), path="swinmae_add/datasets/img/mask1_acdc.jpg")
        show_label(acdc_data_loader.dataset.label_to_img(mask2), path="swinmae_add/datasets/img/mask2_acdc.jpg")
        break
    
    train_dataloader, test_dataloader = get_acdc_loader()
    # print(len(train_dataloader))
    # print(len(test_dataloader))
    # print(len(test_dataloader.dataset))
    for image, label in train_dataloader:
        print(image.shape)
        print(label.shape)
        print(np.unique(label.numpy()))
        show(image[0])
        show_label(train_dataloader.dataset.label_to_img(label))
        break

    for sample in test_dataloader:
        image, label = sample
        print(image.shape)
        print(label.shape)
        print(np.unique(label.numpy()))
        # show(image[0])
        # show_label(label[0].numpy())
        break