from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

PALETTE = np.array([
    [0, 0, 0],
    [255, 255, 255],
])
    
class Transform():
    def __init__(self,mode="train",size=224) -> None:
        self.mode=mode
        self.size=size

    def __call__(self, image:Image.Image,mask:Image.Image) -> Any:
        if self.mode == 'test':
            image, mask = resize(image, mask,self.size)
            image, mask = normalize(image, mask)
            return {
                "image":image,
                "mask":mask
            }
        image, mask = rand_resize(image, mask, (0.5, 2.0))
        image, mask = crop(image, mask, self.size, 255)
        image, mask = hflip(image, mask, p=0.5)

        image, mask = normalize(image, mask)
        return  {
                "image":image,
                "mask":mask
            }
            

def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def rand_resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))
    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask

def resize(img, mask, size):
    img = img.resize((size,size), Image.BILINEAR)
    mask = mask.resize((size,size), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


class ISIC(Dataset):
    def __init__(self, root=r"C:\Users\xcm\Desktop\data\ISIC", split="train", transform=None, index=None):
        super(ISIC, self).__init__()
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
        image = Image.open(self.img_dir[idx]).convert("RGB")
        mask = np.array(Image.open(self.ann_dir[idx]).convert("L"),dtype=np.uint8)
        # image = image.astype('float32') / 255
        mask[mask > 0] = 1
        mask =Image.fromarray(mask)

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
        else:
            with open(self.root + "/test.txt", "r") as f:
                self.sample_list = f.readlines()

        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        self.img_dir = [self.root + "/image/{}.jpg".format(item) for item in self.sample_list]
        self.ann_dir = [self.root + "/gt/{}_segmentation.png".format(item) for item in self.sample_list]

        self.img_dir = np.array(self.img_dir)
        self.ann_dir = np.array(self.ann_dir)


class CustomDataset(Dataset):
    def __init__(self, isic_dataset, transform1=None, transform2=None):
        self.PALETTE = PALETTE
        self.isic_dataset = isic_dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.isic_dataset)

    def __getitem__(self, idx):
        image, mask = self.isic_dataset[idx]  # 获取图像和标签（这里不使用标签）
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


def get_ssl_isic_loader(root=r'C:\Users\xcm\Desktop\data\ISIC', batch_size=4, train_crop_size=(224, 224)):
    """
    :param root:
    :param batch_size: 批次大小
    :param label: 有标签的数量
    :return:
    """
    train_transform = Transform(mode="train",size= train_crop_size[0])
    isic_dataset = ISIC(root=root, split="train")
    # 创建自定义数据集实例
    custom_dataset = CustomDataset(isic_dataset, transform1=train_transform,transform2=train_transform)
    # 创建数据加载器
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True ,drop_last=True)
    return data_loader


def get_isic_loader(root=r'C:\Users\xcm\Desktop\data\ISIC', batch_size=2, train_crop_size=(224, 224)):
    """
    :param root:
    :param batch_size: 批次大小
    :param label: 有标签的数量
    :return:
    """
    train_transform = Transform(mode="train",size= train_crop_size[0])
    test_transform =Transform(mode="test",size= train_crop_size[0])

    train_dataset = ISIC(root=root, split="train", transform=train_transform)
    test_dataset = ISIC(root=root, split="test", transform=test_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_dataloader, test_dataloader


def get_part_isic_loader(root=r'C:\Users\xcm\Desktop\data\ISIC',
                        batch_size=8,
                        train_crop_size=(224, 224),
                        label_num=0.2):
    """
    :param root: 数据集路径
    :param batch_size: 有标注数据批次大小
    :param label_num: 有标签的数量
    :return:
    """
    train_transform = Transform(mode="train",size= train_crop_size[0])
    test_transform =Transform(mode="test",size= train_crop_size[0])

    train_dataset = ISIC(root=root, split="train", transform=train_transform)
    label_length = int(len(train_dataset) * label_num)
    train_label, _ = torch.utils.data.random_split(dataset=train_dataset,
                                                               lengths=[label_length, len(train_dataset) - label_length])

    test_dataset = ISIC(root=root, split="test", transform=test_transform)
    label_loader = DataLoader(train_label, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return label_loader, test_loader


def show(im, path="C:/Users/xcm/Desktop/paper/datasets/img/isic_image.png"):
    im = im.permute(1, 2, 0).numpy()
    image_min, image_max = im.min(), im.max()
    im = (im - image_min) / (image_max - image_min)
    fig = plt.figure()
    plt.imshow(im,cmap='gray')
    plt.show()
    fig.savefig(path)


def show_label(mask, path="C:/Users/xcm/Desktop/paper/datasets/img/isic_mask.jpg"):
    plt.figure()
    plt.imshow(mask)
    plt.show()
    Image.fromarray(mask).save(path)


if __name__ == '__main__':
    isic_data_loader = get_ssl_isic_loader()
    print(len(isic_data_loader))# 453
    print(len(isic_data_loader.dataset))# 1815
    for image1, mask1, image2, mask2 in isic_data_loader:
        print(image1.shape)
        print(image2.shape)
        print(mask1.shape)
        print(mask2.shape)
        print(np.unique(mask1.numpy()))
        print(np.unique(mask2.numpy()))
        show(image1[0], path="C:/Users/xcm/Desktop/paper/datasets/img/isic_image1.png")
        show(image2[0], path="C:/Users/xcm/Desktop/paper/datasets/img/isic_image2.png")
        show_label(isic_data_loader.dataset.label_to_img(mask1[0]), path="C:/Users/xcm/Desktop/paper/datasets/img/isic_mask1.jpg")
        show_label(isic_data_loader.dataset.label_to_img(mask2[0]), path="C:/Users/xcm/Desktop/paper/datasets/img/isic_mask2.jpg")
        break
    
    train_dataloader, test_dataloader = get_isic_loader()
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
