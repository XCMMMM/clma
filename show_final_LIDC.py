import sys
import torch
import matplotlib.pyplot as plt
from model.swinunet import get_swinunet
from datasets.LIDC import get_lidc_loader
sys.path.append('..')
from scipy.ndimage import zoom
from torchvision.utils import make_grid
import numpy as np

PALETTE2 = np.array([
    [0, 0, 0],
    [255, 255, 255],
])

def label_to_img(label):
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    label = label.astype(np.uint8)
    label[label == 255] = 0
    img = PALETTE2[label]
    if len(img.shape) == 4:
        img = torch.tensor(img).permute(0, 3, 1, 2)
        img = make_grid(tensor=img, nrow=8, scale_each=True)
        img = img.permute(1, 2, 0).numpy()

    return img.astype(np.uint8)


# define the utils
def show_image(image, title=''):
    plt.imshow(image,cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir_):
    # build model
    model = get_swinunet(img_size=224, in_channels=3, num_classes=2)
    # load model
    checkpoint = torch.load(chkpt_dir_, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


if __name__ == '__main__':
    # 读取图像
    train_dataloader, test_dataloader = get_lidc_loader(batch_size=4, train_crop_size=[224,224])
    # # 读取模型
    chkpt_dir = r'C:\Users\xcm\Desktop\权重\LIDC\LIDC+pre+contr+conv+ms\LIDC_SwinUnet_30k_224x224_ms_dw_0.5_0.0003\model\model_0.8211.pth'
    model = prepare_model(chkpt_dir)
    print('Model loaded.')

    for i, (img, label_true) in enumerate(test_dataloader):
        img = img.float()
        label_true = label_true.long()
        print(i)
        print(img.shape)
        print(label_true.shape)
        if i == 2:
            label_pred = torch.argmax(torch.softmax(model(img), dim=1), dim=1, keepdim=False)
            fig = plt.figure()
            plt.rcParams['figure.figsize'] = [12, 6]
            for j in range(4):
                plt.subplot(4, 5, 1+5*j)
                show_image(img[j][0], "original")

                plt.subplot(4, 5, 2+5*j)
                show_image(label_pred[j], "label_pred")

                plt.subplot(4, 5, 3+5*j)
                show_image(label_true[j], "label_true")
                
                plt.subplot(4, 5, 4+5*j)
                plt.imshow(img[j][0], cmap='gray')
                plt.imshow(label_to_img(label_pred[j]), alpha=0.7, cmap='viridis') 
                plt.title("pred", fontsize=16)
                plt.axis('off')

                plt.subplot(4, 5, 5+5*j)
                plt.imshow(img[j][0], cmap='gray')
                plt.imshow(label_to_img(label_true[j]), alpha=0.7, cmap='viridis') 
                plt.title("true", fontsize=16)
                plt.axis('off')
            plt.show()
            fig.savefig("img/lidc_paper_0.5_82.11.png")
            break
