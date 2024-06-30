import sys
import torch
import matplotlib.pyplot as plt
from model.swin_mae import swin_mae, contr_mae_vit_base_patch16
from datasets.LIDC import get_lidc_loader, get_ssl_lidc_loader
from torchvision.utils import make_grid
sys.path.append('..')


# define the utils
def show_image(image, title=''):
    # image is [H, W, 3]
    # print(image.shape)
    #assert image.shape[2] == 1
    plt.imshow(image,cmap='gray')
    plt.title(title, fontsize=8)
    plt.axis('off')
    return


def prepare_model(chkpt_dir_):
    # build model
    model = contr_mae_vit_base_patch16(in_channels=3,mask_ratio=0.5) 
    # load model
    checkpoint = torch.load(chkpt_dir_, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img1, img2, model):
    img1 = img1.float()
    img2 = img2.float()
    _, y, mask, _, _, _ , _, _, _  = model(img1, img2)
    # 使用Einstein将y的维度从(N, C, H, W)转置为(N, H, W, C)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach()
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', img1).detach().cpu()
    # masked image
    im_masked = x * (1 - mask)
    # y = y * mask
    # 根据掩码，将原始图像 x 与重建图像 y 进行组合。它在掩码为1的地方粘贴重建的部分，而在掩码为0的地方保留原始部分。
    im_paste = x * (1 - mask) + y * mask
    
    fig = plt.figure()
    for i in range(4):
        plt.subplot(4, 4, 1+4*i)
        show_image(x[i], "original")

        plt.subplot(4, 4, 2+4*i)
        show_image(im_masked[i], "masked")

        plt.subplot(4, 4, 3+4*i)

        show_image(y[i], "reconstruction")

        plt.subplot(4, 4, 4+4*i)
        show_image(im_paste[i], "reconstruction + visible")

    plt.show()
    fig.savefig("img/mask_LIDC_final.png")

if __name__ == '__main__':
    # 读取图像
    data_loader = get_ssl_lidc_loader()
    for img1, label_true1, img2, label_true2 in data_loader:
        print(img1.shape)
        print(img2.shape)
        # 读取模型
        chkpt_dir = r'C:\Users\xcm\Desktop\checkpoint\LIDC\LIDC+pre+contr+conv+ms\LIDC_SwinMAE_30k_224x224_ms_pr\model\supervise_model.pth'
        model_mae = prepare_model(chkpt_dir)
        print('Model loaded.')
        # make random mask reproducible (comment out to make it change)
        torch.manual_seed(2)
        print('MAE with pixel reconstruction:')
        run_one_image(img1, img2, model_mae)
        break