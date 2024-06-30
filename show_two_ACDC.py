import sys
import torch
import matplotlib.pyplot as plt
from model.swin_mae import swin_mae, contr_mae_vit_base_patch16
from datasets.ACDC import get_acdc_loader, get_ssl_acdc_loader
from torchvision.utils import make_grid
from utils import BoxMaskGenerator
sys.path.append('..')


# define the utils
def show_image(image, title=''):
    # image is [H, W, 3]
    # print(image.shape)
    #assert image.shape[2] == 1
    plt.imshow(image,cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir_):
    # build model
    model = contr_mae_vit_base_patch16(in_channels=1,mask_ratio=0.75) 
    # load model
    checkpoint = torch.load(chkpt_dir_, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img1, img2, batch_mix1, batch_mix2, model):
    img1 = img1.float()
    img2 = img2.float()
    batch_mix1 = batch_mix1.float()
    batch_mix2 = batch_mix2.float()
    _, y1, mask1, y2, mask2, _, _, _ , _ = model(batch_mix1, batch_mix2)
    # 使用Einstein将y的维度从(N, C, H, W)转置为(N, H, W, C)
    y1 = torch.einsum('nchw->nhwc', y1).detach().cpu()
    mask1 = mask1.detach()
    mask1 = torch.einsum('nchw->nhwc', mask1).detach().cpu()
    img1 = torch.einsum('nchw->nhwc', img1).detach().cpu()
    x1 = torch.einsum('nchw->nhwc', batch_mix1).detach().cpu()
    # masked image
    im_masked1 = x1 * (1 - mask1)
    # y = y * mask
    # 根据掩码，将原始图像 x 与重建图像 y 进行组合。它在掩码为1的地方粘贴重建的部分，而在掩码为0的地方保留原始部分。
    im_paste1 = x1 * (1 - mask1) + y1 * mask1
    
    y2 = torch.einsum('nchw->nhwc', y2).detach().cpu()
    mask2 = mask2.detach()
    mask2 = torch.einsum('nchw->nhwc', mask2).detach().cpu()
    img2 = torch.einsum('nchw->nhwc', img2).detach().cpu()
    x2 = torch.einsum('nchw->nhwc', batch_mix2).detach().cpu()
    # masked image
    im_masked2 = x2 * (1 - mask2)
    # y = y * mask
    # 根据掩码，将原始图像 x 与重建图像 y 进行组合。它在掩码为1的地方粘贴重建的部分，而在掩码为0的地方保留原始部分。
    im_paste2 = x2 * (1 - mask2) + y2 * mask2

    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.subplot(2, 5, 1)
    show_image(img1[0], "aug")
    plt.subplot(2, 5, 2)
    show_image(x1[0], "cutmix")
    plt.subplot(2, 5, 3)
    show_image(im_masked1[0], "masked")
    plt.subplot(2, 5, 4)
    show_image(y1[0], "reconstruction")
    plt.subplot(2, 5, 5)
    show_image(im_paste1[0], "reconstruction + visible")
    
    plt.subplot(2, 5, 1+5)
    show_image(img2[0], "aug")
    plt.subplot(2, 5, 2+5)
    show_image(x2[0], "cutmix")
    plt.subplot(2, 5, 3+5)
    show_image(im_masked2[0], "masked")
    plt.subplot(2, 5, 4+5)
    show_image(y2[0], "reconstruction")
    plt.subplot(2, 5, 5+5)
    show_image(im_paste2[0], "reconstruction + visible")

    plt.show()
    fig.savefig("C:/Users/xcm/Desktop/paper/ACDC_two.png")

if __name__ == '__main__':
    # 读取图像
    data_loader = get_ssl_acdc_loader()
    
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
    
    for img1, label_true1, img2, label_true2 in data_loader:
        cutmix_mask1 = mask_generator.generate_params(n_masks=img1.shape[0], mask_shape=(224, 224))
        cutmix_mask1 = torch.tensor(cutmix_mask1, dtype=torch.float)
        cutmix_mask2 = mask_generator.generate_params(n_masks=img1.shape[0], mask_shape=(224, 224))
        cutmix_mask2 = torch.tensor(cutmix_mask2, dtype=torch.float)

        batch_mix1 = img1 * (1.0 - cutmix_mask1) + img2 * cutmix_mask1
        batch_mix2 = img2 * (1.0 - cutmix_mask2) + img1 * cutmix_mask2

        print(img1.shape)
        print(img2.shape)
        
        # 读取模型
        chkpt_dir = r'C:\Users\xcm\Desktop\权重\ACDC\ACDC+pre+contr+conv+ms\ACDC_SwinMAE_30k_224x224_ms_pr\model\supervise_model.pth'
        model_mae = prepare_model(chkpt_dir)
        print('Model loaded.')
        #
        # make random mask reproducible (comment out to make it change)
        torch.manual_seed(2)
        print('MAE with pixel reconstruction:')
        run_one_image(img1, img2, batch_mix1, batch_mix2, model_mae)
        break