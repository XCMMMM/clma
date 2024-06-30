import sys
import torch
import matplotlib.pyplot as plt
from model.swinunet import get_swinunet
from datasets.ACDC import get_acdc_loader
sys.path.append('..')
from scipy.ndimage import zoom


# define the utils
def show_image(image, title=''):
    plt.imshow(image,cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir_):
    # build model
    model = get_swinunet(in_channels=1)
    # load model
    checkpoint = torch.load(chkpt_dir_, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

if __name__ == '__main__':
    # 读取图像
    train_dataloader, test_dataloader = get_acdc_loader()
    # # 读取模型
    chkpt_dir = r'C:\Users\xcm\Desktop\essay\checkpoint\ACDC\ACDC+pre+contr+conv+ms\ACDC_SwinUnet_30k_224x224_ms_dw_0.001\model\model_0.9093.pth'
    model = prepare_model(chkpt_dir)
    print('Model loaded.')
    test_crop_size = [224,224]

    for i_batch, sampled_batch in enumerate(test_dataloader):
        image = sampled_batch[0]
        label = sampled_batch[1]
        length = image.shape[1]
        print(i_batch)
        print(image.shape)
        print(label.shape)
        print(length)
        if i_batch == 2:
            fig = plt.figure()
            plt.rcParams['figure.figsize'] = [12, 6]
            for i in range(4):
                slice = image[0, i, :, :].detach().numpy()
                x, y = slice.shape[0], slice.shape[1]
                # 调整了预测标签的大小，以使其与输入图像的原始大小对齐
                slice = zoom(slice, (test_crop_size[0] / x, test_crop_size[1] / y), order=0)
                img = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()
                label_pred = torch.argmax(torch.softmax(model(img), dim=1), dim=1, keepdim=False).squeeze(0)
                label_pred = label_pred.detach().numpy()
                label_pred = zoom(label_pred, (x / test_crop_size[0], y / test_crop_size[1]), order=0)
                label_pred = test_dataloader.dataset.label_to_img(label_pred)

                label_true = label[0, i, :, :].squeeze().detach().numpy()
                label_true = test_dataloader.dataset.label_to_img(label_true)

                plt.subplot(4, 5, 1+5*i)
                show_image(image[0][i], "original")

                plt.subplot(4, 5, 2+5*i)
                show_image(label_pred, "label_pred")

                plt.subplot(4, 5, 3+5*i)
                show_image(label_true, "label_true")
                
                plt.subplot(4, 5, 4+5*i)
                plt.imshow(image[0][i], cmap='gray')
                plt.imshow(label_pred, alpha=0.6, cmap='viridis') 
                plt.title("pred", fontsize=16)
                plt.axis('off')

                plt.subplot(4, 5, 5+5*i)
                plt.imshow(image[0][i], cmap='gray')
                plt.imshow(label_true, alpha=0.6, cmap='viridis') 
                plt.title("true", fontsize=16)
                plt.axis('off')

            plt.show()
            fig.savefig("img/acdc_paper_1_90.93.png")
            break