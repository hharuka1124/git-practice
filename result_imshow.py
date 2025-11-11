import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import numpy as np
from typing import Union

def result_load(path:str) -> list:
    """画像の読み込み + リストへの格納
    Args:
        path (str): 画像のフォルダパス

    Returns:
        _type_: 読み込んだ画像を格納したリスト
    """
    filename = os.listdir(path)
    imgs = []
    for i, name in enumerate(filename):
        img_path = os.path.join(path, name)
        img = Image.open(img_path)
        imgs.append(img)
    return imgs

def result_show(imgs:list, labels:list, preds:list, rows:int, savepath:str):
    """画像, ラベル, 予測ラベルの可視化

    Args:
        imgs (list): 画像のリスト
        labels (list): ラベルのリスト
        preds (list): 予測ラベルのリスト
        rows (int): 可視化する画像の種類数
        savepath (str): 作成したfigureの保存パス
    """
    cols = 6 # 列数を指定
    fig, ax = plt.subplots(rows, cols, figsize=(15, 20))
    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    shows = []
    x = np.arange(len(imgs))
    random_index = np.array(random.choices(x, k=rows*2)).reshape(-1, 2)  # ランダムに画像を選択
    
    for index in random_index:        
        index1, index2 = index
        img1 = imgs[index1]
        label1 = labels[index1]
        pred1 = preds[index1]
        img2 = imgs[index2]
        label2 = labels[index2]
        pred2 = preds[index2]
        shows.append([img1, label1, pred1, img2, label2, pred2])
        titles = ["Image", "Label", "Prediction", "Image", "Label", "Prediction"]
    for i in range(rows):
        for j in range(cols):
            ax[i, j].imshow(shows[i][j])
            ax[i, j].axis("off")

            if i == 0:
                ax[i, j].set_title(titles[j])
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()

def compare_result_show(imgs_folderpath:str, 
                        labels_folderpath:str, 
                        compare_method_folderpath:list[str], 
                        rows:int, 
                        index_list:list[int], 
                        compare_method_name:Union[list[str]|None], 
                        savepath:str) -> None:
    
    """画像, ラベル, 各手法での予測ラベルの可視化, 画像がランダムに選択

    Args:
        imgs_folderpath (str): 画像のフォルダパス
        labels_folderpath (str): ラベルのフォルダパス
        compare_method_folderpath (list[str]): 比較手法の予測ラベルの保存フォルダパスを格納したリスト
        rows (int): 可視化する画像の種類
        index_list (list[int] or None): 表示する画像のインデックスのリスト
        compare_method_name (list[str]): 比較手法の名前(サブタイトルに使用)
        savepath (str): 作成したimshowの保存パス
    """
    assert rows != index_list, "rows and index_lost must have the same length."
    cols = 2 + len(compare_method_folderpath)
    fig, ax = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    if index_list is None:
        nun_imgs = len(os.listdir(imgs_folderpath))
        x = np.arange(nun_imgs)
        index_list = np.array(random.choices(x, k=rows))
    
    subtitles = ["Image", "Label"]
    subtitles.extend(compare_method_name)

    for i in range(rows):
        show_imgs = []
        img_path = os.path.join(imgs_folderpath, "img{0:04}.png".format(index_list[i]))
        label_path = os.path.join(labels_folderpath, "label{0:04}.png".format(index_list[i]))
        img = Image.open(img_path)
        label = Image.open(label_path)
        show_imgs.extend([img, label])
        for folderpath in compare_method_folderpath:
            img_path = os.path.join(folderpath, "pred{0:04}.png".format(index_list[i]))
            img = Image.open(img_path)
            show_imgs.append(img)
        
        for j in range(cols):
            ax[i, j].imshow(show_imgs[j], 'gray')
            ax[i, j].axis("off")

            if i == 0:
                ax[i, j].set_title(subtitles[j], fontsize=18)

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()

def one_row_imshow(show_imgs_path:list[str], subtitles:list[str], savepath:str):
    cols = len(show_imgs_path)
    fig, ax = plt.subplots(1, cols, figsize=(2*cols, 2))

    for i, path in enumerate(show_imgs_path):
        img = np.array(Image.open(path))
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title(subtitles[i], fontsize=18)

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()

if __name__ == "__main__":
    
    imgs_path = "C:/Users/04/OneDrive - 同志社大学/M2/yhara/Spring/Data/masking_background/test_imgs"
    labels_path = "C:/Users/04/OneDrive - 同志社大学/M2/data/08/test_labels"
    preds_path = "C:/Users/04/Desktop/result/pred"
    savepath = "C:/Users/04/Desktop/result.pdf"

    vgg_path = "C:/Users/04/Desktop/compare_encoder/pred/vgg11"
    resnet_path = "C:/Users/04/Desktop/compare_encoder/pred/resnet18"
    mobilenet_path = "C:/Users/04/Desktop/compare_encoder/pred/mobilenet-v2"
    efficientnet_path = "C:/Users/04/Desktop/compare_encoder/pred/efficientnet-b1"
    mixvit_path = "C:/Users/04/Desktop/compare_encoder/pred/mix-vit-b1"
    mobileone_path = "C:/Users/04/Desktop/compare_encoder/pred/mobileone-s1"
    
    compare_method_folderpath = [vgg_path, resnet_path, mobilenet_path, efficientnet_path, mixvit_path, mobileone_path]
    compare_method_name = ["VGG11", "ResNet18", "MobileNet-V2", "EfficientNet-B1", "Mix-ViT-B1", "MobileOne-S1"]
    index_list = [1137, 94, 1497, 379, 453, 750, 804]
    compare_result_show(imgs_path, labels_path, compare_method_folderpath, 7, index_list, compare_method_name, savepath)
    

    