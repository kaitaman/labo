import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

def diplay(x, y=None):
    plt.imshow(x)

    if y:
        plt.title(y)

    plt.show()

def main():

    # meta define -----
    cwd = os.getcwd()
    BATCH_SIZE = 10
    CLASS_LIST = ['bird', 'plane']
    
    # ここにデータのディレクトリを指定してください。
    #   cat, dog が置いてある場所の 1つ上
    data_dir = os.path.join(cwd, "image", "exp", "child-exp")

    data_gen = ImageDataGenerator(#rescale=1.0/255.0,
        zoom_range=0.9)  # ここにお好き変換を追加する

    # gene る
    generator = data_gen.flow_from_directory(data_dir,
                                             target_size=(224, 224),  # 必要ならサイズを変える (channel はいらない)
                                             shuffle=False,
                                             batch_size=BATCH_SIZE,
                                             class_mode='binary')

    # 積む
    iter_n = generator.n//BATCH_SIZE

    for i in range(iter_n):
        tmp_data, tmp_label = next(generator)
        if i == 0:
            data = tmp_data
            label = tmp_label
        else:
            data = np.vstack((data, tmp_data))
            label = np.hstack((label, tmp_label))
            # binary (正解ラベルが 0, 1) の場合は ここを hstack に
            # categorical (one hot) の場合は ここを "v"stack にする
            #    <= 正解ラベルが ([1, 0], [0, 1]) の場合

    #print(label.shape)
    save_dir = os.path.join(cwd, "image", "exp", "zoomed-3")  # お好み part2

    
    for j, class_name in enumerate(CLASS_LIST):
        idx = 0
        for i, each_data in enumerate(data):
            if label[i] == j:
                save_dir_each =  os.path.join(save_dir, '{}'.format(class_name))  # zoomed/brid という dir を作成
                os.makedirs(save_dir_each, exist_ok=True)
                save_file = os.path.join(save_dir_each, "{}.{}.jpg".format(class_name, idx))

                pil_obj = Image.fromarray(each_data.astype('uint8'))  # float の場合は [0,1]/uintの場合は[0,255]で保存
                pil_obj.save(save_file)
                idx += 1

    print("finished.")
