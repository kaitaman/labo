import os

import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from keras import models

def main(model_path, img_path):
    model=load_model(model_path, compile=False)

    model.summary()

    data=img_path
    
    count=0

    for i in range(0, 60):
        # image load
        img_data=os.path.join(data, "plane-exp."+str(i)+".jpg")
        input_size=224
        img=image.load_img(img_data, target_size=(input_size, input_size))
        
        temp_img_array=image.img_to_array(img)
        temp_img_array=temp_img_array.astype('float32')/255.0
        temp_img_array=temp_img_array.reshape((1, 224, 224, 3))

        img_pred=model.predict_classes(temp_img_array)

        if img_pred==[[0]]:
            name="bird"
            count=count+1
        else:
            name="plane"
    
        print('predict_classes : {}'.format(img_pred))

        plt.imshow(img)
        plt.axis(False)
        plt.title('pred : {}'.format(name))
        plt.show()
        
    acc=count/60
    
    print("accuracy : ", acc)

if __name__ == '__main__':
    # directory -----
    cwd = os.getcwd()
    cwd_dir = os.path.dirname(cwd)
    #print("cwd : ", cwd_dir)
    
    cnn_dir = os.path.join(cwd_dir, "study")
    #print("cnn : ", cnn_dir)
    
    data_dir=os.path.join(cnn_dir, "image", "exp")
    print("data : ", data_dir)
    
    # image path
    img_dir=input('画像のパスを入力して下さい>>')
    img_path=os.path.join(data_dir, img_dir)
    print("img : ", img_path)

    # image load
    #input_size=150
    #img=image.load_img(img_path, target_size=(input_size, input_size))
    
    # model path
    model_dir=os.path.join(cnn_dir, "binary_bird_plane_model_3")
    model_path=os.path.join(model_dir, "binary_bird_plane_model_3.h5")
    print("model : ", model_path)

    main(model_path, img_path)

