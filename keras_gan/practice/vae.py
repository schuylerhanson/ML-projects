import numpy as np
import matplotlib.pyplot as plt
import os 
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from PIL import Image
import cv2

#%% Function definitions

def encoder(x):
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(filters=256, kernel_size=(7,7), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    return x

def x_data_format_mnist(x_data):
    x_data = (np.repeat(np.expand_dims(x_data, axis=-1), 3, axis=-1)).astype('float32')
    normalizer = keras.layers.Normalization(axis=None)
    normalizer.adapt(x_data)
    x_data_norm = normalizer(x_data)
    return x_data_norm



#%%

input_shape = (224,224,3)

inputs = keras.Input(shape=input_shape)
outputs = encoder(inputs)
model = keras.Model(inputs, outputs)

fdir = '/mnt/c/Users/schuy/Pictures/yalefaces/yalefaces'
fin = os.path.join(fdir, os.listdir(fdir)[0])
img = np.array(Image.open(fin))
orig_shape = img.shape
img = cv2.resize(img, (128,128))

fout = '/mnt/c/Users/schuy/Documents/ML-projects/plots/resized_img_128x128.png'
fig,ax = plt.subplots()
ax.imshow(img); ax.set_title('Resized image: orig_shape {} resized to {}'.format(orig_shape, img.shape))
fig.savefig(fout)









'''
### Trying to use vgg16 weights as input layer
vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
vgg_out = vgg.outputs[0]
vgg.trainable = False

x = keras.layers.Conv2D(filters=1024, kernel_size=1)(vgg_out)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)

z_mean = keras.layers.Dense(2)(x)
z_log_var = keras.layers.Dense(2)(x)

encoder = keras.Model(inputs=vgg.inputs, outputs=x)

encoder.summary()
'''