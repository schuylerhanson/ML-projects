import os
import numpy as np
from tensorflow import keras 
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

### Possible next steps:
# Try on different dataset, eg.: path = '/mnt/c/Users/schuy/Pictures/yalefaces/test1'
# Play around w architectures, understand model architecture better
# Improve model performance
# Design model performance characterization, eg. confusion matrix? 

#%% Function definitions
def load_img_dataset(path):
    files = [os.path.join(path,x) for x in os.listdir(path)]
    arrs = {}
    for idx,file in enumerate(files):
        arrs[idx] = np.asarray(Image.open(file))
    return arrs

def plot_fct(x_train, y_train, fout):
    os.chdir('../plots')
    fout = 'MNIST_data.png'
    fig,ax = plt.subplots(3,3,figsize=(12,12))
    fig.suptitle('Some MNIST digits data')
    for x in range(3):
        for y in range(3):
            ax[x,y].imshow(x_train[x+y]); ax[x,y].set_title('label = {}'.format(y_train[x+y]))
            ax[x,y].grid()
    fig.savefig(fout)

def model_1(x_data):
    x = keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu')(x_data)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    #x = keras.layers.Dense(4*4*32*10, activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Dense(10, activation='softmax')(x)
    return output

def model_2(x_data):
    output = keras.layers.Dense(28*28*32, activation)
    
def x_data_format(x_data):
    x_data = (np.repeat(np.expand_dims(x_data, axis=-1), 3, axis=-1)).astype('float32')
    normalizer = keras.layers.Normalization(axis=None)
    normalizer.adapt(x_data)
    x_data_norm = normalizer(x_data)
    return x_data_norm

def y_data_format(y_data):
    one_hot = keras.layers.CategoryEncoding(num_tokens=10, output_mode='one_hot')
    return one_hot(y_train.astype('int64'))


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_tensor = x_data_format(x_train)
y_tensor = y_data_format(y_train) 

print('X_TRAIN SHAPE:{} \n X_TRAIN TYPE:{}'.format(x_tensor.shape, type(x_tensor)))

#%% Model inputs

N=25000
x_train = x_tensor[:N]
y_train = y_tensor[:N]

inputs = keras.Input(shape=(28,28,3))
outputs = model_1(inputs)
model = keras.Model(inputs, outputs)

M=25
x_val = x_tensor[-M:]
y_val = y_tensor[-M:]

#%% Model action
model.compile(optimizer='adam', 
    loss='categorical_crossentropy',
    )
history = model.fit(x_train, y_train, batch_size=64, epochs=10)

preds = model.predict(x_val)
print('y_pred:', np.argmax(preds, axis=1), '\n')
print('y_true:', np.argmax(y_train[-M:], axis=1))
