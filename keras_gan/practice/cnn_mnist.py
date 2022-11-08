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

def plot_fct(x_train, y_train, fout, train_data=False, feature_maps=False, 
    show_predictions=False, model_inputs=None, model_outputs=None):
    if not os.path.exists(os.path.dirname(fout)):
        os.makedirs(os.path.dirname(fout))
    fig,ax = plt.subplots(3,3,figsize=(12,12))
    fig.suptitle(os.path.basename(fout).split('.')[0])
    for row in range(3):
        for col in range(3):
            if feature_maps:
                fig,ax = plt.subplots(3,3, figsize=(12,12))
                fig.suptitle(os.path.basename(fout).split('.')[0])
                model_viz = keras.Model(model_inputs, model_outputs)
                feature_maps = model_viz.predict(np.expand_dims(x_train_tensor[0], axis=0))
                ax[row,col].imshow(feature_maps[0,:,:,row+col]); ax[row,col].set_title('feature map {}'.format(row+col))
                ax[row,col].set_xticks([]); ax[row,col].set_yticks([])
                ax[row,col].grid()
            if train_data:
                fig,ax = plt.subplots(3,3,figsize=(12,12))
                fig.suptitle(os.path.basename(fout).split('.')[0])
                ax[row,col].imshow(x_train[row+col]); ax[row,col].set_title('label={}'.format(y_train[row+col]))
                ax[row,col].grid()
            if show_predictions:
                fig,ax = plt.subplots(3,3,figsize=(12,12))
                fig.suptitle('Predictions: {}'.format(os.path.basename(fout).split('.')[0]))
                ax[row,col].imshow(x_train[row+col]); ax[row,col].set_title('label={}'.format(y_train[row+col]))
                ax[row,col].grid()

                fig,ax = plt.subplots(3,3,figsize=(12,12))
                fig.suptitle('Ground Truth: {}'.format(os.path.basename(fout).split('.')[0]))
                ax[row,col].imshow(y_train[row+col]); ax[row,col].set_title('label={}'.format(y_train[row+col]))
                ax[row,col].grid()


    fig.savefig(fout)

def model_1(x, n_classes):
    x = keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(n_classes, activation='relu')(x)
    #x = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)
    return output

def model_2(x, n_classes):
    x = keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2,2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(n_classes, activation='softmax')(x)
    return x
    
def x_data_format_mnist(x_data):
    x_data = (np.repeat(np.expand_dims(x_data, axis=-1), 3, axis=-1)).astype('float32')
    normalizer = keras.layers.Normalization(axis=None)
    normalizer.adapt(x_data)
    x_data_norm = normalizer(x_data)
    return x_data_norm

def x_data_format_cifar(x_data):
    #x_data = (np.repeat(np.expand_dims(x_data, axis=-1), 3, axis=-1)).astype('float32')
    normalizer = keras.layers.Normalization(axis=None)
    normalizer.adapt(x_data)
    x_data_norm = normalizer(x_data)
    return x_data_norm


def y_data_format(y_data, num_tokens=10):
    one_hot = keras.layers.CategoryEncoding(num_tokens, output_mode='one_hot')
    return one_hot(y_data.astype('int64'))

#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

#%% Plot training

fout = '/mnt/c/Users/schuy/Documents/ML-projects/plots/keras_gan/practice/CIFAR_100_predictions.png'
fig,ax = plt.subplots(2,2,figsize=(12,12))
for row in range(2):
    for col in range(2):
        ax[row,col].imshow(x_train[row+col]); ax[row,col].set_title(y_train[row+col])
fig.savefig(fout)

print('Saved img out to {}'.format(fout))


#%% Model inputs
print('\n MODEL INPUTS DEFINITIONS: \n')

x_train_tensor = x_data_format_cifar(x_train)
y_train_tensor = y_data_format(y_train, num_tokens=100) 

x_test_tensor = x_data_format_cifar(x_test)
y_test_tensor = y_data_format(y_test, num_tokens=100)

N = int(x_train.shape[0])
print('N={} TRAINING EXAMPLES \n'.format(N))
x_train = x_train_tensor[:N]
y_train = y_train_tensor[:N]

print(x_train_tensor.shape, y_train_tensor.shape, x_test_tensor.shape, y_test_tensor.shape)

#inputs = keras.Input(shape=(28,28,3))
inputs = keras.Input(shape=(32,32,3))
#outputs = model_1(inputs, n_classes=100)
outputs = model_2(inputs, n_classes=100)
model = keras.Model(inputs, outputs)

model.summary() 

#%% Model action

print('------\n MODEL TRAINING \n------')

model.compile(optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy', 'categorical_accuracy'])

history = model.fit(x_train_tensor, y_train_tensor, 
    validation_data=(x_test_tensor, y_test_tensor),
    batch_size=256, epochs=1, validation_split=0.2)

print('-------\n MODEL PREDICTION \n-------')
preds_onehot = model.predict(x_test_tensor[:50])
y_pred = np.argmax(preds_onehot, axis=1)
y_true = np.argmax(y_test_tensor[:50], axis=1)

print('y_pred:', y_pred, '\n')
print('y_true:', y_true)

#%% Plotting  --> need to figure out how to plot what you want; broken
