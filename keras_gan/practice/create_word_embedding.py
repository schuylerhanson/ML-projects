import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras

def text_cleaner(sample):
    exclude_words = ['the', 'is', 'a']
    sample = sample.split(' ')
    sample = ' '.join([word for word in sample if word not in exclude_words])
    return sample

def word_indexing(context_list):
    flat_list = []
    for elem in context_list:
        flat_list.append(elem[0])
        flat_list.append(elem[1])
    flat_list = set(flat_list)
    word_map = {word:idx for idx,word in enumerate(flat_list)}
    return word_map

def one_hot_encoding(word, word_map, map_len):
    vec = np.zeros((map_len,))
    vec[word_map[word]] = 1
    return vec

def embedding_model(x, map_len=21):
    x = keras.layers.Dense(embed_size, activation='linear')(x)
    x = keras.layers.Dense(units=map_len, activation='softmax')(x)
    return x

samples = [
    'the future king is the prince',
    'daughter is the princess',
    'son is the prince',
    'only a man can be a king',
    'only a woman can be a queen',
    'queen and king rule the realm',
    'the prince is a strong man',
    'the princess is a beautiful woman',
    'the royal family is the king and queen and their children'
]

if __name__ == '__main__':
    
    sample_dict = {}
    for idx,sample in enumerate(samples):
        sample = text_cleaner(sample)
        print('sample:', sample)
        sample = sample.split(' ')
        context_list = []
        if len(sample) > 2:
            for word_idx in range(2, len(sample)):
                context_list.append((sample[word_idx-2], sample[word_idx-1]))
                context_list.append((sample[word_idx-2], sample[word_idx]))
            context_list.append((sample[-2], sample[-1]))
            #sorted_context_list = sorted(context_list, key = lambda x: x[0])
            #print(sorted_context_list)
            #context_list = list(set(sorted_context_list))
            sample_dict[idx] = context_list
        else:
            sample_dict[idx] = [(sample[0], sample[1])]

    #print(sample_dict)

    #print(sample_dict.values(), '\n')
    context_list = list(sample_dict.values())
    context_list = [xx for x in context_list for xx in x]
    word_map = word_indexing(context_list)
    print(word_map)

    data_len = len(context_list) ## no. rows
    map_len = len(word_map.keys()) ## no. cols
    X = np.zeros((data_len, map_len))
    Y = np.zeros((data_len, map_len))

    for idx, pair in enumerate(context_list):
        focus, context = pair[0], pair[1]
        focus_vec = one_hot_encoding(focus, word_map, map_len)
        context_vec = one_hot_encoding(context, word_map, map_len)
        X[idx] = focus_vec
        Y[idx] = context_vec


    #%% NN embedding training
    embed_size = 2

    inputs = keras.Input(shape=(X.shape[1],))
    outputs = embedding_model(inputs, map_len=map_len)
    model = keras.Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X, Y, batch_size=256, epochs=1000, verbose=0)

    weights = model.get_weights()[0]

    print('WEIGHTS (shape,weights):', weights.shape, weights, '\n')

    embedding_dict = {}
    for idx,word in enumerate(word_map.keys()):
        embedding_dict[word] = weights[idx]

    fig,ax = plt.subplots(figsize=(14,14))
    words = list(embedding_dict.keys()) 
    x = [x[0] for x in embedding_dict.values()]
    y = [x[1] for x in embedding_dict.values()]
    ax.scatter(x, y)
    for idx,word in enumerate(words):
        ax.annotate(word, (x[idx], y[idx]))

    fout = '/mnt/c/Users/schuy/Documents/ML-projects/plots/word_embedding.png'
    fig.savefig(fout)