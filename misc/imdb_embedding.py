import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
import nltk
#nltk.download('punkt')
from nltk.util import ngrams
#from nltk.corpus import stopwords
import re
import string

def clean_files(files, stopwords):
    file_dict = {}
    orig_dict = {}
    for idx,file in enumerate(files):
        with open(file) as f:
            contents = f.read()
        orig_dict[os.path.basename(file).split('.')[0]] = contents
        contents = re.sub("\'\w+", '', contents)
        contents = re.sub(r'\w*\d+\w*', '', contents)
        contents = ''.join(word.strip(string.punctuation) for word in contents)
        contents = ' '.join([word for word in contents.split(' ') if word not in stop_words])
        contents = ' '.join([word.lower() for word in contents.split(' ')])
        file_dict[os.path.basename(file).split('.')[0]] = contents
    
    return file_dict

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



fdir = '/mnt/c/Users/schuy/Downloads/imdb_data/imdb_data/test/pos'
files = [os.path.join(fdir,x) for x in np.sort(os.listdir(fdir))]
files = files[:20]

stop_words = nltk.corpus.stopwords.words('english')

file_dict = clean_files(files, stop_words)



## View sample file
#key = list(file_dict.keys())[0]
#sample = file_dict[key]
#token = nltk.word_tokenize(sample)
#bigram =  list(ngrams(token, 2))

## Over multiple reviews
dataset = list(file_dict.values())
dataset_tokens = [nltk.word_tokenize(review) for review in dataset]
dataset_tokens = [xx for x in dataset_tokens for xx in x]
dataset_bigrams = list(ngrams(dataset_tokens, 2))

## Optional use
token_freq = {}
for unique_token in np.unique(dataset_tokens):
    token_freq[unique_token] = np.count_nonzero(np.array(dataset_tokens) == unique_token)

token_freq_tuples = list(token_freq.items())
sorted_hist = sorted(list(token_freq.items()), key= lambda x:x[1])

most_freq_words = sorted_hist[-200:]

## Constructing X,Y
word_map = word_indexing(dataset_bigrams)
data_len = len(dataset_bigrams)
map_len = len(word_map.keys())

X = np.zeros((data_len, map_len))
Y = np.zeros((data_len, map_len))

for idx, pair in enumerate(dataset_bigrams):
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
model.fit(X, Y, batch_size=256, epochs=1000, verbose=1)

weights = model.get_weights()[0]

#print('WEIGHTS (shape,weights):', weights.shape, weights, '\n')

#%% NEEDS FIXING: Plotting embeddings of most freq words \
## Next step is going to be diagnosing embedding performance/characterizing embeddings somehow

embedding_dict = {}
for idx,word in enumerate(word_map.keys()):
    embedding_dict[word] = weights[idx]

fig,ax = plt.subplots(figsize=(14,14))
words = list(embedding_dict.keys()) 
x = [x[0] for x in embedding_dict.values()]
y = [x[1] for x in embedding_dict.values()]
ax.scatter(x, y); ax.set_title('IMDB reviews word embedding, N=20 reviews'); ax.set_xlabel('hidden layer dim x1'); ax.set_ylabel('hidden layer dim x2') 
for idx,word in enumerate(words):
    ax.annotate(word, (x[idx], y[idx]))

fout = '/mnt/c/Users/schuy/Documents/ML-projects/plots/word_embedding_imdb.png'
fig.savefig(fout)