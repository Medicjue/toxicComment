# -*- coding: utf-8 -*-

import pandas as pd
import nltk
import numpy as np
import datetime
import gc
import pickle
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

random_seed = 23

stemmer = SnowballStemmer('english')

def recursive_stem(token):
    new_token = stemmer.stem(token)
    if new_token == token:
        return token
    else:
        return recursive_stem(new_token)

data = pd.read_csv('data/train_wMeta.csv', encoding='utf-8')

comments = data['comment_text'].as_matrix()
tocix_results = data['toxic'].as_matrix()
noun_cnts = data['noun_cnt'].as_matrix()
verb_cnts = data['verb_cnt'].as_matrix()
adj_cnts = data['adj_cnt'].as_matrix()
adv_cnts = data['adv_cnt'].as_matrix()

del(data)
gc.collect()

noun_tokens_comments = []
adj_tokens_comments = []
words = dict()
comments_length = []

print('Start Collect Data')
collect_start = datetime.datetime.now()
for comment in comments:
    comments_length.append(len(comment))
    noun_tokens = []
    adj_tokens = []
    tokens = nltk.word_tokenize(comment)
    pos_tags = nltk.pos_tag(tokens)
    for pos_tag in pos_tags:
        word = recursive_stem(pos_tag[0])
        tag_type = pos_tag[1]
        if 'NN' in tag_type:
            noun_tokens.append(word)
            words[word] = words.get(word, 0) + 1
        elif 'JJ' in tag_type:
            adj_tokens.append(word)
            words[word] = words.get(word, 0) + 1
    noun_tokens_comments.append(noun_tokens)
    adj_tokens_comments.append(adj_tokens)
    
collect_end = datetime.datetime.now()
print('Collect data completed, time consume: {}'.format(collect_end - collect_start))

words = list(words.keys())

del(comments)
gc.collect()
  
conv_start = datetime.datetime.now()
X = []
for index in range(len(comments_length)):
    noun_tokens = noun_tokens_comments[index]
    adj_tokens = adj_tokens_comments[index]
    noun_cnt = noun_cnts[index]
    verb_cnt = verb_cnts[index]
    adj_cnt = adj_cnts[index]
    adv_cnt = adv_cnts[index]
    comment_length = comments_length[index]
    one_hot = np.zeros(len(words) + 1 + 4 + 1, dtype=int)
    for noun_token in noun_tokens:
        noun_token = recursive_stem(noun_token)
        one_hot[words.index(noun_token)] = 1
    for adj_token in adj_tokens:
        adj_token = recursive_stem(adj_token)
        one_hot[words.index(adj_token)] = 1
    meta_index_start = len(words)+1
    one_hot[meta_index_start+0] = noun_cnt
    one_hot[meta_index_start+1] = verb_cnt
    one_hot[meta_index_start+2] = adj_cnt
    one_hot[meta_index_start+3] = adv_cnt
    one_hot[meta_index_start+4] = comment_length
    X.append(one_hot)

Y = tocix_results
conv_end = datetime.datetime.now()
print('Convert data to vector completed, time consume: {}'.format(conv_end - conv_start))

f = open('data/X.list', 'wb')
pickle.dump(X, f)
f.close()

ttl_size = len(X)
train_size =  int(ttl_size * 0.8)
train_x = X[0:train_size]
test_x = X[train_size:]
train_y = Y[0:train_size]
test_y = Y[train_size:]

del(X)
del(Y)
gc.collect()
           
train_start = datetime.datetime.now()
model = RandomForestClassifier(random_state=random_seed, n_jobs=4)
model.fit(train_x, train_y)
train_end = datetime.datetime.now()
print('Train model completed, time consume: {}'.format(train_end - train_start))

predict_y = model.predict(test_x)

f = open('data/predict_y.npy', 'wb')
pickle.dump(predict_y, f)
f.close()

tn, fp, fn, tp = confusion_matrix(test_x, predict_y).ravel()
try:
    precision = tp / (tp+fp)
except:
    precision = 0.0
try:
    recall = tp / (tp+fn)
except:
    recall = 0.0
try:
    f1_score = 2 * precision * recall / (precision+recall)
except:
    f1_score = 0.0
    
print('TN:{}, FP:{}, FN:{}, TP:{}'.format(tn, fp, fn, tp))
print('Precision:{}, Recall:{}, F1-Score:{}'.format(precision, recall, f1_score))
