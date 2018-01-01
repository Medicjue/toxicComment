# -*- coding: utf-8 -*-

import pandas as pd
import nltk
import matplotlib.pyplot as plt
import gc

train_data = pd.read_csv('data/train.csv')
comments = train_data['comment_text'].as_matrix()
nouns = []
vebs = []
adjs = []
advs= []

for comment in comments:
    tokens = nltk.word_tokenize(comment)
    pos_tags = nltk.pos_tag(tokens)
    cnt_n = 0
    cnt_v = 0
    cnt_adj = 0
    cnt_adv = 0
    for pos_tag in pos_tags:
        tag_type = pos_tag[1]
        if 'NN' in tag_type:
            cnt_n += 1
        elif 'VB' in tag_type:
            cnt_v += 1
        elif 'JJ' in tag_type:
            cnt_adj += 1
        elif 'RB' in tag_type:
            cnt_adv += 1
    nouns.append(cnt_n)
    vebs.append(cnt_v)
    adjs.append(cnt_adj)
    advs.append(cnt_adv)
    
train_data['noun_cnt'] = pd.Series(nouns)
train_data['verb_cnt'] = pd.Series(vebs)
train_data['adj_cnt'] = pd.Series(adjs)
train_data['adv_cnt'] = pd.Series(advs)

train_data.to_csv('data/train_wMeta.csv', index=False)

