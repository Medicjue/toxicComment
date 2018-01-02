# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

data = pd.read_csv('data/train_wMeta.csv', encoding='cp950')

output = pd.DataFrame()
output['lbl'] = data['toxic']
output['comment_text'] = data['comment_text'].str.replace('\r', '').str.replace('\n', '').str.replace('\r', '').str.lower().str.slice(0, 1024)


ttl_size = len(output)
train_size = int(ttl_size * 0.8)

output_train = output.iloc[0:train_size, :]
output_train.to_csv('VDCNN_tf/toxic/train.csv', header=False, index=False, encoding='utf8')

output_test = output.iloc[train_size:, :]
output_test.to_csv('VDCNN_tf/toxic/test.csv', header=False, index=False, encoding='utf8')


data = pd.read_csv('data/test.csv')

output = pd.DataFrame()
output['lbl'] = pd.Series(np.zeros(len(data), dtype=int))
output['comment_text'] = data['comment_text'].str.replace('\r', '').str.replace('\n', '').str.replace('\r', '').str.lower().str.slice(0, 1024)
output.to_csv('VDCNN_tf/toxic/inf.csv', header=False, index=False, encoding='utf8')
