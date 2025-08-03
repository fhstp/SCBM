# CPUSET=200-208 NAME=TRANS GPUID="MIG-c5cfb7ff-2cb3-5b5f-9707-39268dc3fce6" 

models = ['FacebookAI/xlm-roberta-base',
         'FacebookAI/xlm-roberta-large',
          'google-bert/bert-base-uncased',
         'google-bert/bert-large-uncased',]


TRAIN_FILE = 'train.csv'
DEV_FILE = 'test.csv'


import pandas as pd
from sklearn.model_selection import StratifiedKFold
from llm.models import train_model_dev
from sklearn.metrics import f1_score
from llm.models import predict, SeqModel
import numpy as np, pickle


train = pd.read_csv(TRAIN_FILE)
dev = pd.read_csv(DEV_FILE)


label_column = 'Class'
history = None   

map_classes = {j:i for i,j in enumerate(list(set(train['Class'].values)))}
train['Class'] = train['Class'].map(map_classes)

stats = {}

skf = StratifiedKFold(n_splits=5, shuffle=True)

for i, (train_index, test_index) in enumerate(skf.split(train['text'], train['Class'])):

	for model_name in models:

		dev = train.iloc[test_index]
		ttrain = train.iloc[train_index]

		history = train_model_dev(model_name=model_name, data_train=ttrain,
								 data_dev=dev,
								epoches=12, batch_size=32, interm_layer_size = 128, 
								lr = 2e-6 if 'large' in model_name else 1e-5,  
								decay=1e-6, output='.', 
								task=label_column)


		model = SeqModel(interm_size = 128, model=model_name, task='offensive')
		model.load('best_model.pt')
		z = predict(model=model, data_dev=dev, outputfile='transformers')

		if model_name not in stats:
			stats[model_name] = []
		stats[model_name].append(f1_score(dev[label_column], z['pred'], average='macro'))

import pickle
with open('transformers.pickle', 'wb') as handle:
	pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)