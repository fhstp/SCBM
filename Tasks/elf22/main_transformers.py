# CPUSET=200-208 NAME=TRANS GPUID="MIG-c5cfb7ff-2cb3-5b5f-9707-39268dc3fce6" 

models = ['FacebookAI/xlm-roberta-base',
         'FacebookAI/xlm-roberta-large',
         'google-bert/bert-base-uncased',
         'google-bert/bert-large-uncased',]

TRAIN_FILE = 'train.csv'
DEV_FILE = 'test.csv'


import pandas as pd
from llm.models import train_model_dev
from sklearn.metrics import f1_score
from llm.models import predict, SeqModel

train = pd.read_csv(TRAIN_FILE)
dev = pd.read_csv(DEV_FILE)

label_column = 'Class'
history = None   

map_classes = {j:i for i,j in enumerate(list(set(train['Class'].values)))}

train['Class'] = train['Class'].map(map_classes)
dev['Class'] = dev['Class'].map(map_classes)


stats = {}
for i in range(5):

	for model_name in models:

		history = train_model_dev(model_name=model_name, data_train=train, data_dev=dev, 
								epoches=20, batch_size=32, interm_layer_size = 128, 
								lr = 2e-6 if 'large' in model_name else 1e-5,  decay=1e-6,
								 output='.', task_size=len(map_classes))


		model = SeqModel(interm_size = 128, model=model_name, task_size=len(map_classes))
		model.load('best_model.pt')
		z = predict(model=model, data_dev=dev)

		if model_name not in stats:
			stats[model_name] = []
		stats[model_name].append(f1_score(dev[label_column], z['pred'], average='macro'))

import pickle
with open('transformers.pickle', 'wb') as handle:
	pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)