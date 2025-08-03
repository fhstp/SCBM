from Mc import SeqModel, Data, train_model_dev, evaluate
import pandas as pd, numpy as np, pickle
from collections import Counter
from tqdm import tqdm

import numpy as np, pandas as pd
import pickle
import seaborn as sns

train_file_name = '../germeval/train.csv'
test_file_name = '../germeval/test.csv'

with open(f'{train_file_name}.pickle', 'rb') as handle:
    data_transformation = pickle.load(handle)
    mapping_train = {data_transformation['id'][i]:np.array(data_transformation['values'][i]) for i in range(len(data_transformation['id']))}


with open(f'{test_file_name}.pickle', 'rb') as handle:
    data_transformation = pickle.load(handle)
    mapping_test = {data_transformation['id'][i]:np.array(data_transformation['values'][i]) for i in range(len(data_transformation['id']))}

test = pd.read_csv(test_file_name, sep=',')
train = pd.read_csv(train_file_name, sep=',')


feature_vectors_train = []
labels_train = []
texts_train = []

feature_vectors_test = []
labels_test = []
texts_test = []


for _, row in train.iterrows():
    feature_vectors_train += [mapping_train[row['id']]]
    labels_train += [row['Class']]
    # texts_train += [ row['Context'] + row['Text']]
    texts_train += [ row['text']]


for _, row in test.iterrows():
    feature_vectors_test += [mapping_test[row['id']]]
    labels_test += [row['Class']]
    texts_test += [row['text']]
    # texts_test += [row['Context'] + row['Text']]


feature_vectors_train = np.array(feature_vectors_train)
feature_vectors_test = np.array(feature_vectors_test)

performance = {'train': [], 'test': []}
feature_sizes = []

for feature_size in tqdm(sorted(list( set(range(1, 244, 8)) | {244}))):
	

	feature_sizes.append(feature_size)
	performance['train'] += [[]]
	performance['test'] += [[]]
	for observation in range(30):

		active_features = np.random.permutation(244)[:feature_size]

		class_map = {c:i for i, c in enumerate(Counter(train['Class']).keys())}

		_ = train_model_dev(data_train=train, data_dev=test, epoches=100, batch_size=128,
                        interm_layer_size = 128, lr = 2e-3,  decay=1e-6, output='weights_germeval', task='Class', 
                        classes_len = len(Counter(train['Class']).keys()), output_name = f"fs={feature_size}",
                        mapping_feat = {'train': mapping_train, 'test' : mapping_test},
                        class_map = class_map, feature_mask = active_features, verbose = False)

		model = SeqModel(interm_size = 128,  classes_len = len(Counter(train['Class']).keys()), 
				   feature_size = feature_size)

		model.load(f"weights_germeval/fs={feature_size}.pt", verbose=False)

		test_performance = evaluate(model, task = 'Class', data_dev = test, bs = 8, 
									mapping_feat = {'train': mapping_train, 'test' : mapping_test},
										class_map = class_map, feature_mask=active_features, 
										verbose = False)

		train_performance = evaluate(model, task = 'Class', data_dev = train, bs = 8, 
									mapping_feat = {'test': mapping_train, 'train' : mapping_test},
										class_map = class_map, feature_mask=active_features, 
										verbose = False)
		del model
		# print(f'Average Performance\nTrain:{train_performance:.4f} \nTest: {test_performance:.4f}')
		performance['train'][-1] += [train_performance]
		performance['test'][-1] += [test_performance]
	with open(f'performance_CONAN.pickle', 'wb') as handle:
		pickle.dump({'performance': performance, 'feature_sizes': feature_sizes}, handle, protocol=pickle.HIGHEST_PROTOCOL)