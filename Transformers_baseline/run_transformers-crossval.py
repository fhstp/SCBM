from fire import Fire
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from models import train_model_dev
from sklearn.metrics import f1_score
from models import predict, SeqModel
import pickle


def main(train_file: str, output_file: str):

	"""
	Train and evaluate transformer models on a given training and development dataset.
	Run 5 training runs to account for variance in model performance for each model and save F1 scores.

	:param train_file: Path to the CSV file containing the training data (expects columns: id, text, Class).
	:param dev_file: Path to the CSV file containing the development data (expects columns: id, text, Class).
	:param output_file: Path to save the pickle file containing F1 scores for each model across runs.
	
	"""

	train = pd.read_csv(train_file)
	mapping = sorted(list(set(train['Class'].values)))
	print(mapping)
	train['Class'] = train['Class'].map(lambda x: mapping.index(x))

	label_column = 'Class'

	models = ['FacebookAI/xlm-roberta-base',
			'FacebookAI/xlm-roberta-large',
			'google-bert/bert-base-uncased',
			'google-bert/bert-large-uncased',]

	stats = {}

	skf = StratifiedKFold(n_splits=5, shuffle=True)

	for _, (train_index, test_index) in enumerate(skf.split(train['text'], train['Class'])):

		for model_name in models:

			dev = train.iloc[test_index]
			ttrain = train.iloc[train_index]

			history = train_model_dev(model_name=model_name, data_train=ttrain,
									data_dev=dev,
									epoches=12, batch_size=32, interm_layer_size = 128, 
									lr = 2e-6 if 'large' in model_name else 1e-5,  
									decay=1e-6, output='.', 
									task=label_column,
									n_classes=len(mapping))


			model = SeqModel(interm_size = 128, model=model_name, task='offensive', n_classes=len(mapping))
			model.load('best_model.pt')
			z = predict(model=model, data_dev=dev, outputfile='transformers')

			if model_name not in stats:
				stats[model_name] = []
			stats[model_name].append(f1_score(dev[label_column], z['pred'], average='macro'))

	# Save results
	with open(output_file, 'wb') as handle:
		pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    Fire(main)
