#%%
import os
from collections import Counter
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0'
 
  
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from transformers import AutoModel, AutoTokenizer, BloomForCausalLM
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, mean_squared_error
import torch, random, numpy as np, os
from tqdm import tqdm
import pickle

import pandas as pd

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


def HugginFaceLoad(model_name):

  if 'bloom' in model_name:
    model = BloomForCausalLM.from_pretrained(model_name)
  else:
    model = AutoModel.from_pretrained(model_name)
    
  tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, TOKENIZERS_PARALLELISM=True)

  return model, tokenizer


def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

class Data(Dataset):

  def __init__(self, data, mapping, class_map = None):

    self.data = data
    self.mapping = mapping
    self.class_map = class_map
    
  def __len__(self):
    return len(self.data['Class'])

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()
      
    ret = {key: self.data.iloc[idx][key] if key != 'Class' else self.class_map[self.data.iloc[idx][key]] for key in self.data.keys()}
    ret['features'] = torch.tensor(self.mapping[ret['id']], dtype=torch.float32)
    
    # ret['text'] = self.data.iloc[idx]['Context']   + " [SEP] " + self.data.iloc[idx]['Text']      
 
    return ret
   

class LossFunction(torch.nn.Module):

  def __init__(self, classes_len = 3):
    super(LossFunction, self).__init__()
    
    self.loss = torch.nn.CrossEntropyLoss()
    self.classes_len = classes_len

      
  def forward(self, outputs, labels, masks = None):

    if masks is not None:
      
      z = [ torch.where(labels == i) for i in range(self.classes_len)] #torch.stack()
      z = torch.stack([masks[i].mean( dim = 0 ) for i in z if len(i[0]) >= 1])

      z = torch.sum(z @ z.T)*1./z.size(-1)
      return self.loss(outputs, labels), z

    return self.loss(outputs, labels)

class SeqModel(torch.nn.Module):

  def __init__(self, interm_size, classes_len):

    super(SeqModel, self).__init__()
		
    self.best_acc = None
    self.max_length = 256
    self.interm_neurons = interm_size
    
    self.normalize_features = torch.nn.LayerNorm(244)
    self.relevance_gate = torch.nn.Sequential(torch.nn.Linear(in_features=244, out_features=244), torch.nn.Sigmoid())

    self.intermediate_plus = torch.nn.Sequential(
                                            torch.nn.Linear(in_features=244, out_features=self.interm_neurons>>1),
                                            torch.nn.LeakyReLU())
    
    self.classifier_plus = torch.nn.Linear(in_features=self.interm_neurons>>1, out_features=classes_len)
    self.loss_criterion = LossFunction()
    
    self.device = torch.device("cuda") if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else torch.device("cpu"))
    self.to(device=self.device)

  def forward(self, data, get_mask = False):
    
    feat = self.normalize_features(data['features'].to(self.device))
    mask = self.relevance_gate(feat)
    feat = feat*mask
    
    enc = self.intermediate_plus(feat)
    output = self.classifier_plus(enc)

    if get_mask:
      return output, mask
    return output 

  def load(self, path):
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Weights Loaded{bcolors.ENDC}") 
    self.load_state_dict(torch.load(path, map_location=self.device), strict=True)

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):
    return torch.optim.RMSprop(self.parameters(), lr, weight_decay=decay)

def measurement(running_stats, task):
    
    p = torch.max(running_stats['outputs'], 1).indices.detach().cpu()
    l = running_stats['labels'].detach().cpu()
    score = f1_score(l, p, average='macro')

    return score


def train_model(model, trainloader, devloader, epoches, lr, decay, output, task, output_name = None):
  
    eloss, eacc, edev_loss, edev_acc = [], [], [], []

    optimizer = model.makeOptimizer(lr=lr, decay=decay)
    batches = len(trainloader)

    for epoch in range(epoches):

        running_stats = {'outputs':None, 'labels':None, 'masks_l2': None}
        model.train()

        itera = tqdm(enumerate(trainloader, 0), total = len(trainloader))
        itera.set_description(f'Epoch: {epoch:3d}')

        for j, data in itera:

            if model.device == 'mps':
               torch.mps.empty_cache()
            elif model.device == 'cuda':
              torch.cuda.empty_cache()         

            labels = data[task].to(model.device)    
            
            optimizer.zero_grad()
            outputs, masks = model(data, get_mask=True)

            loss, loss_masks = model.loss_criterion(outputs, labels, masks)

            if running_stats['outputs'] is None:
                running_stats['outputs'] = outputs.detach().cpu()
                running_stats['labels'] = data[task]
                running_stats['masks_l2'] = loss_masks
            else:
                running_stats['outputs'] = torch.cat((running_stats['outputs'], outputs.detach().cpu()), dim=0)
                running_stats['labels'] = torch.cat((running_stats['labels'], data[task]), dim=0)
                running_stats['masks_l2'] = (running_stats['masks_l2'] + loss_masks)/2

            loss += loss_masks
            loss.backward()
            optimizer.step()
            del outputs
            del loss

            train_loss = model.loss_criterion(running_stats['outputs'], running_stats['labels']).item()
            train_measure = measurement(running_stats, task)
            itera.set_postfix_str(f"loss:{train_loss:.3f} measure:{train_measure:.3f} masks_l2:{running_stats['masks_l2']:.3f}") 

            if j == batches-1:
                eloss += [train_loss]
                eacc += [train_measure]

                model.eval()
                with torch.no_grad():
                    
                    running_dev = {'outputs': None, 'labels': None}
                    for k, data_batch_dev in enumerate(devloader, 0):
                        
                        if model.device == 'mps':
                            torch.mps.empty_cache()
                        elif model.device == 'cuda':
                          torch.cuda.empty_cache() 
  
                        outputs, masks = model(data_batch_dev, get_mask=True)
                        
                        if running_dev['outputs'] is None:
                            running_dev['outputs'] = outputs.detach().cpu()
                            running_dev['labels'] = data_batch_dev[task]
                        else:
                            running_dev['outputs'] = torch.cat((running_dev['outputs'], outputs.detach().cpu()), dim=0)
                            running_dev['labels'] = torch.cat((running_dev['labels'], data_batch_dev[task]), dim=0)

                    dev_loss = model.loss_criterion(running_dev['outputs'], running_dev['labels'])
                    dev_loss = dev_loss.item()
                    # masks_l2 = masks_l2.item()
                    dev_measure = measurement(running_dev, task)
                    
                if model.best_acc is None or model.best_acc < dev_measure:
                    model.save(os.path.join(output, f"{output_name}.pt"))
                    model.best_acc = dev_measure

                itera.set_postfix_str(f"loss:{train_loss:.3f} measure:{train_measure:.3f} masks_l2:{running_stats['masks_l2']:.3f} dev_loss:{dev_loss:.3f} dev_measure: {dev_measure:.3f}") 
                edev_loss += [dev_loss]
                edev_acc += [dev_measure]
    return {'loss': eloss, 'acc': eacc, 'dev_loss': edev_loss, 'dev_acc': edev_acc}
        

def train_model_dev(data_train, data_dev, task = 'classification', epoches = 4, batch_size = 32, 
                    interm_layer_size = 12, lr = 1e-5,  decay=2e-5, output='logs', classes_len = -1,
                    output_name = None, mapping_feat = None, class_map = None):

  history = []

  history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
  model = SeqModel(interm_layer_size, classes_len)
  
  print(model)

  trainloader = DataLoader(Data(data_train, mapping_feat['train'], class_map), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
  devloader = DataLoader(Data(data_dev, mapping_feat['test'], class_map), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)

  history.append(train_model(model, trainloader, devloader, epoches, lr, decay, output, task, output_name = output_name))

  del trainloader
  del model
  del devloader
  return history

def evaluate(model, task, data_dev, bs = 32, mapping_feat = None, class_map = None):
  
    devloader = DataLoader(Data(data_dev, mapping_feat['test'], class_map), batch_size=bs, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
    itera = tqdm(enumerate(devloader, 0), total = len(devloader))

    running_stats = {'out':None, 'label':None}
    for j, data in itera:

        if model.device == 'mps':
            torch.mps.empty_cache()
        elif model.device == 'cuda':
          torch.cuda.empty_cache()            
        outputs = model(data)

        running_stats['out'] = outputs.detach().cpu() if running_stats['out'] is None else  torch.cat((running_stats['out'], outputs.detach().cpu()), dim=0)
        running_stats['label'] = data[task]  if running_stats['label'] is None else torch.cat((running_stats['label'], data[task]), dim=0)

    return f1_score(running_stats['label'].numpy(), running_stats['out'].max(dim=-1).indices.numpy(), average= 'macro')

    
#%%

if __name__ ==  '__main__':

    test_performance = []
    for i in range(5):

        train_file_name = '../conan/train.csv'
        test_file_name = '../conan/test.csv'

        train = pd.read_csv(train_file_name, sep=',').fillna('-1')
        test = pd.read_csv(test_file_name, sep=',').fillna('-1')


        with open(f'{train_file_name}.pickle', 'rb') as handle:
            data_transformation = pickle.load(handle)
            mapping_train = {data_transformation['id'][i]:np.array(data_transformation['values'][i]) for i in range(len(data_transformation['id']))}

        with open(f'{test_file_name}.pickle', 'rb') as handle:
            data_transformation = pickle.load(handle)
            mapping_test = {data_transformation['id'][i]:np.array(data_transformation['values'][i]) for i in range(len(data_transformation['id']))}


        print(len(Counter(train['Class']).keys()))


        class_map = {c:i for i, c in enumerate(Counter(train['Class']).keys())}
        _ = train_model_dev(data_train=train, data_dev=test, epoches=250, batch_size=128,
                        interm_layer_size = 128, lr = 2e-3,  decay=1e-6, output='.', task='Class', 
                        classes_len = len(Counter(train['Class']).keys()), output_name = "Mc+L",
                        mapping_feat = {'train': mapping_train, 'test' : mapping_test},
                        class_map = class_map)


        model = SeqModel(interm_size = 128,  classes_len = len(Counter(train['Class']).keys()))
        model.load(f"Mc+L.pt")

        test_performance += [evaluate(model, task = 'Class', data_dev = test, bs = 8, 
                                        mapping_feat = {'train': mapping_train, 'test' : mapping_test},
                                        class_map = class_map)]

        del model
    # import pickle
    # with open('../conan/Mc+L.pickle', 'wb') as handle:
    #     pickle.dump(test_performance, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # %%
