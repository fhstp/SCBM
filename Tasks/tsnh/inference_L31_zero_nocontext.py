from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import LlamaForCausalLM, pipeline

import torch
import traceback
import pickle, os

import sys, dotenv
sys.path.append('../..')
dotenv.load_dotenv()

repository = "meta-llama/Llama-3.1-8B-Instruct"#
model_id=repository.split("/")[-1]


os.system(f'git clone --progress --verbose https://{os.getenv("HF_USER")}:{os.getenv("HF_TOKEN")}@huggingface.co/{repository} /workspace/{model_id}')

pipe = pipeline("text-generation", model=f"/workspace/{model_id}", device="cuda")
tokenizer = AutoTokenizer.from_pretrained(f"/workspace/{model_id}")
print('Zero-shot Inference with no context!!!!!!!!!')


import json
import numpy as np
import traceback

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
    
def get_inference(sentence):

    prefix = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert in social psychology. When you are asked a question, you prefer to give short, concrete answers.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nConsider what a person "A" states: """
    
    prefix += f'"A": {sentence}\nDoes "A" seem to be counterspeech. Answer in a single word "counterspeech" or "no-counterspeech".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    
    z = pipe(prefix, 
             return_full_text=False,
              max_new_tokens=5, 
             temperature=1e-4, 
             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    # print(z)
    return z[0]['generated_text'], 1

def majority_vote(prompt, votes_num = 1):
    o = []
    votes = []
    for i in range(votes_num):
        
        z = get_inference(prompt)
        if z[0] != "unknown":
            votes += [z[0]]
        o += [z[1]]
        
    if len(votes) == 0:
        votes += ["unknown"]
    return max(set(votes), key=votes.count), np.mean(o)

import pandas as pd
from tqdm import tqdm
import traceback, pickle


FILE_PATH = "TSNH_uniform.csv"

df = pd.read_csv(FILE_PATH, sep=',')
ids = df['id'].tolist()

messages = df['text'].tolist()


runs = []
for i in range(4):

    print(f"{bcolors.OKGREEN}Running inference for {len(messages)} instances!{bcolors.ENDC}")

    itera = tqdm(enumerate(messages), total = len(messages))
    runs += [{'id':[], 'values': []}]
    for j, mess in itera:
        
        probabilities, mean_intents = majority_vote(prompt = mess)
        # print(probabilities)
        runs[-1]['values'] += [probabilities]
        runs[-1]['id'] += [ids[j]]

        itera.set_postfix(mean_intents=mean_intents)
        
    with open(f'{FILE_PATH}.llama31.pickle', 'wb') as handle:
        pickle.dump(runs, handle, protocol=pickle.HIGHEST_PROTOCOL)