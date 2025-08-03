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
    
def get_inference(context, comment):

    prefix = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert in social psychology. When you are asked a question, you prefer to give short, concrete answers.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nConsider what a person "A" states when reacting to a "CONTEXT":\n\n"CONTEXT":"""
    
    prefix += f'{context}\n"A": {comment}\nDoes response message from "A" seem to be a troll?. Answer in a single word "troll" or "no-troll".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    
    z = pipe(prefix, 
             return_full_text=False,
              max_new_tokens=5, 
            #  temperature=1e-4, 
             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    # print(z)
    return z[0]['generated_text'], 1

import pandas as pd
from tqdm import tqdm
import traceback, pickle


FILE_PATH = "test.csv"

df = pd.read_csv(FILE_PATH, sep=',')
ids = df['id'].tolist()

comment = df['text'].tolist()
context = df['context'].tolist()


runs = []
for i in range(4):

    print(f"{bcolors.OKGREEN}Running inference for {len(comment)} instances!{bcolors.ENDC}")

    itera = tqdm(enumerate(zip(context, comment)), total = len(context))
    runs += [{'id':[], 'values': []}]
    for j, (cont, com) in itera:
        
        probabilities = get_inference(comment = cont, context = com)
        # print(probabilities)
        runs[-1]['values'] += [probabilities]
        runs[-1]['id'] += [ids[j]]
        
    with open(f'{FILE_PATH}.llama31.pickle', 'wb') as handle:
        pickle.dump(runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

