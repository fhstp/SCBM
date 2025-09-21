from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import LlamaForCausalLM, pipeline

import pickle, os
import traceback
from fire import Fire

import os, sys, dotenv
sys.path.append('../..')
dotenv.load_dotenv()

repository = "meta-llama/Llama-3.1-8B-Instruct"#
model_id=repository.split("/")[-1]


os.system(f'git clone --progress --verbose https://{os.getenv("HF_USER")}:{os.getenv("HF_TOKEN")}@huggingface.co/{repository} /workspace/{model_id}')

pipe = pipeline("text-generation", model=f"/workspace/{model_id}", device="cuda")

tokenizer = AutoTokenizer.from_pretrained(f"/workspace/{model_id}")

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
    
def get_inference(comment, classes, context = None):

    if context is None:
        prefix = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert in social psychology. When you are asked a question, you prefer to give short, concrete answers.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nConsider what a person "A" states: """
        prefix += f'"A": {comment}\nDoes "A" seem to be {",".join(classes[:-1]) + ", or " + classes[-1]}. Answer with a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    else:
        prefix = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert in social psychology. When you are asked a question, you prefer to give short, concrete answers.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nConsider what a person "A" states when reacting to a "CONTEXT":\n\n"CONTEXT":"""
        prefix += f'{context}\n"A": {comment}\nDoes response message from "A" seem to be one of [{", ".join(classes)}]?. Answer in a single word.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    
    z = pipe(prefix, 
             return_full_text=False,
              max_new_tokens=5, 
             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    
    return z[0]['generated_text']

import pandas as pd
from tqdm import tqdm
import traceback, pickle


def main(file_path: str):

    df = pd.read_csv(file_path, sep=',')
    ids = df['id'].tolist()
    classes = sorted(list(set(df['Class'].values)))
    comment = df['text'].tolist()
    
    if 'context' in df.columns:
        context = df['context'].tolist()
    else:
        context = ['']*len(comment)


    runs = []
    for _ in range(4):

        print(f"{bcolors.OKGREEN}Running inference for {len(comment)} instances!{bcolors.ENDC}")

        itera = tqdm(enumerate(zip(context, comment)), total = len(context))
        runs += [{'id':[], 'values': []}]
        for j, (cont, com) in itera:
            
            if 'context' in df.columns:
                prediction = get_inference(comment = com, context = cont, classes=classes)
            else:
                prediction = get_inference(comment = com, classes=classes)

            runs[-1]['values'] += [prediction]
            runs[-1]['id'] += [ids[j]]
            
        with open(f'{file_path}.llama31.pickle', 'wb') as handle:
            pickle.dump(runs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    Fire(main)